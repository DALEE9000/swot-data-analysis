"""SWOTxAI evolve candidate.

CONTRACT — the harness calls exactly this function; keep the signature:

    train_and_predict(X_train, Y_train, X_test, params) -> np.ndarray

  X_train : float32 (n_train, d) feature matrix; may contain NaN (stencil
            padding at swath edges).
  Y_train : float32 (n_train, 2) targets (u, v) in m/s; NaN marks an invalid
            component (train on the valid entries only).
  X_test  : float32 (n_test, d).
  params  : dict with keys:
              seed          - int, seed all RNGs with this
              device        - "cuda" or "cpu"
              max_epochs    - int, do not train longer than this many epochs
              time_budget_s - float, soft wall-clock budget for training

  Returns predictions for X_test, shape (n_test, 2), finite everywhere.

Everything below the contract may be rewritten freely. Allowed imports:
numpy, torch, math, random, time, copy, typing, dataclasses, collections,
itertools, functools, warnings, sklearn, scipy. No file, network, or OS access.
"""
import copy
import math
import time
import warnings

import numpy as np
import torch
import torch.nn as nn


def _cells(d):
    # Columns are k*k spatial-stencil copies of base features, feature-major.
    # Infer the stencil cell count so the appended blocks stay layout-correct.
    for c in (9, 25, 49):
        if d % c == 0:
            return c
    return 1


def _featurize(X, mean, scale, n_cells):
    """Standardize, gradient-fill missing stencil cells, append missingness,
    spatial std, and explicit per-feature spatial derivatives.

    Feature-major layout: column f * n_cells + c is feature f at stencil
    cell c, so reshaping to (n, F, C) groups each feature's spatial
    neighborhood. Ingredients:
      * per-cell missing fraction (n, C) — lets the gate and experts
        distinguish swath-edge padding from a genuinely calm/average ocean
        state (edge rows are the widest R^2 gap in the diagnostics);
      * per-feature stencil standard deviation (n, F), computed on the
        standardized values over VALID cells only — an explicit front/eddy
        intensity signal, and prime gating information for regime routing.
        Rows with a single valid cell get std 0 ("no local gradient
        information");
      * gradient-preserving imputation: a missing stencil cell at offset
        (di, dj) from the center is filled with 2*center - mirror, where
        mirror is the cell at (-di, -dj) — the first-order Taylor (constant
        local gradient) continuation of the field. Swath-edge padding
        truncates one SIDE of the neighborhood, so the mirror cell is
        usually valid; center-fill (the previous scheme) flattened the
        neighborhood to "locally uniform", erasing exactly the gradient
        signal the geostrophic features and spatial-std block carry.
        Extrapolated fills are clamped to +/-4 standardized sigma so edge
        noise is not amplified; where the mirror is itself missing the fill
        falls back to the center value, and rows whose center is NaN fall
        back to the mean (0 after standardization) via the final nan_to_num.
        The missingness block still marks which cells were padded, so the
        model can discount imputed structure;
      * per-feature central differences (n, 2F): gx = (right - left)/2 and
        gy = (down - up)/2 across the stencil, computed AFTER imputation.
        These are the finite-difference operators a stencil CNN would learn
        as 3x3 kernels — the along/across-track gradients of SSH and the
        geostrophic velocity components, whose cross-feature linear combos
        are shear, strain, divergence and vorticity, i.e. the front/eddy
        dynamics behind the fast-regime deficit. Handing them to the network
        as first-class inputs means the first Linear layer can form those
        physical combos directly instead of rediscovering differencing from
        anonymous columns. The mirror imputation is Taylor-consistent with a
        central difference (2*center - mirror keeps the linear gradient), so
        edge rows get coherent, conservatively damped derivatives; the
        missingness block tells the net when to trust them less.
    """
    Xs = (X.astype(np.float32) - mean) / scale
    n, d = Xs.shape
    F = d // n_cells
    miss = np.isnan(X).reshape(n, F, n_cells).mean(axis=1).astype(np.float32)
    Z = Xs.reshape(n, F, n_cells)
    with warnings.catch_warnings():
        # nanstd emits RuntimeWarning on all-NaN neighborhoods; those rows
        # are handled by the nan_to_num below.
        warnings.simplefilter("ignore")
        spat = np.nanstd(Z, axis=2).astype(np.float32)
    spat = np.nan_to_num(spat, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    # Fill AFTER the std block (which must see true valid cells). Snapshot the
    # point-reflected cells before any in-place filling so late cells never
    # read an already-imputed mirror as if it were data.
    center = Z[:, :, n_cells // 2]
    mirror = Z[:, :, ::-1].copy()  # cell c's mirror across the center is C-1-c
    for c in range(n_cells):
        if c == n_cells // 2:
            continue
        col = Z[:, :, c]
        hole = np.isnan(col)
        if hole.any():
            lin = 2.0 * center - mirror[:, :, c]
            fill = np.where(np.isfinite(lin), np.clip(lin, -4.0, 4.0), center)
            col[hole] = fill[hole]
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    blocks = [Xs, miss, spat]
    # Central differences on the filled, finite stencil. Cells are flattened
    # row-major over the k x k window (the same ordering the mirror trick
    # relies on): the center's horizontal neighbors sit at +/-1, vertical
    # neighbors at +/-k. Orientation/sign conventions don't matter to the
    # net — only that the two directions are consistent and orthogonal.
    k = int(round(math.sqrt(n_cells)))
    if k >= 2 and k * k == n_cells:
        Zf = Xs[:, :F * n_cells].reshape(n, F, n_cells)
        ci = n_cells // 2
        gx = (0.5 * (Zf[:, :, ci + 1] - Zf[:, :, ci - 1])).astype(np.float32)
        gy = (0.5 * (Zf[:, :, ci + k] - Zf[:, :, ci - k])).astype(np.float32)
        blocks += [gx, gy]
    return np.hstack(blocks)


class MoENet(nn.Module):
    """Gated mixture-of-experts over flow regimes.

    Shared trunk -> softmax gate + K small expert heads, each predicting
    (u, v); the output is the gate-weighted sum of expert predictions.

    Rationale: the task mixes qualitatively different input->velocity
    mappings — calm interior (slow, smooth), fronts/eddies (fast, gradient-
    driven), and swath-edge rows (partially imputed stencils). A single MLP
    must serve all three with one set of weights, and the diagnostics show
    the compromise: rmse_speed 0.068 slow vs 0.229 fast, r2 edge-stencil
    ~0.17 below full-stencil. Soft gating lets the network partition rows —
    the trunk representation already contains explicit regime signals
    (spatial std, central-difference gradients, missingness fractions), so
    the gate can route fast/edge rows to experts whose weights are not
    shared with (and hence not dragged toward) the easy majority. Dense
    soft mixing (all K experts evaluated, weights from softmax) keeps the
    forward cheap at K=4 and fully differentiable — no routing tricks.
    """

    def __init__(self, n_inputs, n_experts=4, trunk=(256, 256),
                 expert_hidden=128, dropout=0.1):
        super().__init__()
        layers = []
        d = n_inputs
        for h in trunk:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.gate = nn.Sequential(nn.Linear(d, 64), nn.SiLU(), nn.Linear(64, n_experts))
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d, expert_hidden), nn.LayerNorm(expert_hidden),
                          nn.SiLU(), nn.Dropout(dropout), nn.Linear(expert_hidden, 2))
            for _ in range(n_experts)])
        self.n_experts = n_experts

    def forward(self, x):
        z = self.trunk(x)
        g = torch.softmax(self.gate(z), dim=1)                    # (B, K)
        outs = torch.stack([e(z) for e in self.experts], dim=1)   # (B, K, 2)
        mu = (g.unsqueeze(-1) * outs).sum(dim=1)                  # (B, 2)
        return mu, g


def _moe_loss(mu, gate, target, mask, row_w, lb_coef=0.01):
    """Recency-weighted masked MSE plus a load-balancing penalty.

    The MSE term is the scored quantity, masked to valid components and
    scaled by the per-row recency weight (mean ~1, so the loss scale the
    LR schedule was tuned on is unchanged).

    The load-balance term is the switch-transformer importance penalty:
    K * sum(mean_gate_k^2) - 1, which is 0 when the gate uses experts
    uniformly on average and grows as it collapses onto few experts. Soft
    MoEs trained on skewed regime distributions (slow rows vastly outnumber
    fast ones here) otherwise collapse to one expert and degenerate to the
    plain MLP; a small coefficient keeps all experts alive without forcing
    per-row uniformity, so specialization can still be sharp.
    """
    w = mask * row_w.unsqueeze(1)
    mse = ((mu - target).pow(2) * w).sum() / w.sum().clamp(min=1e-8)
    imp = gate.mean(dim=0)
    lb = gate.shape[1] * (imp * imp).sum() - 1.0
    return mse + lb_coef * lb


def _masked_mse(pred, target, mask):
    diff = (pred - target) * mask
    return diff.pow(2).sum() / mask.sum().clamp(min=1)


def _train_member(member_seed, deadline, device, max_epochs,
                  X_t, Y_t, M_t, W_t, X_v, Y_v, M_v,
                  X_v_cpu, Y_v_cpu, M_v_cpu, W_v_cpu):
    """Train one MoE ensemble member with the recency-weighted masked MSE +
    load-balance loss (early stop on UNWEIGHTED temporal-val masked MSE of
    mu — the scored quantity — then the proven recency fine-tune) within its
    wall-clock slice. Returns (net, best_val)."""
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)

    net = MoENet(X_t.shape[1], n_experts=4, trunk=(256, 256),
                 expert_hidden=128, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    bs = 4096
    patience = 15
    best_val = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    best_epoch = 0
    n_train = len(X_t)

    for epoch in range(1, max_epochs + 1):
        net.train()
        order = torch.from_numpy(rng.permutation(n_train))
        for i in range(0, n_train, bs):
            idx = order[i:i + bs]
            xb, yb, mb = X_t[idx].to(device), Y_t[idx].to(device), M_t[idx].to(device)
            wb = W_t[idx].to(device)
            optimizer.zero_grad()
            mu, g = net(xb)
            loss = _moe_loss(mu, g, yb, mb, wb)
            loss.backward()
            optimizer.step()

        # Model selection and LR scheduling on UNWEIGHTED masked MSE of mu —
        # the scored quantity — NOT on the training loss, whose value mixes
        # in the load-balance term and the recency weighting. Keeping
        # validation unweighted means neither the recency emphasis nor gate
        # bookkeeping can game early stopping.
        net.eval()
        with torch.no_grad():
            val_losses = []
            for i in range(0, len(X_v), bs):
                mu, _ = net(X_v[i:i + bs])
                val_losses.append(_masked_mse(mu, Y_v[i:i + bs], M_v[i:i + bs]).item())
            val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = copy.deepcopy(net.state_dict())
            best_epoch = epoch

        if epoch - best_epoch >= patience:
            break
        if time.time() > deadline:
            break

    net.load_state_dict(best_state)

    # Recency fine-tune (proven +~0.01 in lineage): the validation tail is the
    # most recent — and hence most test-like — 10% of the training window, and
    # the model above never trained on it. After model selection, absorb it
    # with a short low-LR pass over the FULL window (tail included), using the
    # same recency-weighted loss so relative row weighting stays consistent
    # (the tail carries the largest weights, as it should).
    if time.time() < deadline and np.isfinite(best_val):
        X_f = torch.cat([X_t, X_v_cpu])
        Y_f = torch.cat([Y_t, Y_v_cpu])
        M_f = torch.cat([M_t, M_v_cpu])
        W_f = torch.cat([W_t, W_v_cpu])
        ft_opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
        n_all = len(X_f)
        out_of_time = False
        net.train()
        for _ in range(2):
            order = torch.from_numpy(rng.permutation(n_all))
            for bi, i in enumerate(range(0, n_all, bs)):
                idx = order[i:i + bs]
                xb, yb, mb = X_f[idx].to(device), Y_f[idx].to(device), M_f[idx].to(device)
                wb = W_f[idx].to(device)
                ft_opt.zero_grad()
                mu, g = net(xb)
                loss = _moe_loss(mu, g, yb, mb, wb)
                loss.backward()
                ft_opt.step()
                if bi % 100 == 0 and time.time() > deadline:
                    out_of_time = True
                    break
            if out_of_time:
                break

    net.eval()
    return net, best_val


def train_and_predict(X_train, Y_train, X_test, params):
    seed = int(params["seed"])
    torch.manual_seed(seed)
    device = torch.device(params["device"])
    max_epochs = int(params["max_epochs"])
    time_budget_s = float(params["time_budget_s"])
    t0 = time.time()

    # Per-column standardization ignoring NaNs; NaN -> 0 (the feature mean).
    mean = np.nan_to_num(np.nanmean(X_train, axis=0), nan=0.0).astype(np.float32)
    scale = np.nanstd(X_train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)

    n_cells = _cells(X_train.shape[1])
    Xs = _featurize(X_train, mean, scale, n_cells)
    mask = np.isfinite(Y_train).astype(np.float32)
    Ys = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)

    # Temporal validation split: rows are time-ordered (later rows = later in
    # the mission window, regions interleaved), and the test set is a temporal
    # holdout. Validating on the last 10% of the training window makes early
    # stopping and the LR schedule optimize forward-in-time generalization —
    # the quantity the test actually measures — instead of random-split
    # interpolation, which stops too late and overfits the training window.
    n = len(Xs)
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    # Recency weights over the time-ordered window: latest rows weigh 3x the
    # earliest (exponential ramp, normalized to mean 1 so the loss scale — and
    # hence the tuned LR/scheduler behavior — is unchanged). The test window
    # is strictly later than every training row and the late-test diagnostics
    # show within-window drift; a mild geometric emphasis keeps the early
    # window as a physics anchor while biasing the fit toward the most
    # test-like conditions.
    pos = (np.arange(n, dtype=np.float32) / max(n - 1, 1)).astype(np.float32)
    w_all = np.exp(np.log(3.0) * pos).astype(np.float32)
    w_all /= w_all.mean()

    X_t, Y_t, M_t = (torch.from_numpy(a[train_idx]) for a in (Xs, Ys, mask))
    W_t = torch.from_numpy(w_all[train_idx])
    # Keep CPU copies of the validation tail for the post-selection fine-tune.
    X_v_cpu = torch.from_numpy(Xs[val_idx])
    Y_v_cpu = torch.from_numpy(Ys[val_idx])
    M_v_cpu = torch.from_numpy(mask[val_idx])
    W_v_cpu = torch.from_numpy(w_all[val_idx])
    X_v = X_v_cpu.to(device)
    Y_v = Y_v_cpu.to(device)
    M_v = M_v_cpu.to(device)
    del Xs

    # Two-seed deep ensemble with per-member wall-clock slices. Independently
    # seeded MoEs make decorrelated errors — including in the regime
    # partitions their gates learn — and averaging their predictions cancels
    # seed-dependent variance. Member m must finish by 0.95*(m+1)/M of the
    # budget, leaving 5% for test prediction.
    M_ens = 2
    members = []
    for m in range(M_ens):
        deadline = t0 + time_budget_s * 0.95 * (m + 1) / M_ens
        if time.time() > deadline:
            break
        net, best_val = _train_member(
            seed + 101 * m, deadline, device, max_epochs,
            X_t, Y_t, M_t, W_t, X_v, Y_v, M_v,
            X_v_cpu, Y_v_cpu, M_v_cpu, W_v_cpu)
        members.append((net, best_val))

    # Guard: drop members starved by a tight budget (val loss >15% worse than
    # the best member) so the worst case degenerates to a single MoE instead
    # of averaging in an undertrained net.
    finite = [(net, v) for net, v in members if np.isfinite(v)]
    if finite:
        v_best = min(v for _, v in finite)
        keep = [net for net, v in finite if v <= v_best * 1.15]
    else:
        keep = [members[0][0]]

    # Predict on the test set in batches, averaging mu over accepted members.
    out = np.zeros((len(X_test), 2), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_test), 65536):
            xb = torch.from_numpy(_featurize(X_test[i:i + 65536], mean, scale, n_cells)).to(device)
            acc = torch.zeros((xb.shape[0], 2), device=device)
            for net in keep:
                mu, _ = net(xb)
                acc += mu
            out[i:i + 65536] = (acc / len(keep)).cpu().numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
