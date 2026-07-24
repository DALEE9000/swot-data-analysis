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
    """Standardize, gradient-fill missing stencil cells, append missingness +
    spatial std + directional stencil gradients.

    Feature-major layout: column f * n_cells + c is feature f at stencil
    cell c, so reshaping to (n, F, C) groups each feature's spatial
    neighborhood. Ingredients (all proven in the lineage):
      * per-cell missing fraction (n, C) — distinguishes swath-edge padding
        from a genuinely calm/average ocean state;
      * per-feature stencil standard deviation (n, F) over VALID cells only —
        an explicit front/eddy intensity signal (and here, a direct regime
        cue for the mixture gate);
      * gradient-preserving imputation: a missing stencil cell at offset
        (di, dj) from the center is filled with 2*center - mirror (first-order
        Taylor continuation), clamped to +/-4 sigma, falling back to the
        center where the mirror is also missing;
      * directional gradients (n, 2F) computed AFTER imputation: per feature,
        mean right-minus-left column difference (zonal) and top-minus-bottom
        row difference (meridional), each / (k-1) — the discrete derivatives
        geostrophy runs on.
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
    # Directional gradients on the imputed (still-standardized) stencil.
    k = int(round(math.sqrt(n_cells)))
    if k >= 2 and k * k == n_cells:
        Zk = Z.reshape(n, F, k, k)  # row-major: axis 2 = rows, axis 3 = cols
        gx = ((Zk[:, :, :, k - 1] - Zk[:, :, :, 0]).mean(axis=2)
              / float(k - 1)).astype(np.float32)
        gy = ((Zk[:, :, 0, :] - Zk[:, :, k - 1, :]).mean(axis=2)
              / float(k - 1)).astype(np.float32)
        grad = np.hstack([gx, gy])
    else:
        grad = np.zeros((n, 0), dtype=np.float32)
    grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return np.hstack([Xs, miss, spat, grad])


class MoEMLP(nn.Module):
    """Shared trunk + soft-gated mixture of small expert heads.

    One flat MLP must serve slow interior rows, fast fronts/eddies, and
    swath-edge rows with a single set of weights, so gradient updates from
    the dominant slow regime wash out what the rare fast rows need. Here a
    shared trunk learns common representation, a softmax gate learns a
    regime assignment from that representation (the spatial-std and
    directional-gradient inputs are direct regime signals), and each expert
    head fits the (u, v) map for its soft cluster of rows. Dense soft gating
    (all experts evaluated, E=4 heads of 128 units) keeps cost trivial and
    training smooth — no routing discreteness, fully differentiable.

    forward returns (pred, gates): pred (n, 2) is the gate-weighted mixture
    prediction; gates (n, E) are the softmax weights, exposed for the
    load-balance penalty.
    """

    def __init__(self, n_inputs, trunk_hidden=(256, 256), n_experts=4,
                 expert_hidden=128, dropout=0.1):
        super().__init__()
        layers = []
        d = n_inputs
        for h in trunk_hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.gate = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, expert_hidden), nn.LayerNorm(expert_hidden),
                nn.SiLU(), nn.Dropout(dropout), nn.Linear(expert_hidden, 2))
            for _ in range(n_experts)])
        # Zero-init the gate so training starts as a uniform mixture (an
        # implicit ensemble of E heads); specialization emerges only as the
        # gate finds regime structure worth splitting on.
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x):
        z = self.trunk(x)
        gates = torch.softmax(self.gate(z), dim=1)          # (n, E)
        outs = torch.stack([e(z) for e in self.experts], 1)  # (n, E, 2)
        pred = (gates.unsqueeze(-1) * outs).sum(dim=1)       # (n, 2)
        return pred, gates


def _masked_mse(pred, target, mask):
    diff = (pred - target) * mask
    return diff.pow(2).sum() / mask.sum().clamp(min=1)


def _moe_loss(pred, gates, target, mask, balance_coef=0.01):
    """Masked MSE on the mixture prediction (the scored quantity) plus a
    switch-style importance penalty: E * sum(mean_gate^2) is minimized (=1)
    when average expert usage is uniform, so the gate cannot collapse onto
    one expert and silently reduce to the parent's flat MLP. The penalty acts
    only on BATCH-MEAN usage — individual rows are free to gate hard onto a
    single specialist."""
    mse = _masked_mse(pred, target, mask)
    importance = gates.mean(dim=0)
    balance = gates.shape[1] * (importance * importance).sum()
    return mse + balance_coef * (balance - 1.0)


def _train_member(member_seed, deadline, device, max_epochs,
                  X_t, Y_t, M_t, X_v, Y_v, M_v, X_v_cpu, Y_v_cpu, M_v_cpu):
    """Train one MoE ensemble member (early stop on temporal-val masked MSE,
    then the proven recency fine-tune) within its wall-clock slice.
    Returns (net, best_val)."""
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)

    net = MoEMLP(X_t.shape[1], trunk_hidden=(256, 256), n_experts=4,
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
            optimizer.zero_grad()
            pred, gates = net(xb)
            loss = _moe_loss(pred, gates, yb, mb)
            loss.backward()
            optimizer.step()

        # Model selection and LR scheduling on masked MSE of the mixture
        # prediction — the scored quantity — NOT the training loss, which
        # mixes in the balance penalty.
        net.eval()
        with torch.no_grad():
            val_losses = []
            for i in range(0, len(X_v), bs):
                pred, _ = net(X_v[i:i + bs])
                val_losses.append(_masked_mse(pred, Y_v[i:i + bs], M_v[i:i + bs]).item())
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
    # with a short low-LR pass over the FULL window (tail included).
    if time.time() < deadline and np.isfinite(best_val):
        X_f = torch.cat([X_t, X_v_cpu])
        Y_f = torch.cat([Y_t, Y_v_cpu])
        M_f = torch.cat([M_t, M_v_cpu])
        ft_opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
        n_all = len(X_f)
        out_of_time = False
        net.train()
        for _ in range(2):
            order = torch.from_numpy(rng.permutation(n_all))
            for bi, i in enumerate(range(0, n_all, bs)):
                idx = order[i:i + bs]
                xb, yb, mb = X_f[idx].to(device), Y_f[idx].to(device), M_f[idx].to(device)
                ft_opt.zero_grad()
                pred, gates = net(xb)
                loss = _moe_loss(pred, gates, yb, mb)
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

    X_t, Y_t, M_t = (torch.from_numpy(a[train_idx]) for a in (Xs, Ys, mask))
    # Keep CPU copies of the validation tail for the post-selection fine-tune.
    X_v_cpu = torch.from_numpy(Xs[val_idx])
    Y_v_cpu = torch.from_numpy(Ys[val_idx])
    M_v_cpu = torch.from_numpy(mask[val_idx])
    X_v = X_v_cpu.to(device)
    Y_v = Y_v_cpu.to(device)
    M_v = M_v_cpu.to(device)
    del Xs

    # Two-seed deep ensemble with per-member wall-clock slices. Independently
    # seeded MoEs make decorrelated errors — including in their learned gate
    # partitions — and averaging their predictions cancels seed-dependent
    # variance. Member m must finish by 0.95*(m+1)/M of the budget, leaving
    # 5% for test prediction.
    M_ens = 2
    members = []
    for m in range(M_ens):
        deadline = t0 + time_budget_s * 0.95 * (m + 1) / M_ens
        if time.time() > deadline:
            break
        net, best_val = _train_member(
            seed + 101 * m, deadline, device, max_epochs,
            X_t, Y_t, M_t, X_v, Y_v, M_v, X_v_cpu, Y_v_cpu, M_v_cpu)
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

    # Predict on the test set in batches, averaging over accepted members.
    out = np.zeros((len(X_test), 2), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_test), 65536):
            xb = torch.from_numpy(_featurize(X_test[i:i + 65536], mean, scale, n_cells)).to(device)
            acc = torch.zeros((xb.shape[0], 2), device=device)
            for net in keep:
                pred, _ = net(xb)
                acc += pred
            out[i:i + 65536] = (acc / len(keep)).cpu().numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
