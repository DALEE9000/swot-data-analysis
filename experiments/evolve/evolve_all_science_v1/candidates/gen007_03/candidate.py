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
    """Standardize, center-fill missing stencil cells, append missingness + spatial std.

    Unchanged proven plumbing from the parent lineage:
      * per-cell missing fraction (n, C) — distinguishes swath-edge padding
        from a genuinely calm/average ocean state;
      * per-feature stencil standard deviation (n, F) over VALID cells on the
        standardized values — an explicit front/eddy intensity signal (and
        exactly the regime information the MoE gate needs);
      * center-cell imputation — a missing stencil cell is filled with that
        feature's CENTER-cell value ("locally uniform field", the physically
        neutral assumption) rather than the global mean; rows whose center is
        itself NaN fall back to the mean (0 after standardization) via the
        final nan_to_num.
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
    # Center-fill AFTER the std block (which must see true valid cells).
    center = Z[:, :, n_cells // 2]
    for c in range(n_cells):
        if c == n_cells // 2:
            continue
        col = Z[:, :, c]
        hole = np.isnan(col)
        if hole.any():
            col[hole] = center[hole]
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return np.hstack([Xs, miss, spat])


class MoENet(nn.Module):
    """Gated mixture-of-experts over flow regimes.

    Shared trunk -> (a) softmax gate over K experts, (b) K small expert MLPs
    each emitting an (u, v) pair. Output is the gate-weighted sum of expert
    outputs. Soft (dense) routing: every expert runs on every row, so the
    model stays a smooth function of the inputs and trains with plain SGD —
    no routing discreteness tricks needed at this K. The intent is regime
    specialization: slow shelf flow, mid, and fast front/eddy rows have
    genuinely different feature->velocity mappings, and the gate (seeing the
    stencil-std front-intensity block and geostrophic magnitudes through the
    trunk) can dedicate an expert to the steep fast-regime mapping instead of
    averaging it into the calm-ocean majority fit.
    """

    def __init__(self, n_inputs, n_experts=4, trunk_width=256,
                 expert_hidden=128, dropout=0.1, gate_temp=1.0):
        super().__init__()
        self.gate_temp = gate_temp
        self.trunk = nn.Sequential(
            nn.Linear(n_inputs, trunk_width), nn.LayerNorm(trunk_width),
            nn.SiLU(), nn.Dropout(dropout),
        )
        self.gate = nn.Linear(trunk_width, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(trunk_width, expert_hidden), nn.LayerNorm(expert_hidden),
                nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(expert_hidden, 2),
            )
            for _ in range(n_experts)
        ])
        # Near-uniform gate at init so all experts receive gradient early.
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)

    def forward(self, x):
        z = self.trunk(x)
        g = torch.softmax(self.gate(z) / self.gate_temp, dim=-1)   # (B, K)
        outs = torch.stack([e(z) for e in self.experts], dim=1)    # (B, K, 2)
        y = (g.unsqueeze(-1) * outs).sum(dim=1)                    # (B, 2)
        return y, g


def _masked_mse(pred, target, mask):
    diff = (pred - target) * mask
    return diff.pow(2).sum() / mask.sum().clamp(min=1)


def _load_balance(g):
    # Squared mean usage per expert, scaled so a uniform gate scores 1.0.
    # Penalizing it pushes average usage toward uniform and prevents the
    # classic soft-MoE failure of one expert absorbing everything.
    K = g.shape[1]
    return K * (g.mean(dim=0) ** 2).sum()


def _train_member(member_seed, deadline, device, max_epochs,
                  X_t, Y_t, M_t, X_v, Y_v, M_v, X_v_cpu, Y_v_cpu, M_v_cpu):
    """Train one MoE ensemble member (early stop on temporal-val masked MSE,
    then the proven recency fine-tune) within its wall-clock slice.
    Returns (net, best_val)."""
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)

    net = MoENet(X_t.shape[1], n_experts=4, trunk_width=256,
                 expert_hidden=128, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    bs = 4096
    patience = 15
    lb_weight = 0.01
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
            pred, g = net(xb)
            loss = _masked_mse(pred, yb, mb) + lb_weight * _load_balance(g)
            loss.backward()
            optimizer.step()

        # Model selection and LR scheduling on masked MSE alone — the scored
        # quantity — not the load-balance-regularized training loss.
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
                pred, g = net(xb)
                loss = _masked_mse(pred, yb, mb) + lb_weight * _load_balance(g)
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
    # the quantity the test actually measures.
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

    # Two-seed deep ensemble with per-member wall-clock slices (unchanged
    # parent plumbing). Member m must finish by 0.95*(m+1)/M of the budget,
    # leaving 5% for test prediction.
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
