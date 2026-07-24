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

# Base-feature order in the flattened stencil (feature-major blocks of k*k):
# 0 mdt, 1 ssha_filtered, 2 ugos_filtered, 3 vgos_filtered,
# 4 ugosa_filtered, 5 vgosa_filtered, 6 era5_u, 7 era5_v, 8 SST
_N_BASE = 9
_F_UGOS, _F_VGOS, _F_ERA5U, _F_ERA5V = 2, 3, 6, 7


def _augment(X):
    """Append physically-motivated nonlinear features.

    Wind stress on the ocean surface scales ~quadratically with wind speed
    (tau ~ |W| * W), and the wind-driven (Ekman) part of the current — which
    dominates the ageostrophic u component off the US west coast — responds
    to STRESS, not to the raw wind velocity the linear features expose. We
    add per-row: wind speed |W|, pseudo-stress (era5_u*|W|, era5_v*|W|), and
    geostrophic current speed (a nonlinear-advection proxy). Each is computed
    from the NaN-aware mean of that feature's stencil block, so the result is
    independent of the (unknown) within-block cell ordering. Rows where the
    source block is fully NaN yield NaN and are zeroed by standardization.
    """
    d = X.shape[1]
    if d % _N_BASE != 0:
        return X  # unexpected layout — leave the matrix untouched
    k2 = d // _N_BASE

    def block_mean(f):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            return np.nanmean(X[:, f * k2:(f + 1) * k2], axis=1)

    eu = block_mean(_F_ERA5U)
    ev = block_mean(_F_ERA5V)
    wspd = np.sqrt(eu * eu + ev * ev)
    tau_u = eu * wspd
    tau_v = ev * wspd
    gu = block_mean(_F_UGOS)
    gv = block_mean(_F_VGOS)
    gspd = np.sqrt(gu * gu + gv * gv)

    extra = np.stack([wspd, tau_u, tau_v, gspd], axis=1).astype(np.float32)
    return np.concatenate([X.astype(np.float32), extra], axis=1)


def _standardize(X, mean, scale):
    Xs = (X.astype(np.float32) - mean) / scale
    return np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_ridge(Xs, Ys, mask, alpha=10.0):
    """Closed-form ridge per target component on the masked rows.

    Xs is standardized (NaN->0), so features are ~zero-mean; the intercept is
    absorbed by centering y. Returns (B, b): weights (d, 2) and intercepts (2,).
    Exact and deterministic — captures the linear (geostrophic) part of the
    signal on ALL training data with no SGD noise. With the augmented stress
    columns it also captures the leading wind-driven (Ekman) response.
    """
    d = Xs.shape[1]
    B = np.zeros((d, 2), dtype=np.float64)
    b = np.zeros(2, dtype=np.float64)
    chunk = 200_000
    for c in range(2):
        m = mask[:, c] > 0
        n_c = int(m.sum())
        if n_c == 0:
            continue
        ybar = float(Ys[m, c].mean())
        A = np.zeros((d, d), dtype=np.float64)
        g = np.zeros(d, dtype=np.float64)
        idx = np.flatnonzero(m)
        for i0 in range(0, len(idx), chunk):
            ii = idx[i0:i0 + chunk]
            Xc = Xs[ii].astype(np.float64)
            yc = Ys[ii, c].astype(np.float64) - ybar
            A += Xc.T @ Xc
            g += Xc.T @ yc
        A[np.diag_indices(d)] += alpha
        B[:, c] = np.linalg.solve(A, g)
        b[c] = ybar
    return B.astype(np.float32), b.astype(np.float32)


class MLP(nn.Module):
    """Residual-correction MLP: shared trunk, 2-unit head."""

    def __init__(self, n_inputs, hidden=(256, 256, 128), dropout=0.1):
        super().__init__()
        layers = []
        d = n_inputs
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _masked_mse(pred, target, weight):
    """Weighted masked MSE: `weight` carries both the validity mask and the
    per-component metric weight, applied LINEARLY to the squared error
    (sum(w * e^2) / sum(w)). With a binary mask this equals plain masked MSE."""
    err2 = (pred - target).pow(2)
    return (err2 * weight).sum() / weight.sum().clamp(min=1e-8)


def _train_member(X_t, Y_t, M_t, X_v, Y_v, M_v, X_f, Y_f, M_f,
                  n_inputs, member_seed, device, max_epochs, t_end):
    """Train one temporally-validated MLP (parent recipe, unchanged) that must
    finish by wall-clock time t_end. Returns the trained net in eval mode."""
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)
    t_start = time.time()
    # Reserve the last 20% of this member's slice for the recency fine-tune.
    main_t_end = t_start + 0.8 * max(0.0, t_end - t_start)

    net = MLP(n_inputs, hidden=(256, 256, 128), dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    bs = 4096
    patience = 15
    best_val = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    best_epoch = 0
    n_train = len(X_t)
    n = len(X_f)

    for epoch in range(1, max_epochs + 1):
        net.train()
        order = torch.from_numpy(rng.permutation(n_train))
        for i in range(0, n_train, bs):
            idx = order[i:i + bs]
            xb, yb, mb = X_t[idx].to(device), Y_t[idx].to(device), M_t[idx].to(device)
            optimizer.zero_grad()
            loss = _masked_mse(net(xb), yb, mb)
            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            val_losses = []
            for i in range(0, len(X_v), bs):
                val_losses.append(_masked_mse(net(X_v[i:i + bs]), Y_v[i:i + bs], M_v[i:i + bs]).item())
            val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = copy.deepcopy(net.state_dict())
            best_epoch = epoch

        if epoch - best_epoch >= patience:
            break
        if time.time() > main_t_end:
            break

    net.load_state_dict(best_state)

    # Recency fine-tune: the validation tail is the most recent data — closest
    # in time to the held-out test cycles — and was excluded from training
    # above. Briefly fine-tune the selected model on the FULL dataset at low
    # LR, sweeping oldest -> newest within each pass so the freshest samples
    # shape the final weights.
    ft_opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    ft_epochs = min(2, max(0, max_epochs - best_epoch))
    net.train()
    for _ in range(ft_epochs):
        if time.time() > t_end:
            break
        # Shuffle within coarse chronological blocks: keeps batch diversity
        # but preserves the oldest->newest ordering of the pass.
        block = 262144
        for b0 in range(0, n, block):
            blk = np.arange(b0, min(b0 + block, n))
            rng.shuffle(blk)
            blk_t = torch.from_numpy(blk)
            for i in range(0, len(blk_t), bs):
                idx = blk_t[i:i + bs]
                xb, yb, mb = X_f[idx].to(device), Y_f[idx].to(device), M_f[idx].to(device)
                ft_opt.zero_grad()
                loss = _masked_mse(net(xb), yb, mb)
                loss.backward()
                ft_opt.step()
            if time.time() > t_end:
                break

    net.eval()
    return net


def train_and_predict(X_train, Y_train, X_test, params):
    seed = int(params["seed"])
    torch.manual_seed(seed)
    device = torch.device(params["device"])
    max_epochs = int(params["max_epochs"])
    time_budget_s = float(params["time_budget_s"])
    t0 = time.time()

    # PHYSICS FEATURES: append wind-speed / pseudo-stress / current-speed
    # columns (see _augment) so both the ridge stage and the MLPs can use the
    # quadratic wind-stress relation that drives the ageostrophic u component.
    X_train = _augment(X_train)
    X_test = _augment(X_test)

    # Per-column standardization ignoring NaNs; NaN -> 0 (the feature mean).
    mean = np.nan_to_num(np.nanmean(X_train, axis=0), nan=0.0).astype(np.float32)
    scale = np.nanstd(X_train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)

    Xs = _standardize(X_train, mean, scale)
    mask = np.isfinite(Y_train).astype(np.float32)
    Y_raw = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)

    # RIDGE BASELINE: fit the linear map features -> (u, v) in closed form on
    # the full training set. Geostrophic velocity features are near-linearly
    # related to the surface current, so this exact fit captures the dominant
    # signal deterministically; the MLP ensemble then only has to learn the
    # nonlinear/ageostrophic RESIDUAL.
    B, b_int = _fit_ridge(Xs, Y_raw, mask, alpha=10.0)
    lin_train = Xs @ B + b_int
    Ys = np.where(mask > 0, Y_raw - lin_train, 0.0).astype(np.float32)
    del lin_train

    # METRIC-ALIGNED COMPONENT WEIGHTING. The score is mean of per-component
    # R^2 = 1 - mean_c(MSE_c / Var_c), so the loss that exactly matches the
    # metric weights each component's squared error by 1/Var_c of its
    # (residual) target. Plain MSE instead lets the higher-variance component
    # (v) dominate the gradient, starving u — the component with the most
    # headroom. Fold w_c into the mask so training loss, validation early
    # stopping, and the fine-tune all optimize the actual score.
    var_c = np.array([
        float(np.mean(Ys[mask[:, c] > 0, c] ** 2)) if (mask[:, c] > 0).any() else 1.0
        for c in range(2)
    ], dtype=np.float32)
    var_c = np.maximum(var_c, 1e-8)
    w_c = var_c.mean() / var_c
    w_c = np.clip(w_c, 0.2, 5.0).astype(np.float32)
    Wmask = (mask * w_c[None, :]).astype(np.float32)

    # TEMPORAL validation split: rows are flattened in cycle (time) order, and
    # the harness's test split is later cycles. Hold out the last 10% of rows
    # as a contiguous tail so early stopping / LR scheduling select for
    # forward-in-time generalization instead of interleaved memorization.
    n = len(Xs)
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    # Build all tensors ONCE and share them across ensemble members.
    X_t, Y_t, M_t = (torch.from_numpy(a[train_idx]) for a in (Xs, Ys, Wmask))
    X_v = torch.from_numpy(Xs[val_idx]).to(device)
    Y_v = torch.from_numpy(Ys[val_idx]).to(device)
    M_v = torch.from_numpy(Wmask[val_idx]).to(device)
    X_f = torch.from_numpy(Xs)
    Y_f = torch.from_numpy(Ys)
    M_f = torch.from_numpy(Wmask)

    # SEED ENSEMBLE within the wall-clock budget. Member 0 gets half the
    # budget so that, even under a tight clock, the ensemble contains at
    # least one full-strength model; members 1-2 split the remainder and
    # add decorrelated views. Cumulative slice boundaries as budget fractions:
    slice_ends = [0.50, 0.78, 1.0]
    Xp = _standardize(X_test, mean, scale)
    lin_test = (Xp @ B + b_int).astype(np.float64)
    pred_sum = np.zeros((len(Xp), 2), dtype=np.float64)
    n_members = 0

    for m, frac in enumerate(slice_ends):
        t_end = t0 + time_budget_s * frac
        remaining = t_end - time.time()
        # Don't start a member whose slice is already (nearly) gone —
        # a barely-trained net would only dilute the average.
        if m > 0 and remaining < 0.08 * time_budget_s:
            break
        net = _train_member(X_t, Y_t, M_t, X_v, Y_v, M_v, X_f, Y_f, M_f,
                            Xs.shape[1], seed + 1000 * m, device, max_epochs, t_end)
        with torch.no_grad():
            for i in range(0, len(Xp), 65536):
                xb = torch.from_numpy(Xp[i:i + 65536]).to(device)
                pred_sum[i:i + 65536] += net(xb).cpu().numpy()
        n_members += 1
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()

    out = (lin_test + pred_sum / max(1, n_members)).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
