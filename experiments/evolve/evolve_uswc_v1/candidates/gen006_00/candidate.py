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

import numpy as np
import torch
import torch.nn as nn

# Known feature-major layout of the 81-column input: 9 base features x 9
# stencil cells, feature order:
# mdt, ssha_filtered, ugos_filtered, vgos_filtered, ugosa_filtered,
# vgosa_filtered, era5_u, era5_v, SST
_N_BASE = 9
_IDX_UGOS = 2    # ugos_filtered block; vgos_filtered is the following block
_IDX_ERA5_U = 6  # era5_u block; era5_v is the following block


def _augment_physics(X):
    """Append quadratic stress-like features for wind AND geostrophic current.

    Wind: Ekman/upwelling currents respond to wind STRESS tau ~ rho*Cd*|U|*U,
    quadratic in the 10 m wind, so per stencil cell append |W|, |W|*u10,
    |W|*v10 (verified gain in this lineage).
    Current: the surface->subsurface velocity transfer depends nonlinearly on
    current speed (shear, drag ~ |Ug|*Ug, Ekman-layer attenuation), so per
    stencil cell also append |Ug|, |Ug|*ug, |Ug|*vg for the geostrophic pair.
    All computed before standardization so the closed-form ridge can represent
    speed-dependent attenuation/rotation directly. NaNs propagate and are
    handled by the standard NaN->0 imputation. Guarded to the known 81-column
    layout; any other width passes through unchanged.
    """
    X = np.asarray(X, dtype=np.float32)
    if X.shape[1] != _N_BASE * _N_BASE:
        return X
    n_cells = X.shape[1] // _N_BASE

    def blk(i):
        return X[:, i * n_cells:(i + 1) * n_cells]

    u10, v10 = blk(_IDX_ERA5_U), blk(_IDX_ERA5_U + 1)
    wspd = np.sqrt(u10 * u10 + v10 * v10)
    ug, vg = blk(_IDX_UGOS), blk(_IDX_UGOS + 1)
    cspd = np.sqrt(ug * ug + vg * vg)
    return np.concatenate(
        [X,
         wspd.astype(np.float32),
         (wspd * u10).astype(np.float32),
         (wspd * v10).astype(np.float32),
         cspd.astype(np.float32),
         (cspd * ug).astype(np.float32),
         (cspd * vg).astype(np.float32)], axis=1)


def _standardize(X, mean, scale):
    Xs = (X.astype(np.float32) - mean) / scale
    return np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)


def _fit_ridge(Xs, Ys, mask, alpha=10.0):
    """Closed-form ridge per target component on the masked rows.

    Xs is standardized (NaN->0), so features are ~zero-mean; the intercept is
    absorbed by centering y. Returns (B, b): weights (d, 2) and intercepts (2,).
    Exact and deterministic — captures the linear (geostrophic + stress) part
    of the signal on ALL training data with no SGD noise.
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


def _masked_mse(pred, target, mask):
    diff = (pred - target) * mask
    return diff.pow(2).sum() / mask.sum().clamp(min=1)


def _masked_huber(pred, target, mask, delta):
    """Masked Huber loss with per-component delta (shape (2,)).

    Quadratic (matching MSE up to the 0.5 factor) for |err| <= delta_c,
    linear beyond — so heavy-tailed HFR noise and rare ageostrophic spikes
    contribute bounded gradients instead of dominating the update.
    """
    err = (pred - target).abs()
    quad = torch.minimum(err, delta)
    loss = 0.5 * quad.pow(2) + delta * (err - quad)
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def _train_member(X_t, Y_t, M_t, X_v, Y_v, M_v, X_f, Y_f, M_f, delta,
                  n_inputs, member_seed, device, max_epochs, t_end):
    """Train one temporally-validated MLP that must finish by wall-clock time
    t_end. Trains with masked Huber (robust to outlier residuals); validates
    and early-stops on plain masked MSE, the quantity the R^2 metric measures.
    Returns the trained net in eval mode."""
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)
    t_start = time.time()
    # Reserve the last 20% of this member's slice for the recency fine-tune.
    main_t_end = t_start + 0.8 * max(0.0, t_end - t_start)

    net = MLP(n_inputs, hidden=(256, 256, 128), dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    delta_d = delta.to(device)
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
            loss = _masked_huber(net(xb), yb, mb, delta_d)
            loss.backward()
            optimizer.step()

        # Validation / checkpoint selection on plain masked MSE: the score is
        # R^2 (an MSE quantity), so the robust training loss must not leak
        # into model selection.
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
                loss = _masked_huber(net(xb), yb, mb, delta_d)
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

    # PHYSICS AUGMENTATION: append wind pseudo-stress (|W|, |W|*u10, |W|*v10)
    # and geostrophic current-speed (|Ug|, |Ug|*ug, |Ug|*vg) blocks per
    # stencil cell (81 -> 135 columns) before any statistics are computed, so
    # the ridge and the MLP both see the quadratic wind forcing AND the
    # speed-dependent surface->subsurface transfer of the geostrophic flow.
    X_train = _augment_physics(X_train)
    X_test = _augment_physics(X_test)

    # Per-column standardization ignoring NaNs; NaN -> 0 (the feature mean).
    mean = np.nan_to_num(np.nanmean(X_train, axis=0), nan=0.0).astype(np.float32)
    scale = np.nanstd(X_train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)

    Xs = _standardize(X_train, mean, scale)
    mask = np.isfinite(Y_train).astype(np.float32)
    Y_raw = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)

    # RIDGE BASELINE: fit the linear map features -> (u, v) in closed form on
    # the full training set. With the stress and current-speed features
    # appended, the linear fit can represent both the Ekman response to wind
    # stress and speed-dependent attenuation/rotation of the geostrophic
    # current, so this exact fit captures the dominant signal
    # deterministically; the MLP ensemble then only has to learn the
    # remaining nonlinear RESIDUAL.
    B, b_int = _fit_ridge(Xs, Y_raw, mask, alpha=10.0)
    lin_train = Xs @ B + b_int
    Ys = np.where(mask > 0, Y_raw - lin_train, 0.0).astype(np.float32)
    del lin_train

    # ROBUST-LOSS SCALE: one Huber delta per component, set to the std of that
    # component's post-ridge residual. Residuals within ~1 sigma (the
    # predictable bulk) get exact quadratic treatment; the heavy tail (HFR
    # radar outliers, rare ageostrophic spikes) gets linear gradients so it
    # cannot dominate the update the way squared error lets it.
    delta_np = np.array([
        float(np.sqrt(np.mean(Ys[mask[:, c] > 0, c] ** 2))) if (mask[:, c] > 0).any() else 1.0
        for c in range(2)
    ], dtype=np.float32)
    delta_np = np.clip(delta_np, 1e-3, None)
    delta = torch.from_numpy(delta_np)

    # TEMPORAL validation split: rows are flattened in cycle (time) order, and
    # the harness's test split is later cycles. Hold out the last 10% of rows
    # as a contiguous tail so early stopping / LR scheduling select for
    # forward-in-time generalization instead of interleaved memorization.
    n = len(Xs)
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    # Build all tensors ONCE and share them across ensemble members.
    X_t, Y_t, M_t = (torch.from_numpy(a[train_idx]) for a in (Xs, Ys, mask))
    X_v = torch.from_numpy(Xs[val_idx]).to(device)
    Y_v = torch.from_numpy(Ys[val_idx]).to(device)
    M_v = torch.from_numpy(mask[val_idx]).to(device)
    X_f = torch.from_numpy(Xs)
    Y_f = torch.from_numpy(Ys)
    M_f = torch.from_numpy(mask)

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
        net = _train_member(X_t, Y_t, M_t, X_v, Y_v, M_v, X_f, Y_f, M_f, delta,
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
