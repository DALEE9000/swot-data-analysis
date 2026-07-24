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

# Known feature-major layout of the 81-column input: 9 base features x 9
# stencil cells, feature order:
# mdt, ssha_filtered, ugos_filtered, vgos_filtered, ugosa_filtered,
# vgosa_filtered, era5_u, era5_v, SST
_N_BASE = 9
_IDX_UGOS = 2    # ugos_filtered block; vgos_filtered is the following block
_IDX_ERA5_U = 6  # era5_u block; era5_v is the following block


def _augment_physics(X):
    """Append quadratic stress-like features, current-speed features, and
    stencil-spread features.

    Wind: Ekman/upwelling currents respond to wind STRESS tau ~ rho*Cd*|U|*U,
    quadratic in the 10 m wind, so per stencil cell append |W|, |W|*u10,
    |W|*v10 (verified gain in this lineage).
    Current: the surface->subsurface velocity transfer depends nonlinearly on
    current speed (shear, drag ~ |Ug|*Ug, Ekman-layer attenuation), so per
    stencil cell also append |Ug|, |Ug|*ug, |Ug|*vg for the geostrophic pair
    (verified gain in gen006_00).
    Stencil spread (verified gain in gen006_03): per base feature, the
    NaN-aware std across its 9 stencil cells — local front strength /
    mesoscale roughness, the regime variable that modulates
    surface->subsurface transfer. Invariant to stencil cell ordering; all-NaN
    rows yield NaN and fall back to the column mean via standard imputation.
    All computed before standardization so the closed-form ridge can use them
    directly. Guarded to the known 81-column layout; any other width passes
    through unchanged.
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

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        spread = np.concatenate(
            [np.nanstd(blk(i), axis=1, keepdims=True).astype(np.float32)
             for i in range(_N_BASE)], axis=1)

    return np.concatenate(
        [X,
         wspd.astype(np.float32),
         (wspd * u10).astype(np.float32),
         (wspd * v10).astype(np.float32),
         cspd.astype(np.float32),
         (cspd * ug).astype(np.float32),
         (cspd * vg).astype(np.float32),
         spread], axis=1)


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


def _fit_rff(Xs, R, mask, n_val, seed, device):
    """Closed-form random-Fourier-feature kernel ridge on the linear-ridge
    residuals R (the ONE new change).

    phi(x) = sqrt(2/D) * cos(W^T x + b) with W drawn for an RBF kernel at two
    lengthscales (sqrt(d) and 2*sqrt(d) — the median-distance scale of
    standardized data and a smoother octave), so the closed-form fit
    approximates kernel ridge regression: the smooth, stationary-nonlinear
    part of the residual map, learned exactly on ALL rows with no SGD noise.
    Per component: the Gram matrix is accumulated chunk-wise on the GPU in
    fp32 (converted to float64 on CPU per chunk, so precision is safe),
    split into a temporal head and the last-n_val tail; alpha is selected on
    the tail against a STRICT zero-prediction baseline (predicting nothing
    beats a kernel that doesn't generalize forward in time), then the chosen
    alpha is refit on head+tail. Components that don't beat the baseline get
    w=0, making the stage non-degrading by construction.
    Returns (W, b, scale, w_out (D,2), offsets (2,)).
    """
    n, d = Xs.shape
    dev = device
    D = 2048 if dev.type == "cuda" else 512
    rng = np.random.default_rng(seed + 777)
    l1 = math.sqrt(d)
    W_np = np.concatenate(
        [rng.standard_normal((d, D // 2)) / l1,
         rng.standard_normal((d, D - D // 2)) / (2.0 * l1)],
        axis=1).astype(np.float32)
    b_np = rng.uniform(0.0, 2.0 * np.pi, D).astype(np.float32)
    s = math.sqrt(2.0 / D)
    W = torch.from_numpy(W_np).to(dev)
    bb = torch.from_numpy(b_np).to(dev)

    n_head = n - n_val
    A_h = np.zeros((2, D, D), dtype=np.float64)
    A_t = np.zeros((2, D, D), dtype=np.float64)
    g_h = np.zeros((2, D), dtype=np.float64)
    g_t = np.zeros((2, D), dtype=np.float64)
    s_h = np.zeros((2, D), dtype=np.float64)
    s_t = np.zeros((2, D), dtype=np.float64)
    phi_tail = np.empty((n_val, D), dtype=np.float32)
    chunk = 65536

    def accumulate(lo, hi, A_acc, g_acc, s_acc, store):
        pos = 0
        for i0 in range(lo, hi, chunk):
            i1 = min(i0 + chunk, hi)
            xb = torch.from_numpy(Xs[i0:i1]).to(dev)
            Phi = torch.cos(xb @ W + bb) * s
            if store is not None:
                store[pos:pos + (i1 - i0)] = Phi.cpu().numpy()
                pos += i1 - i0
            mb = torch.from_numpy(mask[i0:i1]).to(dev)
            rb = torch.from_numpy(R[i0:i1]).to(dev)
            for c in range(2):
                Pm = Phi * mb[:, c:c + 1]
                A_acc[c] += (Pm.T @ Phi).double().cpu().numpy()
                g_acc[c] += (Phi.T @ (mb[:, c] * rb[:, c])).double().cpu().numpy()
                s_acc[c] += Pm.sum(dim=0).double().cpu().numpy()
                del Pm
            del xb, Phi, mb, rb

    accumulate(0, n_head, A_h, g_h, s_h, None)
    accumulate(n_head, n, A_t, g_t, s_t, phi_tail)
    del W, bb
    if dev.type == "cuda":
        torch.cuda.empty_cache()

    w_out = np.zeros((D, 2), dtype=np.float32)
    off = np.zeros(2, dtype=np.float32)
    m_head, r_head = mask[:n_head], R[:n_head]
    m_tail, r_tail = mask[n_head:], R[n_head:]
    alphas = [3.0, 10.0, 30.0, 100.0]
    for c in range(2):
        nh = float(m_head[:, c].sum())
        mt = m_tail[:, c] > 0
        if nh < 2 * D or not mt.any():
            continue
        rbar = float((r_head[:, c] * m_head[:, c]).sum() / nh)
        gc = g_h[c] - rbar * s_h[c]
        rt = r_tail[mt, c].astype(np.float64)
        Pt = phi_tail[mt]
        # Zero prediction (= parent behavior, MLP handles everything) is the
        # baseline to beat on the temporal tail.
        best_mse, best_alpha = float(np.mean(rt ** 2)), None
        for a in alphas:
            Ac = A_h[c].copy()
            Ac[np.diag_indices(D)] += a
            w = np.linalg.solve(Ac, gc)
            pred = Pt @ w + rbar
            mse = float(np.mean((rt - pred) ** 2))
            if mse < best_mse - 1e-9:
                best_mse, best_alpha = mse, a
        if best_alpha is None:
            continue
        # Final fit on ALL rows (head + tail) at the tail-selected alpha,
        # centered on the full masked residual mean.
        nf = float(mask[:, c].sum())
        rbar_f = float((R[:, c] * mask[:, c]).sum() / nf)
        gf = (g_h[c] + g_t[c]) - rbar_f * (s_h[c] + s_t[c])
        Af = A_h[c] + A_t[c]
        Af[np.diag_indices(D)] += best_alpha
        w_out[:, c] = np.linalg.solve(Af, gf).astype(np.float32)
        off[c] = np.float32(rbar_f)
    del phi_tail
    return W_np, b_np, s, w_out, off


def _rff_predict(X, W_np, b_np, s, w_out, off, dev):
    """Chunked GPU evaluation of the RFF ridge component. Components with
    w=0 (guard rejected) contribute exactly 0."""
    out = np.zeros((len(X), 2), dtype=np.float32)
    if not np.any(w_out):
        return out
    W = torch.from_numpy(W_np).to(dev)
    bb = torch.from_numpy(b_np).to(dev)
    wt = torch.from_numpy(w_out).to(dev)
    with torch.no_grad():
        for i0 in range(0, len(X), 65536):
            xb = torch.from_numpy(X[i0:i0 + 65536]).to(dev)
            Phi = torch.cos(xb @ W + bb) * s
            out[i0:i0 + 65536] = (Phi @ wt).cpu().numpy()
            del xb, Phi
    del W, bb, wt
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return out + off


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


def _masked_mse(pred, target, mask, comp_w):
    """Masked MSE with per-component weights comp_w (shape (2,)).

    comp_w is the inverse residual variance normalized to mean 1: the metric
    averages the scale-invariant R^2 of u and v, so validation/selection must
    value a 1-sigma error in u exactly as much as a 1-sigma error in v.
    """
    diff = (pred - target)
    return (diff.pow(2) * comp_w * mask).sum() / mask.sum().clamp(min=1)


def _masked_huber(pred, target, mask, delta, comp_w):
    """Masked Huber loss with per-component delta and component weights.

    Quadratic (matching MSE up to the 0.5 factor) for |err| <= delta_c,
    linear beyond — so heavy-tailed HFR noise and rare ageostrophic spikes
    contribute bounded gradients instead of dominating the update.
    COMPONENT BALANCE (verified in gen006_02): each component's loss is
    scaled by comp_w_c ~ 1/delta_c^2 (normalized to mean 1, so the overall
    loss magnitude — and the tuned LR — is unchanged). Raw m/s losses let
    the higher-variance v residual dominate the shared trunk's gradients;
    in sigma units both components pull equally, mirroring the mean-of-R^2
    metric.
    """
    err = (pred - target).abs()
    quad = torch.minimum(err, delta)
    loss = (0.5 * quad.pow(2) + delta * (err - quad)) * comp_w
    return (loss * mask).sum() / mask.sum().clamp(min=1)


def _train_member(X_t, Y_t, M_t, X_v, Y_v, M_v, X_f, Y_f, M_f, delta, comp_w,
                  n_inputs, member_seed, device, max_epochs, t_end):
    """Train one temporally-validated MLP that must finish by wall-clock time
    t_end. Trains with component-balanced masked Huber (robust to outlier
    residuals); validates and early-stops on component-balanced masked MSE,
    the quantity the mean-R^2 metric measures. Returns the net in eval mode."""
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)
    t_start = time.time()
    # Reserve the last 20% of this member's slice for the recency fine-tune.
    main_t_end = t_start + 0.8 * max(0.0, t_end - t_start)

    net = MLP(n_inputs, hidden=(256, 256, 128), dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    delta_d = delta.to(device)
    comp_w_d = comp_w.to(device)
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
            loss = _masked_huber(net(xb), yb, mb, delta_d, comp_w_d)
            loss.backward()
            optimizer.step()

        # Validation / checkpoint selection on component-balanced masked MSE:
        # the score averages per-component R^2, so selection must weigh u and
        # v equally in sigma units — but the robust (Huber) shape must not
        # leak into selection.
        net.eval()
        with torch.no_grad():
            val_losses = []
            for i in range(0, len(X_v), bs):
                val_losses.append(_masked_mse(net(X_v[i:i + bs]), Y_v[i:i + bs],
                                              M_v[i:i + bs], comp_w_d).item())
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
                loss = _masked_huber(net(xb), yb, mb, delta_d, comp_w_d)
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

    # PHYSICS AUGMENTATION: append wind pseudo-stress (|W|, |W|*u10, |W|*v10),
    # geostrophic current-speed (|Ug|, |Ug|*ug, |Ug|*vg) blocks per stencil
    # cell, and per-base-feature stencil spread (nanstd over the 9 cells):
    # 81 -> 144 columns, all before any statistics are computed, so the ridge
    # and the MLP both see quadratic wind forcing, speed-dependent transfer,
    # and local front-strength regime information.
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
    # the full training set. With the stress, current-speed, and stencil-
    # spread features appended, the linear fit can represent the Ekman
    # response to wind stress, speed-dependent attenuation/rotation of the
    # geostrophic current, and a front-strength-modulated offset.
    B, b_int = _fit_ridge(Xs, Y_raw, mask, alpha=10.0)
    lin_train = Xs @ B + b_int
    R = np.where(mask > 0, Y_raw - lin_train, 0.0).astype(np.float32)
    del lin_train

    # TEMPORAL split sizes (rows are in cycle/time order; the harness's test
    # split is later cycles): last 10% is the validation tail used by both
    # the RFF alpha selection and the MLP early stopping.
    n = len(Xs)
    n_val = max(1, int(n * 0.1))

    # KERNEL (RFF) RIDGE STAGE — the one new change: fit the smooth
    # stationary-nonlinear part of the residual in closed form; per-component
    # tail-guarded so it can only match or improve the parent's baseline.
    W_rff, b_rff, s_rff, w_rff, off_rff = _fit_rff(Xs, R, mask, n_val, seed, device)
    rff_train = _rff_predict(Xs, W_rff, b_rff, s_rff, w_rff, off_rff, device)
    Ys = np.where(mask > 0, R - rff_train, 0.0).astype(np.float32)
    del R, rff_train

    # ROBUST-LOSS SCALE: one Huber delta per component, set to the std of that
    # component's post-baseline residual. Residuals within ~1 sigma (the
    # predictable bulk) get exact quadratic treatment; the heavy tail (HFR
    # radar outliers, rare ageostrophic spikes) gets linear gradients so it
    # cannot dominate the update the way squared error lets it.
    delta_np = np.array([
        float(np.sqrt(np.mean(Ys[mask[:, c] > 0, c] ** 2))) if (mask[:, c] > 0).any() else 1.0
        for c in range(2)
    ], dtype=np.float32)
    delta_np = np.clip(delta_np, 1e-3, None)
    delta = torch.from_numpy(delta_np)

    # COMPONENT BALANCE WEIGHTS: inverse post-baseline residual variance per
    # component, normalized to mean 1. The metric is the MEAN of R^2(u) and
    # R^2(v) — each R^2 is scale-invariant — but a raw m/s loss weights
    # components by absolute residual variance, letting the larger-variance
    # component dominate the shared trunk. In sigma units both components
    # pull equally; the mean-1 normalization keeps the loss magnitude (and
    # tuned LR) unchanged.
    inv_var = 1.0 / np.maximum(delta_np.astype(np.float64) ** 2, 1e-6)
    comp_w_np = (inv_var / inv_var.mean()).astype(np.float32)
    comp_w = torch.from_numpy(comp_w_np)

    # TEMPORAL validation split for the MLP: contiguous tail, so early
    # stopping / LR scheduling select for forward-in-time generalization
    # instead of interleaved memorization.
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
    lin_test += _rff_predict(Xp, W_rff, b_rff, s_rff, w_rff, off_rff,
                             device).astype(np.float64)
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
                            comp_w, Xs.shape[1], seed + 1000 * m, device,
                            max_epochs, t_end)
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
