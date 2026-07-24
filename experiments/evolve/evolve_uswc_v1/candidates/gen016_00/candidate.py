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
from sklearn.ensemble import HistGradientBoostingRegressor

# Known feature-major layout of the 81-column input: 9 base features x 9
# stencil cells, feature order:
# mdt, ssha_filtered, ugos_filtered, vgos_filtered, ugosa_filtered,
# vgosa_filtered, era5_u, era5_v, SST
_N_BASE = 9
_IDX_UGOS = 2    # ugos_filtered block; vgos_filtered is the following block
_IDX_ERA5_U = 6  # era5_u block; era5_v is the following block

# Weight EMA (verified neutral-or-better in gen011_02): decay of the
# exponential moving average of MLP weights, updated after every step.
_EMA_DECAY = 0.999

# GBM ensemble member (verified +0.012 in gen014_01): wall-clock fraction
# reserved for the tree models and the recency-capped row count they train
# on. Trees are fit on RAW features with NaNs intact (native missing-value
# handling) and are gated per component on the temporal validation tail.
# The budget fraction covers up to two fits per component: the gated
# pre-tail fit and the recency refit on the full span (verified in
# gen014_02).
_GBM_BUDGET_FRAC = 0.26
_GBM_MAX_ROWS = 500_000

# RECENCY-WINDOWED FINE-TUNE (the one new change): the MLP recency
# fine-tune now trains only on the most recent _FT_MAX_ROWS rows of the
# full span — the same verified window size the GBM's recency moves use —
# instead of sweeping the entire dataset. Each pass is ~half as long, so
# the epoch cap rises to _FT_MAX_EPOCHS inside the SAME reserved time
# slice (the t_end deadline still bounds everything), concentrating the
# unchanged fine-tune budget on the rows closest in time to the test
# cycles.
_FT_MAX_ROWS = 500_000
_FT_MAX_EPOCHS = 4


def _augment_physics(X):
    """Append quadratic stress-like features, current-speed features,
    stencil-spread features, and stencil validity fractions.

    Wind: Ekman/upwelling currents respond to wind STRESS tau ~ rho*Cd*|U|*U,
    quadratic in the 10 m wind, so per stencil cell append |W|, |W|*u10,
    |W|*v10 (verified gain in this lineage).
    Current: the surface->subsurface velocity transfer depends nonlinearly on
    current speed (shear, drag ~ |Ug|*Ug, Ekman-layer attenuation), so per
    stencil cell also append |Ug|, |Ug|*ug, |Ug|*vg for the geostrophic pair
    (verified gain in gen006_00).
    Stencil spread (verified gain in gen006_03): per base feature, the
    NaN-aware std across its 9 stencil cells — local front strength /
    mesoscale roughness, the regime variable that modulates surface->
    subsurface transfer. Invariant to the unknown stencil cell ordering.
    Validity fractions (verified gain in gen007_02): per base feature, the
    fraction of finite cells across its 9 stencil cells. After
    standardization every NaN cell is imputed to the column mean, so
    downstream models cannot tell a genuinely mean-valued cell from an
    imputed one — swath-edge rows masquerade as quiescent interior rows.
    These missingness indicators let the ridge learn per-feature offsets for
    partially observed rows and let the MLP gate its correction on data
    quality. Never NaN, ordering-invariant, zero layout risk.
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

    valid_frac = np.concatenate(
        [np.isfinite(blk(i)).mean(axis=1, keepdims=True).astype(np.float32)
         for i in range(_N_BASE)], axis=1)

    return np.concatenate(
        [X,
         wspd.astype(np.float32),
         (wspd * u10).astype(np.float32),
         (wspd * v10).astype(np.float32),
         cspd.astype(np.float32),
         (cspd * ug).astype(np.float32),
         (cspd * vg).astype(np.float32),
         spread,
         valid_frac], axis=1)


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
    the quantity the mean-R^2 metric measures.

    Weight EMA: an exponential moving average of the network weights (decay
    _EMA_DECAY) is updated after every optimizer step. Each epoch BOTH the
    raw net and the EMA net are scored on the clean validation tail, and the
    checkpoint keeps whichever is better; because the raw net remains a
    selection candidate, the scheme cannot underperform plain checkpointing.
    Returns the net (loaded with the best-found weights) in eval mode."""
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)
    t_start = time.time()
    # Reserve the last 20% of this member's slice for the recency fine-tune.
    main_t_end = t_start + 0.8 * max(0.0, t_end - t_start)

    net = MLP(n_inputs, hidden=(256, 256, 128), dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    # EMA shadow weights + a second net used only to evaluate the EMA.
    ema_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    ema_net = MLP(n_inputs, hidden=(256, 256, 128), dropout=0.1).to(device)
    ema_net.eval()

    delta_d = delta.to(device)
    comp_w_d = comp_w.to(device)
    bs = 4096
    patience = 15
    best_val = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    best_epoch = 0
    n_train = len(X_t)
    n = len(X_f)

    def _validate(model):
        losses = []
        with torch.no_grad():
            for i in range(0, len(X_v), bs):
                losses.append(_masked_mse(model(X_v[i:i + bs]), Y_v[i:i + bs],
                                          M_v[i:i + bs], comp_w_d).item())
        return float(np.mean(losses))

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
            with torch.no_grad():
                for k, v in net.state_dict().items():
                    ema_state[k].mul_(_EMA_DECAY).add_(v, alpha=1.0 - _EMA_DECAY)

        # Validation / checkpoint selection on component-balanced masked MSE:
        # the score averages per-component R^2, so selection must weigh u and
        # v equally in sigma units — but the robust (Huber) shape must not
        # leak into selection. Both the raw and the EMA weights are scored;
        # the better of the two becomes the checkpoint candidate. The LR
        # scheduler follows the RAW validation loss, since that reflects the
        # trajectory the optimizer is actually driving.
        net.eval()
        val_raw = _validate(net)
        ema_net.load_state_dict(ema_state)
        val_ema = _validate(ema_net)
        val_loss = min(val_raw, val_ema)
        scheduler.step(val_raw)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            if val_ema < val_raw:
                best_state = copy.deepcopy(ema_state)
            else:
                best_state = copy.deepcopy(net.state_dict())
            best_epoch = epoch

        if epoch - best_epoch >= patience:
            break
        if time.time() > main_t_end:
            break

    net.load_state_dict(best_state)

    # RECENCY-WINDOWED FINE-TUNE (the one new change): the validation tail is
    # the most recent data — closest in time to the held-out test cycles —
    # and was excluded from training above. Fine-tune the selected model at
    # low LR on the most recent _FT_MAX_ROWS rows of the FULL span (rows are
    # in cycle/time order, so this window ends at the freshest sample and
    # includes the tail), sweeping oldest -> newest within each pass so the
    # freshest samples shape the final weights. Previously this fine-tune
    # swept the entire span, spending half of each pass on the stalest
    # cycles; the GBM's two verified recency wins both used exactly this
    # bounded window. Passes are ~half as long now, so the epoch cap rises
    # to _FT_MAX_EPOCHS inside the same reserved time slice — the t_end
    # deadline still bounds total fine-tune wall-clock exactly as before.
    ft_opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
    ft_epochs = min(_FT_MAX_EPOCHS, max(0, max_epochs - best_epoch))
    ft_start = max(0, n - _FT_MAX_ROWS)
    net.train()
    for _ in range(ft_epochs):
        if time.time() > t_end:
            break
        # Shuffle within coarse chronological blocks: keeps batch diversity
        # but preserves the oldest->newest ordering of the pass.
        block = 262144
        for b0 in range(ft_start, n, block):
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


def _make_gbm(seed):
    return HistGradientBoostingRegressor(
        max_iter=100, learning_rate=0.08, max_leaf_nodes=63,
        min_samples_leaf=40, l2_regularization=1.0, max_bins=255,
        early_stopping=False, random_state=seed)


def _fit_gbm_members(X_raw, X_test_raw, Ys, mask, train_idx, val_idx, seed,
                     deadline):
    """Fit one HistGradientBoostingRegressor per component on the post-ridge
    residuals — a DECORRELATED model family for the ensemble (verified gain
    in gen014_01; recency refit verified in gen014_02).

    Trees see the RAW augmented features with NaNs intact: HistGBR handles
    missing values natively by learning a per-split direction for missings,
    a fundamentally different treatment of swath-edge cells than the
    mean-imputation the ridge and MLPs rely on, so its errors decorrelate
    from theirs.

    Gate-then-refit: the gating fit uses the most recent _GBM_MAX_ROWS valid
    rows of the PRE-validation span, so the untouched temporal tail can gate
    inclusion — a component's GBM joins the residual average only if it
    beats the zero-residual predictor (ridge alone) on that tail, so a
    failed family dilutes nothing. Per component that PASSES the gate, the
    GBM is refit with identical hyperparameters on the most recent
    _GBM_MAX_ROWS valid rows of the FULL span — the same window size,
    shifted forward to include the tail — and the refit model predicts the
    test set. A deadline check before the refit falls back to the gated
    pre-tail model when the clock is tight.

    Returns (pred_test (n_test, 2) float64, included (2,) bool).
    """
    pred_test = np.zeros((len(X_test_raw), 2), dtype=np.float64)
    included = np.zeros(2, dtype=bool)
    for c in range(2):
        if time.time() > deadline:
            break
        rows = train_idx[mask[train_idx, c] > 0]
        if len(rows) < 10_000:
            continue
        rows = rows[-_GBM_MAX_ROWS:]
        gbm = _make_gbm(seed + 31 * c)
        gbm.fit(X_raw[rows], Ys[rows, c])
        vm = mask[val_idx, c] > 0
        if not vm.any():
            del gbm
            continue
        vv = val_idx[vm]
        pv = gbm.predict(X_raw[vv])
        y_val = Ys[vv, c].astype(np.float64)
        mse_gbm = float(np.mean((pv - y_val) ** 2))
        mse_zero = float(np.mean(y_val ** 2))
        if mse_gbm < mse_zero:
            included[c] = True
            pred_test[:, c] = gbm.predict(X_test_raw)
            # RECENCY REFIT: same hyperparameters, same window size, but the
            # window now ends at the last row of the full span — rows are in
            # cycle (time) order, so this is the data closest to the test
            # cycles. Only attempted when the clock allows; otherwise the
            # gated pre-tail prediction above stands.
            if time.time() < deadline:
                full_rows = np.flatnonzero(mask[:, c] > 0)[-_GBM_MAX_ROWS:]
                gbm_r = _make_gbm(seed + 31 * c + 7)
                gbm_r.fit(X_raw[full_rows], Ys[full_rows, c])
                pred_test[:, c] = gbm_r.predict(X_test_raw)
                del gbm_r
        del gbm
    return pred_test, included


def train_and_predict(X_train, Y_train, X_test, params):
    seed = int(params["seed"])
    torch.manual_seed(seed)
    device = torch.device(params["device"])
    max_epochs = int(params["max_epochs"])
    time_budget_s = float(params["time_budget_s"])
    t0 = time.time()

    # PHYSICS + QUALITY AUGMENTATION: append wind pseudo-stress (|W|, |W|*u10,
    # |W|*v10), geostrophic current-speed (|Ug|, |Ug|*ug, |Ug|*vg) blocks per
    # stencil cell, per-base-feature stencil spread (nanstd over the 9 cells),
    # and per-base-feature stencil validity fractions: 81 -> 153 columns, all
    # before any statistics are computed, so the ridge, the MLP, and the GBM
    # all see quadratic wind forcing, speed-dependent transfer, local
    # front-strength regime information, and explicit data-quality indicators.
    X_train = _augment_physics(X_train)   # raw, NaNs preserved (GBM input)
    X_test = _augment_physics(X_test)     # raw, NaNs preserved (GBM input)

    # Per-column standardization ignoring NaNs; NaN -> 0 (the feature mean).
    mean = np.nan_to_num(np.nanmean(X_train, axis=0), nan=0.0).astype(np.float32)
    scale = np.nanstd(X_train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)

    Xs = _standardize(X_train, mean, scale)
    mask = np.isfinite(Y_train).astype(np.float32)
    Y_raw = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)

    # RIDGE BASELINE: fit the linear map features -> (u, v) in closed form on
    # the full training set. With the stress, current-speed, stencil-spread,
    # and validity-fraction features appended, the linear fit can represent
    # the Ekman response to wind stress, speed-dependent attenuation/rotation
    # of the geostrophic current, a front-strength-modulated offset, and a
    # per-feature correction for partially observed (mean-imputed) rows; the
    # nonlinear members then only have to learn the remaining RESIDUAL.
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

    # COMPONENT BALANCE WEIGHTS: inverse post-ridge residual variance per
    # component, normalized to mean 1. The metric is the MEAN of R^2(u) and
    # R^2(v) — each R^2 is scale-invariant — but a raw m/s loss weights
    # components by absolute residual variance, letting the larger-variance
    # component dominate the shared trunk. In sigma units both components
    # pull equally; the mean-1 normalization keeps the loss magnitude (and
    # tuned LR) unchanged.
    inv_var = 1.0 / np.maximum(delta_np.astype(np.float64) ** 2, 1e-6)
    comp_w_np = (inv_var / inv_var.mean()).astype(np.float32)
    comp_w = torch.from_numpy(comp_w_np)

    # TEMPORAL validation split: rows are flattened in cycle (time) order, and
    # the harness's test split is later cycles. Hold out the last 10% of rows
    # as a contiguous tail so early stopping / LR scheduling / GBM gating
    # select for forward-in-time generalization instead of memorization.
    n = len(Xs)
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    # GBM MEMBERS: a tree-family residual model per component, gated on the
    # pre-tail fit and then recency-refit on the full span (see
    # _fit_gbm_members), capped to a fixed slice of the wall clock so the
    # MLP ensemble keeps the lion's share.
    gbm_deadline = t0 + _GBM_BUDGET_FRAC * time_budget_s
    gbm_pred_test, gbm_included = _fit_gbm_members(
        X_train, X_test, Ys, mask, train_idx, val_idx, seed, gbm_deadline)

    # Build all tensors ONCE and share them across ensemble members.
    X_t, Y_t, M_t = (torch.from_numpy(a[train_idx]) for a in (Xs, Ys, mask))
    X_v = torch.from_numpy(Xs[val_idx]).to(device)
    Y_v = torch.from_numpy(Ys[val_idx]).to(device)
    M_v = torch.from_numpy(mask[val_idx]).to(device)
    X_f = torch.from_numpy(Xs)
    Y_f = torch.from_numpy(Ys)
    M_f = torch.from_numpy(mask)

    # SEED ENSEMBLE within the REMAINING wall-clock budget (after the GBM
    # slice). Member 0 gets half of what's left so that, even under a tight
    # clock, the ensemble contains at least one full-strength model;
    # members 1-2 split the remainder and add decorrelated views.
    slice_ends = [0.50, 0.78, 1.0]
    Xp = _standardize(X_test, mean, scale)
    lin_test = (Xp @ B + b_int).astype(np.float64)
    pred_sum = np.zeros((len(Xp), 2), dtype=np.float64)
    n_members = 0

    t_ml0 = time.time()
    remaining_total = max(1.0, t0 + time_budget_s - t_ml0)
    for m, frac in enumerate(slice_ends):
        t_end = t_ml0 + remaining_total * frac
        remaining = t_end - time.time()
        # Don't start a member whose slice is already (nearly) gone —
        # a barely-trained net would only dilute the average.
        if m > 0 and remaining < 0.08 * remaining_total:
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

    # COMBINE: per component, average the MLP members plus (if it passed the
    # validation gate) the GBM as one extra equal-weight member, all on top
    # of the exact ridge baseline. (The blind equal-weight combine is kept:
    # the tail-fitted combine tried in gen015_02 scored below this scheme.)
    out = lin_test.copy()
    for c in range(2):
        total = pred_sum[:, c].copy()
        denom = n_members
        if gbm_included[c]:
            total += gbm_pred_test[:, c]
            denom += 1
        out[:, c] += total / max(1, denom)
    out = out.astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
