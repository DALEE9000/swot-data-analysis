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
from scipy.optimize import nnls
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

# Budgeted cosine LR (verified neutral-or-better in gen016_01): each MLP
# member anneals its learning rate from _LR_MAX to _LR_MIN following a
# cosine over the elapsed wall-clock fraction of its own training slice, so
# every member finishes in a converged low-LR regime exactly when its slice
# expires.
_LR_MAX = 1e-3
_LR_MIN = 2e-5

# ARCHITECTURE-DIVERSE MLP ENSEMBLE (verified in gen018_02): member 0 keeps
# the exact verified architecture so the ensemble always contains one
# full-strength known-good model; member 1 is wider-and-shallower (a
# smoother, lower-frequency fit of the residual), member 2 deeper-and-
# narrower (a more compositional fit). Different depths/widths induce
# genuinely different inductive biases, decorrelating member errors far
# more than seed noise does. Dropout stays at the verified 0.1 everywhere.
_MEMBER_ARCHS = [
    (256, 256, 128),        # member 0: the verified baseline architecture
    (384, 192),             # member 1: wider, shallower
    (192, 192, 192, 96),    # member 2: deeper, narrower
]

# ROTATED-TARGET MEMBER (verified direction in gen019_00): member
# _ROT_MEMBER trains on the post-ridge residuals rotated 45 degrees in the
# (u, v) plane — it predicts (u+v)/sqrt(2) and (v-u)/sqrt(2) — and its
# predictions are rotated back before entering the combine. A rotated-basis
# learner splits capacity and errors along different physical axes (roughly
# along-/across-shore for USWC), decorrelating its per-component errors
# from every axis-aligned member. Rotated targets require BOTH components
# valid, so the member gets its own joint mask and its own rotated Huber
# deltas / component weights; with too few joint-valid rows it silently
# falls back to the unrotated targets.
_ROT_MEMBER = 2
_ROT_MIN_ROWS = 100_000
_ROT_C = math.sqrt(0.5)  # cos(45 deg) == sin(45 deg)

# DIVERSE GBM FAMILIES (verified in gen018_03): family 0 is the exact
# verified configuration and always runs first. Family 1 is deliberately
# diverse along two axes at once: deeper trees with stronger shrinkage and
# heavier regularization (different function class), and a strided data
# window — every 2nd row of the last 1M valid rows — so it trains on twice
# the time span at the same row count (different data view). Both families
# run the verified gate-then-refit protocol independently; the included
# families are AVERAGED into the single tree ensemble member. Under a tight
# clock family 1 is simply skipped.
_GBM_FAMILIES = [
    dict(max_iter=100, learning_rate=0.08, max_leaf_nodes=63,
         min_samples_leaf=40, l2_regularization=1.0, stride=1),
    dict(max_iter=70, learning_rate=0.12, max_leaf_nodes=127,
         min_samples_leaf=80, l2_regularization=2.0, stride=2),
]

# Wall-clock cap for ALL tree fits (both families, gate + recency refit
# each). This is a deadline, not a reservation — the MLP slices scale to
# the time actually remaining when the trees finish, so unused tree budget
# flows back to the MLPs.
_GBM_BUDGET_FRAC = 0.38
_GBM_MAX_ROWS = 500_000

# TAIL-STACKED COMBINE (THE ONE NEW CHANGE): every prior generation averaged
# the ensemble members with EQUAL weights per component, yet member quality
# is measurably unequal and component-dependent (the tree member alone was
# worth +0.012; the rotated and deeper MLPs are deliberately weaker fits of
# the axis-aligned residual). Per component, a non-negative least squares
# fit on the temporal validation tail learns the member weights instead:
# the design matrix holds each member's LEAK-FREE tail prediction — for
# MLPs the selected pre-fine-tune checkpoint (trained strictly before the
# tail), for the tree member the gated pre-tail fits — and the target is
# the post-ridge residual on the tail's valid rows. NNLS on a
# later-in-time tail directly optimizes the forward-in-time criterion the
# score measures, and because its weights need not sum to 1 it doubles as
# a variance-calibrating shrinkage of the residual correction, which
# MSE/R^2 rewards when member predictions are noisy. The fitted weights
# are shrunk _STACK_UNIFORM_W of the way back toward the uniform average
# to bound tail-overfitting, guarded on finiteness / a sane weight sum /
# a minimum tail row count, and applied to the recency-refit test
# predictions; on any guard failure the combine falls back to the exact
# verified equal-weight average, so this cannot lose the champion's
# behavior — it can only reweight it.
_STACK_MIN_ROWS = 20_000
_STACK_UNIFORM_W = 0.5   # fraction of the final weight kept on uniform
_STACK_MAX_WSUM = 5.0    # reject clearly pathological NNLS solutions


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
                  n_inputs, hidden, member_seed, device, max_epochs, t_end):
    """Train one temporally-validated MLP that must finish by wall-clock time
    t_end. Trains with component-balanced masked Huber (robust to outlier
    residuals); validates and early-stops on component-balanced masked MSE,
    the quantity the mean-R^2 metric measures.

    This function is TARGET-BASIS AGNOSTIC: the caller supplies the target
    tensors, masks, deltas, and component weights, so the rotated-basis
    member (see _ROT_MEMBER) reuses this code path unchanged — every
    verified mechanism below applies identically in either basis.

    ARCHITECTURE DIVERSITY (verified in gen018_02): `hidden` is the member's
    own hidden-layer shape from _MEMBER_ARCHS. Member 0 keeps the verified
    (256, 256, 128); the others use wider-shallower / deeper-narrower
    variants so the seed ensemble averages genuinely decorrelated inductive
    biases instead of seed-noise-only clones.

    Budgeted cosine LR (verified in gen016_01): before each epoch the
    learning rate is set to a cosine interpolation between _LR_MAX and
    _LR_MIN driven by the fraction of this member's wall-clock slice already
    spent, so the member is annealed to a low LR exactly as its slice
    expires, regardless of how many epochs that turns out to be.

    Weight EMA: an exponential moving average of the network weights (decay
    _EMA_DECAY) is updated after every optimizer step. Each epoch BOTH the
    raw net and the EMA net are scored on the clean validation tail, and the
    checkpoint keeps whichever is better; because the raw net remains a
    selection candidate, the scheme cannot underperform plain checkpointing.

    Returns (net, val_pred): the net (loaded with the best-found weights,
    then recency fine-tuned) in eval mode, and val_pred — the SELECTED
    pre-fine-tune checkpoint's predictions on the validation tail
    (float32, (n_val, 2), in this member's target basis). That checkpoint
    never trained on the tail, so val_pred is the leak-free signal the
    tail-stacked combine fits member weights on."""
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)
    t_start = time.time()
    # Reserve the last 20% of this member's slice for the recency fine-tune.
    main_t_end = t_start + 0.8 * max(0.0, t_end - t_start)

    net = MLP(n_inputs, hidden=hidden, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=_LR_MAX, weight_decay=1e-4)

    # EMA shadow weights + a second net used only to evaluate the EMA.
    ema_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    ema_net = MLP(n_inputs, hidden=hidden, dropout=0.1).to(device)
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
    main_span = max(1e-6, main_t_end - t_start)

    def _validate(model):
        losses = []
        with torch.no_grad():
            for i in range(0, len(X_v), bs):
                losses.append(_masked_mse(model(X_v[i:i + bs]), Y_v[i:i + bs],
                                          M_v[i:i + bs], comp_w_d).item())
        return float(np.mean(losses))

    for epoch in range(1, max_epochs + 1):
        # Budgeted cosine annealing: LR follows the elapsed fraction of this
        # member's wall-clock slice, so the schedule completes exactly when
        # the slice does — no epoch-count assumption needed.
        frac = min(1.0, max(0.0, (time.time() - t_start) / main_span))
        lr = _LR_MIN + 0.5 * (_LR_MAX - _LR_MIN) * (1.0 + math.cos(math.pi * frac))
        for g in optimizer.param_groups:
            g["lr"] = lr

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
        # the better of the two becomes the checkpoint candidate.
        net.eval()
        val_raw = _validate(net)
        ema_net.load_state_dict(ema_state)
        val_ema = _validate(ema_net)
        val_loss = min(val_raw, val_ema)

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

    # LEAK-FREE TAIL PREDICTIONS for the stacked combine: the selected
    # checkpoint has only ever trained on pre-tail rows, so its tail
    # predictions are an honest out-of-time sample of this member's skill —
    # captured NOW, before the recency fine-tune touches the tail.
    net.eval()
    with torch.no_grad():
        vp = [net(X_v[i:i + bs]).cpu().numpy() for i in range(0, len(X_v), bs)]
    val_pred = (np.concatenate(vp, axis=0) if vp
                else np.zeros((0, 2), dtype=np.float32))

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
    return net, val_pred


def _make_gbm(seed, fam):
    return HistGradientBoostingRegressor(
        max_iter=fam["max_iter"], learning_rate=fam["learning_rate"],
        max_leaf_nodes=fam["max_leaf_nodes"],
        min_samples_leaf=fam["min_samples_leaf"],
        l2_regularization=fam["l2_regularization"], max_bins=255,
        early_stopping=False, random_state=seed)


def _fit_gbm_members(X_raw, X_test_raw, Ys, mask, train_idx, val_idx, seed,
                     deadline):
    """Fit the tree-family residual models on the post-ridge residuals — the
    DECORRELATED model family of the ensemble (verified +0.012 in gen014_01;
    recency refit verified in gen014_02; diverse families verified in
    gen018_03).

    Trees see the RAW augmented features with NaNs intact: HistGBR handles
    missing values natively by learning a per-split direction for missings,
    a fundamentally different treatment of swath-edge cells than the
    mean-imputation the ridge and MLPs rely on, so its errors decorrelate
    from theirs.

    Gate-then-refit, per family and component: the gating fit uses that
    family's window over the PRE-validation span, so the untouched temporal
    tail gates inclusion — a family joins only if it beats the zero-residual
    predictor (ridge alone) on that tail, so a failed family dilutes
    nothing. A family that passes is refit with identical hyperparameters on
    its window over the FULL span — shifted forward to include the tail, the
    rows closest in time to the test cycles — and the refit model predicts
    the test set. Deadline checks before every fit fall back gracefully.

    The included families are AVERAGED into ONE tree prediction per
    component. For the tail-stacked combine, the GATED (pre-tail) fits'
    predictions on the full validation tail are averaged the same way —
    those models never saw the tail, so they are the tree member's
    leak-free stacking signal.

    Returns (pred_test (n_test, 2) float64, included (2,) bool,
             val_pred (n_val, 2) float64).
    """
    n_test = len(X_test_raw)
    n_val = len(val_idx)
    pred_sum = np.zeros((n_test, 2), dtype=np.float64)
    val_sum = np.zeros((n_val, 2), dtype=np.float64)
    cnt = np.zeros(2, dtype=np.int64)
    for f, fam in enumerate(_GBM_FAMILIES):
        if time.time() > deadline:
            break
        stride = int(fam["stride"])
        window = _GBM_MAX_ROWS * stride
        for c in range(2):
            if time.time() > deadline:
                break
            rows = train_idx[mask[train_idx, c] > 0]
            if len(rows) < 10_000:
                continue
            rows = rows[-window:][::stride]
            gbm = _make_gbm(seed + 31 * c + 101 * f, fam)
            gbm.fit(X_raw[rows], Ys[rows, c])
            vm = mask[val_idx, c] > 0
            if not vm.any():
                del gbm
                continue
            # Predict the WHOLE tail once: the valid subset gates inclusion,
            # the full vector feeds the stacked combine if the gate passes.
            pv_full = gbm.predict(X_raw[val_idx])
            pv = pv_full[vm]
            y_val = Ys[val_idx, c][vm].astype(np.float64)
            mse_gbm = float(np.mean((pv - y_val) ** 2))
            mse_zero = float(np.mean(y_val ** 2))
            if mse_gbm < mse_zero:
                pred_c = gbm.predict(X_test_raw)
                # RECENCY REFIT: same hyperparameters, same window size, but
                # the window now ends at the last row of the full span —
                # rows are in cycle (time) order, so this is the data
                # closest to the test cycles. Only attempted when the clock
                # allows; otherwise the gated pre-tail prediction stands.
                if time.time() < deadline:
                    full_rows = np.flatnonzero(
                        mask[:, c] > 0)[-window:][::stride]
                    gbm_r = _make_gbm(seed + 31 * c + 101 * f + 7, fam)
                    gbm_r.fit(X_raw[full_rows], Ys[full_rows, c])
                    pred_c = gbm_r.predict(X_test_raw)
                    del gbm_r
                pred_sum[:, c] += pred_c
                val_sum[:, c] += pv_full
                cnt[c] += 1
            del gbm
    pred_test = np.zeros((n_test, 2), dtype=np.float64)
    val_pred = np.zeros((n_val, 2), dtype=np.float64)
    for c in range(2):
        if cnt[c] > 0:
            pred_test[:, c] = pred_sum[:, c] / cnt[c]
            val_pred[:, c] = val_sum[:, c] / cnt[c]
    return pred_test, cnt > 0, val_pred


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
    # as a contiguous tail so early stopping / LR scheduling / GBM gating /
    # combine-weight stacking select for forward-in-time generalization
    # instead of memorization.
    n = len(Xs)
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    # GBM MEMBERS: two diverse tree families per component, each gated on the
    # pre-tail fit and recency-refit on the full span (see _fit_gbm_members),
    # averaged into ONE tree ensemble member and capped to a slice of the
    # wall clock; unused tree time flows back to the MLP ensemble because
    # the MLP slices are computed from the time actually remaining.
    gbm_deadline = t0 + _GBM_BUDGET_FRAC * time_budget_s
    gbm_pred_test, gbm_included, gbm_val_pred = _fit_gbm_members(
        X_train, X_test, Ys, mask, train_idx, val_idx, seed, gbm_deadline)

    # Build all tensors ONCE and share them across ensemble members.
    X_t, Y_t, M_t = (torch.from_numpy(a[train_idx]) for a in (Xs, Ys, mask))
    X_v = torch.from_numpy(Xs[val_idx]).to(device)
    Y_v = torch.from_numpy(Ys[val_idx]).to(device)
    M_v = torch.from_numpy(mask[val_idx]).to(device)
    X_f = torch.from_numpy(Xs)
    Y_f = torch.from_numpy(Ys)
    M_f = torch.from_numpy(mask)

    # ROTATED-BASIS TARGETS (verified direction in gen019_00): build the
    # 45-degree-rotated residuals, joint-validity mask, and rotated deltas/
    # component weights for member _ROT_MEMBER. Rotation mixes u and v, so a
    # rotated target is valid only where BOTH components are; the joint mask
    # is applied to both rotated columns. Enabled only when enough
    # joint-valid rows exist — otherwise the member silently trains on the
    # unrotated targets.
    c_, s_ = _ROT_C, _ROT_C
    joint = (mask[:, 0] * mask[:, 1]).astype(np.float32)
    rot_ok = int(joint.sum()) >= _ROT_MIN_ROWS
    if rot_ok:
        Ys_rot = np.stack([c_ * Ys[:, 0] + s_ * Ys[:, 1],
                           -s_ * Ys[:, 0] + c_ * Ys[:, 1]], axis=1)
        mask_rot = np.repeat(joint[:, None], 2, axis=1).astype(np.float32)
        Ys_rot = np.where(mask_rot > 0, Ys_rot, 0.0).astype(np.float32)
        delta_rot_np = np.array([
            float(np.sqrt(np.mean(Ys_rot[mask_rot[:, cc] > 0, cc] ** 2)))
            if (mask_rot[:, cc] > 0).any() else 1.0
            for cc in range(2)
        ], dtype=np.float32)
        delta_rot_np = np.clip(delta_rot_np, 1e-3, None)
        delta_rot = torch.from_numpy(delta_rot_np)
        inv_var_r = 1.0 / np.maximum(delta_rot_np.astype(np.float64) ** 2, 1e-6)
        comp_w_rot = torch.from_numpy(
            (inv_var_r / inv_var_r.mean()).astype(np.float32))
        Yr_t = torch.from_numpy(Ys_rot[train_idx])
        Mr_t = torch.from_numpy(mask_rot[train_idx])
        Yr_v = torch.from_numpy(Ys_rot[val_idx]).to(device)
        Mr_v = torch.from_numpy(mask_rot[val_idx]).to(device)
        Yr_f = torch.from_numpy(Ys_rot)
        Mr_f = torch.from_numpy(mask_rot)

    # SEED + ARCHITECTURE + TARGET-BASIS ENSEMBLE within the REMAINING
    # wall-clock budget (after the GBM slice). Member 0 gets half of what's
    # left AND the verified architecture, so even under a tight clock the
    # ensemble contains at least one full-strength known-good model; members
    # 1-2 split the remainder with diverse architectures, and member
    # _ROT_MEMBER additionally trains in the rotated target basis for a
    # decorrelated error axis. Each member's test prediction is kept
    # SEPARATELY (not summed) so the tail-stacked combine can weight them.
    slice_ends = [0.50, 0.78, 1.0]
    Xp = _standardize(X_test, mean, scale)
    lin_test = (Xp @ B + b_int).astype(np.float64)
    member_test = []   # per-member (n_test, 2) float64, (u, v) basis
    member_val = []    # per-member (n_val, 2) float64, (u, v) basis

    t_ml0 = time.time()
    remaining_total = max(1.0, t0 + time_budget_s - t_ml0)
    for m, frac in enumerate(slice_ends):
        t_end = t_ml0 + remaining_total * frac
        remaining = t_end - time.time()
        # Don't start a member whose slice is already (nearly) gone —
        # a barely-trained net would only dilute the average.
        if m > 0 and remaining < 0.08 * remaining_total:
            break
        hidden = _MEMBER_ARCHS[m % len(_MEMBER_ARCHS)]
        use_rot = rot_ok and (m == _ROT_MEMBER)
        if use_rot:
            net, val_pred = _train_member(
                X_t, Yr_t, Mr_t, X_v, Yr_v, Mr_v, X_f, Yr_f, Mr_f, delta_rot,
                comp_w_rot, Xs.shape[1], hidden, seed + 1000 * m, device,
                max_epochs, t_end)
            # Rotate the tail predictions back to the (u, v) basis (inverse
            # of the target rotation, i.e. the transpose).
            val_pred = np.stack([c_ * val_pred[:, 0] - s_ * val_pred[:, 1],
                                 s_ * val_pred[:, 0] + c_ * val_pred[:, 1]],
                                axis=1)
        else:
            net, val_pred = _train_member(
                X_t, Y_t, M_t, X_v, Y_v, M_v, X_f, Y_f, M_f, delta, comp_w,
                Xs.shape[1], hidden, seed + 1000 * m, device, max_epochs,
                t_end)
        pred_m = np.zeros((len(Xp), 2), dtype=np.float64)
        with torch.no_grad():
            for i in range(0, len(Xp), 65536):
                xb = torch.from_numpy(Xp[i:i + 65536]).to(device)
                p = net(xb).cpu().numpy()
                if use_rot:
                    p = np.stack([c_ * p[:, 0] - s_ * p[:, 1],
                                  s_ * p[:, 0] + c_ * p[:, 1]], axis=1)
                pred_m[i:i + 65536] = p
        member_test.append(pred_m)
        member_val.append(val_pred.astype(np.float64))
        del net
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # TAIL-STACKED COMBINE (the one new change): per component, learn
    # non-negative member weights on the temporal validation tail from the
    # LEAK-FREE tail predictions (pre-fine-tune MLP checkpoints, gated
    # pre-tail GBM fits), shrink them _STACK_UNIFORM_W of the way back
    # toward the verified uniform average, and apply them to the
    # recency-refit test predictions. Guards on tail size, finiteness, and
    # a sane weight sum; any failure falls back to the exact equal-weight
    # combine of the champion.
    out = lin_test.copy()
    for c in range(2):
        cols_test = [p[:, c] for p in member_test]
        cols_val = [p[:, c] for p in member_val]
        if gbm_included[c]:
            cols_test.append(gbm_pred_test[:, c])
            cols_val.append(gbm_val_pred[:, c])
        k = len(cols_test)
        if k == 0:
            continue
        w = np.full(k, 1.0 / k, dtype=np.float64)
        vm = mask[val_idx, c] > 0
        if k >= 2 and int(vm.sum()) >= _STACK_MIN_ROWS:
            P = np.stack([cv[vm] for cv in cols_val], axis=1)
            yv = Ys[val_idx, c][vm].astype(np.float64)
            try:
                w_fit, _ = nnls(P, yv)
                if (np.all(np.isfinite(w_fit))
                        and 1e-6 < float(w_fit.sum()) < _STACK_MAX_WSUM):
                    w = _STACK_UNIFORM_W * w + (1.0 - _STACK_UNIFORM_W) * w_fit
            except Exception:
                pass
        out[:, c] += np.stack(cols_test, axis=1) @ w
    out = out.astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
