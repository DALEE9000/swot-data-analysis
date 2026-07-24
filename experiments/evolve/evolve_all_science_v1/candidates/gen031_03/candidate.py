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
import math
import time
import warnings

import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge


def _cells(d):
    # Columns are k*k spatial-stencil copies of base features, feature-major.
    for c in (9, 25, 49):
        if d % c == 0:
            return c
    return 1


def _featurize(X, mean, scale, n_cells):
    """Proven lineage plumbing, unchanged: standardize, gradient-preserving
    mirror imputation, per-cell missing fraction, per-feature stencil std,
    first/second-order spatial derivatives, gradient magnitude, and the
    nonlinear flow-regime block (speeds, vorticity, strains, Okubo-Weiss,
    geostrophic advection) when F == 8. The tree core consumes the whole
    vector flat; the missingness and regime columns give it explicit
    split variables for edge/interior and slow/fast gating."""
    Xs = (X.astype(np.float32) - mean) / scale
    n, d = Xs.shape
    F = d // n_cells
    miss = np.isnan(X).reshape(n, F, n_cells).mean(axis=1).astype(np.float32)
    Z = Xs.reshape(n, F, n_cells)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spat = np.nanstd(Z, axis=2).astype(np.float32)
    spat = np.nan_to_num(spat, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    center = Z[:, :, n_cells // 2]
    mirror = Z[:, :, ::-1].copy()
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
    k = int(round(math.sqrt(n_cells)))
    if k >= 2 and k * k == n_cells:
        Zf = Xs[:, :F * n_cells].reshape(n, F, n_cells)
        ci = n_cells // 2
        gx = (0.5 * (Zf[:, :, ci + 1] - Zf[:, :, ci - 1])).astype(np.float32)
        gy = (0.5 * (Zf[:, :, ci + k] - Zf[:, :, ci - k])).astype(np.float32)
        blocks += [gx, gy]
        if k >= 3:
            lap = (Zf[:, :, ci + 1] + Zf[:, :, ci - 1] + Zf[:, :, ci + k]
                   + Zf[:, :, ci - k] - 4.0 * Zf[:, :, ci]).astype(np.float32)
            inv2 = np.float32(0.5 / math.sqrt(2.0))
            gp = (inv2 * (Zf[:, :, ci + k + 1] - Zf[:, :, ci - k - 1])).astype(np.float32)
            gq = (inv2 * (Zf[:, :, ci - k + 1] - Zf[:, :, ci + k - 1])).astype(np.float32)
            blocks += [lap, gp, gq]
        gmag = np.sqrt(gx * gx + gy * gy).astype(np.float32)
        blocks.append(gmag)
        if F == 8:
            cu, cv = Zf[:, 2, ci], Zf[:, 3, ci]
            au, av = Zf[:, 4, ci], Zf[:, 5, ci]
            wu, wv = Zf[:, 6, ci], Zf[:, 7, ci]
            spd = np.sqrt(cu * cu + cv * cv)
            spda = np.sqrt(au * au + av * av)
            spdw = np.sqrt(wu * wu + wv * wv)
            vort = gx[:, 3] - gy[:, 2]
            sn = gx[:, 2] - gy[:, 3]
            ss = gx[:, 3] + gy[:, 2]
            ow = np.clip(sn * sn + ss * ss - vort * vort, -16.0, 16.0)
            reg = np.stack([spd, spda, spdw, vort, sn, ss, ow],
                           axis=1).astype(np.float32)
            adv = np.clip(cu[:, None] * gx + cv[:, None] * gy,
                          -16.0, 16.0).astype(np.float32)
            blocks += [reg, adv]
    return np.hstack(blocks)


def _recency_subsample(rows, w, cap, rng):
    """Recency-biased sample WITHOUT replacement via the Gumbel top-k trick:
    key = log(w) + Gumbel noise, keep the cap largest keys. Equivalent to
    sequential sampling proportional to w, O(n) — replaces per-row
    sample_weight so the fitted trees see the same recency emphasis as the
    lineage's weighted loss while bounding per-iteration boosting cost."""
    if len(rows) <= cap:
        return rows, w[rows]
    g = rng.gumbel(size=len(rows)).astype(np.float32)
    keys = np.log(np.maximum(w[rows], 1e-12)).astype(np.float32) + g
    keep = np.argpartition(keys, -cap)[-cap:]
    return rows[keep], None


def train_and_predict(X_train, Y_train, X_test, params):
    seed = int(params["seed"])
    max_epochs = int(params["max_epochs"])
    time_budget_s = float(params["time_budget_s"])
    t0 = time.time()
    rng = np.random.default_rng(seed)

    mean = np.nan_to_num(np.nanmean(X_train, axis=0), nan=0.0).astype(np.float32)
    scale = np.nanstd(X_train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)

    n_cells = _cells(X_train.shape[1])
    Xs = _featurize(X_train, mean, scale, n_cells)
    mask = np.isfinite(Y_train)
    Ys = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)
    n = len(Xs)

    # ---- Ridge baseline + residual learning (proven lineage plumbing) ----
    # The linear map extrapolates the dominant geostrophic signal into the
    # fast tail, where tree piecewise-constant leaves (like saturating
    # activations) cannot extrapolate; the trees fit only the residual.
    sub = rng.choice(n, size=min(n, 1_500_000), replace=False)
    W_lin = np.zeros((Xs.shape[1], 2), dtype=np.float32)
    b_lin = np.zeros(2, dtype=np.float32)
    for c in range(2):
        rows = sub[mask[sub, c]]
        if len(rows) > 1000:
            ridge = Ridge(alpha=1.0)
            ridge.fit(Xs[rows], Ys[rows, c])
            W_lin[:, c] = ridge.coef_.astype(np.float32)
            b_lin[c] = np.float32(ridge.intercept_)

    def _baseline(Xf):
        return np.clip(Xf @ W_lin + b_lin, -3.0, 3.0).astype(np.float32)

    base_tr = np.empty((n, 2), dtype=np.float32)
    for i in range(0, n, 1_000_000):
        base_tr[i:i + 1_000_000] = _baseline(Xs[i:i + 1_000_000])
    Yr = np.where(mask, Ys - base_tr, 0.0).astype(np.float32)
    del base_tr

    # Temporal validation split: last 10% of the time-ordered window, so
    # early stopping optimizes forward-in-time generalization.
    n_val = max(1, int(n * 0.1))
    n_tr = n - n_val

    # Recency emphasis: latest rows 3x the earliest (exponential ramp), as
    # in the lineage — applied here via biased subsampling, not loss weights.
    pos = (np.arange(n, dtype=np.float32) / max(n - 1, 1)).astype(np.float32)
    w_all = np.exp(np.log(3.0) * pos).astype(np.float32)

    # ---- THE WILDCARD CORE: per-component boosted-tree ensembles ----
    # Warm-start chunked boosting: each chunk adds CHUNK trees, then checks
    # temporal-val MSE and the wall clock. One chunk plays the role of one
    # "epoch" (capped by max_epochs). Stop after 2 stale chunks. warm_start
    # cannot roll back to the best chunk; the small learning rate bounds the
    # overshoot to <=2 chunks of trees.
    CHUNK = 25
    ROW_CAP = 2_400_000
    train_deadline = t0 + time_budget_s * 0.80  # reserve tail for predict
    models = [None, None]
    for c in range(2):
        comp_deadline = min(train_deadline,
                            t0 + time_budget_s * 0.80 * (c + 1) / 2.0)
        if time.time() > comp_deadline - 60.0:
            continue  # baseline-only for this component
        rows_t = np.nonzero(mask[:n_tr, c])[0]
        rows_v = np.nonzero(mask[n_tr:, c])[0] + n_tr
        if len(rows_t) < 1000 or len(rows_v) < 100:
            continue
        rows_t, sw = _recency_subsample(rows_t, w_all, ROW_CAP, rng)
        Xt, yt = Xs[rows_t], Yr[rows_t, c]
        vsub = rows_v if len(rows_v) <= 300_000 else \
            rng.choice(rows_v, size=300_000, replace=False)
        Xv, yv = Xs[vsub], Yr[vsub, c]
        model = HistGradientBoostingRegressor(
            loss="squared_error", learning_rate=0.06,
            max_leaf_nodes=255, min_samples_leaf=100,
            l2_regularization=1.0, max_bins=255,
            warm_start=True, early_stopping=False,
            random_state=seed + 17 * c, max_iter=CHUNK)
        best_val = float(np.mean(yv ** 2))  # residual-zero reference
        stale = 0
        iters = 0
        for _ in range(max(1, max_epochs)):
            iters += CHUNK
            model.set_params(max_iter=iters)
            if sw is not None:
                model.fit(Xt, yt, sample_weight=sw)
            else:
                model.fit(Xt, yt)
            v = float(np.mean((model.predict(Xv) - yv) ** 2))
            if v < best_val - 1e-7:
                best_val = v
                stale = 0
            else:
                stale += 1
            if stale >= 2 or time.time() > comp_deadline:
                break
        # Guard: only keep the trees if they beat baseline-only on the
        # temporal val split; worst case degenerates to the ridge baseline.
        if best_val < float(np.mean(yv ** 2)):
            models[c] = model
    del Xs, Yr

    # Predict: ridge baseline + tree residuals, chunked over the test set.
    out = np.zeros((len(X_test), 2), dtype=np.float32)
    for i in range(0, len(X_test), 200_000):
        Xf = _featurize(X_test[i:i + 200_000], mean, scale, n_cells)
        pred = _baseline(Xf)
        for c in range(2):
            if models[c] is not None:
                pred[:, c] += models[c].predict(Xf).astype(np.float32)
        out[i:i + 200_000] = pred
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
