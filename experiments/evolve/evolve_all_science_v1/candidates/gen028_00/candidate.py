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
from sklearn.linear_model import Ridge


def _cells(d):
    # Columns are k*k spatial-stencil copies of base features, feature-major.
    # Infer the stencil cell count so the appended blocks stay layout-correct.
    for c in (9, 25, 49):
        if d % c == 0:
            return c
    return 1


def _featurize(X, mean, scale, n_cells):
    """Standardize, gradient-fill missing stencil cells, append missingness,
    spatial std, explicit per-feature spatial derivatives (first AND second
    order), and NONLINEAR flow-regime features.

    Proven plumbing (unchanged from the lineage):
      * per-cell missing fraction (n, C);
      * per-feature stencil std over VALID cells (n, F) — front/eddy signal;
      * gradient-preserving mirror imputation (2*center - mirror, clamped to
        +/-4 sigma, center fallback) so swath-edge truncation keeps the
        first-order gradient instead of flattening the neighborhood;
      * per-feature central differences gx, gy (n, 2F);
      * per-feature 5-point Laplacian and the two diagonal central
        differences (n, 3F);
      * |grad f| per feature (n, F);
      * flow-regime block when F == 8 (speeds, vorticity, strains,
      Okubo-Weiss, geostrophic advection).

    Layout matters downstream: the conv-wide-and-deep net slices the FIRST
    F*C columns as the raw standardized/imputed stencil image and the NEXT
    C columns as the per-cell missingness mask channel; everything after
    the image block (mask included) feeds the dense path. The MLP member
    consumes the whole vector flat.

    CRITICALLY for this generation: the edge-truncation augmentation feeds
    synthetic NaN'd rows through THIS exact function, so every downstream
    signal (mask channel, missing fractions, mirror-imputed gradients,
    nan-std) reacts to augmented rows precisely as it does to real
    swath-edge rows at test time.
    """
    Xs = (X.astype(np.float32) - mean) / scale
    n, d = Xs.shape
    F = d // n_cells
    miss = np.isnan(X).reshape(n, F, n_cells).mean(axis=1).astype(np.float32)
    Z = Xs.reshape(n, F, n_cells)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spat = np.nanstd(Z, axis=2).astype(np.float32)
    spat = np.nan_to_num(spat, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    # Fill AFTER the std block (which must see true valid cells). Snapshot the
    # point-reflected cells before any in-place filling so late cells never
    # read an already-imputed mirror as if it were data.
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
            # Pooled layout: [mdt, ssha, ugos, vgos, ugosa, vgosa,
            # era5_u, era5_v]; center-cell standardized values.
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


class ConvWideDeep(nn.Module):
    """Conv path over the raw stencil image + dense path over the engineered
    scalars, joined into one trunk (unchanged from the parent).

    Conv path: first F*C input columns reshaped to an (F, k, k) image with
    the per-cell missing fraction appended as an explicit mask channel, so
    filters learn edge-aware local operators the fixed central differences
    cannot express. Two padded 3x3 convolutions; with k=3 the second
    layer's receptive field covers the whole window (~20k conv params).

    Dense path: everything from the mask block onward through a small MLP —
    the proven engineered features kept as a shortcut.

    Heads: mu (ridge-residual predictions) and logvar (anchored-NLL
    weighting), as in the whole lineage.
    """

    def __init__(self, d_total, n_feat, n_cells, dropout=0.1,
                 logvar_init=-3.5):
        super().__init__()
        self.F = n_feat
        self.C = n_cells
        self.k = int(round(math.sqrt(n_cells)))
        self.img_dim = n_feat * n_cells
        ch1, ch2 = 32, 48
        self.conv = nn.Sequential(
            nn.Conv2d(n_feat + 1, ch1, 3, padding=1), nn.SiLU(),
            nn.Conv2d(ch1, ch2, 3, padding=1), nn.SiLU(),
        )
        self.conv_proj = nn.Sequential(
            nn.Linear(ch2 * n_cells, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Dropout(dropout),
        )
        dd = d_total - self.img_dim  # mask block + all engineered scalars
        self.dense = nn.Sequential(
            nn.Linear(dd, 256), nn.LayerNorm(256), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(256, 128), nn.LayerNorm(128), nn.SiLU(), nn.Dropout(dropout),
        )
        self.trunk = nn.Sequential(
            nn.Linear(256, 128), nn.LayerNorm(128), nn.SiLU(), nn.Dropout(dropout),
        )
        self.mu = nn.Linear(128, 2)
        self.logvar = nn.Linear(128, 2)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, logvar_init)

    def forward(self, x):
        b = x.shape[0]
        img = x[:, :self.img_dim].view(b, self.F, self.k, self.k)
        msk = x[:, self.img_dim:self.img_dim + self.C].view(b, 1, self.k, self.k)
        zc = self.conv_proj(self.conv(torch.cat([img, msk], dim=1)).flatten(1))
        zd = self.dense(x[:, self.img_dim:])
        z = self.trunk(torch.cat([zc, zd], dim=1))
        return self.mu(z), self.logvar(z)


class HeteroMLP(nn.Module):
    """Lineage MLP core, kept as a first-class ensemble member: its flat
    fully-connected inductive bias over the whole engineered vector
    decorrelates errors from the conv member's local translation-shared
    filters far more than a second seed of one architecture would — that
    decorrelation is what the prediction average harvests. Notably the
    stronger-r2_u member in the lineage, which the component weighting
    leans on."""

    def __init__(self, n_inputs, hidden=(256, 256, 128), dropout=0.1,
                 logvar_init=-3.5):
        super().__init__()
        layers = []
        d = n_inputs
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.mu = nn.Linear(d, 2)
        self.logvar = nn.Linear(d, 2)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, logvar_init)

    def forward(self, x):
        z = self.trunk(x)
        return self.mu(z), self.logvar(z)


def _make_net(d_total, n_feat, n_cells, device, arch):
    """arch is "conv" or "mlp". Non-square stencils force the MLP (the conv
    reshape would be invalid), so the pooled k=3 dataset gets one of each
    and degenerate layouts get two MLPs — never a crash."""
    k = int(round(math.sqrt(n_cells)))
    if arch == "conv" and n_cells > 1 and k * k == n_cells:
        return ConvWideDeep(d_total, n_feat, n_cells, dropout=0.1).to(device)
    return HeteroMLP(d_total, hidden=(256, 256, 128), dropout=0.1).to(device)


def _masked_nll_anchored(mu, logvar, target, mask, row_w, comp_w):
    """Gaussian NLL plus an unweighted MSE anchor, per valid component,
    scaled by a per-row recency weight AND a per-component inverse-variance
    weight (kept from the parent). The score is mean per-component R^2 =
    mean_c(1 - MSE_c/Var_c), so a unit of squared error on the
    lower-variance component (u) buys more R^2 than the same error on v;
    comp_w = (1/Var_u, 1/Var_v) mean-1 normalized makes the loss's exchange
    rate between u and v errors match the metric's. The anchor keeps a
    gradient floor on variance-downweighted rows, as in the lineage."""
    logvar = logvar.clamp(-7.0, 2.0)
    err2 = (mu - target).pow(2)
    per = 0.5 * (logvar + err2 * torch.exp(-logvar)) + err2
    w = mask * row_w.unsqueeze(1) * comp_w.unsqueeze(0)
    return (per * w).sum() / w.sum().clamp(min=1e-8)


def _masked_mse(pred, target, mask, comp_w):
    """Component-weighted masked MSE. With comp_w proportional to the
    inverse target variances this is (up to a constant) the mean per-
    component R^2 deficit — the scored quantity — so early stopping,
    raw-vs-EMA selection, the LR schedule, and the member guard all
    optimize the metric directly instead of pooled MSE."""
    w = mask * comp_w.unsqueeze(0)
    diff2 = (pred - target).pow(2) * w
    return diff2.sum() / w.sum().clamp(min=1e-8)


class _EMA:
    """Exponential moving average of a net's weights along the SGD
    trajectory (SWA-family); shadow sits in the flatter basin center,
    transfers better across the temporal train->test shift."""

    def __init__(self, net, decay):
        self.decay = decay
        self.net = copy.deepcopy(net)
        for p in self.net.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def update(self, net):
        for pe, pn in zip(self.net.parameters(), net.parameters()):
            pe.lerp_(pn, 1.0 - self.decay)
        for be, bn in zip(self.net.buffers(), net.buffers()):
            be.copy_(bn)


def _val_mse(net, X_v, Y_v, M_v, bs, comp_w):
    net.eval()
    with torch.no_grad():
        losses = []
        for i in range(0, len(X_v), bs):
            mu, _ = net(X_v[i:i + bs])
            losses.append(_masked_mse(mu, Y_v[i:i + bs], M_v[i:i + bs],
                                      comp_w).item())
    return float(np.mean(losses))


def _train_member(member_seed, arch, deadline, device, max_epochs,
                  n_feat, n_cells, comp_w,
                  X_t, Y_t, M_t, W_t, X_v, Y_v, M_v,
                  X_v_cpu, Y_v_cpu, M_v_cpu, W_v_cpu):
    """Train one ensemble member of the requested architecture with the
    recency- and component-weighted anchored heteroscedastic NLL; selection
    over raw-vs-EMA on component-weighted temporal-val masked MSE (the
    scored quantity); then the proven recency fine-tune ending on a
    short-horizon EMA. Returns (net, best_val)."""
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)

    net = _make_net(X_t.shape[1], n_feat, n_cells, device, arch)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    bs = 4096
    patience = 15
    best_val = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    best_epoch = 0
    n_train = len(X_t)

    ema = _EMA(net, decay=0.999)

    for epoch in range(1, max_epochs + 1):
        net.train()
        order = torch.from_numpy(rng.permutation(n_train))
        for i in range(0, n_train, bs):
            idx = order[i:i + bs]
            xb, yb, mb = X_t[idx].to(device), Y_t[idx].to(device), M_t[idx].to(device)
            wb = W_t[idx].to(device)
            optimizer.zero_grad()
            mu, lv = net(xb)
            loss = _masked_nll_anchored(mu, lv, yb, mb, wb, comp_w)
            loss.backward()
            optimizer.step()
            ema.update(net)

        val_raw = _val_mse(net, X_v, Y_v, M_v, bs, comp_w)
        val_ema = _val_mse(ema.net, X_v, Y_v, M_v, bs, comp_w)
        scheduler.step(val_raw)
        val_loss = min(val_raw, val_ema)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            src = net if val_raw <= val_ema else ema.net
            best_state = copy.deepcopy(src.state_dict())
            best_epoch = epoch

        if epoch - best_epoch >= patience:
            break
        if time.time() > deadline:
            break

    net.load_state_dict(best_state)

    # Recency fine-tune (proven +~0.01 in lineage): absorb the held-out
    # most-recent 10% with a short low-LR pass over the full window, ending
    # on a short-horizon EMA of the fine-tune trajectory.
    if time.time() < deadline and np.isfinite(best_val):
        X_f = torch.cat([X_t, X_v_cpu])
        Y_f = torch.cat([Y_t, Y_v_cpu])
        M_f = torch.cat([M_t, M_v_cpu])
        W_f = torch.cat([W_t, W_v_cpu])
        ft_opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
        ft_ema = _EMA(net, decay=0.998)
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
                mu, lv = net(xb)
                loss = _masked_nll_anchored(mu, lv, yb, mb, wb, comp_w)
                loss.backward()
                ft_opt.step()
                ft_ema.update(net)
                if bi % 100 == 0 and time.time() > deadline:
                    out_of_time = True
                    break
            if out_of_time:
                break
        net.load_state_dict(ft_ema.net.state_dict())

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
    n_feat = X_train.shape[1] // n_cells

    # Per-row NaN flag BEFORE featurize consumes the NaNs: full-stencil rows
    # (no NaN anywhere) are the augmentation candidates — for them we KNOW
    # the target is supervised by a complete neighborhood, so truncating
    # them synthetically yields (truncated input, trusted target) pairs.
    row_full = ~np.isnan(X_train).any(axis=1)

    Xs = _featurize(X_train, mean, scale, n_cells)
    mask = np.isfinite(Y_train).astype(np.float32)
    Ys = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)
    n = len(Xs)

    # Per-component inverse-variance weights (kept from the parent).
    # Score = mean_c(1 - MSE_c/Var_c): weighting each component's squared
    # error by 1/Var_c (of the ORIGINAL targets — the additive ridge
    # baseline preserves MSE identity, and R^2 is computed on the final
    # predictions) makes both the training loss and every validation-based
    # selection decision optimize mean R^2 rather than pooled MSE, which
    # implicitly favors the higher-variance component. Mean-1 normalized so
    # the tuned loss scale is unchanged; clipped to [0.5, 2] relative so a
    # pathological variance ratio cannot destabilize training.
    cw = np.ones(2, dtype=np.float32)
    for c in range(2):
        vc = float(np.var(Y_train[mask[:, c] > 0, c])) if mask[:, c].sum() > 1000 else 0.0
        cw[c] = 1.0 / vc if vc > 1e-8 else 1.0
    cw = np.clip(cw / cw.mean(), 0.5, 2.0).astype(np.float32)
    cw /= cw.mean()
    comp_w = torch.from_numpy(cw).to(device)

    # ---- Linear baseline + residual learning (unchanged) ----
    # Ridge from the full engineered block to (u, v) on a seeded subsample;
    # the nets fit the residual. The linear map extrapolates the dominant
    # geostrophic signal into the fast tail where bounded activations
    # saturate; baseline clipped to +/-3 m/s. Additive per-row baseline
    # means residual val MSE == final-prediction val MSE, so all selection
    # still optimizes the scored quantity.
    rng0 = np.random.default_rng(seed)
    sub = rng0.choice(n, size=min(n, 1_500_000), replace=False)
    W_lin = np.zeros((Xs.shape[1], 2), dtype=np.float32)
    b_lin = np.zeros(2, dtype=np.float32)
    for c in range(2):
        rows = sub[mask[sub, c] > 0]
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
    Ys = np.where(mask > 0, Ys - base_tr, 0.0).astype(np.float32)
    del base_tr

    # Temporal validation split: last 10% of the time-ordered window, so
    # early stopping and the LR schedule optimize forward-in-time
    # generalization — what the temporal test holdout measures.
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    # Recency weights: latest rows weigh 3x the earliest (exponential ramp,
    # mean-1 normalized so the tuned loss scale is unchanged).
    pos = (np.arange(n, dtype=np.float32) / max(n - 1, 1)).astype(np.float32)
    w_all = np.exp(np.log(3.0) * pos).astype(np.float32)
    w_all /= w_all.mean()

    # ---- Edge-truncation augmentation (THE one change vs the parent) ----
    # The edge-stencil bin is the parent's widest deficit (r2_u 0.287 vs
    # 0.476 full-stencil) and loss upweighting of real edge rows measured
    # neutral-to-negative in the sibling lineage — the bottleneck is
    # supervision density, not gradient allocation. Fix it with data: take
    # 12% of FULL-stencil training rows (never validation rows), NaN out one
    # entire stencil side — left/right column or top/bottom row of the k x k
    # window, the truncation pattern real swath edges produce — and run them
    # through the byte-identical featurize, so mirror imputation, the
    # missingness mask channel, per-cell missing fractions, and nan-std all
    # respond exactly as they do for genuine edge rows. Residual targets
    # come from the SAME ridge baseline evaluated on the AUGMENTED features
    # (additivity preserved: net + baseline still reconstructs the original
    # target). Augmented rows inherit their source rows' recency weights and
    # are appended to the training tensors only; validation, member
    # selection, the loss form, and the prediction path are untouched.
    aug = None
    k_st = int(round(math.sqrt(n_cells)))
    if k_st >= 3 and k_st * k_st == n_cells:
        rngA = np.random.default_rng(seed + 7)
        cand = train_idx[row_full[train_idx]]
        n_aug = min(len(cand), int(0.12 * len(train_idx)))
        if n_aug > 1000:
            sel = rngA.choice(cand, size=n_aug, replace=False)
            X_a = X_train[sel].astype(np.float32).copy()
            pat = rngA.integers(0, 4, size=n_aug)
            for p in range(4):
                if p == 0:
                    cells = [r * k_st for r in range(k_st)]                  # left col
                elif p == 1:
                    cells = [r * k_st + (k_st - 1) for r in range(k_st)]     # right col
                elif p == 2:
                    cells = list(range(k_st))                                # top row
                else:
                    cells = list(range((k_st - 1) * k_st, n_cells))          # bottom row
                cols = [f * n_cells + c for f in range(n_feat) for c in cells]
                rows_p = np.where(pat == p)[0]
                if len(rows_p):
                    X_a[np.ix_(rows_p, cols)] = np.nan
            Xa_parts = []
            for i in range(0, n_aug, 500_000):
                Xa_parts.append(_featurize(X_a[i:i + 500_000], mean, scale, n_cells))
            Xa = np.vstack(Xa_parts) if len(Xa_parts) > 1 else Xa_parts[0]
            del Xa_parts, X_a
            Ya = Y_train[sel].astype(np.float32)
            Ma = np.isfinite(Ya).astype(np.float32)
            Ya = np.nan_to_num(Ya, nan=0.0)
            base_a = np.empty((n_aug, 2), dtype=np.float32)
            for i in range(0, n_aug, 1_000_000):
                base_a[i:i + 1_000_000] = _baseline(Xa[i:i + 1_000_000])
            Ya = np.where(Ma > 0, Ya - base_a, 0.0).astype(np.float32)
            del base_a
            aug = (Xa, Ya, Ma, w_all[sel].copy())

    X_t, Y_t, M_t = (torch.from_numpy(a[train_idx]) for a in (Xs, Ys, mask))
    W_t = torch.from_numpy(w_all[train_idx])
    X_v_cpu = torch.from_numpy(Xs[val_idx])
    Y_v_cpu = torch.from_numpy(Ys[val_idx])
    M_v_cpu = torch.from_numpy(mask[val_idx])
    W_v_cpu = torch.from_numpy(w_all[val_idx])
    X_v = X_v_cpu.to(device)
    Y_v = Y_v_cpu.to(device)
    M_v = M_v_cpu.to(device)
    del Xs

    if aug is not None:
        Xa, Ya, Ma, Wa = aug
        X_t = torch.cat([X_t, torch.from_numpy(Xa)])
        Y_t = torch.cat([Y_t, torch.from_numpy(Ya)])
        M_t = torch.cat([M_t, torch.from_numpy(Ma)])
        W_t = torch.cat([W_t, torch.from_numpy(Wa)])
        del Xa, Ya, Ma, Wa, aug

    # Two-member ARCHITECTURE-diverse deep ensemble with per-member
    # wall-clock slices (unchanged from the parent): member 0 is the
    # conv-wide-and-deep core, member 1 is the lineage HeteroMLP. Different
    # inductive biases decorrelate errors far more than different seeds of
    # one architecture, so the same-cost average cancels more variance,
    # concentrated on the noisy edge-stencil and fast-front rows. Distinct
    # seeds per member so data order and init both differ.
    archs = ["conv", "mlp"]
    M_ens = 2
    members = []
    for m in range(M_ens):
        deadline = t0 + time_budget_s * 0.95 * (m + 1) / M_ens
        if time.time() > deadline:
            break
        net, best_val = _train_member(
            seed + 101 * m, archs[m % len(archs)], deadline, device, max_epochs,
            n_feat, n_cells, comp_w,
            X_t, Y_t, M_t, W_t, X_v, Y_v, M_v,
            X_v_cpu, Y_v_cpu, M_v_cpu, W_v_cpu)
        members.append((net, best_val))

    # Guard: drop members starved by a tight budget (val loss >15% worse
    # than the best, in the component-weighted metric) so the worst case
    # degenerates to the single-conv parent instead of averaging in an
    # undertrained or weaker-arch net.
    finite = [(net, v) for net, v in members if np.isfinite(v)]
    if finite:
        v_best = min(v for _, v in finite)
        keep = [net for net, v in finite if v <= v_best * 1.15]
    else:
        keep = [members[0][0]]

    # Predict: ridge baseline + averaged residual mu over accepted members.
    out = np.zeros((len(X_test), 2), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_test), 65536):
            Xf = _featurize(X_test[i:i + 65536], mean, scale, n_cells)
            base = _baseline(Xf)
            xb = torch.from_numpy(Xf).to(device)
            acc = torch.zeros((xb.shape[0], 2), device=device)
            for net in keep:
                mu, _ = net(xb)
                acc += mu
            out[i:i + 65536] = base + (acc / len(keep)).cpu().numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
