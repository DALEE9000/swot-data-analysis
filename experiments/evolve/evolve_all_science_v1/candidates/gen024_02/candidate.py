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


def _edge_masks(k):
    """NEW: swath-edge truncation patterns for a k x k stencil, as cell-index
    arrays (row-major). Real edge rows lose a whole side of the window (the
    swath boundary is a line), or two adjacent sides at a corner — never the
    center cell, which is the observed grid point itself. These eight
    patterns are what test-time edge missingness actually looks like."""
    idx = np.arange(k * k).reshape(k, k)
    left, right = idx[:, 0], idx[:, -1]
    top, bot = idx[0, :], idx[-1, :]
    pats = [left, right, top, bot,
            np.union1d(left, top), np.union1d(left, bot),
            np.union1d(right, top), np.union1d(right, bot)]
    ci = (k * k) // 2
    return [np.setdiff1d(p, [ci]).astype(np.int64) for p in pats]


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
    """Lineage MLP core, first-class ensemble member (unchanged from the
    parent). Its inductive bias (flat fully-connected over the whole
    engineered vector) differs from the conv core's, so its errors
    decorrelate far more than a second seed of the same architecture —
    that decorrelation is what the prediction average harvests."""

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


def _masked_nll_anchored(mu, logvar, target, mask, row_w):
    """Gaussian NLL plus an unweighted MSE anchor, per valid component,
    scaled by a per-row recency weight (unchanged from the lineage: the
    anchor keeps a gradient floor on variance-downweighted rows; row_w
    biases the fit toward the most test-like late-window rows)."""
    logvar = logvar.clamp(-7.0, 2.0)
    err2 = (mu - target).pow(2)
    per = 0.5 * (logvar + err2 * torch.exp(-logvar)) + err2
    w = mask * row_w.unsqueeze(1)
    return (per * w).sum() / w.sum().clamp(min=1e-8)


def _masked_mse(pred, target, mask):
    diff = (pred - target) * mask
    return diff.pow(2).sum() / mask.sum().clamp(min=1)


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


def _val_mse(net, X_v, Y_v, M_v, bs):
    net.eval()
    with torch.no_grad():
        losses = []
        for i in range(0, len(X_v), bs):
            mu, _ = net(X_v[i:i + bs])
            losses.append(_masked_mse(mu, Y_v[i:i + bs], M_v[i:i + bs]).item())
    return float(np.mean(losses))


def _train_member(member_seed, arch, deadline, device, max_epochs,
                  n_feat, n_cells,
                  X_t, Y_t, M_t, W_t, X_v, Y_v, M_v,
                  X_v_cpu, Y_v_cpu, M_v_cpu, W_v_cpu):
    """Train one ensemble member of the requested architecture with the
    recency-weighted anchored heteroscedastic NLL; selection over raw-vs-EMA
    on unweighted temporal-val masked MSE; then the proven recency fine-tune
    ending on a short-horizon EMA. Identical to the parent. Returns
    (net, best_val)."""
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
            loss = _masked_nll_anchored(mu, lv, yb, mb, wb)
            loss.backward()
            optimizer.step()
            ema.update(net)

        val_raw = _val_mse(net, X_v, Y_v, M_v, bs)
        val_ema = _val_mse(ema.net, X_v, Y_v, M_v, bs)
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
                loss = _masked_nll_anchored(mu, lv, yb, mb, wb)
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
    Xs = _featurize(X_train, mean, scale, n_cells)
    mask = np.isfinite(Y_train).astype(np.float32)
    Ys = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)
    n = len(Xs)

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

    # ---- NEW: edge-simulation augmentation ----
    # The edge-stencil bins are the weakest (r2_u 0.287 vs 0.476 full), yet
    # edge rows are a minority of training data and their supervision never
    # tells the net what the truncated side actually contained. Simulate the
    # test-time failure mode with full information: take fully-valid-stencil
    # TRAIN rows, NaN out one realistic swath-edge pattern (a side, or two
    # adjacent sides at a corner — never the observed center cell), run them
    # through the IDENTICAL featurize path (mirror fill, mask channel,
    # derivatives all recomputed on the truncated view), and append them to
    # the train split with the source rows' targets and recency weights.
    # Residual targets are computed against the ridge baseline of the MASKED
    # features, so prediction composition (baseline + net) stays exact.
    # The temporal val split stays pure real rows: selection still measures
    # the true forward distribution.
    k_st = int(round(math.sqrt(n_cells)))
    aug = None
    if k_st >= 2 and k_st * k_st == n_cells:
        full_ok = ~np.isnan(X_train).any(axis=1)
        elig = train_idx[full_ok[train_idx]]
        if len(elig) > 10_000:
            n_aug = min(500_000, int(0.15 * len(train_idx)), len(elig))
            src = np.sort(rng0.choice(elig, size=n_aug, replace=False))
            Xa_raw = X_train[src].copy()
            masks_e = _edge_masks(k_st)
            pat = rng0.integers(0, len(masks_e), size=n_aug)
            for pi, cells in enumerate(masks_e):
                rows = np.where(pat == pi)[0]
                if len(rows):
                    cols = (np.arange(n_feat)[:, None] * n_cells
                            + cells[None, :]).ravel()
                    Xa_raw[np.ix_(rows, cols)] = np.nan
            Xa = _featurize(Xa_raw, mean, scale, n_cells)
            del Xa_raw
            Ya = np.where(mask[src] > 0,
                          np.nan_to_num(Y_train[src].astype(np.float32),
                                        nan=0.0) - _baseline(Xa),
                          0.0).astype(np.float32)
            aug = (Xa, Ya, mask[src].copy(), w_all[src].copy())

    if aug is not None:
        Xa, Ya, Ma, Wa = aug
        X_t = torch.from_numpy(np.concatenate([Xs[train_idx], Xa]))
        Y_t = torch.from_numpy(np.concatenate([Ys[train_idx], Ya]))
        M_t = torch.from_numpy(np.concatenate([mask[train_idx], Ma]))
        W_t = torch.from_numpy(np.concatenate([w_all[train_idx], Wa]))
        del Xa, Ya, Ma, Wa, aug
    else:
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

    # Two-member ARCHITECTURE-diverse deep ensemble with per-member
    # wall-clock slices (unchanged from the parent): member 0 conv-wide-and-
    # deep, member 1 lineage HeteroMLP; distinct seeds so data order and
    # init both differ. Both members now see the augmented edge rows.
    archs = ["conv", "mlp"]
    M_ens = 2
    members = []
    for m in range(M_ens):
        deadline = t0 + time_budget_s * 0.95 * (m + 1) / M_ens
        if time.time() > deadline:
            break
        net, best_val = _train_member(
            seed + 101 * m, archs[m % len(archs)], deadline, device, max_epochs,
            n_feat, n_cells,
            X_t, Y_t, M_t, W_t, X_v, Y_v, M_v,
            X_v_cpu, Y_v_cpu, M_v_cpu, W_v_cpu)
        members.append((net, best_val))

    # Guard: drop members starved by a tight budget (val loss >15% worse
    # than the best) so the worst case degenerates to the single-conv
    # parent instead of averaging in an undertrained or weaker-arch net.
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
