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

    Layout matters downstream: the attention core slices the FIRST F*C
    columns as the per-cell values and the NEXT C columns as the per-cell
    missing fraction (token feature AND key-padding mask); everything from
    the mask block onward feeds the dense path. Imputed values in masked
    cells are never attended, so the fill only serves the derivative and
    dense blocks, exactly as in the lineage.
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


class StencilAttnNet(nn.Module):
    """WILDCARD CORE — mask-aware set-attention over stencil cells.

    Each stencil cell is a token: its F standardized (possibly imputed)
    feature values plus its missing fraction, linearly embedded to d_model
    with a learned per-position embedding. A 2-layer transformer encoder
    attends with a key-padding mask that EXCLUDES cells whose missing
    fraction exceeds 0.5, so swath-edge neighborhoods are processed over
    exactly the cells that exist — no fixed-stencil operator (conv or
    central difference) sees imputed pseudo-data. This is aimed squarely at
    the edge-stencil bins, the lineage's worst. The center token output and
    a masked mean over valid tokens are pooled and joined with the proven
    dense path over the engineered scalars; mu/logvar heads as in the
    lineage.
    """

    def __init__(self, d_total, n_feat, n_cells, d_model=64, dropout=0.1,
                 logvar_init=-3.5):
        super().__init__()
        self.F = n_feat
        self.C = n_cells
        self.img_dim = n_feat * n_cells
        self.tok_in = nn.Linear(n_feat + 1, d_model)
        self.pos = nn.Parameter(torch.randn(1, n_cells, d_model) * 0.02)
        enc = nn.TransformerEncoderLayer(
            d_model, nhead=4, dim_feedforward=128, dropout=dropout,
            activation="gelu", batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(enc, num_layers=2,
                                             enable_nested_tensor=False)
        self.attn_proj = nn.Sequential(
            nn.Linear(2 * d_model, 128), nn.LayerNorm(128), nn.SiLU(),
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
        vals = x[:, :self.img_dim].view(b, self.F, self.C).transpose(1, 2)
        miss = x[:, self.img_dim:self.img_dim + self.C]
        tok = self.tok_in(torch.cat([vals, miss.unsqueeze(-1)], dim=-1))
        tok = tok + self.pos
        pad = miss > 0.5
        pad = pad.clone()
        pad[:, self.C // 2] = False  # center cell always attended
        h = self.encoder(tok, src_key_padding_mask=pad)
        valid = (~pad).float().unsqueeze(-1)
        pooled = (h * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1.0)
        zc = self.attn_proj(torch.cat([h[:, self.C // 2], pooled], dim=-1))
        zd = self.dense(x[:, self.img_dim:])
        z = self.trunk(torch.cat([zc, zd], dim=1))
        return self.mu(z), self.logvar(z)


class HeteroMLP(nn.Module):
    """Fallback core for degenerate (non-stencil) layouts only."""

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


def _make_net(d_total, n_feat, n_cells, device):
    if n_cells > 1:
        return StencilAttnNet(d_total, n_feat, n_cells, dropout=0.1).to(device)
    return HeteroMLP(d_total, hidden=(256, 256, 128), dropout=0.1).to(device)


def _masked_nll_anchored(mu, logvar, target, mask, row_w, comp_w):
    """Gaussian NLL plus an unweighted MSE anchor, per valid component,
    scaled by a per-row recency weight AND a per-component inverse-variance
    weight (both proven in the lineage). The anchor keeps a gradient floor
    on variance-downweighted rows."""
    logvar = logvar.clamp(-7.0, 2.0)
    err2 = (mu - target).pow(2)
    per = 0.5 * (logvar + err2 * torch.exp(-logvar)) + err2
    w = mask * row_w.unsqueeze(1) * comp_w.unsqueeze(0)
    return (per * w).sum() / w.sum().clamp(min=1e-8)


def _masked_mse(pred, target, mask, comp_w):
    """Component-weighted masked MSE — (up to a constant) the mean
    per-component R^2 deficit, so all selection optimizes the metric."""
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


def _train_member(member_seed, deadline, device, max_epochs,
                  n_feat, n_cells, comp_w,
                  X_t, Y_t, M_t, W_t, X_v, Y_v, M_v,
                  X_v_cpu, Y_v_cpu, M_v_cpu, W_v_cpu):
    """Train the attention core with the recency- and component-weighted
    anchored heteroscedastic NLL; selection over raw-vs-EMA on
    component-weighted temporal-val masked MSE; then the proven recency
    fine-tune ending on a short-horizon EMA. Returns (net, best_val)."""
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)

    net = _make_net(X_t.shape[1], n_feat, n_cells, device)
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
        out_of_time = False
        for bi, i in enumerate(range(0, n_train, bs)):
            idx = order[i:i + bs]
            xb, yb, mb = X_t[idx].to(device), Y_t[idx].to(device), M_t[idx].to(device)
            wb = W_t[idx].to(device)
            optimizer.zero_grad()
            mu, lv = net(xb)
            loss = _masked_nll_anchored(mu, lv, yb, mb, wb, comp_w)
            loss.backward()
            optimizer.step()
            ema.update(net)
            if bi % 200 == 0 and time.time() > deadline:
                out_of_time = True
                break

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
        if out_of_time or time.time() > deadline:
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
    Xs = _featurize(X_train, mean, scale, n_cells)
    mask = np.isfinite(Y_train).astype(np.float32)
    Ys = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)
    n = len(Xs)

    # Per-component inverse-variance weights (proven): makes the loss's
    # exchange rate between u and v errors match mean-R^2's. Mean-1
    # normalized, clipped to [0.5, 2] relative.
    cw = np.ones(2, dtype=np.float32)
    for c in range(2):
        vc = float(np.var(Y_train[mask[:, c] > 0, c])) if mask[:, c].sum() > 1000 else 0.0
        cw[c] = 1.0 / vc if vc > 1e-8 else 1.0
    cw = np.clip(cw / cw.mean(), 0.5, 2.0).astype(np.float32)
    cw /= cw.mean()
    comp_w = torch.from_numpy(cw).to(device)

    # ---- Linear baseline + residual learning (unchanged) ----
    # Ridge from the full engineered block to (u, v) on a seeded subsample;
    # the net fits the residual. The linear map extrapolates the dominant
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

    # Single attention-core member with the whole budget: the wildcard
    # spends its capacity on the new structure rather than an ensemble, so
    # the measured score isolates the architecture's contribution.
    deadline = t0 + time_budget_s * 0.95
    net, best_val = _train_member(
        seed + 101, deadline, device, max_epochs,
        n_feat, n_cells, comp_w,
        X_t, Y_t, M_t, W_t, X_v, Y_v, M_v,
        X_v_cpu, Y_v_cpu, M_v_cpu, W_v_cpu)
    keep = [net]

    # Predict: ridge baseline + residual mu.
    out = np.zeros((len(X_test), 2), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_test), 65536):
            Xf = _featurize(X_test[i:i + 65536], mean, scale, n_cells)
            base = _baseline(Xf)
            xb = torch.from_numpy(Xf).to(device)
            acc = torch.zeros((xb.shape[0], 2), device=device)
            for net_k in keep:
                mu, _ = net_k(xb)
                acc += mu
            out[i:i + 65536] = base + (acc / len(keep)).cpu().numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
