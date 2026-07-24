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
    for c in (9, 25, 49):
        if d % c == 0:
            return c
    return 1


def _featurize(X, mean, scale, n_cells):
    """Standardize, gradient-fill missing stencil cells, append missingness,
    spatial std, explicit per-feature spatial derivatives (first AND second
    order), and NONLINEAR flow-regime features. Unchanged proven plumbing
    from the lineage (see gen031_00); layout contract downstream:
    first F*C columns = standardized/imputed stencil image, next C columns =
    per-cell missing fraction, remainder = engineered scalars."""
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


class StencilFormer(nn.Module):
    """WILDCARD CORE — mask-aware self-attention over stencil cells.

    Each of the C = k*k stencil cells becomes a token: its F standardized
    (possibly mirror-imputed) feature values concatenated with that cell's
    missing fraction, linearly embedded and tagged with a learned positional
    embedding. A 2-layer pre-norm transformer encoder lets every cell attend
    to every other cell CONDITIONED ON MISSINGNESS: on edge-stencil rows the
    attention can learn to route information around imputed cells and
    re-estimate local structure from the valid subset — a per-row adaptive
    receptive field that neither translation-shared conv filters nor a flat
    MLP over fixed central differences can express. Pooling keeps both the
    masked context (mean over tokens) and the prediction site (center token).

    The dense path over the engineered scalars (mask block onward) is kept
    from the lineage as a proven shortcut; mu/logvar heads unchanged."""

    def __init__(self, d_total, n_feat, n_cells, d_model=64, n_heads=4,
                 n_layers=2, dropout=0.1, logvar_init=-3.5):
        super().__init__()
        self.F = n_feat
        self.C = n_cells
        self.img_dim = n_feat * n_cells
        self.embed = nn.Linear(n_feat + 1, d_model)
        self.pos = nn.Parameter(torch.zeros(1, n_cells, d_model))
        nn.init.normal_(self.pos, std=0.02)
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=3 * d_model,
            dropout=dropout, activation="gelu", batch_first=True,
            norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, num_layers=n_layers)
        self.attn_proj = nn.Sequential(
            nn.Linear(2 * d_model, 128), nn.LayerNorm(128), nn.SiLU(),
            nn.Dropout(dropout),
        )
        dd = d_total - self.img_dim  # mask block + engineered scalars
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
        # feature-major image (b, F, C) -> tokens (b, C, F)
        tok = x[:, :self.img_dim].view(b, self.F, self.C).permute(0, 2, 1)
        msk = x[:, self.img_dim:self.img_dim + self.C].unsqueeze(-1)  # (b,C,1)
        h = self.encoder(self.embed(torch.cat([tok, msk], dim=-1)) + self.pos)
        za = self.attn_proj(torch.cat([h.mean(dim=1), h[:, self.C // 2]], dim=-1))
        zd = self.dense(x[:, self.img_dim:])
        z = self.trunk(torch.cat([za, zd], dim=1))
        return self.mu(z), self.logvar(z)


class HeteroMLP(nn.Module):
    """Lineage MLP, kept ONLY as the fallback for degenerate stencil layouts
    (n_cells == 1) where tokenization is meaningless."""

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
        return StencilFormer(d_total, n_feat, n_cells).to(device)
    return HeteroMLP(d_total).to(device)


def _masked_nll_anchored(mu, logvar, target, mask, row_w, comp_w):
    """Gaussian NLL + unweighted MSE anchor per valid component, scaled by
    per-row recency weight and per-component inverse-variance weight
    (unchanged from the lineage; matches the loss's u/v exchange rate to
    the mean-R^2 metric)."""
    logvar = logvar.clamp(-7.0, 2.0)
    err2 = (mu - target).pow(2)
    per = 0.5 * (logvar + err2 * torch.exp(-logvar)) + err2
    w = mask * row_w.unsqueeze(1) * comp_w.unsqueeze(0)
    return (per * w).sum() / w.sum().clamp(min=1e-8)


def _masked_mse(pred, target, mask, comp_w):
    """Component-weighted masked MSE — the scored quantity up to a constant;
    all selection decisions optimize this (unchanged)."""
    w = mask * comp_w.unsqueeze(0)
    diff2 = (pred - target).pow(2) * w
    return diff2.sum() / w.sum().clamp(min=1e-8)


class _EMA:
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
            mu, _ = net(X_v[i:i + bs].float())
            losses.append(_masked_mse(mu, Y_v[i:i + bs], M_v[i:i + bs],
                                      comp_w).item())
    return float(np.mean(losses))


def _train_member(member_seed, deadline, device, max_epochs,
                  n_feat, n_cells, comp_w,
                  X_t, Y_t, M_t, W_t, X_v, Y_v, M_v, W_v):
    """Train the single StencilFormer with the lineage's proven loop:
    recency- and component-weighted anchored heteroscedastic NLL, raw-vs-EMA
    selection on component-weighted temporal-val masked MSE, plateau LR
    schedule, patience-15 early stop, then the recency fine-tune ending on a
    short-horizon EMA. Tensors may arrive GPU-resident fp16 (see caller);
    per-batch .float() casts are no-ops on the CPU fp32 fallback path."""
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
        order = torch.from_numpy(rng.permutation(n_train)).to(X_t.device)
        for i in range(0, n_train, bs):
            idx = order[i:i + bs]
            xb = X_t[idx].to(device).float()
            yb = Y_t[idx].to(device)
            mb = M_t[idx].to(device)
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
    # most-recent 10% with a short low-LR pass, ending on a short-horizon
    # EMA; virtual concat of train/val tensors, no full copy.
    if time.time() < deadline and np.isfinite(best_val):
        n_t = len(X_t)
        n_all = n_t + len(X_v)
        ft_opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
        ft_ema = _EMA(net, decay=0.998)
        out_of_time = False
        net.train()
        for _ in range(2):
            order = torch.from_numpy(rng.permutation(n_all))
            for bi, i in enumerate(range(0, n_all, bs)):
                idx = order[i:i + bs]
                a = idx < n_t
                i1 = idx[a].to(X_t.device)
                i2 = (idx[~a] - n_t).to(X_v.device)
                xb = torch.cat([X_t[i1].to(device).float(),
                                X_v[i2].to(device).float()])
                yb = torch.cat([Y_t[i1].to(device), Y_v[i2].to(device)])
                mb = torch.cat([M_t[i1].to(device), M_v[i2].to(device)])
                wb = torch.cat([W_t[i1].to(device), W_v[i2].to(device)])
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

    # Per-component inverse-variance weights (unchanged from the lineage).
    cw = np.ones(2, dtype=np.float32)
    for c in range(2):
        vc = float(np.var(Y_train[mask[:, c] > 0, c])) if mask[:, c].sum() > 1000 else 0.0
        cw[c] = 1.0 / vc if vc > 1e-8 else 1.0
    cw = np.clip(cw / cw.mean(), 0.5, 2.0).astype(np.float32)
    cw /= cw.mean()
    comp_w = torch.from_numpy(cw).to(device)

    # ---- Linear baseline + residual learning (unchanged) ----
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

    # Temporal validation split: last 10% of the time-ordered window.
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    # Recency weights: latest rows weigh 3x the earliest.
    pos = (np.arange(n, dtype=np.float32) / max(n - 1, 1)).astype(np.float32)
    w_all = np.exp(np.log(3.0) * pos).astype(np.float32)
    w_all /= w_all.mean()

    X_t, Y_t, M_t = (torch.from_numpy(a[train_idx]) for a in (Xs, Ys, mask))
    W_t = torch.from_numpy(w_all[train_idx])
    X_v = torch.from_numpy(Xs[val_idx])
    Y_v = torch.from_numpy(Ys[val_idx])
    M_v = torch.from_numpy(mask[val_idx])
    W_v = torch.from_numpy(w_all[val_idx])
    del Xs

    # fp16 GPU staging (unchanged from gen031_00); CPU fallback on OOM.
    if device.type == "cuda":
        try:
            X_t = X_t.to(device, dtype=torch.float16)
            Y_t = Y_t.to(device)
            M_t = M_t.to(device)
            W_t = W_t.to(device)
            X_v = X_v.to(device, dtype=torch.float16)
        except RuntimeError:
            torch.cuda.empty_cache()
            X_t = X_t.cpu()
            Y_t = Y_t.cpu()
            M_t = M_t.cpu()
            W_t = W_t.cpu()
            X_v = X_v.cpu().to(device)
        Y_v = Y_v.to(device)
        M_v = M_v.to(device)
        W_v = W_v.to(device)

    # WILDCARD: ONE StencilFormer member with the FULL wall-clock budget
    # (no ensemble). The slot measures the new architecture at its best
    # rather than splitting the budget across members; the attention core
    # is slower per epoch than the conv, so a half-budget slice would
    # confound "arch is worse" with "arch was starved".
    deadline = t0 + time_budget_s * 0.95
    net, best_val = _train_member(
        seed + 101, deadline, device, max_epochs,
        n_feat, n_cells, comp_w,
        X_t, Y_t, M_t, W_t, X_v, Y_v, M_v, W_v)

    # Predict: ridge baseline + transformer residual mu.
    out = np.zeros((len(X_test), 2), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_test), 65536):
            Xf = _featurize(X_test[i:i + 65536], mean, scale, n_cells)
            base = _baseline(Xf)
            xb = torch.from_numpy(Xf).to(device)
            mu, _ = net(xb)
            out[i:i + 65536] = base + mu.cpu().numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
