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
    """Lineage-proven feature block, unchanged: standardize, gradient-fill
    missing stencil cells (mirror imputation), append per-cell missingness,
    per-feature spatial std, first/second-order spatial derivatives,
    gradient magnitude, and the nonlinear flow-regime block (speeds,
    vorticity, strains, Okubo-Weiss, advection) when F == 8."""
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


class PolarNet(nn.Module):
    """NEW predictive core: polar target decomposition.

    A shared trunk feeds two heads:
      * spd  -> scalar log-speed s (clamped to [-7, 1.6]; exp(s) <= ~5 m/s,
        strictly positive, multiplicative — relative errors, not absolute,
        drive the magnitude channel);
      * dirn -> raw 2-vector normalized to a unit direction d_hat (the
        smooth unit-circle problem, decoupled from magnitude).
    Prediction: (u, v) = exp(s) * d_hat. Cartesian MSE nets shrink
    magnitudes toward the mean in the fast tail; the explicit exp-decoded
    speed channel plus a log-space auxiliary speed loss removes that
    shrinkage where the parent is weakest.
    """

    def __init__(self, d_in, hidden=(512, 256, 128), dropout=0.1):
        super().__init__()
        layers = []
        d = d_in
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU(),
                       nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.spd = nn.Linear(d, 1)
        self.dirn = nn.Linear(d, 2)
        # Bias log-speed toward the typical ~0.15 m/s so early training
        # starts near the bulk of the speed distribution.
        nn.init.zeros_(self.spd.weight)
        nn.init.constant_(self.spd.bias, math.log(0.15))

    def forward(self, x):
        z = self.trunk(x)
        s = self.spd(z).clamp(-7.0, 1.6)
        speed = torch.exp(s)
        d = self.dirn(z)
        d = d / (d.norm(dim=1, keepdim=True) + 1e-6)
        return speed * d, s.squeeze(1)


def _polar_loss(uv, logs, target, mask, row_w, logspd_t, spd_m):
    """Recency-weighted masked component MSE (trains one-valid-component
    rows, keeps selection aligned with the scored quantity) plus a 0.2x
    log-space auxiliary speed loss on both-valid rows — the term that makes
    relative speed error, not absolute, the magnitude objective."""
    err2 = (uv - target).pow(2)
    w = mask * row_w.unsqueeze(1)
    comp = (err2 * w).sum() / w.sum().clamp(min=1e-8)
    aw = spd_m * row_w
    aux = ((logs - logspd_t).pow(2) * aw).sum() / aw.sum().clamp(min=1e-8)
    return comp + 0.2 * aux


def _masked_mse(pred, target, mask):
    diff = (pred - target) * mask
    return diff.pow(2).sum() / mask.sum().clamp(min=1)


class _EMA:
    """Exponential moving average of weights along the SGD trajectory
    (SWA-family); shadow sits in the flatter basin center, transfers better
    across the temporal train->test shift. Unchanged from the lineage."""

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
            uv, _ = net(X_v[i:i + bs])
            losses.append(_masked_mse(uv, Y_v[i:i + bs], M_v[i:i + bs]).item())
    return float(np.mean(losses))


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
    Xs = _featurize(X_train, mean, scale, n_cells)
    mask = np.isfinite(Y_train).astype(np.float32)
    Yt = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)
    n = len(Xs)

    # ---- Ridge baseline, repurposed as INPUT features ----
    # Additive residualization is meaningless in polar space, so the linear
    # geostrophic extrapolation enters as conditioning instead: three extra
    # columns (base_u, base_v, base_speed) appended after featurization.
    rng0 = np.random.default_rng(seed)
    sub = rng0.choice(n, size=min(n, 1_500_000), replace=False)
    W_lin = np.zeros((Xs.shape[1], 2), dtype=np.float32)
    b_lin = np.zeros(2, dtype=np.float32)
    for c in range(2):
        rows = sub[mask[sub, c] > 0]
        if len(rows) > 1000:
            ridge = Ridge(alpha=1.0)
            ridge.fit(Xs[rows], Yt[rows, c])
            W_lin[:, c] = ridge.coef_.astype(np.float32)
            b_lin[c] = np.float32(ridge.intercept_)

    def _base_cols(Xf):
        b = np.clip(Xf @ W_lin + b_lin, -3.0, 3.0).astype(np.float32)
        bs_ = np.sqrt(b[:, 0:1] ** 2 + b[:, 1:2] ** 2)
        return np.hstack([b, bs_]).astype(np.float32)

    def _with_base(Xf):
        return np.hstack([Xf, _base_cols(Xf)])

    parts = []
    for i in range(0, n, 1_000_000):
        parts.append(_with_base(Xs[i:i + 1_000_000]))
    Xs = np.vstack(parts)
    del parts

    # Auxiliary log-speed targets on both-valid rows.
    both = (mask[:, 0] * mask[:, 1]).astype(np.float32)
    spd_t = np.sqrt(Yt[:, 0] ** 2 + Yt[:, 1] ** 2)
    logspd_t = np.log(np.clip(spd_t, 1e-3, None)).astype(np.float32)
    logspd_t = np.where(both > 0, logspd_t, 0.0).astype(np.float32)

    # Temporal validation split: last 10% of the time-ordered window.
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    # Recency weights: latest rows weigh 3x the earliest (mean-1 normalized).
    pos = (np.arange(n, dtype=np.float32) / max(n - 1, 1)).astype(np.float32)
    w_all = np.exp(np.log(3.0) * pos).astype(np.float32)
    w_all /= w_all.mean()

    X_t = torch.from_numpy(Xs[train_idx])
    Y_t = torch.from_numpy(Yt[train_idx])
    M_t = torch.from_numpy(mask[train_idx])
    W_t = torch.from_numpy(w_all[train_idx])
    S_t = torch.from_numpy(logspd_t[train_idx])
    B_t = torch.from_numpy(both[train_idx])
    X_v_cpu = torch.from_numpy(Xs[val_idx])
    Y_v_cpu = torch.from_numpy(Yt[val_idx])
    M_v_cpu = torch.from_numpy(mask[val_idx])
    W_v_cpu = torch.from_numpy(w_all[val_idx])
    S_v_cpu = torch.from_numpy(logspd_t[val_idx])
    B_v_cpu = torch.from_numpy(both[val_idx])
    X_v = X_v_cpu.to(device)
    Y_v = Y_v_cpu.to(device)
    M_v = M_v_cpu.to(device)
    del Xs

    # ---- Single polar net gets the whole budget ----
    torch.manual_seed(seed + 1)
    rng = np.random.default_rng(seed + 1)
    net = PolarNet(X_t.shape[1], hidden=(512, 256, 128), dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=5)

    bs = 4096
    patience = 15
    deadline = t0 + time_budget_s * 0.90
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
            xb = X_t[idx].to(device)
            yb = Y_t[idx].to(device)
            mb = M_t[idx].to(device)
            wb = W_t[idx].to(device)
            sb = S_t[idx].to(device)
            bb = B_t[idx].to(device)
            optimizer.zero_grad()
            uv, logs = net(xb)
            loss = _polar_loss(uv, logs, yb, mb, wb, sb, bb)
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

    # Recency fine-tune (proven in lineage): absorb the held-out most-recent
    # 10% with a short low-LR pass, ending on a short-horizon EMA.
    if time.time() < deadline and np.isfinite(best_val):
        X_f = torch.cat([X_t, X_v_cpu])
        Y_f = torch.cat([Y_t, Y_v_cpu])
        M_f = torch.cat([M_t, M_v_cpu])
        W_f = torch.cat([W_t, W_v_cpu])
        S_f = torch.cat([S_t, S_v_cpu])
        B_f = torch.cat([B_t, B_v_cpu])
        ft_opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
        ft_ema = _EMA(net, decay=0.998)
        n_all = len(X_f)
        out_of_time = False
        net.train()
        for _ in range(2):
            order = torch.from_numpy(rng.permutation(n_all))
            for bi, i in enumerate(range(0, n_all, bs)):
                idx = order[i:i + bs]
                xb = X_f[idx].to(device)
                yb = Y_f[idx].to(device)
                mb = M_f[idx].to(device)
                wb = W_f[idx].to(device)
                sb = S_f[idx].to(device)
                bb = B_f[idx].to(device)
                ft_opt.zero_grad()
                uv, logs = net(xb)
                loss = _polar_loss(uv, logs, yb, mb, wb, sb, bb)
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

    # Predict: polar net output directly (baseline is an input, not additive).
    out = np.zeros((len(X_test), 2), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_test), 65536):
            Xf = _with_base(_featurize(X_test[i:i + 65536], mean, scale, n_cells))
            xb = torch.from_numpy(Xf).to(device)
            uv, _ = net(xb)
            out[i:i + 65536] = uv.cpu().numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
