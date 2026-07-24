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

    The mixture-of-experts core consumes the whole vector flat; the gate in
    particular sees the flow-regime block, which is exactly the signal it
    needs to partition rows by dynamical regime.
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


class MoENet(nn.Module):
    """NEW CORE: gated mixture-of-experts over flow regimes.

    A shared trunk embeds the full engineered vector; a softmax gate routes
    each row to K=4 small experts, each with its own (mu, logvar) heads, and
    the outputs are combined by the gate weights. Rationale: one monolithic
    net spreads its capacity across regimes whose dynamics differ (slow
    interior RMSE 0.067 vs fast fronts 0.227 in the parent's diagnostics);
    conditional computation lets an expert specialize on fast eddy/front
    rows and another on edge/missing rows without a brittle hand-made
    split — the gate learns the partition from the flow-regime and
    missingness features already in the vector. All experts are evaluated
    densely (K is tiny; batch stays one matmul per expert) so there is no
    routing sparsity to destabilize training. logvar is combined as the
    gate-weighted average — smooth in the gate and consistent with the
    anchored-NLL weighting used throughout the lineage."""

    def __init__(self, d_in, n_experts=4, d_trunk=256, d_exp=128,
                 dropout=0.1, logvar_init=-3.5):
        super().__init__()
        self.K = n_experts
        self.trunk = nn.Sequential(
            nn.Linear(d_in, d_trunk), nn.LayerNorm(d_trunk), nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(d_trunk, d_trunk), nn.LayerNorm(d_trunk), nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(d_trunk, n_experts)
        nn.init.zeros_(self.gate.weight)
        nn.init.zeros_(self.gate.bias)
        self.experts = nn.ModuleList([
            nn.Sequential(nn.Linear(d_trunk, d_exp), nn.LayerNorm(d_exp),
                          nn.SiLU(), nn.Dropout(dropout))
            for _ in range(n_experts)
        ])
        self.mu_heads = nn.ModuleList(
            [nn.Linear(d_exp, 2) for _ in range(n_experts)])
        self.lv_heads = nn.ModuleList(
            [nn.Linear(d_exp, 2) for _ in range(n_experts)])
        for h in self.lv_heads:
            nn.init.zeros_(h.weight)
            nn.init.constant_(h.bias, logvar_init)

    def forward(self, x):
        z = self.trunk(x)
        g = torch.softmax(self.gate(z), dim=1)          # (b, K)
        mus, lvs = [], []
        for k in range(self.K):
            e = self.experts[k](z)
            mus.append(self.mu_heads[k](e))
            lvs.append(self.lv_heads[k](e))
        mus = torch.stack(mus, dim=1)                   # (b, K, 2)
        lvs = torch.stack(lvs, dim=1)
        gw = g.unsqueeze(-1)
        mu = (gw * mus).sum(dim=1)
        lv = (gw * lvs).sum(dim=1)
        return mu, lv, g


def _load_balance(g):
    """Importance-based load-balancing penalty. imp_k = mean gate weight of
    expert k over the batch; K * sum(imp^2) is minimized (=1) at uniform
    usage and grows as the gate collapses onto few experts. Returned shifted
    so a balanced gate contributes 0 loss."""
    imp = g.mean(dim=0)
    return g.shape[1] * (imp * imp).sum() - 1.0


def _masked_nll_anchored(mu, logvar, target, mask, row_w, comp_w):
    """Gaussian NLL plus an unweighted MSE anchor, per valid component,
    scaled by a per-row recency weight AND a per-component inverse-variance
    weight (unchanged from the lineage). comp_w = (1/Var_u, 1/Var_v) mean-1
    normalized makes the loss's exchange rate between u and v errors match
    the mean-R^2 metric's. The anchor keeps a gradient floor on
    variance-downweighted rows."""
    logvar = logvar.clamp(-7.0, 2.0)
    err2 = (mu - target).pow(2)
    per = 0.5 * (logvar + err2 * torch.exp(-logvar)) + err2
    w = mask * row_w.unsqueeze(1) * comp_w.unsqueeze(0)
    return (per * w).sum() / w.sum().clamp(min=1e-8)


def _masked_mse(pred, target, mask, comp_w):
    """Component-weighted masked MSE — (up to a constant) the mean
    per-component R^2 deficit, so all validation-based selection optimizes
    the scored quantity directly."""
    w = mask * comp_w.unsqueeze(0)
    diff2 = (pred - target).pow(2) * w
    return diff2.sum() / w.sum().clamp(min=1e-8)


class _EMA:
    """Exponential moving average of the net's weights along the SGD
    trajectory; the shadow sits in a flatter basin center and transfers
    better across the temporal train->test shift."""

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
            mu, _, _ = net(X_v[i:i + bs])
            losses.append(_masked_mse(mu, Y_v[i:i + bs], M_v[i:i + bs],
                                      comp_w).item())
    return float(np.mean(losses))


def _train_member(member_seed, deadline, device, max_epochs, comp_w,
                  X_t, Y_t, M_t, W_t, X_v, Y_v, M_v,
                  X_v_cpu, Y_v_cpu, M_v_cpu, W_v_cpu):
    """Train one MoE member with the recency- and component-weighted
    anchored heteroscedastic NLL plus the load-balancing penalty.

    Optimization plumbing unchanged from the reference: budget-aware cosine
    anneal sized from the measured first-epoch cost so the member finishes
    its wall-clock slice fully annealed; raw-vs-EMA selection on the
    component-weighted temporal-val masked MSE; patience-15 backstop; short
    low-LR recency fine-tune over the full window ending on a short-horizon
    EMA."""
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)

    net = MoENet(X_t.shape[1], n_experts=4, dropout=0.1).to(device)
    lr0, lr_floor = 1e-3, 3e-5
    lb_coef = 0.02
    optimizer = torch.optim.AdamW(net.parameters(), lr=lr0, weight_decay=1e-4)

    bs = 4096
    patience = 15
    best_val = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    best_epoch = 0
    n_train = len(X_t)

    ema = _EMA(net, decay=0.999)

    t_start = time.time()
    total_T = None  # cosine horizon, estimated after epoch 1

    for epoch in range(1, max_epochs + 1):
        net.train()
        order = torch.from_numpy(rng.permutation(n_train))
        for i in range(0, n_train, bs):
            idx = order[i:i + bs]
            xb, yb, mb = X_t[idx].to(device), Y_t[idx].to(device), M_t[idx].to(device)
            wb = W_t[idx].to(device)
            optimizer.zero_grad()
            mu, lv, g = net(xb)
            loss = (_masked_nll_anchored(mu, lv, yb, mb, wb, comp_w)
                    + lb_coef * _load_balance(g))
            loss.backward()
            optimizer.step()
            ema.update(net)

        val_raw = _val_mse(net, X_v, Y_v, M_v, bs, comp_w)
        val_ema = _val_mse(ema.net, X_v, Y_v, M_v, bs, comp_w)
        val_loss = min(val_raw, val_ema)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            src = net if val_raw <= val_ema else ema.net
            best_state = copy.deepcopy(src.state_dict())
            best_epoch = epoch

        # Budget-aware cosine anneal: size the horizon from the measured
        # first-epoch cost, then set the LR for the NEXT epoch. Floor of 3
        # epochs guards a degenerate estimate; cap at max_epochs.
        if total_T is None:
            ep_time = max(time.time() - t_start, 1e-6)
            left = max(0.0, deadline - time.time())
            total_T = min(max_epochs, epoch + max(3, int(left / ep_time)))
        frac = min(1.0, epoch / max(total_T, 1))
        new_lr = lr_floor + 0.5 * (lr0 - lr_floor) * (1.0 + math.cos(math.pi * frac))
        for gpar in optimizer.param_groups:
            gpar["lr"] = new_lr

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
                mu, lv, g = net(xb)
                loss = (_masked_nll_anchored(mu, lv, yb, mb, wb, comp_w)
                        + lb_coef * _load_balance(g))
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
    Xs = _featurize(X_train, mean, scale, n_cells)
    mask = np.isfinite(Y_train).astype(np.float32)
    Ys = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)
    n = len(Xs)

    # Per-component inverse-variance weights (unchanged from the lineage):
    # makes both the training loss and every validation-based selection
    # decision optimize mean per-component R^2 rather than pooled MSE.
    # Mean-1 normalized; clipped to [0.5, 2] relative.
    cw = np.ones(2, dtype=np.float32)
    for c in range(2):
        vc = float(np.var(Y_train[mask[:, c] > 0, c])) if mask[:, c].sum() > 1000 else 0.0
        cw[c] = 1.0 / vc if vc > 1e-8 else 1.0
    cw = np.clip(cw / cw.mean(), 0.5, 2.0).astype(np.float32)
    cw /= cw.mean()
    comp_w = torch.from_numpy(cw).to(device)

    # ---- Linear baseline + residual learning (unchanged) ----
    # Ridge from the full engineered block to (u, v) on a seeded subsample;
    # the MoE fits the residual. The linear map extrapolates the dominant
    # geostrophic signal into the fast tail; baseline clipped to +/-3 m/s.
    # Additive per-row baseline means residual val MSE == final-prediction
    # val MSE, so all selection still optimizes the scored quantity.
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
    # early stopping optimizes forward-in-time generalization.
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    # Recency weights: latest rows weigh 3x the earliest (exponential ramp,
    # mean-1 normalized).
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

    # Two-member seed-diverse ensemble of the NEW MoE core, per-member
    # wall-clock slices (ensembling is proven plumbing; the core inside is
    # what changed). Distinct seeds so data order, init, and the learned
    # gate partition all differ between members.
    M_ens = 2
    members = []
    for m in range(M_ens):
        deadline = t0 + time_budget_s * 0.95 * (m + 1) / M_ens
        if time.time() > deadline:
            break
        net, best_val = _train_member(
            seed + 101 * m, deadline, device, max_epochs, comp_w,
            X_t, Y_t, M_t, W_t, X_v, Y_v, M_v,
            X_v_cpu, Y_v_cpu, M_v_cpu, W_v_cpu)
        members.append((net, best_val))

    # Guard: drop members starved by a tight budget (val loss >15% worse
    # than the best) so the worst case degenerates to a single MoE instead
    # of averaging in an undertrained net.
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
                mu, _, _ = net(xb)
                acc += mu
            out[i:i + 65536] = base + (acc / len(keep)).cpu().numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
