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

    Kept from the parent (proven plumbing):
      * per-cell missing fraction (n, C);
      * per-feature stencil std over VALID cells (n, F) — front/eddy signal;
      * gradient-preserving mirror imputation (2*center - mirror, clamped to
        +/-4 sigma, center fallback) so swath-edge truncation keeps the
        first-order gradient instead of flattening the neighborhood.

    THE CHANGE — appended derived blocks, computed AFTER imputation (so the
    same code path serves full and edge rows):
      * per-feature central differences gx, gy (n, 2F);
      * per-feature 5-point Laplacian and the two diagonal central
        differences (n, 3F) — curvature/second-order structure;
      * |grad f| per feature (n, F);
      * flow-regime block when F == 8 (pooled layout [mdt, ssha, ugos, vgos,
        ugosa, vgosa, era5_u, era5_v]): speeds of the geostrophic /
        anomaly / wind vectors, vorticity, normal and shear strain,
        Okubo-Weiss, and geostrophic advection u·grad(f) per feature.
        Fronts and eddies — the fast-regime rows where the parent is worst —
        are defined by exactly these gradient combinations; giving them
        explicitly spares the MLP from spending capacity rediscovering
        finite differences, and vorticity/strain carry rotational signal
        the u component (the weaker one) needs.

    Layout: [Xs (F*C), miss (C), spat (F), gx (F), gy (F), lap (F), gp (F),
    gq (F), gmag (F), reg (7), adv (F)] — _augment_edges depends on these
    offsets and recomputes every derived block for synthetic edge rows.
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


# The four swath-edge padding geometries of a 3x3 row-major stencil: a full
# left/right column or top/bottom row is truncated. Real padding at swath
# edges takes exactly these shapes, so augmentation samples from them.
_SIDES = torch.tensor([[0, 3, 6], [2, 5, 8], [0, 1, 2], [6, 7, 8]])


def _augment_edges(xb, F, C, p=0.15):
    """Convert a random subset of FULL-stencil batch rows into synthetic
    swath-edge rows, reproducing the inference-time featurization exactly.

    Unchanged mechanics from the parent: drop one side of the 3x3
    neighborhood, mirror-fill it (2*center - mirror, clamped to +/-4 sigma —
    the same rule _featurize applies to real padding), set the missingness
    block to 1 on the dropped cells, recompute the per-feature spatial std
    over the 6 surviving cells. Real edge rows are never re-augmented and
    validation is untouched.

    NEW: the derived blocks (gx, gy, lap, gp, gq, gmag, and the F==8
    flow-regime block) are recomputed here from the filled neighborhood —
    _featurize derives them AFTER imputation, so at a real edge they are
    one-sided/extrapolated differences; leaving the full-stencil values in
    an augmented row would hand the model derivative features a genuine
    edge row could never have, teaching it exactly nothing about edges.
    Recomputing keeps the augmented (input -> truth) pairs consistent with
    what inference produces at real swath edges.
    """
    if C != 9 or p <= 0.0:
        return xb
    d = F * C
    full = xb[:, d:d + C].sum(dim=1) <= 1e-6
    sel = full & (torch.rand(xb.shape[0], device=xb.device) < p)
    idx = sel.nonzero(as_tuple=True)[0]
    m = idx.numel()
    if m == 0:
        return xb
    Z0 = xb[idx, :d].view(m, F, C)
    cells = _SIDES.to(xb.device)[torch.randint(0, 4, (m,), device=xb.device)]
    keep = torch.ones(m, C, device=xb.device)
    keep.scatter_(1, cells, 0.0)
    k = keep.unsqueeze(1)  # (m, 1, C)
    cnt = k.sum(dim=2)
    mean = (Z0 * k).sum(dim=2) / cnt
    var = ((Z0 - mean.unsqueeze(2)).pow(2) * k).sum(dim=2) / cnt
    spat = var.clamp(min=0.0).sqrt()
    center = Z0[:, :, C // 2]
    mirror_idx = (C - 1 - cells).unsqueeze(1).expand(m, F, 3)
    mirror = Z0.gather(2, mirror_idx)
    fill = (2.0 * center.unsqueeze(2) - mirror).clamp(-4.0, 4.0)
    Z = Z0.scatter(2, cells.unsqueeze(1).expand(m, F, 3), fill)
    new = xb[idx].clone()
    new[:, :d] = Z.reshape(m, d)
    new[:, d:d + C] = 1.0 - keep
    new[:, d + C:d + C + F] = spat
    # Recompute derived blocks on the filled neighborhood (layout mirrors
    # _featurize: gx, gy, lap, gp, gq, gmag, then reg + adv when F == 8).
    ci = C // 2
    kk = 3
    gx = 0.5 * (Z[:, :, ci + 1] - Z[:, :, ci - 1])
    gy = 0.5 * (Z[:, :, ci + kk] - Z[:, :, ci - kk])
    lap = (Z[:, :, ci + 1] + Z[:, :, ci - 1] + Z[:, :, ci + kk]
           + Z[:, :, ci - kk] - 4.0 * Z[:, :, ci])
    inv2 = 0.5 / math.sqrt(2.0)
    gp = inv2 * (Z[:, :, ci + kk + 1] - Z[:, :, ci - kk - 1])
    gq = inv2 * (Z[:, :, ci - kk + 1] - Z[:, :, ci + kk - 1])
    gmag = torch.sqrt(gx * gx + gy * gy)
    o = d + C + F
    new[:, o:o + F] = gx
    new[:, o + F:o + 2 * F] = gy
    new[:, o + 2 * F:o + 3 * F] = lap
    new[:, o + 3 * F:o + 4 * F] = gp
    new[:, o + 4 * F:o + 5 * F] = gq
    new[:, o + 5 * F:o + 6 * F] = gmag
    o2 = o + 6 * F
    if F == 8 and xb.shape[1] >= o2 + 7 + F:
        cu, cv = Z[:, 2, ci], Z[:, 3, ci]
        au, av = Z[:, 4, ci], Z[:, 5, ci]
        wu, wv = Z[:, 6, ci], Z[:, 7, ci]
        spd = torch.sqrt(cu * cu + cv * cv)
        spda = torch.sqrt(au * au + av * av)
        spdw = torch.sqrt(wu * wu + wv * wv)
        vort = gx[:, 3] - gy[:, 2]
        sn = gx[:, 2] - gy[:, 3]
        ss = gx[:, 3] + gy[:, 2]
        ow = (sn * sn + ss * ss - vort * vort).clamp(-16.0, 16.0)
        new[:, o2:o2 + 7] = torch.stack([spd, spda, spdw, vort, sn, ss, ow],
                                        dim=1)
        adv = (cu.unsqueeze(1) * gx + cv.unsqueeze(1) * gy).clamp(-16.0, 16.0)
        new[:, o2 + 7:o2 + 7 + F] = adv
    xb[idx] = new
    return xb


class HeteroMLP(nn.Module):
    """Joint (u, v) MLP with a heteroscedastic head.

    Shared trunk -> two heads:
      mu     : (u, v) point predictions (what the harness scores)
      logvar : per-row, per-component log aleatoric variance, used only to
               weight the training loss. Hard/noisy rows (swath-edge stencils,
               fast fronts/eddies) get high predicted variance and therefore
               small gradients, so they stop distorting the fit on the
               well-conditioned majority of rows.
    """

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
        # Start near the marginal target log-variance so the NLL is well
        # scaled from step 1 (u, v std ~0.15-0.2 m/s -> log var ~ -3.5) and
        # early training behaves like plain (scaled) MSE.
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, logvar_init)

    def forward(self, x):
        z = self.trunk(x)
        return self.mu(z), self.logvar(z)


def _masked_nll_anchored(mu, logvar, target, mask, row_w):
    """Gaussian NLL plus an unweighted MSE anchor, per valid component,
    scaled by a per-row recency weight (unchanged from the parent).

    The NLL alone lets the variance head shrink gradients on hard rows by
    exp(-logvar); the MSE anchor restores a gradient floor on downweighted
    rows so the mu head keeps fitting them while the variance head handles
    calibration and relative weighting. row_w is the mean-1 recency weight.
    """
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
    trajectory (SWA-family). Near convergence, SGD with a finite LR bounces
    around a minimum; the running average sits in the flatter center of
    that basin, which transfers better across the temporal train->test
    shift, and averages out the per-step edge-augmentation jitter."""

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


def _train_member(member_seed, deadline, device, max_epochs, F, C,
                  X_t, Y_t, M_t, W_t, X_v, Y_v, M_v,
                  X_v_cpu, Y_v_cpu, M_v_cpu, W_v_cpu):
    """Train one ensemble member with the recency-weighted anchored
    heteroscedastic NLL and train-time edge augmentation. Each epoch, BOTH
    the raw net and its weight-EMA shadow are scored on the UNWEIGHTED
    temporal-val masked MSE of mu — the scored quantity — and model
    selection keeps the better of the two; the LR schedule still steps on
    the raw net's val loss (its training signal). Then the proven recency
    fine-tune runs (same augmentation), ending on a short-horizon EMA of
    the fine-tune trajectory rather than the last minibatch's weights.
    Returns (net, best_val). Unchanged from the parent."""
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)

    net = HeteroMLP(X_t.shape[1], hidden=(256, 256, 128), dropout=0.1).to(device)
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
            xb = _augment_edges(xb, F, C)
            optimizer.zero_grad()
            mu, lv = net(xb)
            loss = _masked_nll_anchored(mu, lv, yb, mb, wb)
            loss.backward()
            optimizer.step()
            ema.update(net)

        # Model selection on UNWEIGHTED masked MSE of mu — the scored
        # quantity. Validation stays unaugmented so neither the recency
        # emphasis nor the edge augmentation can game early stopping.
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
    # most-recent 10% with a short low-LR pass over the full window (same
    # augmentation), ending on a short-horizon EMA of the fine-tune
    # trajectory. If the deadline cuts the pass short the EMA has barely
    # moved, so the worst case degenerates to no fine-tune.
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
                xb = _augment_edges(xb, F, C)
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

    # Temporal validation split: last 10% of the time-ordered window, so
    # early stopping and the LR schedule optimize forward-in-time
    # generalization — the quantity the temporal test holdout measures.
    n = len(Xs)
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    # Recency weights over the time-ordered window: latest rows weigh 3x the
    # earliest (exponential ramp, mean-1 normalized so the tuned loss scale
    # is unchanged).
    pos = (np.arange(n, dtype=np.float32) / max(n - 1, 1)).astype(np.float32)
    w_all = np.exp(np.log(3.0) * pos).astype(np.float32)
    w_all /= w_all.mean()

    X_t, Y_t, M_t = (torch.from_numpy(a[train_idx]) for a in (Xs, Ys, mask))
    W_t = torch.from_numpy(w_all[train_idx])
    # Keep CPU copies of the validation tail for the post-selection fine-tune.
    X_v_cpu = torch.from_numpy(Xs[val_idx])
    Y_v_cpu = torch.from_numpy(Ys[val_idx])
    M_v_cpu = torch.from_numpy(mask[val_idx])
    W_v_cpu = torch.from_numpy(w_all[val_idx])
    X_v = X_v_cpu.to(device)
    Y_v = Y_v_cpu.to(device)
    M_v = M_v_cpu.to(device)
    del Xs

    # Two-seed deep ensemble with per-member wall-clock slices (unchanged).
    # Member m must finish by 0.95*(m+1)/M of the budget, leaving 5% for
    # test prediction.
    M_ens = 2
    members = []
    for m in range(M_ens):
        deadline = t0 + time_budget_s * 0.95 * (m + 1) / M_ens
        if time.time() > deadline:
            break
        net, best_val = _train_member(
            seed + 101 * m, deadline, device, max_epochs, n_feat, n_cells,
            X_t, Y_t, M_t, W_t, X_v, Y_v, M_v,
            X_v_cpu, Y_v_cpu, M_v_cpu, W_v_cpu)
        members.append((net, best_val))

    # Guard: drop members starved by a tight budget (val loss >15% worse than
    # the best member) so the worst case degenerates to the single-model
    # anchored parent instead of averaging in an undertrained net.
    finite = [(net, v) for net, v in members if np.isfinite(v)]
    if finite:
        v_best = min(v for _, v in finite)
        keep = [net for net, v in finite if v <= v_best * 1.15]
    else:
        keep = [members[0][0]]

    # Predict on the test set in batches, averaging mu over accepted members.
    out = np.zeros((len(X_test), 2), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_test), 65536):
            xb = torch.from_numpy(_featurize(X_test[i:i + 65536], mean, scale, n_cells)).to(device)
            acc = torch.zeros((xb.shape[0], 2), device=device)
            for net in keep:
                mu, _ = net(xb)
                acc += mu
            out[i:i + 65536] = (acc / len(keep)).cpu().numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
