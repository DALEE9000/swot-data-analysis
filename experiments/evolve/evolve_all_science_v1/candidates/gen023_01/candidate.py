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

    Proven plumbing (unchanged from the lineage):
      * per-cell missing fraction (n, C);
      * per-feature stencil std over VALID cells (n, F) — front/eddy signal;
      * gradient-preserving mirror imputation (2*center - mirror, clamped to
        +/-4 sigma, center fallback) so swath-edge truncation keeps the
        first-order gradient instead of flattening the neighborhood;
      * per-feature central differences gx, gy (n, 2F) — the geostrophic
        gradient operators;
      * per-feature 5-point Laplacian (prop. to relative vorticity for SSH)
        and the two diagonal central differences (n, 3F);
      * |grad f| = sqrt(gx^2 + gy^2) per feature (n, F) — front intensity;
      * flow-regime block when F == 8 (speeds, vorticity, strains,
        Okubo-Weiss, geostrophic advection), quadratics clipped to +/-16.
    """
    Xs = (X.astype(np.float32) - mean) / scale
    n, d = Xs.shape
    F = d // n_cells
    miss = np.isnan(X).reshape(n, F, n_cells).mean(axis=1).astype(np.float32)
    Z = Xs.reshape(n, F, n_cells)
    with warnings.catch_warnings():
        # nanstd emits RuntimeWarning on all-NaN neighborhoods; those rows
        # are handled by the nan_to_num below.
        warnings.simplefilter("ignore")
        spat = np.nanstd(Z, axis=2).astype(np.float32)
    spat = np.nan_to_num(spat, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    # Fill AFTER the std block (which must see true valid cells). Snapshot the
    # point-reflected cells before any in-place filling so late cells never
    # read an already-imputed mirror as if it were data.
    center = Z[:, :, n_cells // 2]
    mirror = Z[:, :, ::-1].copy()  # cell c's mirror across the center is C-1-c
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
    # Derivatives on the filled, finite stencil. Cells are flattened
    # row-major over the k x k window: the center's horizontal neighbors sit
    # at +/-1, vertical neighbors at +/-k, diagonals at +/-(k+1) and
    # +/-(k-1). Orientation/sign conventions only need to be consistent.
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
        # ---- nonlinear flow-regime block ----
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


def _masked_nll_anchored(mu, logvar, target, mask, row_w, comp_w):
    """Gaussian NLL plus a METRIC-ALIGNED MSE anchor, per valid component,
    scaled by a per-row recency weight.

    The NLL alone lets the variance head shrink gradients on hard rows by
    exp(-logvar) — the anchor restores a gradient floor on downweighted
    rows so the mu head keeps fitting them (unchanged from the lineage).

    NEW: the anchor is weighted per COMPONENT by comp_w (shape (2,),
    inverse target variance normalized to mean 1). The scored metric is the
    mean of per-component R^2, i.e. each component's squared error counts
    inversely to its own variance — but a raw-m/s anchor lets the higher-
    variance component dominate the anchor gradient. comp_w makes the
    anchor's implied objective proportional to (2 - r2_u - r2_v), so the
    gradient floor is split between u and v the way the score splits
    credit. The NLL term self-normalizes via exp(-logvar) and is left
    untouched. row_w (mean ~1) still multiplies the whole per-component
    loss and the normalizer, keeping the loss scale the tuned LR schedule
    expects.
    """
    logvar = logvar.clamp(-7.0, 2.0)
    err2 = (mu - target).pow(2)
    per = 0.5 * (logvar + err2 * torch.exp(-logvar)) + err2 * comp_w
    w = mask * row_w.unsqueeze(1)
    return (per * w).sum() / w.sum().clamp(min=1e-8)


def _masked_mse(pred, target, mask, comp_w):
    """Masked MSE with per-component inverse-variance weights (mean-1
    normalized). This is the selection criterion everywhere — early
    stopping, raw-vs-EMA choice, LR scheduling, starved-member gating — so
    every decision in the stack now optimizes a quantity monotone in the
    scored mean-of-R^2 instead of raw m/s MSE, which over-weights the
    higher-variance component."""
    w = mask * comp_w
    diff2 = (pred - target).pow(2) * w
    return diff2.sum() / w.sum().clamp(min=1e-8)


class _EMA:
    """Exponential moving average of a net's weights along the SGD
    trajectory (SWA-family). Near convergence, SGD with a finite LR bounces
    around a minimum; the running average sits in the flatter center of
    that basin, which transfers better across the temporal train->test
    shift and smooths the seed/minibatch noise that hits edge-stencil and
    fast-front rows hardest. Weight-space averaging is complementary to the
    lineage's prediction-space two-seed ensemble and costs one extra copy
    of a ~100k-parameter net plus one lerp per optimizer step.
    """

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
            losses.append(
                _masked_mse(mu, Y_v[i:i + bs], M_v[i:i + bs], comp_w).item())
    return float(np.mean(losses))


def _train_member(member_seed, deadline, device, max_epochs,
                  X_t, Y_t, M_t, W_t, X_v, Y_v, M_v,
                  X_v_cpu, Y_v_cpu, M_v_cpu, W_v_cpu, comp_w):
    """Train one ensemble member with the recency-weighted anchored
    heteroscedastic NLL (anchor now component-balanced). Each epoch, BOTH
    the raw net and its weight-EMA shadow are scored on the UNWEIGHTED-in-
    time, variance-normalized temporal-val masked MSE of mu — the scored
    quantity — and model selection keeps the better of the two; the LR
    schedule still steps on the raw net's val loss. Then the proven recency
    fine-tune runs, ending on a short-horizon EMA of the fine-tune
    trajectory. Returns (net, best_val)."""
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

    # EMA horizon ~1000 steps (decay 0.999) ~= 1.2 epochs at bs 4096 —
    # long enough to average over the LR-plateau bounce, short enough to
    # track ReduceLROnPlateau's regime changes.
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

        # Model selection on the variance-normalized masked MSE of mu — a
        # monotone proxy for the scored mean R^2 — NOT on the training
        # loss. Both the raw and EMA weights compete; the scheduler keeps
        # stepping on the raw net's val loss so LR decisions reflect the
        # weights actually being optimized.
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

    # Recency fine-tune (proven +~0.01 in lineage): the validation tail is the
    # most recent — and hence most test-like — 10% of the training window, and
    # the model above never trained on it. After model selection, absorb it
    # with a short low-LR pass over the FULL window (tail included), using the
    # same recency-weighted anchored loss so relative row weighting stays
    # consistent. The pass has no validation-based selection, so it returns a
    # short-horizon EMA (decay 0.998) — recency absorption is kept, last-batch
    # noise is not. If the deadline cuts the pass short, the EMA has barely
    # moved from the selected weights, so the worst case degenerates to no
    # fine-tune.
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
    Xs = _featurize(X_train, mean, scale, n_cells)
    mask = np.isfinite(Y_train).astype(np.float32)
    Ys = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)

    # ---- metric-aligned per-component weights (the one new mechanism) ----
    # Score = mean of per-component R^2 = 1 - 0.5 * sum_c MSE_c / var_c, so
    # the metric prices each component's squared error at 1/var_c. Weight
    # both the anchor loss and every selection MSE accordingly (normalized
    # to mean 1 so the tuned loss/LR scale is unchanged; clipped to [1/3, 3]
    # so a degenerate variance estimate cannot blow up the loss). If
    # var_u == var_v this is exactly the parent.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tvar = np.nanvar(Y_train.astype(np.float64), axis=0)
    tvar = np.where(np.isfinite(tvar) & (tvar > 1e-8), tvar, 1.0)
    cw = 1.0 / tvar
    cw = cw / cw.mean()
    cw = np.clip(cw, 1.0 / 3.0, 3.0).astype(np.float32)
    comp_w = torch.from_numpy(cw).to(device)  # shape (2,), broadcasts over rows

    # Temporal validation split: rows are time-ordered (later rows = later in
    # the mission window, regions interleaved), and the test set is a temporal
    # holdout. Validating on the last 10% of the training window makes early
    # stopping and the LR schedule optimize forward-in-time generalization —
    # the quantity the test actually measures.
    n = len(Xs)
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    # Recency weights over the time-ordered window: latest rows weigh 3x the
    # earliest (exponential ramp, normalized to mean 1 so the loss scale — and
    # hence the tuned LR/scheduler behavior — is unchanged).
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

    # Two-seed deep ensemble with per-member wall-clock slices. Independently
    # seeded nets make decorrelated errors and averaging their mu predictions
    # cancels seed-dependent variance; the gain concentrates on edge-stencil
    # and fast-front rows where single-net outputs are noisiest. Member m must
    # finish by 0.95*(m+1)/M of the budget, leaving 5% for test prediction.
    M_ens = 2
    members = []
    for m in range(M_ens):
        deadline = t0 + time_budget_s * 0.95 * (m + 1) / M_ens
        if time.time() > deadline:
            break
        net, best_val = _train_member(
            seed + 101 * m, deadline, device, max_epochs,
            X_t, Y_t, M_t, W_t, X_v, Y_v, M_v,
            X_v_cpu, Y_v_cpu, M_v_cpu, W_v_cpu, comp_w)
        members.append((net, best_val))

    # Guard: drop members starved by a tight budget (val loss >15% worse than
    # the best member — val loss is now the variance-normalized MSE, so this
    # gate is metric-aligned too) so the worst case degenerates to the
    # single-model anchored parent instead of averaging in an undertrained
    # net.
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
