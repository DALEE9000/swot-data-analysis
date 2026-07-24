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
    spatial std, and explicit per-feature spatial derivatives (first AND
    second order). Unchanged proven plumbing from the lineage:
      * per-cell missing fraction (n, C);
      * per-feature stencil std over VALID cells (n, F) — front/eddy signal;
      * gradient-preserving mirror imputation (2*center - mirror, clamped to
        +/-4 sigma, center fallback) so swath-edge truncation keeps the
        first-order gradient instead of flattening the neighborhood;
      * per-feature central differences gx, gy (n, 2F) — the geostrophic
        gradient operators;
      * per-feature 5-point Laplacian (prop. to relative vorticity for SSH)
        and the two diagonal central differences (n, 3F), completing the
        local deformation/strain tensor. Mirror imputation is Taylor-linear,
        so imputed cells contribute zero second difference — a conservative
        "no measured curvature" default at swath edges.
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
    # at +/-1, vertical at +/-k, diagonals at +/-(k+1) and +/-(k-1).
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
    return np.hstack(blocks)


def _ridge_fit(Xs, Ys, mask, rng, n_sub=1_000_000, lam_rel=1e-3):
    """Closed-form masked ridge (u, v) on a subsample of the training window.

    Geostrophic balance is a LINEAR map from SSH gradients to velocity, and
    the augmented feature vector already contains those gradients (gx, gy,
    diagonals) plus the raw geostrophic velocity estimates — so a linear
    model captures the physics backbone of this task. The ridge solution
    initializes the wide path of the wide-and-deep net below: training then
    starts AT the physics solution rather than at noise.

    Per component: restrict to rows with a valid target, center X and y,
    solve (X'X + lam*n*I) w = X'y in float64, recover the intercept from the
    means. lam_rel=1e-3 (relative to n) only conditions near-singular
    directions of the standardized design; the fit is effectively OLS.
    A 1M-row subsample is plenty for a ~130-dim linear fit and keeps the
    normal-equation matmul to a few seconds.
    """
    n, d = Xs.shape
    if n > n_sub:
        idx = rng.choice(n, size=n_sub, replace=False)
    else:
        idx = np.arange(n)
    Xd = Xs[idx].astype(np.float64)
    Yd = Ys[idx]
    Md = mask[idx]
    W = np.zeros((2, d), dtype=np.float64)
    B = np.zeros(2, dtype=np.float64)
    for comp in range(2):
        rows = Md[:, comp] > 0
        if rows.sum() < d + 2:
            continue
        Xc = Xd[rows]
        yc = Yd[rows, comp].astype(np.float64)
        xm = Xc.mean(axis=0)
        ym = yc.mean()
        Xc = Xc - xm
        yc = yc - ym
        A = Xc.T @ Xc
        A[np.diag_indices(d)] += lam_rel * len(yc)
        w = np.linalg.solve(A, Xc.T @ yc)
        W[comp] = w
        B[comp] = ym - xm @ w
    return W.astype(np.float32), B.astype(np.float32)


class WideDeep(nn.Module):
    """Wide-and-deep joint (u, v) net with a heteroscedastic head.

    Two additive paths trained jointly end-to-end:
      wide : one Linear(n_inputs -> 2) straight from the augmented features
             to the targets — the geostrophic/linear backbone. Initialized
             from the closed-form ridge solution and trained with a 10x
             smaller LR and NO weight decay (decaying physics coefficients
             toward zero is the wrong prior), so it adapts slowly and stays
             a stable, time-robust anchor — linear maps generalize across
             the temporal train->test shift better than a deep net's
             co-adapted features, which is the late-test weakness.
      deep : the lineage's trunk (LayerNorm/SiLU/Dropout MLP) with a
             ZERO-initialized mu head, so at step 0 the model's output is
             exactly the ridge prediction and the trunk is forced to learn
             only the nonlinear/ageostrophic RESIDUAL (fronts, eddies, wind-
             driven flow) — the part of the signal the fast-regime deficit
             lives in.
      logvar : per-row, per-component log aleatoric variance from the trunk,
             used only to weight the training loss (proven plumbing).
    """

    def __init__(self, n_inputs, hidden=(256, 256, 128), dropout=0.1,
                 logvar_init=-3.5):
        super().__init__()
        self.wide = nn.Linear(n_inputs, 2)
        layers = []
        d = n_inputs
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.deep_mu = nn.Linear(d, 2)
        self.logvar = nn.Linear(d, 2)
        # Zero-init the deep mu head: the model starts exactly at the ridge
        # solution loaded into `wide`. The trunk still learns from step 1 —
        # deep_mu.weight receives a nonzero gradient (dL/dmu * z), and once
        # it moves off zero the trunk gets signal (standard zero-init-last-
        # layer residual trick).
        nn.init.zeros_(self.deep_mu.weight)
        nn.init.zeros_(self.deep_mu.bias)
        # Start near the marginal target log-variance so the NLL is well
        # scaled from step 1 (u, v std ~0.15-0.2 m/s -> log var ~ -3.5).
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, logvar_init)

    def forward(self, x):
        z = self.trunk(x)
        return self.wide(x) + self.deep_mu(z), self.logvar(z)

    def param_groups(self, lr_deep, lr_wide, wd_deep):
        wide_ids = {id(p) for p in self.wide.parameters()}
        deep = [p for p in self.parameters() if id(p) not in wide_ids]
        return [
            {"params": list(self.wide.parameters()), "lr": lr_wide, "weight_decay": 0.0},
            {"params": deep, "lr": lr_deep, "weight_decay": wd_deep},
        ]


def _masked_nll_anchored(mu, logvar, target, mask, row_w):
    """Gaussian NLL plus an unweighted MSE anchor, per valid component,
    scaled by a per-row recency weight (proven plumbing, unchanged).

    The NLL alone lets the variance head shrink gradients on hard rows; the
    scored metric (plain R^2) weights all rows equally, so the MSE anchor
    (weight 1 vs typical NLL precision exp(3.5) ~ 33) restores a gradient
    floor on downweighted rows. row_w is the recency weight (mean ~1), so
    rows later in the time-ordered window count more, matching the strictly
    later test split.
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
    trajectory (SWA-family, proven plumbing). p_ema <- decay*p_ema +
    (1-decay)*p each optimizer step; the average sits in the flatter center
    of the basin, which transfers better across the temporal shift."""

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


def _train_member(member_seed, deadline, device, max_epochs,
                  X_t, Y_t, M_t, W_t, X_v, Y_v, M_v,
                  X_v_cpu, Y_v_cpu, M_v_cpu, W_v_cpu, ridge_w, ridge_b):
    """Train one wide-and-deep member with the recency-weighted anchored
    heteroscedastic NLL. The wide path is loaded with the ridge solution
    before training. Model selection (raw vs weight-EMA, unweighted
    temporal-val masked MSE of mu) and the recency fine-tune are the
    lineage's proven procedure, unchanged. Returns (net, best_val)."""
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)

    net = WideDeep(X_t.shape[1], hidden=(256, 256, 128), dropout=0.1).to(device)
    with torch.no_grad():
        net.wide.weight.copy_(torch.from_numpy(ridge_w).to(device))
        net.wide.bias.copy_(torch.from_numpy(ridge_b).to(device))
    optimizer = torch.optim.AdamW(net.param_groups(lr_deep=1e-3, lr_wide=1e-4, wd_deep=1e-4))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    bs = 4096
    patience = 15
    best_val = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    best_epoch = 0
    n_train = len(X_t)

    # EMA horizon ~1000 steps (decay 0.999) ~= 1.2 epochs at bs 4096.
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

        # Model selection on UNWEIGHTED masked MSE of mu — the scored
        # quantity. Both raw and EMA weights compete; the scheduler steps on
        # the raw net's val loss (its actual training signal).
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

    # Recency fine-tune (proven +~0.01 in lineage): absorb the never-trained
    # validation tail with a short low-LR pass over the FULL window under the
    # same loss, returning a short-horizon EMA of the fine-tune trajectory.
    # Wide/deep LR ratio is preserved (1e-5 / 1e-4) so the linear backbone
    # stays slow-moving during absorption too. If the deadline cuts the pass
    # short, the EMA has barely moved, degenerating to no fine-tune.
    if time.time() < deadline and np.isfinite(best_val):
        X_f = torch.cat([X_t, X_v_cpu])
        Y_f = torch.cat([Y_t, Y_v_cpu])
        M_f = torch.cat([M_t, M_v_cpu])
        W_f = torch.cat([W_t, W_v_cpu])
        ft_opt = torch.optim.AdamW(net.param_groups(lr_deep=1e-4, lr_wide=1e-5, wd_deep=1e-4))
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
    Xs = _featurize(X_train, mean, scale, n_cells)
    mask = np.isfinite(Y_train).astype(np.float32)
    Ys = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)

    # Temporal validation split (proven): validate on the last 10% of the
    # time-ordered training window so early stopping and the LR schedule
    # optimize forward-in-time generalization.
    n = len(Xs)
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    # Closed-form ridge on the training window (excluding the validation
    # tail, which model selection must not have seen even through an init)
    # -> initialization for every member's wide path.
    rng0 = np.random.default_rng(seed)
    ridge_w, ridge_b = _ridge_fit(Xs[:n - n_val], Ys[:n - n_val],
                                  mask[:n - n_val], rng0)

    # Recency weights over the time-ordered window (proven): latest rows
    # weigh 3x the earliest, exponential ramp, normalized to mean 1.
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

    # Two-seed deep ensemble with per-member wall-clock slices (proven).
    # Members share the ridge init (it is deterministic given the data) but
    # differ in trunk init and minibatch order, so their residual nets make
    # decorrelated errors; averaging mu cancels seed-dependent variance.
    M_ens = 2
    members = []
    for m in range(M_ens):
        deadline = t0 + time_budget_s * 0.95 * (m + 1) / M_ens
        if time.time() > deadline:
            break
        net, best_val = _train_member(
            seed + 101 * m, deadline, device, max_epochs,
            X_t, Y_t, M_t, W_t, X_v, Y_v, M_v,
            X_v_cpu, Y_v_cpu, M_v_cpu, W_v_cpu, ridge_w, ridge_b)
        members.append((net, best_val))

    # Guard: drop members starved by a tight budget (val loss >15% worse than
    # the best member) so the worst case degenerates to a single wide-and-deep
    # model instead of averaging in an undertrained net.
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
