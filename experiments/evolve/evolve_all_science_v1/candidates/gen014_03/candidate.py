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
import torch.nn.functional as fnn


def _cells(d):
    # Columns are k*k spatial-stencil copies of base features, feature-major.
    # Infer the stencil cell count so the appended blocks stay layout-correct.
    for c in (9, 25, 49):
        if d % c == 0:
            return c
    return 1


def _featurize(X, mean, scale, n_cells):
    """Standardize, gradient-fill missing stencil cells, append missingness,
    spatial std, and explicit per-feature spatial derivatives.

    Feature-major layout: column f * n_cells + c is feature f at stencil
    cell c, so reshaping to (n, F, C) groups each feature's spatial
    neighborhood. Ingredients:
      * per-cell missing fraction (n, C) — distinguishes swath-edge padding
        from a genuinely calm/average ocean state;
      * per-feature stencil standard deviation (n, F) over VALID cells only —
        an explicit front/eddy intensity signal;
      * gradient-preserving imputation: a missing stencil cell at offset
        (di, dj) is filled with 2*center - mirror (first-order Taylor
        continuation), clamped to +/-4 sigma, falling back to the center
        value where the mirror is itself missing;
      * per-feature central differences (n, 2F) computed AFTER imputation —
        the finite-difference operators whose cross-feature linear combos
        are shear, strain, divergence and vorticity.
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
    # Central differences on the filled, finite stencil. Cells are flattened
    # row-major over the k x k window: horizontal neighbors at +/-1,
    # vertical neighbors at +/-k relative to the center.
    k = int(round(math.sqrt(n_cells)))
    if k >= 2 and k * k == n_cells:
        Zf = Xs[:, :F * n_cells].reshape(n, F, n_cells)
        ci = n_cells // 2
        gx = (0.5 * (Zf[:, :, ci + 1] - Zf[:, :, ci - 1])).astype(np.float32)
        gy = (0.5 * (Zf[:, :, ci + k] - Zf[:, :, ci - k])).astype(np.float32)
        blocks += [gx, gy]
    return np.hstack(blocks)


class PolarHeteroMLP(nn.Module):
    """Joint (u, v) MLP whose point prediction is POLAR-FACTORIZED.

    Shared trunk -> three heads:
      logspeed : scalar log |velocity|. exp(.) gives the predicted speed.
                 Predicting log-speed makes magnitude errors multiplicative:
                 fast-front rows (the parent's worst RMSE bin by 3x) get
                 gradients proportional to their relative — not absolute —
                 error, instead of being shrunk toward the marginal mean as
                 a direct-MSE (u, v) head does.
      dir      : unnormalized 2-vector; L2-normalized to a unit direction.
                 The unit constraint decouples orientation from magnitude —
                 direction error can no longer be "paid down" by shrinking
                 speed, and vice versa.
      logvar   : per-row, per-component log aleatoric variance, used only to
                 weight the training loss (unchanged from the lineage).

    mu = exp(logspeed) * unit(dir), so single-valid-component rows still
    train through the masked reconstruction loss exactly as before: the
    valid component of mu simply receives the gradient, which backpropagates
    into both factors.
    """

    def __init__(self, n_inputs, hidden=(256, 256, 128), dropout=0.1,
                 logvar_init=-3.5, logspeed_init=-2.0):
        super().__init__()
        layers = []
        d = n_inputs
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.logspeed = nn.Linear(d, 1)
        self.dir = nn.Linear(d, 2)
        self.logvar = nn.Linear(d, 2)
        # Marginal speed is ~0.1-0.2 m/s -> log speed ~ -2, so the speed
        # factor starts near the data scale; logvar init as in the lineage
        # (target std ~0.15-0.2 m/s -> log var ~ -3.5) so early training
        # behaves like plain scaled MSE.
        nn.init.zeros_(self.logspeed.weight)
        nn.init.constant_(self.logspeed.bias, logspeed_init)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, logvar_init)

    def forward(self, x):
        z = self.trunk(x)
        # Clamp keeps exp well-behaved: speed in [~0.007, ~4.5] m/s covers
        # any physical surface current while bounding gradients.
        ls = self.logspeed(z).clamp(-5.0, 1.5)
        n = fnn.normalize(self.dir(z), dim=1, eps=1e-6)
        mu = torch.exp(ls) * n
        return mu, self.logvar(z), ls


def _polar_loss(mu, logvar, logspeed, target, mask, row_w, aux_w=0.2):
    """Masked anchored heteroscedastic NLL on the reconstructed (u, v),
    plus a log-speed auxiliary on rows where BOTH components are valid.

    The reconstruction term is byte-identical to the lineage's proven loss
    (NLL + unweighted MSE anchor, recency-weighted): it handles single-valid
    rows, keeps the variance head's relative row weighting, and keeps the
    MSE anchor's gradient floor on downweighted rows.

    The auxiliary term supervises the speed factor directly in log space:
    (logspeed - log|y|)^2 on both-valid rows. This is the piece direct-MSE
    training cannot provide — under MSE, a 0.3 m/s error on a 1 m/s jet and
    on a 0.1 m/s drift cost the same absolute amount, so the rare fast rows
    contribute almost nothing per-row relative to their dynamic range and
    the net learns to underpredict them. In log space every row's magnitude
    error is relative, so the speed head gets an equal-strength signal
    across regimes. Weight 0.2 keeps the scored reconstruction dominant.
    """
    logvar = logvar.clamp(-7.0, 2.0)
    err2 = (mu - target).pow(2)
    per = 0.5 * (logvar + err2 * torch.exp(-logvar)) + err2
    w = mask * row_w.unsqueeze(1)
    recon = (per * w).sum() / w.sum().clamp(min=1e-8)
    both = mask[:, 0] * mask[:, 1]
    bw = both * row_w
    # target was nan_to_num'ed; `both` excludes rows where the 0-fill would
    # corrupt the speed. Floor inside sqrt/log keeps zero-velocity rows finite.
    sp = torch.sqrt((target * target).sum(dim=1).clamp(min=1e-8))
    aux = ((logspeed.squeeze(1) - torch.log(sp + 1e-3)).pow(2) * bw).sum() \
        / bw.sum().clamp(min=1e-8)
    return recon + aux_w * aux


def _masked_mse(pred, target, mask):
    diff = (pred - target) * mask
    return diff.pow(2).sum() / mask.sum().clamp(min=1)


class _EMA:
    """Exponential moving average of a net's weights along the SGD
    trajectory (SWA-family). Near convergence the running average sits in
    the flatter center of the loss basin, which transfers better across the
    temporal train->test shift and smooths seed/minibatch noise."""

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
            mu = net(X_v[i:i + bs])[0]
            losses.append(_masked_mse(mu, Y_v[i:i + bs], M_v[i:i + bs]).item())
    return float(np.mean(losses))


def _train_member(member_seed, deadline, device, max_epochs,
                  X_t, Y_t, M_t, W_t, X_v, Y_v, M_v,
                  X_v_cpu, Y_v_cpu, M_v_cpu, W_v_cpu):
    """Train one ensemble member with the polar-factorized head and the
    recency-weighted anchored loss + log-speed auxiliary. Each epoch, BOTH
    the raw net and its weight-EMA shadow are scored on the UNWEIGHTED
    temporal-val masked MSE of mu — the scored quantity — and model
    selection keeps the better of the two; the LR schedule steps on the raw
    net's val loss. Then the proven recency fine-tune runs, ending on a
    short-horizon EMA of the fine-tune trajectory. Returns (net, best_val)."""
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)

    net = PolarHeteroMLP(X_t.shape[1], hidden=(256, 256, 128), dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
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
            mu, lv, ls = net(xb)
            loss = _polar_loss(mu, lv, ls, yb, mb, wb)
            loss.backward()
            optimizer.step()
            ema.update(net)

        # Model selection on UNWEIGHTED masked MSE of mu — the scored
        # quantity — NOT on the training loss (which mixes in the variance
        # term, recency weighting, and the log-speed auxiliary).
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

    # Recency fine-tune (proven +~0.01 in lineage): absorb the most recent —
    # most test-like — 10% of the window with a short low-LR pass over the
    # FULL window, same loss, ending on a short-horizon EMA of the pass.
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
                mu, lv, ls = net(xb)
                loss = _polar_loss(mu, lv, ls, yb, mb, wb)
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

    # Temporal validation split: rows are time-ordered and the test set is a
    # temporal holdout, so validating on the last 10% of the training window
    # makes early stopping and the LR schedule optimize forward-in-time
    # generalization — the quantity the test actually measures.
    n = len(Xs)
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    # Recency weights over the time-ordered window: latest rows weigh 3x the
    # earliest (exponential ramp, normalized to mean 1 so the tuned loss
    # scale is unchanged).
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

    # Two-seed deep ensemble with per-member wall-clock slices. Member m must
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
            X_v_cpu, Y_v_cpu, M_v_cpu, W_v_cpu)
        members.append((net, best_val))

    # Guard: drop members starved by a tight budget (val loss >15% worse than
    # the best member) so the worst case degenerates to a single polar net.
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
                mu = net(xb)[0]
                acc += mu
            out[i:i + 65536] = (acc / len(keep)).cpu().numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
