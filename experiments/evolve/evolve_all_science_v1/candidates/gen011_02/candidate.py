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
    """Standardize, gradient-fill missing stencil cells, append missingness +
    spatial std + directional stencil gradients.

    Feature-major layout: column f * n_cells + c is feature f at stencil
    cell c, so reshaping to (n, F, C) groups each feature's spatial
    neighborhood. Ingredients:
      * per-cell missing fraction (n, C) — lets both the mu and logvar heads
        distinguish swath-edge padding from a genuinely calm/average ocean
        state (edge rows are the widest R^2 gap in the diagnostics);
      * per-feature stencil standard deviation (n, F), computed on the
        standardized values over VALID cells only — an explicit front/eddy
        intensity signal, and the conditioning information the
        heteroscedastic variance head needs. Rows with a single valid cell
        get std 0 ("no local gradient information");
      * gradient-preserving imputation: a missing stencil cell at offset
        (di, dj) from the center is filled with 2*center - mirror, where
        mirror is the cell at (-di, -dj) — the first-order Taylor (constant
        local gradient) continuation of the field. Swath-edge padding
        truncates one SIDE of the neighborhood, so the mirror cell is
        usually valid; center-fill flattened the neighborhood to "locally
        uniform", erasing exactly the gradient signal the geostrophic
        features and spatial-std block carry. Extrapolated fills are clamped
        to +/-4 standardized sigma; where the mirror is itself missing the
        fill falls back to the center value, and rows whose center is NaN
        fall back to the mean (0 after standardization) via the final
        nan_to_num. The missingness block still marks padded cells;
      * directional gradients (n, 2F), computed AFTER imputation: per
        feature, the mean right-minus-left column difference (zonal, gx) and
        the mean top-minus-bottom row difference (meridional, gy) of the
        k x k stencil, each divided by (k-1). These are the discrete
        first-order derivatives the ocean physics runs on — SSH gradients
        are geostrophic shear (u is tied to the meridional SSH gradient, the
        weaker scored component), and gradients of the velocity features
        encode vorticity/strain, the front/eddy dynamics behind the
        fast-regime deficit. The MLP could in principle assemble each from 9
        signed columns, but supplying them explicitly removes that burden —
        the same reason the spatial-std block earned its keep. Computing
        after mirror-fill means edge rows get their linearly-continued
        gradient rather than a truncated one.
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
    # Directional gradients on the imputed (still-standardized) stencil.
    k = int(round(math.sqrt(n_cells)))
    if k >= 2 and k * k == n_cells:
        Zk = Z.reshape(n, F, k, k)  # row-major: axis 2 = rows, axis 3 = cols
        gx = ((Zk[:, :, :, k - 1] - Zk[:, :, :, 0]).mean(axis=2)
              / float(k - 1)).astype(np.float32)
        gy = ((Zk[:, :, 0, :] - Zk[:, :, k - 1, :]).mean(axis=2)
              / float(k - 1)).astype(np.float32)
        grad = np.hstack([gx, gy])
    else:
        grad = np.zeros((n, 0), dtype=np.float32)
    grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return np.hstack([Xs, miss, spat, grad])


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


class _EMA:
    """Exponential moving average of a net's parameters.

    Late-trajectory SGD iterates orbit the local minimum with minibatch
    noise; their running average sits nearer the basin center and
    generalizes better — a free along-trajectory ensemble, complementary to
    the cross-seed ensemble (which cancels a different noise source). Decay
    is warmed up as min(decay, (1+n)/(10+n)) so early averages track the
    fast-moving young network instead of its random init. The model has no
    buffers (LayerNorm affine terms are parameters), so averaging
    parameters covers the full state.
    """

    def __init__(self, net, decay):
        self.decay = decay
        self.n = 0
        self.shadow = [p.detach().clone() for p in net.parameters()]

    @torch.no_grad()
    def update(self, net):
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        for s, p in zip(self.shadow, net.parameters()):
            s.mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, net):
        for s, p in zip(self.shadow, net.parameters()):
            p.copy_(s)

    def clone_state(self):
        return [s.clone() for s in self.shadow]


@torch.no_grad()
def _load_params(net, state):
    for p, s in zip(net.parameters(), state):
        p.copy_(s)


def _masked_nll_anchored(mu, logvar, target, mask):
    """Gaussian NLL plus an unweighted MSE anchor, per valid component.

    The NLL alone lets the variance head shrink gradients on hard rows by
    exp(-logvar) — up to ~exp(-2) under the clamp — which trades edge/fast-
    regime accuracy for the easy majority. The scored metric (plain R^2)
    weights all rows equally, so that trade is a training/metric mismatch.
    The MSE anchor (weight 1 vs typical NLL precision exp(3.5) ~ 33) is a
    ~6% perturbation on well-fit rows but restores a gradient floor on
    downweighted rows, so the mu head keeps fitting them while the variance
    head still handles calibration and relative weighting.
    """
    logvar = logvar.clamp(-7.0, 2.0)
    err2 = (mu - target).pow(2)
    per = 0.5 * (logvar + err2 * torch.exp(-logvar)) + err2
    return (per * mask).sum() / mask.sum().clamp(min=1)


def _masked_mse(pred, target, mask):
    diff = (pred - target) * mask
    return diff.pow(2).sum() / mask.sum().clamp(min=1)


def _train_member(member_seed, deadline, device, max_epochs,
                  X_t, Y_t, M_t, X_v, Y_v, M_v, X_v_cpu, Y_v_cpu, M_v_cpu):
    """Train one ensemble member with the anchored heteroscedastic NLL.

    Model selection, LR scheduling, and early stopping all run on the EMA
    weights' temporal-val masked MSE — the EMA copy is what gets deployed,
    so it is what must be selected on. The proven recency fine-tune then
    runs with its own faster-decay EMA restarted from the selected weights,
    so absorbing the validation tail does not leave the final weights
    hostage to the last few minibatches of the low-LR pass. Returns
    (net, best_val)."""
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)

    net = HeteroMLP(X_t.shape[1], hidden=(256, 256, 128), dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    ema = _EMA(net, decay=0.999)

    bs = 4096
    patience = 15
    best_val = float("inf")
    best_state = ema.clone_state()
    best_epoch = 0
    n_train = len(X_t)

    for epoch in range(1, max_epochs + 1):
        net.train()
        order = torch.from_numpy(rng.permutation(n_train))
        for i in range(0, n_train, bs):
            idx = order[i:i + bs]
            xb, yb, mb = X_t[idx].to(device), Y_t[idx].to(device), M_t[idx].to(device)
            optimizer.zero_grad()
            mu, lv = net(xb)
            loss = _masked_nll_anchored(mu, lv, yb, mb)
            loss.backward()
            optimizer.step()
            ema.update(net)

        # Model selection and LR scheduling on masked MSE of the EMA mu — the
        # deployed quantity — NOT on the training loss, whose value mixes in
        # the variance term. Swap EMA weights in for the eval, then restore
        # the raw weights so optimization continues undisturbed.
        net.eval()
        raw = [p.detach().clone() for p in net.parameters()]
        ema.copy_to(net)
        with torch.no_grad():
            val_losses = []
            for i in range(0, len(X_v), bs):
                mu, _ = net(X_v[i:i + bs])
                val_losses.append(_masked_mse(mu, Y_v[i:i + bs], M_v[i:i + bs]).item())
            val_loss = float(np.mean(val_losses))
        _load_params(net, raw)
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = ema.clone_state()
            best_epoch = epoch

        if epoch - best_epoch >= patience:
            break
        if time.time() > deadline:
            break

    _load_params(net, best_state)

    # Recency fine-tune (proven +~0.01 in lineage): the validation tail is the
    # most recent — and hence most test-like — 10% of the training window, and
    # the model above never trained on it. After model selection, absorb it
    # with a short low-LR pass over the FULL window (tail included), using the
    # same anchored loss so relative row weighting stays consistent. A fresh
    # faster-decay EMA (0.998, ~500-step horizon vs ~1900 fine-tune steps)
    # restarts from the selected weights and is loaded at the end, so the
    # deployed net averages over the recency pass instead of stopping wherever
    # the time budget happened to cut the last minibatch.
    if time.time() < deadline and np.isfinite(best_val):
        X_f = torch.cat([X_t, X_v_cpu])
        Y_f = torch.cat([Y_t, Y_v_cpu])
        M_f = torch.cat([M_t, M_v_cpu])
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
                ft_opt.zero_grad()
                mu, lv = net(xb)
                loss = _masked_nll_anchored(mu, lv, yb, mb)
                loss.backward()
                ft_opt.step()
                ft_ema.update(net)
                if bi % 100 == 0 and time.time() > deadline:
                    out_of_time = True
                    break
            if out_of_time:
                break
        ft_ema.copy_to(net)

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

    # Temporal validation split: rows are time-ordered (later rows = later in
    # the mission window, regions interleaved), and the test set is a temporal
    # holdout. Validating on the last 10% of the training window makes early
    # stopping and the LR schedule optimize forward-in-time generalization —
    # the quantity the test actually measures — instead of random-split
    # interpolation, which stops too late and overfits the training window.
    n = len(Xs)
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    X_t, Y_t, M_t = (torch.from_numpy(a[train_idx]) for a in (Xs, Ys, mask))
    # Keep CPU copies of the validation tail for the post-selection fine-tune.
    X_v_cpu = torch.from_numpy(Xs[val_idx])
    Y_v_cpu = torch.from_numpy(Ys[val_idx])
    M_v_cpu = torch.from_numpy(mask[val_idx])
    X_v = X_v_cpu.to(device)
    Y_v = Y_v_cpu.to(device)
    M_v = M_v_cpu.to(device)
    del Xs

    # Two-seed deep ensemble with per-member wall-clock slices. Independently
    # seeded nets make decorrelated errors — including in their learned
    # variance surfaces — and averaging their mu predictions cancels
    # seed-dependent variance; the gain concentrates on edge-stencil and
    # fast-front rows where single-net outputs are noisiest. Member m must
    # finish by 0.95*(m+1)/M of the budget, leaving 5% for test prediction.
    M_ens = 2
    members = []
    for m in range(M_ens):
        deadline = t0 + time_budget_s * 0.95 * (m + 1) / M_ens
        if time.time() > deadline:
            break
        net, best_val = _train_member(
            seed + 101 * m, deadline, device, max_epochs,
            X_t, Y_t, M_t, X_v, Y_v, M_v, X_v_cpu, Y_v_cpu, M_v_cpu)
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
