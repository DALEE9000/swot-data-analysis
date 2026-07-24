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
    """Standardize, center-fill missing stencil cells, append missingness,
    spatial std, and directional stencil gradients.

    Feature-major layout: column f * n_cells + c is feature f at stencil
    cell c, so reshaping to (n, F, C) groups each feature's spatial
    neighborhood. Four ingredients:
      * per-cell missing fraction (n, C) — lets both the mu and logvar heads
        distinguish swath-edge padding from a genuinely calm/average ocean
        state (edge rows are the widest R^2 gap in the diagnostics);
      * per-feature stencil standard deviation (n, F), computed on the
        standardized values over VALID cells only — an isotropic front/eddy
        intensity signal, and the conditioning information the
        heteroscedastic variance head needs. Rows with a single valid cell
        get std 0 ("no local gradient information");
      * center-cell imputation: a missing stencil cell is filled with that
        feature's CENTER-cell value rather than the global mean. Mean-fill
        hands every swath-edge row a neighborhood that looks like "average
        ocean" welded onto its real center — a systematic distortion on
        exactly the rows with the worst held-out R^2. Center-fill degrades an
        edge row toward "locally uniform field", which is the physically
        neutral assumption; the missingness block still marks which cells
        were padded. Rows whose center is itself NaN fall back to the mean
        (0 after standardization) via the final nan_to_num;
      * per-feature DIRECTIONAL gradients (n, 2F): mean first difference
        across the stencil along each grid axis, computed on the
        center-filled values. The std block is isotropic — it measures how
        much local variation exists but not which way it points. Directional
        gradients supply the dynamics the fast-regime rows are missing:
        SSH gradients encode geostrophic shear, and gradients of the
        velocity components are vorticity/strain proxies — exactly what
        separates a fast frontal jet from calm water with identical
        center-cell values. Center-fill makes padded directions read as
        zero gradient (locally uniform), consistent with the imputation
        philosophy, and the missingness block still flags those rows.
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
    # Center-fill AFTER the std block (which must see true valid cells).
    center = Z[:, :, n_cells // 2]
    for c in range(n_cells):
        if c == n_cells // 2:
            continue
        col = Z[:, :, c]
        hole = np.isnan(col)
        if hole.any():
            col[hole] = center[hole]
    # Directional gradients AFTER center-fill: rows whose neighborhood was
    # padded read as locally uniform (gradient 0). Rows with a NaN center
    # are still NaN here and get zeroed by nan_to_num.
    k = int(round(math.sqrt(n_cells)))
    if k * k == n_cells and k >= 2:
        G = Z.reshape(n, F, k, k)
        gx = (G[:, :, :, -1] - G[:, :, :, 0]).mean(axis=2) / (k - 1)
        gy = (G[:, :, -1, :] - G[:, :, 0, :]).mean(axis=2) / (k - 1)
        grad = np.concatenate([gx, gy], axis=1).astype(np.float32)
        grad = np.nan_to_num(grad, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    else:
        grad = np.zeros((n, 0), dtype=np.float32)
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


def _masked_nll(mu, logvar, target, mask):
    # Gaussian NLL per valid component; clamp keeps exp() finite and stops
    # the model from "explaining away" everything with infinite variance.
    logvar = logvar.clamp(-7.0, 2.0)
    per = 0.5 * (logvar + (mu - target).pow(2) * torch.exp(-logvar))
    return (per * mask).sum() / mask.sum().clamp(min=1)


def _masked_mse(pred, target, mask):
    diff = (pred - target) * mask
    return diff.pow(2).sum() / mask.sum().clamp(min=1)


def _train_member(member_seed, deadline, device, max_epochs,
                  X_t, Y_t, M_t, X_v, Y_v, M_v,
                  X_v_cpu, Y_v_cpu, M_v_cpu):
    """Train one ensemble member — byte-for-byte the parent's training loop
    (heteroscedastic NLL, ReduceLROnPlateau, patience-15 early stopping on
    temporal-val masked MSE, recency fine-tune), bounded by the member's own
    wall-clock deadline instead of the global budget. Seed controls init AND
    the per-epoch shuffle order, so members explore genuinely different SGD
    trajectories; their averaged predictions cancel the fit noise that is
    largest on the ill-conditioned edge-stencil and fast-regime rows."""
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

    for epoch in range(1, max_epochs + 1):
        net.train()
        order = torch.from_numpy(rng.permutation(n_train))
        for i in range(0, n_train, bs):
            idx = order[i:i + bs]
            xb, yb, mb = X_t[idx].to(device), Y_t[idx].to(device), M_t[idx].to(device)
            optimizer.zero_grad()
            mu, lv = net(xb)
            loss = _masked_nll(mu, lv, yb, mb)
            loss.backward()
            optimizer.step()

        # Model selection and LR scheduling on masked MSE of mu — the scored
        # quantity — NOT on the NLL, whose value mixes in the variance term.
        net.eval()
        with torch.no_grad():
            val_losses = []
            for i in range(0, len(X_v), bs):
                mu, _ = net(X_v[i:i + bs])
                val_losses.append(_masked_mse(mu, Y_v[i:i + bs], M_v[i:i + bs]).item())
            val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = copy.deepcopy(net.state_dict())
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
    # same heteroscedastic NLL so noisy rows stay downweighted.
    if time.time() < deadline and np.isfinite(best_val):
        X_f = torch.cat([X_t, X_v_cpu])
        Y_f = torch.cat([Y_t, Y_v_cpu])
        M_f = torch.cat([M_t, M_v_cpu])
        ft_opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
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
                loss = _masked_nll(mu, lv, yb, mb)
                loss.backward()
                ft_opt.step()
                if bi % 100 == 0 and time.time() > deadline:
                    out_of_time = True
                    break
            if out_of_time:
                break

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
    # the quantity the test actually measures.
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

    # THE ONE CHANGE vs the parent: a seed-diverse deep ensemble with
    # per-member wall-clock slices, grafted from the inspiration lineage.
    # The parent used ~294s of the budget — early stopping fired long before
    # the wall clock — so two more independently seeded members fit in the
    # unused time at zero marginal cost to any member's convergence. Each
    # member differs in init and shuffle order only; averaging their mu
    # predictions cancels per-fit variance, which the diagnostics localize
    # to the edge-stencil rows (r2_u 0.284 vs 0.459 full stencil) and the
    # fast-speed bin — exactly where independently seeded fits disagree
    # most. Members that early-stop cheaply leave their slack to later
    # members via the shared clock check at loop top.
    M_ens = 3
    members = []
    for m in range(M_ens):
        deadline = t0 + time_budget_s * 0.95 * (m + 1) / M_ens
        if time.time() > deadline:
            break
        net, best_val = _train_member(
            seed + 101 * m, deadline, device, max_epochs,
            X_t, Y_t, M_t, X_v, Y_v, M_v,
            X_v_cpu, Y_v_cpu, M_v_cpu)
        members.append((net, best_val))

    # Guard (from the inspiration): drop members starved by a tight budget
    # (val loss >15% worse than the best) so the worst case degenerates to
    # the single-net parent instead of averaging in an undertrained member.
    finite = [(net, v) for net, v in members if np.isfinite(v)]
    if finite:
        v_best = min(v for _, v in finite)
        keep = [net for net, v in finite if v <= v_best * 1.15]
    else:
        keep = [members[0][0]]

    # Predict on the test set in batches; average mu over accepted members.
    out = np.empty((len(X_test), 2), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_test), 65536):
            xb = torch.from_numpy(_featurize(X_test[i:i + 65536], mean, scale, n_cells)).to(device)
            acc = torch.zeros((xb.shape[0], 2), device=device)
            for net in keep:
                mu, _ = net(xb)
                acc += mu
            out[i:i + 65536] = (acc / len(keep)).cpu().numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
