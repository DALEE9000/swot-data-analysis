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
    for c in (9, 25, 49):
        if d % c == 0:
            return c
    return 1


def _featurize(X, mean, scale, n_cells):
    """Standardize, gradient-fill missing stencil cells, and emit ONE
    superset feature array both ensemble members slice column views of:

      [Xs (F*C) | mask_full (F*C) | spat (F) | miss (C) | gx (F) | gy (F)]

    * Xs — standardized values, gradient-mirror imputed (proven in lineage):
      a missing cell at offset (di, dj) from the center is filled with
      2*center - mirror (first-order Taylor continuation), clamped to +/-4
      sigma, falling back to the center where the mirror is also missing and
      to the mean (0) where the center itself is NaN.
    * mask_full — per-feature per-cell NaN mask (the CNN member's mask image:
      WHICH side of each feature's neighborhood is padded).
    * spat — per-feature stencil std over VALID cells (front/eddy intensity;
      computed before imputation so filled cells cannot inflate it).
    * miss — per-cell missing fraction averaged over features (the MLP
      member's scalar missingness summary, as in the parent).
    * gx, gy — per-feature central differences on the filled stencil (the
      parent's engineered derivative features; the mirror imputation is
      Taylor-consistent with them, so edge rows get coherent, conservatively
      damped derivatives). Emitted only when the stencil is a k x k grid
      with k >= 2.
    """
    Xs = (X.astype(np.float32) - mean) / scale
    n, d = Xs.shape
    F = d // n_cells
    nan_flags = np.isnan(X).reshape(n, F, n_cells)
    mask_full = nan_flags.astype(np.float32)
    miss = mask_full.mean(axis=1)  # (n, C)
    Z = Xs.reshape(n, F, n_cells)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        spat = np.nanstd(Z, axis=2).astype(np.float32)
    spat = np.nan_to_num(spat, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    # Fill AFTER the std block (which must see true valid cells). Snapshot
    # mirrors before in-place filling so late cells never read imputed data.
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
    blocks = [Xs, mask_full.reshape(n, F * n_cells), spat, miss]
    k = int(round(math.sqrt(n_cells)))
    if k >= 2 and k * k == n_cells:
        Zf = Xs[:, :F * n_cells].reshape(n, F, n_cells)
        ci = n_cells // 2
        gx = (0.5 * (Zf[:, :, ci + 1] - Zf[:, :, ci - 1])).astype(np.float32)
        gy = (0.5 * (Zf[:, :, ci + k] - Zf[:, :, ci - k])).astype(np.float32)
        blocks += [gx, gy]
    return np.hstack(blocks)


def _member_cols(F, n_cells, total):
    """Column views into the superset array for each member architecture.

    MLP: [Xs, spat, miss, gx, gy] — everything except the full mask image
         (contiguous tail from spat onward), matching the parent's inputs.
    CNN: [Xs, mask_full, spat] — the first 2*F*C + F columns, exactly the
         layout StencilCNN.forward slices.
    """
    FC = F * n_cells
    cols_mlp = np.concatenate([np.arange(0, FC), np.arange(2 * FC, total)])
    cols_cnn = np.arange(0, 2 * FC + F)
    return torch.from_numpy(cols_mlp), torch.from_numpy(cols_cnn)


class HeteroMLP(nn.Module):
    """Joint (u, v) MLP with a heteroscedastic head (the parent, verbatim).

    Shared trunk -> mu head (scored point predictions) and logvar head
    (per-row, per-component log aleatoric variance, used only to weight the
    training loss so hard/noisy rows stop distorting the fit)."""

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


class StencilCNN(nn.Module):
    """Per-feature 2D stencil CNN with a heteroscedastic head (the proven
    gen012_03 architecture, verbatim). Values and the per-feature NaN-mask
    image are stacked into 2F channels; two small convolutions turn the
    k x k neighborhood into a 128-dim descriptor — the 3x3 kernels are
    learnable finite-difference operators, so shear/strain/vorticity
    stencils are first-class, and the mask channels show WHICH side of the
    neighborhood is padded."""

    def __init__(self, F, n_cells, dropout=0.1, logvar_init=-3.5):
        super().__init__()
        self.F = F
        self.C = n_cells
        self.k = max(1, int(round(math.sqrt(n_cells))))
        self.conv = nn.Sequential(
            nn.Conv2d(2 * F, 64, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(64, 128, kernel_size=self.k),
            nn.SiLU(),
        )
        d_in = 128 + F
        self.head = nn.Sequential(
            nn.Linear(d_in, 192), nn.LayerNorm(192), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(192, 128), nn.LayerNorm(128), nn.SiLU(), nn.Dropout(dropout),
        )
        self.mu = nn.Linear(128, 2)
        self.logvar = nn.Linear(128, 2)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, logvar_init)

    def forward(self, x):
        n = x.shape[0]
        F, C, k = self.F, self.C, self.k
        vals = x[:, :F * C].view(n, F, k, k)
        mask = x[:, F * C:2 * F * C].view(n, F, k, k)
        spat = x[:, 2 * F * C:]
        h = self.conv(torch.cat([vals, mask], dim=1)).flatten(1)
        z = self.head(torch.cat([h, spat], dim=1))
        return self.mu(z), self.logvar(z)


def _masked_nll_anchored(mu, logvar, target, mask, row_w):
    """Gaussian NLL plus an unweighted MSE anchor, per valid component,
    scaled by a per-row recency weight (all proven in lineage). The anchor
    keeps a gradient floor on rows the variance head downweights; row_w
    (mean ~1) emphasizes later, more test-like rows without changing the
    loss scale the optimizer was tuned on."""
    logvar = logvar.clamp(-7.0, 2.0)
    err2 = (mu - target).pow(2)
    per = 0.5 * (logvar + err2 * torch.exp(-logvar)) + err2
    w = mask * row_w.unsqueeze(1)
    return (per * w).sum() / w.sum().clamp(min=1e-8)


def _masked_mse(pred, target, mask):
    diff = (pred - target) * mask
    return diff.pow(2).sum() / mask.sum().clamp(min=1)


def _train_member(make_net, cols, member_seed, deadline, device, max_epochs,
                  X_t, Y_t, M_t, W_t, X_v, Y_v, M_v,
                  X_v_cpu, Y_v_cpu, M_v_cpu, W_v_cpu):
    """Train one ensemble member (architecture supplied by make_net, inputs
    the `cols` view of the superset features) with the recency-weighted
    anchored heteroscedastic NLL — early stop on UNWEIGHTED temporal-val
    masked MSE of mu (the scored quantity), then the proven recency
    fine-tune — within its wall-clock slice. Returns (net, cols, best_val)."""
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)

    net = make_net().to(device)
    cols = cols.to(device)
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
        out_of_time = False
        for bi, i in enumerate(range(0, n_train, bs)):
            idx = order[i:i + bs]
            xb = X_t[idx].to(device)[:, cols]
            yb, mb = Y_t[idx].to(device), M_t[idx].to(device)
            wb = W_t[idx].to(device)
            optimizer.zero_grad()
            mu, lv = net(xb)
            loss = _masked_nll_anchored(mu, lv, yb, mb, wb)
            loss.backward()
            optimizer.step()
            # The CNN member costs more per batch than the MLP; check the
            # deadline mid-epoch so a member cannot blow its slice.
            if bi % 200 == 0 and time.time() > deadline:
                out_of_time = True
                break

        # Model selection and LR scheduling on UNWEIGHTED masked MSE of mu —
        # the scored quantity — not the training loss, whose value mixes in
        # the variance term and the recency weighting.
        net.eval()
        with torch.no_grad():
            val_losses = []
            for i in range(0, len(X_v), bs):
                mu, _ = net(X_v[i:i + bs][:, cols])
                val_losses.append(_masked_mse(mu, Y_v[i:i + bs], M_v[i:i + bs]).item())
            val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = copy.deepcopy(net.state_dict())
            best_epoch = epoch

        if epoch - best_epoch >= patience:
            break
        if out_of_time or time.time() > deadline:
            break

    net.load_state_dict(best_state)

    # Recency fine-tune (proven +~0.01 in lineage): absorb the held-out
    # validation tail with a short low-LR pass over the FULL window using the
    # same recency-weighted anchored loss.
    if time.time() < deadline and np.isfinite(best_val):
        X_f = torch.cat([X_t, X_v_cpu])
        Y_f = torch.cat([Y_t, Y_v_cpu])
        M_f = torch.cat([M_t, M_v_cpu])
        W_f = torch.cat([W_t, W_v_cpu])
        ft_opt = torch.optim.AdamW(net.parameters(), lr=1e-4, weight_decay=1e-4)
        n_all = len(X_f)
        out_of_time = False
        net.train()
        for _ in range(2):
            order = torch.from_numpy(rng.permutation(n_all))
            for bi, i in enumerate(range(0, n_all, bs)):
                idx = order[i:i + bs]
                xb = X_f[idx].to(device)[:, cols]
                yb, mb = Y_f[idx].to(device), M_f[idx].to(device)
                wb = W_f[idx].to(device)
                ft_opt.zero_grad()
                mu, lv = net(xb)
                loss = _masked_nll_anchored(mu, lv, yb, mb, wb)
                loss.backward()
                ft_opt.step()
                if bi % 100 == 0 and time.time() > deadline:
                    out_of_time = True
                    break
            if out_of_time:
                break

    net.eval()
    return net, cols, best_val


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
    F = X_train.shape[1] // n_cells
    Xs = _featurize(X_train, mean, scale, n_cells)
    total_cols = Xs.shape[1]
    cols_mlp, cols_cnn = _member_cols(F, n_cells, total_cols)
    mask = np.isfinite(Y_train).astype(np.float32)
    Ys = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)

    # Temporal validation split: rows are time-ordered and the test set is a
    # temporal holdout, so validate on the last 10% of the training window to
    # make early stopping optimize forward-in-time generalization.
    n = len(Xs)
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    # Recency weights: latest rows weigh 3x the earliest (exponential ramp,
    # mean-normalized), proven against within-window drift in the lineage.
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

    # Heterogeneous two-member deep ensemble with per-member wall-clock
    # slices: member 1 is the parent's derivative-augmented MLP, member 2 the
    # proven StencilCNN. Different inductive biases (engineered central
    # differences + scalar missingness vs learned finite-difference kernels +
    # full mask images) decorrelate errors far more than a seed change, with
    # the gain concentrated on edge-stencil and fast-front rows where the two
    # architectures disagree most. The MLP trains first so a tight budget
    # degenerates to the proven parent, not the slower CNN.
    specs = [
        (lambda: HeteroMLP(len(cols_mlp), hidden=(256, 256, 128), dropout=0.1), cols_mlp),
        (lambda: StencilCNN(F, n_cells, dropout=0.1), cols_cnn),
    ]
    members = []
    for m, (make_net, cols) in enumerate(specs):
        deadline = t0 + time_budget_s * 0.95 * (m + 1) / len(specs)
        if time.time() > deadline:
            break
        net, dcols, best_val = _train_member(
            make_net, cols, seed + 101 * m, deadline, device, max_epochs,
            X_t, Y_t, M_t, W_t, X_v, Y_v, M_v,
            X_v_cpu, Y_v_cpu, M_v_cpu, W_v_cpu)
        members.append((net, dcols, best_val))

    # Guard: drop members starved by a tight budget (val loss >15% worse than
    # the best) so the worst case degenerates to the single anchored MLP.
    finite = [(net, c, v) for net, c, v in members if np.isfinite(v)]
    if finite:
        v_best = min(v for _, _, v in finite)
        keep = [(net, c) for net, c, v in finite if v <= v_best * 1.15]
    else:
        keep = [(members[0][0], members[0][1])]

    # Predict on the test set in batches, averaging mu over accepted members.
    out = np.zeros((len(X_test), 2), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_test), 65536):
            xb = torch.from_numpy(_featurize(X_test[i:i + 65536], mean, scale, n_cells)).to(device)
            acc = torch.zeros((xb.shape[0], 2), device=device)
            for net, c in keep:
                mu, _ = net(xb[:, c])
                acc += mu
            out[i:i + 65536] = (acc / len(keep)).cpu().numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
