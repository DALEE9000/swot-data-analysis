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
    """Standardize, gradient-fill missing stencil cells, append missingness + spatial std.

    Identical to the reference plumbing (proven across the lineage):
      * per-cell missing fraction (n, C);
      * per-feature stencil std over VALID cells (n, F);
      * gradient-preserving imputation 2*center - mirror, clamped to +/-4
        sigma, falling back to center then to 0 (the standardized mean).
    Output layout: [Xs (F*C), miss (C), spat (F)].
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
    return np.hstack([Xs, miss, spat])


class StencilCNN(nn.Module):
    """CNN over the k x k stencil with a heteroscedastic head.

    The flat-MLP lineage feeds the stencil as a permuted 1-D vector, so the
    net must learn spatial-derivative structure with no locality prior. The
    quantities that determine (u, v) — geostrophic shear, vorticity, strain,
    SSH gradients — are local differential operators on exactly this grid.
    Structure:
      * input image (F+1, s, s): the F standardized/imputed features plus the
        per-cell missing-fraction as an explicit spatial channel, so swath-edge
        padding is seen WHERE it is, not just how much of it there is;
      * Conv 3x3 (pad 1) -> SiLU -> Conv s x s collapse -> 96 responses:
        a bank of learned finite-difference operators over all features;
      * head on [conv responses, aux (miss, spat-std)] -> mu, logvar. The
        logvar head weights the training loss only (anchored NLL), exactly as
        in the reference.
    """

    def __init__(self, n_feat, n_cells, dropout=0.1, logvar_init=-3.5):
        super().__init__()
        self.n_feat = n_feat
        self.n_cells = n_cells
        self.side = max(1, int(round(math.sqrt(n_cells))))
        c_in = n_feat + 1
        self.conv = nn.Sequential(
            nn.Conv2d(c_in, 48, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(48, 96, kernel_size=self.side),
            nn.SiLU(),
        )
        n_aux = n_cells + n_feat
        layers = []
        d = 96 + n_aux
        for h in (192, 128):
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        self.head = nn.Sequential(*layers)
        self.mu = nn.Linear(d, 2)
        self.logvar = nn.Linear(d, 2)
        nn.init.zeros_(self.logvar.weight)
        nn.init.constant_(self.logvar.bias, logvar_init)

    def forward(self, x):
        b = x.shape[0]
        fc = self.n_feat * self.n_cells
        grid = x[:, :fc].view(b, self.n_feat, self.side, self.side)
        missch = x[:, fc:fc + self.n_cells].view(b, 1, self.side, self.side)
        z = self.conv(torch.cat([grid, missch], dim=1)).flatten(1)
        aux = x[:, fc:]
        h = self.head(torch.cat([z, aux], dim=1))
        return self.mu(h), self.logvar(h)


def _masked_nll_anchored(mu, logvar, target, mask):
    """Gaussian NLL plus an unweighted MSE anchor, per valid component
    (reference loss: variance head handles relative weighting, the anchor
    keeps a gradient floor on downweighted edge/fast rows so mu still fits
    them — the scored R^2 weights all rows equally)."""
    logvar = logvar.clamp(-7.0, 2.0)
    err2 = (mu - target).pow(2)
    per = 0.5 * (logvar + err2 * torch.exp(-logvar)) + err2
    return (per * mask).sum() / mask.sum().clamp(min=1)


def _masked_mse(pred, target, mask):
    diff = (pred - target) * mask
    return diff.pow(2).sum() / mask.sum().clamp(min=1)


def _train_member(member_seed, deadline, device, max_epochs, n_feat, n_cells,
                  X_t, Y_t, M_t, X_v, Y_v, M_v, X_v_cpu, Y_v_cpu, M_v_cpu):
    """Train one ensemble member (anchored NLL, early stop on temporal-val
    masked MSE of mu, recency fine-tune) within its wall-clock slice."""
    torch.manual_seed(member_seed)
    rng = np.random.default_rng(member_seed)

    net = StencilCNN(n_feat, n_cells, dropout=0.1).to(device)
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
            loss = _masked_nll_anchored(mu, lv, yb, mb)
            loss.backward()
            optimizer.step()

        # Model selection / LR schedule on masked MSE of mu — the scored
        # quantity — not the training loss.
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

    # Recency fine-tune (proven +~0.01): absorb the most recent 10% (the
    # validation tail) with a short low-LR pass over the full window.
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
                loss = _masked_nll_anchored(mu, lv, yb, mb)
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

    mean = np.nan_to_num(np.nanmean(X_train, axis=0), nan=0.0).astype(np.float32)
    scale = np.nanstd(X_train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)

    n_cells = _cells(X_train.shape[1])
    n_feat = X_train.shape[1] // n_cells
    Xs = _featurize(X_train, mean, scale, n_cells)
    mask = np.isfinite(Y_train).astype(np.float32)
    Ys = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)

    # Temporal validation split: last 10% of the time-ordered training window,
    # so early stopping / LR schedule optimize forward-in-time generalization
    # (the quantity the temporal-holdout test measures).
    n = len(Xs)
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    X_t, Y_t, M_t = (torch.from_numpy(a[train_idx]) for a in (Xs, Ys, mask))
    X_v_cpu = torch.from_numpy(Xs[val_idx])
    Y_v_cpu = torch.from_numpy(Ys[val_idx])
    M_v_cpu = torch.from_numpy(mask[val_idx])
    X_v = X_v_cpu.to(device)
    Y_v = Y_v_cpu.to(device)
    M_v = M_v_cpu.to(device)
    del Xs

    # Two-seed deep ensemble with per-member wall-clock slices (proven).
    M_ens = 2
    members = []
    for m in range(M_ens):
        deadline = t0 + time_budget_s * 0.95 * (m + 1) / M_ens
        if time.time() > deadline:
            break
        net, best_val = _train_member(
            seed + 101 * m, deadline, device, max_epochs, n_feat, n_cells,
            X_t, Y_t, M_t, X_v, Y_v, M_v, X_v_cpu, Y_v_cpu, M_v_cpu)
        members.append((net, best_val))

    # Guard: drop members starved by a tight budget (>15% worse than best).
    finite = [(net, v) for net, v in members if np.isfinite(v)]
    if finite:
        v_best = min(v for _, v in finite)
        keep = [net for net, v in finite if v <= v_best * 1.15]
    else:
        keep = [members[0][0]]

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
