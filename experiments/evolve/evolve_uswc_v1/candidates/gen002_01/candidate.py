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
import time

import numpy as np
import torch
import torch.nn as nn


def _standardize(X, mean, scale):
    Xs = (X.astype(np.float32) - mean) / scale
    return np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0)


def _ridge_baseline(Xs, Y, mask, lam=10.0):
    """Closed-form per-component ridge on standardized features (+ intercept).

    Solves (X^T W X + lam*I) beta = X^T W y with W the 0/1 validity mask of
    each target component. Returns beta of shape (d+1, 2); the last row is
    the intercept. Accumulated in float64 in chunks so the 1M-row normal
    equations stay exact and memory-light (A is only (d+1)x(d+1)).
    """
    n, d = Xs.shape
    betas = np.zeros((d + 1, 2), dtype=np.float64)
    chunk = 262144
    for c in range(2):
        A = np.zeros((d + 1, d + 1), dtype=np.float64)
        b = np.zeros(d + 1, dtype=np.float64)
        for i in range(0, n, chunk):
            Xc = Xs[i:i + chunk].astype(np.float64)
            Xc = np.concatenate([Xc, np.ones((len(Xc), 1))], axis=1)
            w = mask[i:i + chunk, c].astype(np.float64)
            yc = np.nan_to_num(Y[i:i + chunk, c].astype(np.float64), nan=0.0) * w
            Xw = Xc * w[:, None]
            A += Xw.T @ Xc
            b += Xc.T @ yc
        A[np.arange(d), np.arange(d)] += lam  # don't regularize the intercept
        betas[:, c] = np.linalg.solve(A, b)
    return betas.astype(np.float32)


def _ridge_predict(Xs, betas):
    return Xs @ betas[:-1] + betas[-1]


class MLP(nn.Module):
    """Joint (u, v) residual MLP: shared trunk, 2-unit head."""

    def __init__(self, n_inputs, hidden=(256, 256, 128), dropout=0.1):
        super().__init__()
        layers = []
        d = n_inputs
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


def _masked_mse(pred, target, mask):
    diff = (pred - target) * mask
    return diff.pow(2).sum() / mask.sum().clamp(min=1)


def train_and_predict(X_train, Y_train, X_test, params):
    seed = int(params["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(params["device"])
    max_epochs = int(params["max_epochs"])
    time_budget_s = float(params["time_budget_s"])
    t0 = time.time()

    # Per-column standardization ignoring NaNs; NaN -> 0 (the feature mean).
    mean = np.nan_to_num(np.nanmean(X_train, axis=0), nan=0.0).astype(np.float32)
    scale = np.nanstd(X_train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)

    Xs = _standardize(X_train, mean, scale)
    mask = np.isfinite(Y_train).astype(np.float32)

    # STAGE 1 — linear ridge baseline in m/s. The linear map from (stencil)
    # geostrophic velocities / SSH to surface currents is the physically
    # dominant signal and extrapolates across the temporal split far more
    # stably than a deep net. It is fit closed-form in a few seconds.
    betas = _ridge_baseline(Xs, Y_train, mask, lam=10.0)
    base_train = _ridge_predict(Xs, betas)

    # STAGE 2 — the MLP models only the residual Y - ridge(X). Per-component
    # standardization of the residuals keeps the parent's property that the
    # masked MSE weights u and v equally in variance units (i.e. optimizes
    # mean R^2 directly), now applied to the part the net actually has to learn.
    R = Y_train.astype(np.float32) - base_train
    r_mean = np.nan_to_num(np.nanmean(R, axis=0), nan=0.0).astype(np.float32)
    r_scale = np.nanstd(R, axis=0)
    r_scale = np.where(np.isfinite(r_scale) & (r_scale > 1e-6), r_scale, 1.0).astype(np.float32)
    Rs = np.nan_to_num((R - r_mean) / r_scale, nan=0.0)

    # TEMPORAL validation split: rows are flattened in cycle (time) order and
    # the test split is later cycles, so hold out the last 10% of rows as a
    # contiguous tail. Early stopping / LR scheduling then select for
    # forward-in-time generalization of the residual model.
    n = len(Xs)
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    X_t, R_t, M_t = (torch.from_numpy(a[train_idx]) for a in (Xs, Rs, mask))
    X_v = torch.from_numpy(Xs[val_idx]).to(device)
    R_v = torch.from_numpy(Rs[val_idx]).to(device)
    M_v = torch.from_numpy(mask[val_idx]).to(device)

    net = MLP(Xs.shape[1], hidden=(256, 256, 128), dropout=0.1).to(device)
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
            xb, rb, mb = X_t[idx].to(device), R_t[idx].to(device), M_t[idx].to(device)
            optimizer.zero_grad()
            loss = _masked_mse(net(xb), rb, mb)
            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            val_losses = []
            for i in range(0, len(X_v), bs):
                val_losses.append(_masked_mse(net(X_v[i:i + bs]), R_v[i:i + bs], M_v[i:i + bs]).item())
            val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = copy.deepcopy(net.state_dict())
            best_epoch = epoch

        if epoch - best_epoch >= patience:
            break
        if time.time() - t0 > time_budget_s:
            break

    net.load_state_dict(best_state)
    net.eval()

    # Predict: ridge baseline + de-standardized residual correction.
    Xp = _standardize(X_test, mean, scale)
    out = _ridge_predict(Xp, betas).astype(np.float32)
    with torch.no_grad():
        for i in range(0, len(Xp), 65536):
            xb = torch.from_numpy(Xp[i:i + 65536]).to(device)
            out[i:i + 65536] += net(xb).cpu().numpy() * r_scale + r_mean
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
