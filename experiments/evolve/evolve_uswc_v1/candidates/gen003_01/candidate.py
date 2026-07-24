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


def _fit_ridge(Xs, Ys, mask, alpha=10.0):
    """Closed-form ridge per target component on the masked rows.

    Xs is standardized (NaN->0), so features are ~zero-mean; the intercept is
    absorbed by centering y. Returns (B, b): weights (d, 2) and intercepts (2,).
    Exact and deterministic — captures the linear (geostrophic) part of the
    signal on ALL training data with no SGD noise. Normal equations are
    accumulated in float64 chunks so peak memory stays small.
    """
    d = Xs.shape[1]
    B = np.zeros((d, 2), dtype=np.float64)
    b = np.zeros(2, dtype=np.float64)
    chunk = 200_000
    for c in range(2):
        m = mask[:, c] > 0
        n_c = int(m.sum())
        if n_c == 0:
            continue
        ybar = float(Ys[m, c].mean())
        A = np.zeros((d, d), dtype=np.float64)
        g = np.zeros(d, dtype=np.float64)
        idx = np.flatnonzero(m)
        for i0 in range(0, len(idx), chunk):
            ii = idx[i0:i0 + chunk]
            Xc = Xs[ii].astype(np.float64)
            yc = Ys[ii, c].astype(np.float64) - ybar
            A += Xc.T @ Xc
            g += Xc.T @ yc
        A[np.diag_indices(d)] += alpha
        B[:, c] = np.linalg.solve(A, g)
        b[c] = ybar
    return B.astype(np.float32), b.astype(np.float32)


class MLP(nn.Module):
    """Residual-correction MLP: shared trunk, 2-unit head."""

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
    Y_raw = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)

    # RIDGE BASELINE: fit the linear map features -> (u, v) in closed form on
    # the full training set. Geostrophic velocity features are near-linearly
    # related to the surface current, so this exact fit captures the dominant
    # signal deterministically; the MLP then only has to learn the smaller
    # nonlinear/ageostrophic RESIDUAL, and the final predictor can never be
    # worse than the linear model plus whatever the net adds.
    B, b_int = _fit_ridge(Xs, Y_raw, mask, alpha=10.0)
    lin_train = Xs @ B + b_int
    Ys = np.where(mask > 0, Y_raw - lin_train, 0.0).astype(np.float32)
    del lin_train

    # TEMPORAL validation split: rows are flattened in cycle (time) order, and
    # the harness's test split is later cycles. Hold out the last 10% of rows
    # as a contiguous tail so early stopping / LR scheduling select for
    # forward-in-time generalization instead of interleaved memorization.
    n = len(Xs)
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    X_t, Y_t, M_t = (torch.from_numpy(a[train_idx]) for a in (Xs, Ys, mask))
    X_v = torch.from_numpy(Xs[val_idx]).to(device)
    Y_v = torch.from_numpy(Ys[val_idx]).to(device)
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
            xb, yb, mb = X_t[idx].to(device), Y_t[idx].to(device), M_t[idx].to(device)
            optimizer.zero_grad()
            loss = _masked_mse(net(xb), yb, mb)
            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            val_losses = []
            for i in range(0, len(X_v), bs):
                val_losses.append(_masked_mse(net(X_v[i:i + bs]), Y_v[i:i + bs], M_v[i:i + bs]).item())
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

    # Predict: exact linear baseline + learned residual correction.
    Xp = _standardize(X_test, mean, scale)
    lin_test = (Xp @ B + b_int).astype(np.float64)
    out = np.empty((len(Xp), 2), dtype=np.float64)
    with torch.no_grad():
        for i in range(0, len(Xp), 65536):
            xb = torch.from_numpy(Xp[i:i + 65536]).to(device)
            out[i:i + 65536] = net(xb).cpu().numpy()
    out = (lin_test + out).astype(np.float32)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
