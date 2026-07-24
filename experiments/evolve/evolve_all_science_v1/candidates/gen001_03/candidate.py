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


class StencilCNN(nn.Module):
    """Treat each row as an (n_feat, k, k) image plus per-feature validity
    masks; small conv trunk + MLP head, joint (u, v) output."""

    def __init__(self, n_feat, k, width=64, head=(256, 128), dropout=0.1):
        super().__init__()
        self.n_feat = n_feat
        self.k = k
        in_ch = 2 * n_feat  # features + validity masks
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(),
            nn.Conv2d(width, width, 3, padding=1),
            nn.GroupNorm(8, width),
            nn.SiLU(),
        )
        layers = []
        d = width * k * k
        for h in head:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 2))
        self.head = nn.Sequential(*layers)

    def forward(self, x_flat, m_flat):
        # Columns are feature-major: [f0s0..f0s{k*k-1}, f1s0, ...]
        b = x_flat.shape[0]
        img = x_flat.view(b, self.n_feat, self.k, self.k)
        msk = m_flat.view(b, self.n_feat, self.k, self.k)
        z = self.conv(torch.cat([img, msk], dim=1))
        return self.head(z.flatten(1))


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

    d = X_train.shape[1]
    # Infer (n_feat, k) from feature-major stencil layout.
    k = 3 if d % 9 == 0 else 1
    n_feat = d // (k * k)

    # Per-column standardization ignoring NaNs; NaN -> 0 (the feature mean).
    mean = np.nan_to_num(np.nanmean(X_train, axis=0), nan=0.0).astype(np.float32)
    scale = np.nanstd(X_train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)

    # Validity masks kept as uint8 on CPU (float conversion per batch).
    Fm = np.isfinite(X_train).astype(np.uint8)
    Xs = _standardize(X_train, mean, scale)
    mask = np.isfinite(Y_train).astype(np.float32)
    Ys = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)

    # Temporal validation split: rows are time-ordered (later rows = later in
    # the mission window, regions interleaved), and the test set is a temporal
    # holdout. Validating on the last 10% of the training window makes early
    # stopping and the LR schedule optimize forward-in-time generalization.
    n = len(Xs)
    n_val = max(1, int(n * 0.1))
    train_idx = np.arange(0, n - n_val)
    val_idx = np.arange(n - n_val, n)

    X_t = torch.from_numpy(Xs[train_idx])
    F_t = torch.from_numpy(Fm[train_idx])
    Y_t = torch.from_numpy(Ys[train_idx])
    M_t = torch.from_numpy(mask[train_idx])
    X_v = torch.from_numpy(Xs[val_idx]).to(device)
    F_v = torch.from_numpy(Fm[val_idx]).to(device).float()
    Y_v = torch.from_numpy(Ys[val_idx]).to(device)
    M_v = torch.from_numpy(mask[val_idx]).to(device)
    del Xs, Fm

    net = StencilCNN(n_feat, k).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    bs = 4096
    patience = 15
    best_val = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    best_epoch = 0
    n_train = len(X_t)
    out_of_time = False

    for epoch in range(1, max_epochs + 1):
        net.train()
        order = torch.from_numpy(rng.permutation(n_train))
        for bi, i in enumerate(range(0, n_train, bs)):
            idx = order[i:i + bs]
            xb = X_t[idx].to(device)
            fb = F_t[idx].to(device).float()
            yb, mb = Y_t[idx].to(device), M_t[idx].to(device)
            optimizer.zero_grad()
            loss = _masked_mse(net(xb, fb), yb, mb)
            loss.backward()
            optimizer.step()
            if bi % 100 == 0 and time.time() - t0 > time_budget_s:
                out_of_time = True
                break

        net.eval()
        with torch.no_grad():
            val_losses = []
            for i in range(0, len(X_v), bs):
                val_losses.append(
                    _masked_mse(net(X_v[i:i + bs], F_v[i:i + bs]),
                                Y_v[i:i + bs], M_v[i:i + bs]).item())
            val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = copy.deepcopy(net.state_dict())
            best_epoch = epoch

        if out_of_time or epoch - best_epoch >= patience:
            break
        if time.time() - t0 > time_budget_s:
            break

    net.load_state_dict(best_state)
    net.eval()

    # Predict on the test set in batches.
    out = np.empty((len(X_test), 2), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_test), 65536):
            raw = X_test[i:i + 65536]
            xb = torch.from_numpy(_standardize(raw, mean, scale)).to(device)
            fb = torch.from_numpy(np.isfinite(raw).astype(np.float32)).to(device)
            out[i:i + 65536] = net(xb, fb).cpu().numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
