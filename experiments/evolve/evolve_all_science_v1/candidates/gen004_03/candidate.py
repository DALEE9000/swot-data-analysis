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
    """Standardize (NaN -> feature mean/0) and append the per-cell missingness
    map. Layout stays flat: [F*C standardized features | C missing fractions];
    the network reshapes to an image per batch. Feature-major layout means
    column f * n_cells + c is feature f at stencil cell c, so view(n, F, C)
    is the correct channel/spatial factorization with zero copying.
    """
    Xs = (X.astype(np.float32) - mean) / scale
    n, d = Xs.shape
    miss = np.isnan(X).reshape(n, d // n_cells, n_cells).mean(axis=1).astype(np.float32)
    Xs = np.nan_to_num(Xs, nan=0.0, posinf=0.0, neginf=0.0, copy=False)
    return np.hstack([Xs, miss])


class StencilCNN(nn.Module):
    """Treat each row as an (F+1)-channel k x k image: F standardized base
    features plus one missingness channel, on the spatial stencil grid.
    Two padded 3x3 convs share weights across grid positions — finite-
    difference-like operators (gradients, Laplacians, shear) fall out of the
    first layer naturally instead of being relearned per column as in an MLP.
    The dense head then mixes the resulting local-dynamics maps into (u, v).
    """

    def __init__(self, n_feat, k, width=64, dropout=0.1):
        super().__init__()
        self.n_feat = n_feat
        self.k = k
        in_ch = n_feat + 1  # +1 missingness channel
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, width, 3, padding=1),
            nn.GroupNorm(8, width), nn.SiLU(),
            nn.Conv2d(width, 2 * width, 3, padding=1),
            nn.GroupNorm(8, 2 * width), nn.SiLU(),
        )
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(2 * width * k * k, 256),
            nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        # x: flat (b, F*k*k + k*k); split and fold into image form.
        F, k = self.n_feat, self.k
        feat = x[:, :F * k * k].view(-1, F, k, k)
        miss = x[:, F * k * k:].view(-1, 1, k, k)
        return self.head(self.conv(torch.cat([feat, miss], dim=1)))


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
    deadline = t0 + time_budget_s * 0.95  # leave 5% for test prediction

    # Per-column standardization ignoring NaNs; NaN -> 0 (the feature mean).
    mean = np.nan_to_num(np.nanmean(X_train, axis=0), nan=0.0).astype(np.float32)
    scale = np.nanstd(X_train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)

    n_cells = _cells(X_train.shape[1])
    k = int(round(math.sqrt(n_cells)))
    n_feat = X_train.shape[1] // n_cells
    Xs = _featurize(X_train, mean, scale, n_cells)
    mask = np.isfinite(Y_train).astype(np.float32)
    Ys = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)

    # Temporal validation split (inherited, proven): rows are time-ordered and
    # the test set is a temporal holdout, so validating on the last 10% of the
    # training window makes early stopping optimize forward-in-time
    # generalization rather than random-split interpolation.
    n = len(Xs)
    n_val = max(1, int(n * 0.1))
    X_t = torch.from_numpy(Xs[:n - n_val])
    Y_t = torch.from_numpy(Ys[:n - n_val])
    M_t = torch.from_numpy(mask[:n - n_val])
    X_v_cpu = torch.from_numpy(Xs[n - n_val:])
    Y_v_cpu = torch.from_numpy(Ys[n - n_val:])
    M_v_cpu = torch.from_numpy(mask[n - n_val:])
    X_v = X_v_cpu.to(device)
    Y_v = Y_v_cpu.to(device)
    M_v = M_v_cpu.to(device)
    del Xs

    net = StencilCNN(n_feat, k, width=64, dropout=0.1).to(device)
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
                val_losses.append(
                    _masked_mse(net(X_v[i:i + bs]), Y_v[i:i + bs], M_v[i:i + bs]).item())
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

    # Recency fine-tune (proven in gen002_00, +0.010 fitness): the validation
    # tail is the most recent — and hence most test-like — 10% of the training
    # window, never trained on above. After model selection, absorb it with a
    # short low-LR pass over the FULL window; low LR + few epochs keeps the
    # selected solution basin intact while adapting to the freshest state.
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
                loss = _masked_mse(net(xb), yb, mb)
                loss.backward()
                ft_opt.step()
                if bi % 100 == 0 and time.time() > deadline:
                    out_of_time = True
                    break
            if out_of_time:
                break

    net.eval()
    out = np.zeros((len(X_test), 2), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_test), 65536):
            xb = torch.from_numpy(
                _featurize(X_test[i:i + 65536], mean, scale, n_cells)).to(device)
            out[i:i + 65536] = net(xb).cpu().numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
