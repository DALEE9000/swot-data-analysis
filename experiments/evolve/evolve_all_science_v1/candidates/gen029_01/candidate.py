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
    """Standardize, then handle stencil missingness explicitly (THE ONE
    CHANGE vs the parent, grafted from the proven top lineage):

      * gradient-preserving mirror imputation: a missing stencil cell is
        filled with 2*center - mirror (point reflection through the center),
        clamped to +/-4 sigma, falling back to the center value — so a
        swath-edge neighborhood keeps its first-order spatial gradient
        instead of being flattened toward the feature mean (NaN->0), which
        destroyed the local structure the geostrophic relation needs;
      * per-cell missing fraction (n, C) appended — the gate and experts can
        see WHICH cells are fabricated and route/discount accordingly;
      * per-feature stencil std over VALID cells only (n, F) appended —
        a real-data spatial-variability signal computed before any filling.

    Everything else in the module (MoE, temporal split, recency fine-tune)
    is unchanged from the parent.
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
    # Fill AFTER the std block (which must see true valid cells). Snapshot
    # the point-reflected cells before any in-place filling so late cells
    # never read an already-imputed mirror as if it were data.
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


class MoE(nn.Module):
    """Shared trunk + softmax-gated mixture of expert heads over flow regimes.

    All experts are evaluated densely (cheap at this width); the gate learns to
    route rows — e.g. energetic fronts/eddies vs quiescent background, and now
    also edge-degraded vs full-stencil rows via the explicit missingness
    features — to specialized heads instead of forcing one MLP to average
    across regimes.
    """

    def __init__(self, n_inputs, n_experts=4, trunk_hidden=(256, 256),
                 expert_hidden=128, dropout=0.1):
        super().__init__()
        layers = []
        d = n_inputs
        for h in trunk_hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), nn.SiLU(), nn.Dropout(dropout)]
            d = h
        self.trunk = nn.Sequential(*layers)
        self.gate = nn.Linear(d, n_experts)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(d, expert_hidden), nn.LayerNorm(expert_hidden),
                nn.SiLU(), nn.Dropout(dropout), nn.Linear(expert_hidden, 2),
            )
            for _ in range(n_experts)
        ])
        self.n_experts = n_experts

    def forward(self, x):
        h = self.trunk(x)
        g = torch.softmax(self.gate(h), dim=1)                  # (B, E)
        outs = torch.stack([e(h) for e in self.experts], dim=1)  # (B, E, 2)
        out = (g.unsqueeze(-1) * outs).sum(dim=1)                # (B, 2)
        return out, g


def _masked_mse(pred, target, mask):
    diff = (pred - target) * mask
    return diff.pow(2).sum() / mask.sum().clamp(min=1)


def _balance_penalty(gate_probs):
    """KL(mean gate usage || uniform): keeps experts from collapsing to one."""
    p = gate_probs.mean(dim=0).clamp(min=1e-8)
    return (p * (p * gate_probs.shape[1]).log()).sum()


def train_and_predict(X_train, Y_train, X_test, params):
    seed = int(params["seed"])
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    device = torch.device(params["device"])
    max_epochs = int(params["max_epochs"])
    time_budget_s = float(params["time_budget_s"])
    t0 = time.time()

    # Per-column standardization ignoring NaNs; missing cells then get the
    # gradient-preserving mirror fill inside _featurize (not the mean).
    mean = np.nan_to_num(np.nanmean(X_train, axis=0), nan=0.0).astype(np.float32)
    scale = np.nanstd(X_train, axis=0)
    scale = np.where(np.isfinite(scale) & (scale > 1e-6), scale, 1.0).astype(np.float32)

    n_cells = _cells(X_train.shape[1])
    Xs = _featurize(X_train, mean, scale, n_cells)
    mask = np.isfinite(Y_train).astype(np.float32)
    Ys = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)

    # Temporal validation split (proven in gen001_01): validate on the last 10%
    # of the time-ordered training window so early stopping and the LR schedule
    # optimize forward-in-time generalization, matching the temporal test.
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

    net = MoE(X_t.shape[1], n_experts=4, trunk_hidden=(256, 256),
              expert_hidden=128, dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    bs = 4096
    patience = 15
    balance_coef = 1e-2
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
            pred, g = net(xb)
            loss = _masked_mse(pred, yb, mb) + balance_coef * _balance_penalty(g)
            loss.backward()
            optimizer.step()

        net.eval()
        with torch.no_grad():
            val_losses = []
            for i in range(0, len(X_v), bs):
                pred, _ = net(X_v[i:i + bs])
                val_losses.append(_masked_mse(pred, Y_v[i:i + bs], M_v[i:i + bs]).item())
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

    # Recency fine-tune (proven in gen002_00): after model selection, absorb
    # the most recent — most test-like — 10% with a short low-LR pass over the
    # full window. Low LR + few epochs keeps the selected basin intact.
    if time.time() - t0 < time_budget_s:
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
                pred, g = net(xb)
                loss = _masked_mse(pred, yb, mb) + balance_coef * _balance_penalty(g)
                loss.backward()
                ft_opt.step()
                if bi % 100 == 0 and time.time() - t0 > time_budget_s:
                    out_of_time = True
                    break
            if out_of_time:
                break

    net.eval()

    # Predict on the test set in batches (same featurization as training).
    out = np.empty((len(X_test), 2), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_test), 65536):
            Xf = _featurize(X_test[i:i + 65536], mean, scale, n_cells)
            xb = torch.from_numpy(Xf).to(device)
            out[i:i + 65536] = net(xb)[0].cpu().numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
