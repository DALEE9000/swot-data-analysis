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


class MLP(nn.Module):
    """Joint (u, v) MLP: shared trunk, 2-unit head."""

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


class EMA:
    """Exponential moving average of model weights (Polyak averaging).

    Averaged weights sit in flatter, wider minima than any single SGD
    checkpoint, which transfers better under the distribution drift between
    the training window and the temporally held-out test cycles.
    """

    def __init__(self, net, decay=0.999):
        self.decay = decay
        self.shadow = {k: v.detach().clone() for k, v in net.state_dict().items()}

    @torch.no_grad()
    def update(self, net):
        for k, v in net.state_dict().items():
            if v.dtype.is_floating_point:
                self.shadow[k].mul_(self.decay).add_(v.detach(), alpha=1.0 - self.decay)
            else:
                self.shadow[k].copy_(v)

    def state_copy(self):
        return {k: v.clone() for k, v in self.shadow.items()}


def _eval_masked_mse(net, X_v, Y_v, M_v, bs):
    net.eval()
    with torch.no_grad():
        losses = []
        for i in range(0, len(X_v), bs):
            losses.append(_masked_mse(net(X_v[i:i + bs]), Y_v[i:i + bs], M_v[i:i + bs]).item())
    return float(np.mean(losses))


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
    Ys = np.nan_to_num(Y_train.astype(np.float32), nan=0.0)

    # Temporal validation split: rows are time-ordered (regions interleaved)
    # and the test set is a temporal holdout, so validating on the last 10%
    # of the training window makes early stopping and the LR schedule
    # optimize forward-in-time generalization rather than random-split
    # interpolation.
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

    net = MLP(X_t.shape[1], hidden=(256, 256, 128), dropout=0.1).to(device)
    optimizer = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)
    ema = EMA(net, decay=0.999)

    bs = 4096
    patience = 15
    best_val = float("inf")
    best_state = ema.state_copy()
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
            ema.update(net)

        # Validate, checkpoint, and schedule on the EMA weights: they are the
        # weights we will actually deploy, so model selection must see them.
        raw_state = copy.deepcopy(net.state_dict())
        net.load_state_dict(ema.shadow)
        val_loss = _eval_masked_mse(net, X_v, Y_v, M_v, bs)
        net.load_state_dict(raw_state)
        scheduler.step(val_loss)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = ema.state_copy()
            best_epoch = epoch

        if epoch - best_epoch >= patience:
            break
        if time.time() - t0 > time_budget_s:
            break

    # Restore the best EMA weights into both the live net and the EMA shadow,
    # so the fine-tune below starts from — and keeps averaging around — the
    # selected solution.
    net.load_state_dict(best_state)
    ema.shadow = {k: v.clone() for k, v in best_state.items()}

    # Recency fine-tune: the validation tail is the most recent — and hence
    # most test-like — 10% of the training window, never trained on above.
    # Absorb it with a short low-LR pass over the FULL window. There is no
    # validation signal during this phase by construction; continuing the EMA
    # keeps the pass conservative by averaging away any drift from the
    # selected basin.
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
                loss = _masked_mse(net(xb), yb, mb)
                loss.backward()
                ft_opt.step()
                ema.update(net)
                if bi % 100 == 0 and time.time() - t0 > time_budget_s:
                    out_of_time = True
                    break
            if out_of_time:
                break

    # Deploy the averaged weights.
    net.load_state_dict(ema.shadow)
    net.eval()

    # Predict on the test set in batches.
    out = np.empty((len(X_test), 2), dtype=np.float32)
    with torch.no_grad():
        for i in range(0, len(X_test), 65536):
            xb = torch.from_numpy(_standardize(X_test[i:i + 65536], mean, scale)).to(device)
            out[i:i + 65536] = net(xb).cpu().numpy()
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0)
