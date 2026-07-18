from __future__ import annotations

import copy
import json
import time
from typing import Callable

import numpy as np
import torch

from swotxai.models.ann.dataset import fit_scaler
from swotxai.models.ann.model import MLP, ANNRegressor, resolve_device

ProgressCb = Callable[[str, float, str], None]


def _masked_mse(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    diff = (pred - target) * mask
    return diff.pow(2).sum() / mask.sum().clamp(min=1)


def train_ann(X: np.ndarray, Y: np.ndarray, config, cb: ProgressCb | None = None) -> ANNRegressor:
    """Train the joint (u, v) MLP with early stopping.

    Tensors stay on the CPU and batches are moved to the device per step,
    so dataset size is bounded by RAM rather than the (possibly 4 GB) GPU.
    """
    torch.manual_seed(config.random_state)
    rng = np.random.default_rng(config.random_state)
    device = resolve_device(config.ann_device)

    mean, scale = fit_scaler(X)
    Xs = np.nan_to_num((X.astype(np.float32) - mean) / scale, nan=0.0, posinf=0.0, neginf=0.0)
    mask = np.isfinite(Y).astype(np.float32)
    Ys = np.nan_to_num(Y.astype(np.float32), nan=0.0)

    # Train / validation split (validation drives LR schedule + early stopping)
    n = len(Xs)
    perm = rng.permutation(n)
    n_val = max(1, int(n * config.ann_val_fraction))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    X_t = torch.from_numpy(Xs[train_idx])
    Y_t = torch.from_numpy(Ys[train_idx])
    M_t = torch.from_numpy(mask[train_idx])
    X_v = torch.from_numpy(Xs[val_idx]).to(device)
    Y_v = torch.from_numpy(Ys[val_idx]).to(device)
    M_v = torch.from_numpy(mask[val_idx]).to(device)

    net = MLP(
        n_inputs=Xs.shape[1],
        hidden=tuple(int(h) for h in config.ann_hidden_layers),
        dropout=config.ann_dropout,
        activation=config.ann_activation,
    ).to(device)

    optimizer = torch.optim.AdamW(net.parameters(), lr=config.ann_lr, weight_decay=config.ann_weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, patience=5)

    best_val = float("inf")
    best_state = copy.deepcopy(net.state_dict())
    best_epoch = 0
    history = []
    bs = config.ann_batch_size
    n_train = len(X_t)
    t0 = time.time()

    for epoch in range(1, config.ann_max_epochs + 1):
        ep_start = time.time()
        net.train()
        order = torch.from_numpy(rng.permutation(n_train))
        train_loss, n_batches = 0.0, 0
        for i in range(0, n_train, bs):
            idx = order[i:i + bs]
            xb, yb, mb = X_t[idx].to(device), Y_t[idx].to(device), M_t[idx].to(device)
            optimizer.zero_grad()
            loss = _masked_mse(net(xb), yb, mb)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            n_batches += 1
        train_loss /= max(n_batches, 1)

        net.eval()
        with torch.no_grad():
            val_losses = []
            for i in range(0, len(X_v), bs):
                val_losses.append(
                    _masked_mse(net(X_v[i:i + bs]), Y_v[i:i + bs], M_v[i:i + bs]).item()
                )
            val_loss = float(np.mean(val_losses))
        scheduler.step(val_loss)
        entry = {
            "epoch":      epoch,
            "train_loss": round(train_loss, 6),
            "val_loss":   round(val_loss, 6),
            "lr":         optimizer.param_groups[0]["lr"],
            "epoch_s":    round(time.time() - ep_start, 2),
            "elapsed_s":  round(time.time() - t0, 1),
        }
        history.append(entry)

        if val_loss < best_val - 1e-7:
            best_val = val_loss
            best_state = copy.deepcopy(net.state_dict())
            best_epoch = epoch

        if cb:
            frac = min(epoch / config.ann_max_epochs, 0.99)
            cb("train", frac,
               f"Epoch {epoch}/{config.ann_max_epochs} — train {train_loss:.5f}, val {val_loss:.5f} "
               f"(best {best_val:.5f} @ {best_epoch}) [{device.type}]")
            # Structured per-epoch event for live monitors (e.g. the GUI's
            # training panel); consumers that only understand text ignore it.
            cb("train_epoch", frac, json.dumps({
                **entry,
                "max_epochs":    config.ann_max_epochs,
                "best_epoch":    best_epoch,
                "best_val_loss": round(best_val, 6),
                "device":        device.type,
            }))

        if epoch - best_epoch >= config.ann_patience:
            if cb:
                cb("train", min(epoch / config.ann_max_epochs, 0.99),
                   f"Early stopping at epoch {epoch} (no val improvement for {config.ann_patience} epochs).")
            break

    net.load_state_dict(best_state)
    net.to("cpu")

    meta = {
        "n_inputs":    Xs.shape[1],
        "hidden":      [int(h) for h in config.ann_hidden_layers],
        "dropout":     config.ann_dropout,
        "activation":   config.ann_activation,
        # Fixed architecture/training choices, recorded so experiment records
        # stay unambiguous if these defaults ever change:
        "norm":         "LayerNorm",
        "optimizer":    "AdamW",
        "loss":         "masked_mse",
        "lr_scheduler": "ReduceLROnPlateau(factor=0.5, patience=5)",
        "lr":          config.ann_lr,
        "weight_decay": config.ann_weight_decay,
        "batch_size":  config.ann_batch_size,
        "features":    list(config.features),
        "stencil_k":   config.stencil_k,
        "random_state": config.random_state,
        "best_epoch":  best_epoch,
        "best_val_loss": best_val,
        "history":     history,
        "train_seconds": time.time() - t0,
        "device":      device.type,
    }
    return ANNRegressor(net, mean, scale, meta)


def permutation_importance(
    model_view,
    X: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    stencil_k: int,
    random_state: int = 42,
    max_samples: int = 50_000,
) -> dict[str, float]:
    """Grouped permutation importance for one velocity component.

    All k² stencil columns belonging to a base feature are permuted
    together (columns are feature-major per rf_flattening_stencil), and
    importance is the resulting increase in MSE, normalised to sum to 1 —
    directly comparable to the RF impurity importances shown in the GUI.
    """
    rng = np.random.default_rng(random_state)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.float32)
    if len(X) > max_samples:
        sub = rng.choice(len(X), size=max_samples, replace=False)
        X, y = X[sub], y[sub]

    k2 = stencil_k ** 2
    base_mse = float(np.mean((model_view.predict(X) - y) ** 2))

    deltas = []
    for i, name in enumerate(feature_names):
        Xp = X.copy()
        block = slice(i * k2, (i + 1) * k2)
        Xp[:, block] = Xp[rng.permutation(len(Xp)), block]
        mse = float(np.mean((model_view.predict(Xp) - y) ** 2))
        deltas.append(max(mse - base_mse, 0.0))

    total = sum(deltas)
    if total <= 0:
        return {name: 0.0 for name in feature_names}
    return {name: d / total for name, d in zip(feature_names, deltas)}
