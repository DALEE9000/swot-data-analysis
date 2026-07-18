from __future__ import annotations

from typing import Callable

import numpy as np

from swotxai.config import SWOTConfig
from swotxai.pipeline.io_utils import _cached

ProgressCb = Callable[[str, float, str], None]


def _views(reg):
    from swotxai.models.ann.model import ANNComponentView
    return ANNComponentView(reg, "u"), ANNComponentView(reg, "v")


def step_train(
    config: SWOTConfig,
    flattened: dict,
    cb: ProgressCb,
    use_cache: bool,
) -> tuple:
    from swotxai.models.ann.model import ANNRegressor
    from swotxai.models.ann.dataset import concat_for_ann
    from swotxai.models.ann.training import train_ann

    cache_path = config.cache_path("ann_model")

    if _cached(cache_path, use_cache):
        reg = ANNRegressor.load(cache_path)
        meta = reg.meta
        expected = {
            "n_inputs":   len(config.features) * config.stencil_k ** 2,
            "hidden":     [int(h) for h in config.ann_hidden_layers],
            "dropout":    config.ann_dropout,
            "activation": config.ann_activation,
            "features":   list(config.features),
            "stencil_k":  config.stencil_k,
        }
        stale = None
        for key, want in expected.items():
            if meta.get(key) != want:
                stale = f"{key} {meta.get(key)} → {want}"
                break
        if stale:
            cb("train", 0.0, f"Cached ANN stale ({stale}) — retraining...")
            cache_path.unlink(missing_ok=True)
            config.cache_path("inference").unlink(missing_ok=True)
        else:
            cb("train", 0.0, "Loading trained ANN from cache...")
            cb("train", 1.0, f"Loaded from cache (best epoch {meta.get('best_epoch')}, "
                             f"val loss {meta.get('best_val_loss', float('nan')):.5f}).")
            return _views(reg)

    cb("train", 0.0, "Concatenating training data (joint u/v with masked targets)...")
    X, Y = concat_for_ann(flattened, training_percentage=0.8)
    cb("train", 0.05, f"Training MLP on {len(X):,} rows × {X.shape[1]} inputs "
                      f"(hidden={list(config.ann_hidden_layers)})...")

    reg = train_ann(X, Y, config, cb=cb)
    reg.save(cache_path)
    cb("train", 1.0, f"Training complete — best epoch {reg.meta['best_epoch']}, "
                     f"val loss {reg.meta['best_val_loss']:.5f}.")
    return _views(reg)


def step_evaluate(
    config: SWOTConfig,
    model_u,
    model_v,
    flattened: dict,
    cb: ProgressCb,
) -> dict:
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from swotxai.data_utils import concat_flattened
    from swotxai.models.ann.training import permutation_importance

    # Same evaluation protocol as the RF backend so metrics are comparable.
    cb("evaluate", 0.0, "Computing evaluation metrics...")
    X_u, X_v, y_u, y_v = concat_flattened(flattened, training_percentage=1.0)
    _, X_test_u, _, y_test_u = train_test_split(X_u, y_u, test_size=0.2, random_state=config.random_state)
    _, X_test_v, _, y_test_v = train_test_split(X_v, y_v, test_size=0.2, random_state=config.random_state)

    pred_u = model_u.predict(X_test_u)
    pred_v = model_v.predict(X_test_v)

    meta = getattr(getattr(model_u, "base", None), "meta", {}) or {}
    feature_names = meta.get("features", config.features)
    k = meta.get("stencil_k", config.stencil_k)

    cb("evaluate", 0.5, "Computing permutation feature importances...")
    fi_u = permutation_importance(model_u, X_test_u, y_test_u, feature_names, k, config.random_state)
    fi_v = permutation_importance(model_v, X_test_v, y_test_v, feature_names, k, config.random_state)

    metrics = {
        "rmse_u": float(np.sqrt(mean_squared_error(y_test_u, pred_u))),
        "rmse_v": float(np.sqrt(mean_squared_error(y_test_v, pred_v))),
        "r2_u":   float(r2_score(y_test_u, pred_u)),
        "r2_v":   float(r2_score(y_test_v, pred_v)),
        "feature_importance_u": fi_u,
        "feature_importance_v": fi_v,
    }
    cb("evaluate", 1.0, f"R²(u)={metrics['r2_u']:.3f}  R²(v)={metrics['r2_v']:.3f}")
    return metrics
