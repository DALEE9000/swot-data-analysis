from __future__ import annotations

import numpy as np
from typing import Callable

from swotxai.config import SWOTConfig
from swotxai.pipeline.io_utils import _save, _load, _cached, _save_model, _load_model

ProgressCb = Callable[[str, float, str], None]


def step_flatten(
    config: SWOTConfig,
    hfr_interp_data: dict,
    swot_features: dict,
    cb: ProgressCb,
    use_cache: bool,
) -> dict:
    cache_path = config.cache_path("flattened")

    effective_features = config.features
    for ds_list in swot_features.values():
        for ds in (ds_list if isinstance(ds_list, list) else [ds_list]):
            if ds is not None:
                effective_features = [f for f in config.features if f in ds]
                missing = [f for f in config.features if f not in ds]
                if missing:
                    cb("flatten", 0.0, f"Features not in data (skipped): {missing}")
                break
        else:
            continue
        break

    expected_n_cols = len(effective_features) * config.stencil_k ** 2
    if _cached(cache_path, use_cache):
        result = _load(cache_path)
        try:
            first_df = next(
                item["df"]
                for items in result.values()
                for item in (items if isinstance(items, list) else [items])
                if item is not None
            )
            if first_df.shape[1] == expected_n_cols:
                cb("flatten", 0.0, "Loading flattened data from cache...")
                cb("flatten", 1.0, "Loaded from cache.")
                return result
            cb("flatten", 0.0,
               f"Cache has {first_df.shape[1]} cols, expected {expected_n_cols} — rebuilding...")
            cache_path.unlink(missing_ok=True)
        except StopIteration:
            pass

    from swotxai.data_utils import rf_flattening_stencil

    keys = [t for t in swot_features if swot_features[t]]
    n = len(keys)
    cb("flatten", 0.0,
       f"Flattening {n} cycles (stencil k={config.stencil_k}, {len(effective_features)} features → {expected_n_cols} cols)...")

    flattened = {}
    for i, t in enumerate(keys):
        cb("flatten", (i + 1) / n, f"Cycle {t}  ({i + 1}/{n})")
        hfr_list  = hfr_interp_data.get(t, [])
        swot_list = swot_features[t]
        if not hfr_list or not swot_list:
            continue
        flat_list = []
        for hfr_ds, swot_ds in zip(hfr_list, swot_list):
            if hfr_ds is None or swot_ds is None:
                continue
            hfr_ds = hfr_ds.compute(scheduler="threads")
            flat_list.append(
                rf_flattening_stencil(swot_ds, hfr_ds["u"], hfr_ds["v"], effective_features, config.stencil_k)
            )
        flattened[t] = flat_list

    _save(flattened, cache_path)
    cb("flatten", 1.0, "Flattening complete.")
    return flattened


def step_train(
    config: SWOTConfig,
    flattened: dict,
    cb: ProgressCb,
    use_cache: bool,
) -> tuple:
    cache_path_u = config.cache_path("rf_u")
    cache_path_v = config.cache_path("rf_v")

    if _cached(cache_path_u, use_cache) and _cached(cache_path_v, use_cache):
        rf_u = _load_model(cache_path_u)
        rf_v = _load_model(cache_path_v)
        expected_n_cols     = len(config.features) * config.stencil_k ** 2
        cached_n_cols       = getattr(rf_u, "n_features_in_",  None)
        cached_n_estimators = getattr(rf_u, "n_estimators",    None)
        cached_max_depth    = getattr(rf_u, "max_depth",        None)
        stale = None
        if cached_n_cols != expected_n_cols:
            stale = f"feature count {cached_n_cols} → {expected_n_cols}"
        elif cached_n_estimators != config.n_estimators:
            stale = f"n_estimators {cached_n_estimators} → {config.n_estimators}"
        elif cached_max_depth != config.max_depth:
            stale = f"max_depth {cached_max_depth} → {config.max_depth}"
        if stale:
            cb("train", 0.0, f"Cached model stale ({stale}) — retraining...")
            cache_path_u.unlink(missing_ok=True)
            cache_path_v.unlink(missing_ok=True)
            config.cache_path("inference").unlink(missing_ok=True)
        else:
            cb("train", 0.0, "Loading trained models from cache...")
            cb("train", 1.0, "Loaded from cache.")
            return rf_u, rf_v

    from swotxai.data_utils import concat_flattened
    from swotxai.training import train as rf_train

    cb("train", 0.0, "Concatenating training data...")
    X_u, X_v, y_u, y_v = concat_flattened(flattened, training_percentage=0.8)

    if config.use_lgbm:
        backend = "LightGBM (CUDA)"
    elif config.use_gpu:
        backend = "cuML (GPU)"
    else:
        backend = f"sklearn (CPU, n_jobs={config.sklearn_n_jobs})"

    cb("train", 0.3, f"Training RF for u-velocity (n_estimators={config.n_estimators}, backend={backend})...")
    rf_u = rf_train(X_u, y_u, config)

    cb("train", 0.7, f"Training RF for v-velocity (backend={backend})...")
    rf_v = rf_train(X_v, y_v, config)

    _save_model(rf_u, cache_path_u)
    _save_model(rf_v, cache_path_v)

    def _fi(rf):
        saved = getattr(rf, "_feature_importances_saved", None)
        if saved is not None:
            return np.asarray(saved)
        raw = getattr(rf, "feature_importances_", None)
        return np.asarray(raw) if raw is not None else None

    _save({
        "features":              config.features,
        "stencil_k":             config.stencil_k,
        "feature_importances_u": _fi(rf_u),
        "feature_importances_v": _fi(rf_v),
    }, config.cache_path("rf_meta"))
    cb("train", 1.0, "Training complete.")
    return rf_u, rf_v


def step_evaluate(
    config: SWOTConfig,
    rf_u,
    rf_v,
    flattened: dict,
    cb: ProgressCb,
) -> dict:
    from sklearn.metrics import mean_squared_error, r2_score
    from sklearn.model_selection import train_test_split
    from swotxai.data_utils import concat_flattened
    from swotxai.training import predict as rf_predict

    cb("evaluate", 0.0, "Computing evaluation metrics...")
    X_u, X_v, y_u, y_v = concat_flattened(flattened, training_percentage=1.0)
    _, X_test_u, _, y_test_u = train_test_split(X_u, y_u, test_size=0.2, random_state=config.random_state)
    _, X_test_v, _, y_test_v = train_test_split(X_v, y_v, test_size=0.2, random_state=config.random_state)

    pred_u = rf_predict(rf_u, X_test_u)
    pred_v = rf_predict(rf_v, X_test_v)

    meta_path = config.cache_path("rf_meta")
    if meta_path.exists():
        meta = _load(meta_path)
        train_features = meta["features"]
        k = meta["stencil_k"]
        fi_raw_u = meta.get("feature_importances_u")
        fi_raw_v = meta.get("feature_importances_v")
    else:
        train_features = config.features
        k = config.stencil_k
        fi_raw_u = getattr(rf_u, "feature_importances_", None)
        fi_raw_v = getattr(rf_v, "feature_importances_", None)
        if fi_raw_u is not None:
            fi_raw_u = np.asarray(fi_raw_u)
        if fi_raw_v is not None:
            fi_raw_v = np.asarray(fi_raw_v)

    n_base_features = len(train_features)
    feature_names = train_features
    if fi_raw_u is not None and fi_raw_v is not None:
        n_features = len(fi_raw_u) // (k * k)
        fi_u = fi_raw_u.reshape(n_features, k * k).mean(axis=1)
        fi_v = fi_raw_v.reshape(n_features, k * k).mean(axis=1)
        if n_features != n_base_features:
            feature_names = [f"feature_{i}" for i in range(n_features)]
    else:
        fi_u = np.zeros(n_base_features)
        fi_v = np.zeros(n_base_features)

    metrics = {
        "rmse_u": float(np.sqrt(mean_squared_error(y_test_u, pred_u))),
        "rmse_v": float(np.sqrt(mean_squared_error(y_test_v, pred_v))),
        "r2_u":   float(r2_score(y_test_u, pred_u)),
        "r2_v":   float(r2_score(y_test_v, pred_v)),
        "feature_importance_u": dict(zip(feature_names, fi_u.tolist())),
        "feature_importance_v": dict(zip(feature_names, fi_v.tolist())),
    }
    cb("evaluate", 1.0, f"R²(u)={metrics['r2_u']:.3f}  R²(v)={metrics['r2_v']:.3f}")
    return metrics


def step_inference(
    config: SWOTConfig,
    rf_u,
    rf_v,
    swot_regridded: dict,
    hfr_interp_data: dict,
    flattened: dict,
    cb: ProgressCb,
    use_cache: bool,
) -> tuple[dict, dict]:
    from swotxai.data_utils import build_frame_dicts

    predictions = ["ssv_pred_u", "ssv_pred_v", "ssv_pred"]
    frames      = list(range(config.cycles_start, config.cycles_end + 1))
    cache_path  = config.cache_path("inference")

    if _cached(cache_path, use_cache):
        swot_dict, hfr_dict = _load(cache_path)

        def _input_has(var):
            return any(
                var in ds
                for ds_list in swot_regridded.values()
                for ds in (ds_list if isinstance(ds_list, list) else [ds_list])
                if ds is not None
            )

        def _cache_has(var):
            return any(
                var in ds
                for ds_list in swot_dict.values()
                for ds in (ds_list if isinstance(ds_list, list) else [ds_list])
                if ds is not None
            )

        stale_reason = None
        if _input_has("era5_u") and not _cache_has("era5_u"):
            stale_reason = "missing ERA5"
        elif _input_has("SST") and not _cache_has("SST"):
            stale_reason = "missing SST"
        elif not _cache_has("gos_filtered"):
            stale_reason = "missing gos_filtered"

        if stale_reason:
            cb("inference", 0.0, f"Cache {stale_reason} — rebuilding...")
            cache_path.unlink(missing_ok=True)
        else:
            n_valid = sum(1 for v in swot_dict.values() for ds in v if ds is not None)
            cb("inference", 0.0, "Loading prediction dicts from cache...")
            cb("inference", 1.0, f"Loaded from cache — {n_valid} valid entries.")
            return swot_dict, hfr_dict

    cb("inference", 0.0, "Building prediction dicts...")
    swot_dict, hfr_dict = build_frame_dicts(
        rf_u, rf_v, swot_regridded, hfr_interp_data, flattened,
        frames=frames, predictions=predictions,
    )
    _save((swot_dict, hfr_dict), cache_path)
    n_valid = sum(1 for v in swot_dict.values() for ds in v if ds is not None)
    cb("inference", 1.0, f"{n_valid} valid entries.")
    return swot_dict, hfr_dict
