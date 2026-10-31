"""Model-agnostic ML steps (flatten, inference).

The model-specific train / evaluate steps live in the backend packages
(swotxai.models.rf.steps, swotxai.models.ann.steps) and are dispatched by
the orchestrator via swotxai.models.get_backend()."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Callable

import numpy as np

from swotxai.config import SWOTConfig
from swotxai.pipeline.io_utils import _save, _load, _cached

ProgressCb = Callable[[str, float, str], None]

# One lock per flattened-cache file: parallel batch jobs whose configs share a
# flatten stem serialize on it — the first computes and saves, the rest load
# the cache instead of racing it (Windows raises WinError 32 on such races).
_FLATTEN_LOCKS: dict[str, threading.Lock] = {}
_FLATTEN_LOCKS_GUARD = threading.Lock()


def _flatten_lock(path: Path) -> threading.Lock:
    key = str(Path(path).absolute())
    with _FLATTEN_LOCKS_GUARD:
        return _FLATTEN_LOCKS.setdefault(key, threading.Lock())


def pass_time(ds):
    """Scalar pass time of a per-pass dataset as datetime64[ns], or None."""
    try:
        return np.datetime64(np.atleast_1d(ds.coords["time"].values)[0], "ns")
    except (KeyError, AttributeError, TypeError, ValueError):
        return None


_PAIR_MAX_HOURS = 1.0   # HFR is hourly; a pass crosses a box in ~1 min, so a true match is <= 0.5 h away


def _pair_by_time(hfr_list: list, swot_list: list, cb: ProgressCb) -> list:
    """Pair each SWOT pass with the HFR entry nearest in time (within _PAIR_MAX_HOURS).

    The HFR colocation build drops passes it cannot match, so the two lists
    are not position-aligned; pairing by list position (the pre-2026-10-29
    behaviour) shifted every later pass of a thinned cycle onto the wrong
    radar field (offsets up to +447 h in prvi). Entries without a usable
    time coordinate fall back to positional pairing for backward
    compatibility; unmatched SWOT passes are skipped and counted.
    """
    _t = pass_time
    hfr_times = [(_t(h) if h is not None else None) for h in hfr_list]
    if all(t is None for t in hfr_times):
        return list(zip(hfr_list, swot_list))
    pairs, skipped = [], 0
    for i, swot_ds in enumerate(swot_list):
        st = _t(swot_ds) if swot_ds is not None else None
        if st is None:
            pairs.append((hfr_list[i] if i < len(hfr_list) else None, swot_ds))
            continue
        best, best_dt = None, None
        for h, ht in zip(hfr_list, hfr_times):
            if ht is None:
                continue
            dt = abs((ht - st) / np.timedelta64(1, "h"))
            if best_dt is None or dt < best_dt:
                best, best_dt = h, dt
        if best is not None and best_dt <= _PAIR_MAX_HOURS:
            pairs.append((best, swot_ds))
        else:
            skipped += 1
    if skipped:
        cb("flatten", 0.0, f"{skipped} SWOT pass(es) had no HFR field within {_PAIR_MAX_HOURS:g} h - skipped")
    return pairs


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

    with _flatten_lock(cache_path):
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
            for hfr_ds, swot_ds in _pair_by_time(hfr_list, swot_list, cb):
                if hfr_ds is None or swot_ds is None:
                    continue
                hfr_ds = hfr_ds.compute(scheduler="threads")
                entry = rf_flattening_stencil(swot_ds, hfr_ds["u"], hfr_ds["v"], effective_features, config.stencil_k)
                entry["time"] = pass_time(swot_ds)   # additive: lets the evolve harness apply a temporal buffer
                flat_list.append(entry)
            flattened[t] = flat_list

        _save(flattened, cache_path)
    cb("flatten", 1.0, "Flattening complete.")
    return flattened


def step_inference(
    config: SWOTConfig,
    model_u,
    model_v,
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
        model_u, model_v, swot_regridded, hfr_interp_data, flattened,
        frames=frames, predictions=predictions,
    )
    _save((swot_dict, hfr_dict), cache_path)
    n_valid = sum(1 for v in swot_dict.values() for ds in v if ds is not None)
    cb("inference", 1.0, f"{n_valid} valid entries.")
    return swot_dict, hfr_dict
