from __future__ import annotations

import pickle
import time
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr

from swotxai.config import SWOTConfig

# Type alias for the progress callback
# Called as: progress_cb(step_name, fraction_0_to_1, message)
ProgressCb = Callable[[str, float, str], None]

STEPS = [
    "load_swot",
    "regrid",
    "load_era5",
    "load_goes",
    "interp_sources",
    "load_hfr",
    "interp_hfr",
    "flatten",
    "train",
    "evaluate",
    "inference",
    "animate",
]

SHARED_STEPS  = ["load_swot", "regrid", "load_era5", "load_goes", "interp_sources", "load_hfr", "interp_hfr"]
PER_JOB_STEPS = ["flatten", "train", "evaluate", "inference", "animate"]


def _noop_cb(step: str, frac: float, msg: str) -> None:
    print(f"[{step}] {int(frac * 100):3d}%  {msg}")


def _save(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _load(path: Path):
    with open(path, "rb") as f:
        return pickle.load(f)


def _cached(path: Path, use_cache: bool):
    return use_cache and path.exists()


def _save_model(model, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.suffix == ".joblib":
        import joblib
        joblib.dump(model, path)
    else:
        _save(model, path)


def _load_model(path: Path):
    if path.suffix == ".joblib":
        import joblib
        return joblib.load(path)
    return _load(path)


def _load_s3_pkl(s3_path: str):
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    with fs.open(s3_path) as f:
        return pickle.load(f)


def _s3_exists(s3_path: str) -> bool:
    try:
        import s3fs
        fs = s3fs.S3FileSystem(anon=True)
        return fs.exists(s3_path)
    except Exception:
        return False


def _save_s3_pkl(obj, s3_path: str) -> None:
    """Save a pkl to S3. Requires AWS credentials (anon=False)."""
    import s3fs
    fs = s3fs.S3FileSystem(anon=False)
    with fs.open(s3_path, "wb") as f:
        pickle.dump(obj, f)


def step_load_preset_swot(config: SWOTConfig, cb: ProgressCb) -> tuple[dict, dict]:
    """Stream pre-processed SWOT regridded pkl from S3 into memory (no local save)."""
    cb("load_swot", 0.0, "Streaming preset SWOT pkl from S3...")
    swot_regridded = _load_s3_pkl(config.swot_pkl_path)
    cb("load_swot", 1.0, f"Loaded {len(swot_regridded)} cycles.")
    cb("regrid", 1.0, "Skipped — using preset pkl.")
    return {}, swot_regridded


def step_load_preset_hfr(config: SWOTConfig, cb: ProgressCb) -> dict:
    """Stream pre-processed HFR interpolated pkl from S3 into memory (no local save)."""
    cb("load_hfr", 0.0, "Streaming preset HFR pkl from S3...")
    hfr_interp = _load_s3_pkl(config.hfr_pkl_path)
    cb("load_hfr", 1.0, "Loaded preset HFR.")
    cb("interp_hfr", 1.0, "Skipped — using preset pkl.")
    return hfr_interp


# ---------------------------------------------------------------------------
# Individual step runners
# ---------------------------------------------------------------------------

def step_load_swot(config: SWOTConfig, cb: ProgressCb, use_cache: bool) -> dict:
    cache_path = config.cache_path("cycle_data")
    if _cached(cache_path, use_cache):
        cb("load_swot", 0.0, "Loading from cache...")
        result = _load(cache_path)
        cb("load_swot", 1.0, f"Loaded {len(result)} cycles from cache.")
        return result

    from swot import data_loaders
    from swot.download_swaths import find_swaths

    science = (config.mission == "science")
    sph_path = Path(config.sph_science_path if science else config.sph_calval_path)
    cycles_start, cycles_end = config.cycles_start, config.cycles_end

    if not sph_path.exists():
        raise FileNotFoundError(f"Orbit file not found: {sph_path}. Set sph_calval_path / sph_science_path in your config.")

    cycles   = [str(c).zfill(3) for c in range(cycles_start, cycles_end + 1)]
    lat_lims = [config.sw_corner[1], config.ne_corner[1]]

    cb("load_swot", 0.0, "Finding SWOT pass IDs for domain...")
    pass_ids = find_swaths(
        sw_corner=config.sw_corner,
        ne_corner=config.ne_corner,
        path_to_sph_file=sph_path,
    )
    cb("load_swot", 0.01, f"Found {len(pass_ids)} passes. Loading {len(cycles)} cycles...")

    cycle_data = {}
    n = len(cycles)
    for i, cycle in enumerate(cycles):
        cb("load_swot", (i + 1) / n, f"Cycle {cycle}  ({i + 1}/{n})")
        cycle_data[cycle] = data_loaders.load_cycle(
            path=config.swot_path, fields=None,
            cycle=cycle, pass_ids=pass_ids,
            subset=True, lats=lat_lims,
        )

    _save(cycle_data, cache_path)
    cb("load_swot", 1.0, f"Loaded {n} cycles.")
    return cycle_data


def step_regrid(config: SWOTConfig, cycle_data: dict, cb: ProgressCb, use_cache: bool) -> dict:
    cache_path = config.cache_path("swot_regridded")
    if _cached(cache_path, use_cache):
        cb("regrid", 0.0, "Loading regridded SWOT from cache...")
        result = _load(cache_path)
        cb("regrid", 1.0, "Loaded from cache.")
        return result

    from swotxai.swotxai_utils import swot_regrid

    cycles = list(cycle_data.items())
    n = len(cycles)
    cb("regrid", 0.0, f"Regridding {n} cycles onto lat/lon grid...")

    regridded = {}
    for i, (t, ds_list) in enumerate(cycles):
        cb("regrid", (i + 1) / n, f"Cycle {t}  ({i + 1}/{n})")
        regridded[t] = [swot_regrid(ds) for ds in ds_list if ds is not None]

    _save(regridded, cache_path)
    cb("regrid", 1.0, f"Regridded {n} cycles.")
    return regridded


def step_load_era5(config: SWOTConfig, cb: ProgressCb, use_cache: bool) -> xr.Dataset | None:
    if not config.era5_path and not config.era5_pkl_path:
        cb("load_era5", 1.0, "ERA5 path not set — skipping.")
        return None

    # 1. pkl fast path — try loading directly (avoids silent _s3_exists auth failures)
    pkl_path = config.era5_pkl_path
    if use_cache and pkl_path:
        try:
            cb("load_era5", 0.0, f"Loading ERA5 pkl from {pkl_path}...")
            era5 = _load_s3_pkl(pkl_path) if pkl_path.startswith("s3://") else _load(Path(pkl_path))
            cb("load_era5", 1.0, "ERA5 loaded from pkl.")
            return era5
        except FileNotFoundError:
            cb("load_era5", 0.0, f"ERA5 pkl not found at {pkl_path} — falling back to source.")

    # 2. Local run-scoped cache
    cache_path = config.cache_path("era5")
    if _cached(cache_path, use_cache):
        result = _load(cache_path)
        if "era5_u" in result:
            cb("load_era5", 0.0, "Loading ERA5 from cache...")
            cb("load_era5", 1.0, "Loaded from cache.")
            return result
        cb("load_era5", 0.0, "ERA5 cache missing era5_u — reloading from source...")
        cache_path.unlink(missing_ok=True)

    # 3. Load from source (slow — NetCDF from S3 or disk)
    if not config.era5_path:
        cb("load_era5", 1.0, "ERA5 pkl not found and era5_path not set — skipping.")
        return None

    cb("load_era5", 0.0, f"Loading ERA5 from source: {config.era5_path}...")
    if config.era5_path.startswith("s3://"):
        import s3fs
        fs = s3fs.S3FileSystem(anon=True)
        with fs.open(config.era5_path) as f:
            era5 = xr.open_dataset(f, engine="h5netcdf").load()
    else:
        era5 = xr.open_dataset(config.era5_path, engine="netcdf4").load()

    cb("load_era5", 0.5, f"ERA5 raw vars: {list(era5.data_vars)}  dims: {dict(era5.dims)}")

    if "isobaricInhPa" in era5.dims:
        era5 = era5.sel(isobaricInhPa=1000)
    u_name = next((v for v in era5.data_vars if v in ("u", "u10", "ugrd10m")), None)
    v_name = next((v for v in era5.data_vars if v in ("v", "v10", "vgrd10m")), None)
    if u_name and v_name:
        era5 = era5.rename({u_name: "era5_u", v_name: "era5_v"})
    elif "era5_u" not in era5:
        raise ValueError(
            f"Cannot find wind variables in ERA5 file. Available: {list(era5.data_vars)}"
        )

    lon_coord = next((c for c in era5.coords if "lon" in c.lower()), None)
    if lon_coord and float(era5[lon_coord].max()) > 180:
        era5[lon_coord] = (era5[lon_coord] + 180) % 360 - 180
        era5 = era5.sortby(lon_coord)

    # Save to local run-scoped cache
    _save(era5, cache_path)

    # Save to era5_pkl_path (S3 or local) for cross-machine reuse
    if pkl_path:
        cb("load_era5", 0.9, f"Saving ERA5 pkl to {pkl_path}...")
        try:
            if pkl_path.startswith("s3://"):
                _save_s3_pkl(era5, pkl_path)
            else:
                _save(era5, Path(pkl_path))
            cb("load_era5", 1.0, f"ERA5 loaded and cached to {pkl_path}.")
        except Exception as e:
            cb("load_era5", 1.0, f"ERA5 loaded (pkl upload failed: {e}).")
    else:
        cb("load_era5", 1.0, "ERA5 loaded.")
    return era5


def step_load_goes(config: SWOTConfig, cb: ProgressCb, use_cache: bool) -> xr.Dataset | None:
    if not config.goes_nc_path:
        cb("load_goes", 1.0, "GOES dir not set — skipping.")
        return None

    cache_path = config.cache_path("goes")
    if _cached(cache_path, use_cache):
        cb("load_goes", 0.0, "Loading GOES SST from cache...")
        result = _load(cache_path)
        cb("load_goes", 1.0, "Loaded from cache.")
        return result

    goes_path_str = config.goes_nc_path

    # S3 path
    if goes_path_str.startswith("s3://"):
        cb("load_goes", 0.0, f"Loading GOES SST from S3: {goes_path_str}...")
        if goes_path_str.endswith(".pkl"):
            ds_g = _load_s3_pkl(goes_path_str)
            if not isinstance(ds_g, xr.Dataset):
                cb("load_goes", 1.0, f"GOES pkl contains {type(ds_g).__name__}, expected xr.Dataset — skipping.")
                return None
        else:
            import s3fs
            fs = s3fs.S3FileSystem(anon=True)
            with fs.open(goes_path_str) as f:
                ds_g = xr.open_dataset(f, engine="h5netcdf").load()
            sst_var = next((v for v in ds_g.data_vars if v.upper() == "SST"), None)
            if sst_var is None:
                cb("load_goes", 1.0, f"No SST variable in S3 file. Available: {list(ds_g.data_vars)}")
                return None
            if sst_var != "SST":
                ds_g = ds_g.rename({sst_var: "SST"})
        _save(ds_g, cache_path)
        cb("load_goes", 1.0, f"GOES SST loaded from S3 ({len(ds_g.coords)} coords, vars={list(ds_g.data_vars)}).")
        return ds_g

    goes_path = Path(goes_path_str)

    # Single pre-processed .nc file (local)
    if goes_path.is_file() and goes_path.suffix in (".nc", ".nc4"):
        cb("load_goes", 0.0, f"Loading GOES SST from {goes_path.name}...")
        ds_g = xr.open_dataset(goes_path, engine="netcdf4").load()
        sst_var = next((v for v in ds_g.data_vars if v.upper() == "SST"), None)
        if sst_var is None:
            cb("load_goes", 1.0,
               f"No SST variable in {goes_path.name}. Available: {list(ds_g.data_vars)}")
            return None
        if sst_var != "SST":
            ds_g = ds_g.rename({sst_var: "SST"})
        _save(ds_g, cache_path)
        cb("load_goes", 1.0, f"GOES SST loaded from {goes_path.name}.")
        return ds_g

    # Directory of raw GOES-16/17/18 scan files
    from swotxai.animation_utils import build_goes_index
    cb("load_goes", 0.0, f"Scanning GOES files in {config.goes_nc_path}...")
    goes_index, _ = build_goes_index(config.goes_nc_path)
    if not goes_index:
        cb("load_goes", 1.0, "No GOES .nc files found — skipping.")
        return None

    n = len(goes_index)
    times, sst_arrays = [], []
    for i, (t, fpath) in enumerate(sorted(goes_index.items())):
        cb("load_goes", (i + 1) / n, f"GOES {i + 1}/{n}: {Path(fpath).name}")
        ds_g = xr.open_dataset(fpath, engine="netcdf4")
        sst_var = next((v for v in ds_g.data_vars if v.upper() == "SST"), None)
        if sst_var is None:
            ds_g.close()
            continue
        sst_arrays.append(ds_g[[sst_var]].rename({sst_var: "SST"}).load())
        times.append(t)
        ds_g.close()

    if not sst_arrays:
        cb("load_goes", 1.0, "No SST variable found in any GOES file — skipping.")
        return None

    goes_ds = xr.concat(sst_arrays, dim=pd.DatetimeIndex(times, name="time"))
    _save(goes_ds, cache_path)
    cb("load_goes", 1.0, f"GOES SST loaded: {len(times)} files.")
    return goes_ds


def step_interp_sources(
    config: SWOTConfig,
    cycle_data: dict,
    swot_regridded: dict,
    era5: xr.Dataset | None,
    goes_ds: xr.Dataset | None,
    cb: ProgressCb,
    use_cache: bool,
) -> dict:
    cache_path = config.cache_path("swot_features")
    if _cached(cache_path, use_cache):
        result = _load(cache_path)
        expects_era5 = era5 is not None and "era5_u" in era5
        has_era5 = any(
            ds is not None and "era5_u" in ds
            for ds_list in result.values()
            for ds in (ds_list if isinstance(ds_list, list) else [ds_list])
        )
        expects_sst = goes_ds is not None
        has_sst = any(
            ds is not None and "SST" in ds
            for ds_list in result.values()
            for ds in (ds_list if isinstance(ds_list, list) else [ds_list])
        )
        if expects_era5 and not has_era5:
            cb("interp_sources", 0.0, "Cache missing ERA5 features — rebuilding...")
            cache_path.unlink(missing_ok=True)
        elif expects_sst and not has_sst:
            cb("interp_sources", 0.0, "Cache missing SST features — rebuilding...")
            cache_path.unlink(missing_ok=True)
        else:
            cb("interp_sources", 0.0, "Loading interpolated features from cache...")
            cb("interp_sources", 1.0, "Loaded from cache.")
            return result

    sources = []
    if era5 is not None and "era5_u" in era5:
        sources.append({"ds": era5, "vars": ["era5_u", "era5_v"],
                        "lat": "latitude", "lon": "longitude", "time": "time"})
    if goes_ds is not None:
        lat_key = next((c for c in goes_ds.coords if c.lower() in ("lat", "latitude")), None) or "y"
        lon_key = next((c for c in goes_ds.coords if c.lower() in ("lon", "longitude")), None) or "x"
        cb("interp_sources", 0.0, f"GOES coords detected: lat='{lat_key}' lon='{lon_key}', vars={list(goes_ds.data_vars)}, dims={dict(goes_ds.dims)}")
        sources.append({"ds": goes_ds, "vars": ["SST"],
                        "lat": lat_key, "lon": lon_key, "time": "time",
                        "max_dist_deg": 0.05, "save_time_as": "goes_time"})

    if not sources:
        cb("interp_sources", 1.0, "No external sources configured — using raw SWOT features.")
        return swot_regridded

    cb("interp_sources", 0.0, f"Interpolating {[s['vars'] for s in sources]} onto SWOT grid...")
    from swotxai.swotxai_utils import interp_to_swot
    swot_features = interp_to_swot(cycle_data, swot_regridded, sources)
    _save(swot_features, cache_path)
    cb("interp_sources", 1.0, "Interpolation complete.")
    return swot_features


def step_load_hfr(config: SWOTConfig, cb: ProgressCb, use_cache: bool) -> xr.Dataset | None:
    if not config.hfr_path:
        cb("load_hfr", 1.0, "HFR path not set — skipping.")
        return None

    cache_path = config.cache_path("hfr")
    if _cached(cache_path, use_cache):
        cb("load_hfr", 0.0, "Loading HFR from cache...")
        result = _load(cache_path)
        cb("load_hfr", 1.0, "Loaded from cache.")
        return result

    cb("load_hfr", 0.0, f"Loading HFR from {config.hfr_path}...")
    if config.hfr_path.startswith("s3://"):
        import s3fs
        fs = s3fs.S3FileSystem(anon=True)
        with fs.open(config.hfr_path.replace("s3://", "")) as f:
            hfr = xr.open_dataset(f, engine="h5netcdf").load()
    else:
        hfr = xr.open_dataset(config.hfr_path, engine="netcdf4").load()
    _save(hfr, cache_path)
    cb("load_hfr", 1.0, "HFR loaded.")
    return hfr


def step_interp_hfr(
    config: SWOTConfig,
    hfr: xr.Dataset | None,
    cycle_data: dict,
    swot_regridded: dict,
    cb: ProgressCb,
    use_cache: bool,
) -> dict | None:
    cache_path = config.cache_path("hfr_interp")
    if _cached(cache_path, use_cache):
        cb("interp_hfr", 0.0, "Loading interpolated HFR from cache...")
        result = _load(cache_path)
        cb("interp_hfr", 1.0, "Loaded from cache.")
        return result

    if hfr is None:
        cb("interp_hfr", 1.0, "No HFR data — skipping.")
        return None

    from swotxai.swotxai_utils import hfr_on_swot

    cb("interp_hfr", 0.0, "Applying 24-hour rolling mean to HFR...")
    hfr_rolling = hfr.rolling(time=24, center=True, min_periods=1).mean()

    keys = [t for t in swot_regridded if swot_regridded[t]]
    n = len(keys)
    cb("interp_hfr", 0.01, f"Interpolating HFR onto {n} cycles...")

    hfr_interp_data = {}
    for i, t in enumerate(keys):
        cb("interp_hfr", (i + 1) / n, f"Cycle {t}  ({i + 1}/{n})")
        orig_list   = cycle_data.get(t, [])
        regrid_list = swot_regridded[t]
        interp_list = []
        for orig_ds, regrid_ds in zip(orig_list, regrid_list):
            if orig_ds is None or regrid_ds is None:
                continue
            first_time  = orig_ds["time"].values[0]
            hfr_at_time = hfr_rolling.sel(time=first_time, method="nearest")
            result = hfr_on_swot(hfr_at_time, regrid_ds)
            if result is not None:
                interp_list.append(result)
        hfr_interp_data[t] = interp_list

    _save(hfr_interp_data, cache_path)
    cb("interp_hfr", 1.0, "HFR interpolation complete.")
    return hfr_interp_data


def step_flatten(
    config: SWOTConfig,
    hfr_interp_data: dict,
    swot_features: dict,
    cb: ProgressCb,
    use_cache: bool,
) -> dict:
    cache_path = config.cache_path("flattened")

    # Determine which configured features are actually present in the data
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

    from swotxai.swotxai_utils import rf_flattening_stencil

    keys = [t for t in swot_features if swot_features[t]]
    n = len(keys)
    cb("flatten", 0.0, f"Flattening {n} cycles (stencil k={config.stencil_k}, {len(effective_features)} features → {expected_n_cols} cols)...")

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
        expected_n_cols = len(config.features) * config.stencil_k ** 2
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

    from swotxai.swotxai_utils import concat_flattened, random_forest

    cb("train", 0.0, "Concatenating training data...")
    X_u, X_v, y_u, y_v = concat_flattened(flattened, training_percentage=0.8)

    backend = "cuML (GPU)" if config.use_gpu else f"sklearn (CPU, n_jobs={config.sklearn_n_jobs})"
    cb("train", 0.3, f"Training RF for u-velocity (n_estimators={config.n_estimators}, backend={backend})...")
    rf_u = random_forest(X_u, y_u, config.n_estimators, config.max_depth, config.random_state, n_jobs=config.sklearn_n_jobs, use_gpu=config.use_gpu)

    cb("train", 0.7, f"Training RF for v-velocity (backend={backend})...")
    rf_v = random_forest(X_v, y_v, config.n_estimators, config.max_depth, config.random_state, n_jobs=config.sklearn_n_jobs, use_gpu=config.use_gpu)

    _save_model(rf_u, cache_path_u)
    _save_model(rf_v, cache_path_v)
    _save({"features": config.features, "stencil_k": config.stencil_k}, config.cache_path("rf_meta"))
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
    from swotxai.swotxai_utils import concat_flattened

    cb("evaluate", 0.0, "Computing evaluation metrics...")
    X_u, X_v, y_u, y_v = concat_flattened(flattened, training_percentage=1.0)

    _, X_test_u, _, y_test_u = train_test_split(X_u, y_u, test_size=0.2, random_state=config.random_state)
    _, X_test_v, _, y_test_v = train_test_split(X_v, y_v, test_size=0.2, random_state=config.random_state)

    try:
        from cuml.ensemble import RandomForestRegressor as cuRF
        _is_cuml = isinstance(rf_u, cuRF)
    except ImportError:
        _is_cuml = False

    if _is_cuml:
        pred_u = np.asarray(rf_u.predict(np.asarray(X_test_u, dtype="float32")))
        pred_v = np.asarray(rf_v.predict(np.asarray(X_test_v, dtype="float32")))
        y_test_u = np.asarray(y_test_u, dtype="float32")
        y_test_v = np.asarray(y_test_v, dtype="float32")
    else:
        pred_u = rf_u.predict(X_test_u)
        pred_v = rf_v.predict(X_test_v)

    meta_path = config.cache_path("rf_meta")
    if meta_path.exists():
        meta = _load(meta_path)
        train_features = meta["features"]
        k = meta["stencil_k"]
    else:
        train_features = config.features
        k = config.stencil_k
    n_features = len(rf_u.feature_importances_) // (k * k)
    fi_u = rf_u.feature_importances_.reshape(n_features, k * k).mean(axis=1)
    fi_v = rf_v.feature_importances_.reshape(n_features, k * k).mean(axis=1)
    feature_names = train_features if len(train_features) == n_features else [f"feature_{i}" for i in range(n_features)]

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
    from swotxai.swotxai_utils import build_frame_dicts

    predictions = ["ssv_pred_u", "ssv_pred_v", "ssv_pred"]
    frames = list(range(config.cycles_start, config.cycles_end + 1))

    cache_path = config.cache_path("inference")
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
    summary = {k: [("ok" if ds is not None else "None") for ds in v] for k, v in swot_dict.items()}
    n_valid = sum(1 for v in swot_dict.values() for ds in v if ds is not None)
    cb("inference", 1.0, f"{n_valid} valid entries — {summary}")
    return swot_dict, hfr_dict


def step_animate(
    config: SWOTConfig,
    swot_dict: dict,
    hfr_dict: dict,
    cycle_data: dict,
    swot_regridded: dict,
    hfr_interp_data: dict,
    metrics: dict,
    cb: ProgressCb,
) -> list[str]:
    from swotxai.animation_utils import from_cycle_dict, generate_frames, assemble_animations_by_pass

    cb("animate", 0.0, "Building animation panels...")

    lon_bounds = [config.sw_corner[0], config.ne_corner[0]]
    lat_bounds = [config.sw_corner[1], config.ne_corner[1]]
    frames     = list(range(config.cycles_start, config.cycles_end + 1))

    # --- Time helpers (defined first so panels can reference them) ---

    def _hfr_time(cycle: int, j: int) -> pd.Timestamp | None:
        try:
            h_list = hfr_interp_data.get(str(cycle).zfill(3), [])
            if j >= len(h_list) or h_list[j] is None:
                return None
            t = h_list[j].coords["time"].values
            return pd.Timestamp(t.flat[0] if hasattr(t, "flat") else t)
        except Exception:
            return None

    def _swot_time(cycle: int, j: int) -> pd.Timestamp | None:
        key = str(cycle).zfill(3)
        # Check swot_dict then swot_regridded — both carry the time coord when built from
        # new pkls, but a stale inference cache means swot_dict entries are old objects
        # without time while swot_regridded (freshly loaded from S3) has it.
        for src in (swot_dict, swot_regridded):
            try:
                ds_list = src.get(key, [])
                if j < len(ds_list) and ds_list[j] is not None:
                    ds = ds_list[j]
                    t_da = (ds.coords["time"] if "time" in ds.coords
                            else ds["time"] if "time" in ds else None)
                    if t_da is not None:
                        t = np.atleast_1d(t_da.values)
                        valid = t[~np.isnat(t.astype("datetime64[ns]"))]
                        if len(valid):
                            return pd.Timestamp(valid[len(valid) // 2])
            except Exception:
                pass
        # Non-preset: raw cycle_data
        try:
            ds_list = cycle_data.get(key, [])
            if j < len(ds_list) and ds_list[j] is not None:
                t = ds_list[j]["time"].values
                return pd.Timestamp(t.flat[0] if hasattr(t, "flat") else t)
        except Exception:
            pass
        # Last resort: use HFR time as proxy
        return _hfr_time(cycle, j)

    # --- Panels ---

    panels = [
        {
            "title": "SWOT Geostrophic Velocity",
            "data_fn": from_cycle_dict(swot_dict, "gos_filtered"),
            "time_fn": _swot_time,
            "cmap": "viridis", "vmin": 0, "vmax": 2,
            "colorbar_label": "m/s",
        },
        {
            "title": "HFR Ground Truth SSV",
            "data_fn": from_cycle_dict(hfr_dict, "ssv"),
            "time_fn": _hfr_time,
            "cmap": "viridis", "vmin": 0, "vmax": 0.3,
            "colorbar_label": "m/s",
        },
        {
            "title": "SWOT Inferred SSV",
            "data_fn": from_cycle_dict(swot_dict, "ssv_pred"),
            "time_fn": _swot_time,
            "cmap": "viridis", "vmin": 0, "vmax": 0.3,
            "colorbar_label": "m/s",
        },
    ]

    def _era5_time(cycle: int, j: int) -> pd.Timestamp | None:
        key = str(cycle).zfill(3)
        try:
            ds_list = swot_dict.get(key, [])
            if j < len(ds_list) and ds_list[j] is not None:
                ds = ds_list[j]
                for coord in ("valid_time", "time"):
                    if coord in ds.coords:
                        t = np.atleast_1d(ds.coords[coord].values)
                        valid = t[~np.isnat(t.astype("datetime64[ns]"))]
                        if len(valid):
                            return pd.Timestamp(valid[0])
        except Exception:
            pass
        return None

    has_era5_ssv = any(
        "era5_ssv" in ds
        for ds_list in swot_dict.values()
        for ds in ds_list
        if ds is not None
    )
    if has_era5_ssv:
        panels.append({
            "title": "ERA5 Wind Speed",
            "data_fn": from_cycle_dict(swot_dict, "era5_ssv"),
            "time_fn": _era5_time,
            "cmap": "Blues", "vmin": 0, "vmax": 15,
            "colorbar_label": "m/s",
        })

    sst_vals = [
        v for ds_list in swot_dict.values()
        for ds in ds_list if ds is not None and "SST" in ds
        for v in ds["SST"].values.ravel() if np.isfinite(v)
    ]
    has_sst = bool(sst_vals)
    if has_sst:
        sst_vmin = float(np.percentile(sst_vals, 2))
        sst_vmax = float(np.percentile(sst_vals, 98))
        cb("animate", 0.1, f"SST global range (2nd–98th pct): {sst_vmin:.2f} – {sst_vmax:.2f}")

        def _goes_time(cycle: int, j: int) -> pd.Timestamp | None:
            key = str(cycle).zfill(3)
            try:
                ds_list = swot_dict.get(key, [])
                if j < len(ds_list) and ds_list[j] is not None:
                    ds = ds_list[j]
                    if "goes_time" in ds.coords:
                        return pd.Timestamp(ds.coords["goes_time"].values)
            except Exception:
                pass
            return None

        panels.append({
            "title": "GOES SST",
            "data_fn": from_cycle_dict(swot_dict, "SST"),
            "time_fn": _goes_time,
            "cmap": "RdYlBu_r", "vmin": sst_vmin, "vmax": sst_vmax,
            "colorbar_label": "SST",
        })

    # Move SWOT Inferred SSV to the centre position
    inferred_idx = next(i for i, p in enumerate(panels) if p["title"] == "SWOT Inferred SSV")
    inferred_panel = panels.pop(inferred_idx)
    panels.insert(len(panels) // 2, inferred_panel)

    _r2u   = metrics.get("r2_u",   float("nan"))
    _r2v   = metrics.get("r2_v",   float("nan"))
    _rmseu = metrics.get("rmse_u", float("nan"))
    _rmsev = metrics.get("rmse_v", float("nan"))
    _stats = f"R²: u={_r2u:.3f} v={_r2v:.3f}  |  RMSE: u={_rmseu:.4f} v={_rmsev:.4f} m/s"

    def title_fn(cycle: int, j: int) -> str:
        return f"Cycle {cycle}, Pass {j}  ({_stats})"

    n_valid = sum(1 for v in swot_dict.values() for ds in v if ds is not None)
    cb("animate", 0.1, f"{n_valid} valid pass entries in swot_dict. Generating frames...")

    # Pre-flight: log details of the first valid entry so failures are visible in Streamlit
    for key, ds_list in swot_dict.items():
        for j, ds in enumerate(ds_list):
            if ds is not None:
                avail_vars = list(ds.data_vars)
                has_gos = "gos_filtered" in ds
                n_gos_valid = 0
                lon_range = lat_range = "N/A"
                if has_gos:
                    try:
                        vals = ds["gos_filtered"].values
                        n_gos_valid = int(np.sum(~np.isnan(vals)))
                        lon_range = f"[{float(np.nanmin(ds['lon'].values)):.2f}, {float(np.nanmax(ds['lon'].values)):.2f}]"
                        lat_range = f"[{float(np.nanmin(ds['lat'].values)):.2f}, {float(np.nanmax(ds['lat'].values)):.2f}]"
                    except Exception:
                        pass
                cb("animate", 0.1,
                   f"Preflight cycle={key} pass={j}: vars={avail_vars}, "
                   f"gos_filtered={'yes' if has_gos else 'MISSING'}, "
                   f"n_valid={n_gos_valid}, lon={lon_range}, lat={lat_range}")
                break
        else:
            continue
        break

    frame_log_msgs: list[str] = []
    _skip_count = [0]
    def _frame_log(msg: str) -> None:
        frame_log_msgs.append(msg)
        print(msg)
        if "Skipping" in msg:
            _skip_count[0] += 1
            if _skip_count[0] <= 5:
                cb("animate", 0.5, msg)
            elif _skip_count[0] == 6:
                cb("animate", 0.5, f"(further skip messages suppressed...)")
        elif any(kw in msg for kw in ("White preview", "Frame generation complete", "Preflight")):
            cb("animate", 0.5, msg)

    n_panels = len(panels)
    frame_files = generate_frames(
        panels=panels,
        frames=frames,
        frame_dir=config.frame_dir,
        lon_bounds=lon_bounds,
        lat_bounds=lat_bounds,
        title_fn=title_fn,
        figsize=(n_panels * 5.5, 5.5),
        wspace=0.12,
        dpi=config.dpi,
        log_fn=_frame_log,
    )

    cb("animate", 0.9, f"Generated {len(frame_files)} frames. Assembling animations by pass...")
    if not frame_files:
        raise RuntimeError("No frames were generated — all cycles were skipped. Check that swot_dict and hfr_dict have valid data.")
    anim_paths = assemble_animations_by_pass(frame_files, config.animation_output, fps=config.fps)
    paths = [str(p) for p in anim_paths.values()]
    cb("animate", 1.0, f"Animations saved: {', '.join(paths)}")
    return paths


# ---------------------------------------------------------------------------
# Shared steps (data loading / interpolation — independent of ML hyperparams)
# ---------------------------------------------------------------------------

_SHARED_CACHE_KEYS = ["cycle_data", "swot_regridded", "era5", "goes", "swot_features", "hfr", "hfr_interp"]


def _cleanup_shared_cache(config: SWOTConfig) -> None:
    """Delete shared step cache files now that per-job outputs are saved."""
    for name in _SHARED_CACHE_KEYS:
        p = config.cache_path(name)
        if p.exists():
            p.unlink()


def run_shared_steps(
    config: SWOTConfig,
    progress_cb: ProgressCb | None = None,
    use_cache: bool = True,
) -> dict:
    """
    Run load_swot → interp_hfr (the six shared data steps).
    Results are cached to disk and deleted once per-job outputs are saved.

    Returns a dict with keys:
        cycle_data, swot_regridded, era5, swot_features, hfr, hfr_interp_data
    """
    cb = progress_cb or _noop_cb
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    results: dict = {}

    def _run(name: str, fn):
        t = time.time()
        out = fn()
        cb(name, 1.0, f"{name} done in {time.time() - t:.1f}s")
        return out

    if config.swot_pkl_path:
        results["cycle_data"], results["swot_regridded"] = step_load_preset_swot(config, cb)
    else:
        results["cycle_data"]     = _run("load_swot", lambda: step_load_swot(config, cb, use_cache))
        results["swot_regridded"] = _run("regrid",    lambda: step_regrid(config, results["cycle_data"], cb, use_cache))

    results["era5"] = _run("load_era5", lambda: step_load_era5(config, cb, use_cache))
    results["goes"] = _run("load_goes", lambda: step_load_goes(config, cb, use_cache))
    results["swot_features"] = _run("interp_sources", lambda: step_interp_sources(
        config, results["cycle_data"], results["swot_regridded"],
        results.get("era5"), results.get("goes"), cb, use_cache,
    ))

    if config.hfr_pkl_path:
        results["hfr"]             = None
        results["hfr_interp_data"] = step_load_preset_hfr(config, cb)
    else:
        if _cached(config.cache_path("hfr_interp"), use_cache):
            results["hfr"] = None
            cb("load_hfr", 1.0, "Skipped — hfr_interp cache exists.")
        else:
            results["hfr"] = _run("load_hfr", lambda: step_load_hfr(config, cb, use_cache))
        results["hfr_interp_data"] = _run("interp_hfr", lambda: step_interp_hfr(
            config, results.get("hfr"), results["cycle_data"], results["swot_regridded"], cb, use_cache,
        ))

    return results


# ---------------------------------------------------------------------------
# Per-job steps (ML-specific — run once per hyperparameter configuration)
# ---------------------------------------------------------------------------

def run_per_job_steps(
    config: SWOTConfig,
    shared: dict,
    progress_cb: ProgressCb | None = None,
    use_cache: bool = True,
) -> dict:
    """
    Run flatten → animate using pre-loaded shared intermediates.

    Parameters
    ----------
    shared : dict
        Output of run_shared_steps (keys: cycle_data, swot_regridded, era5,
        swot_features, hfr_interp_data).

    Returns a dict with keys:
        flattened, rf_u, rf_v, metrics, swot_dict, hfr_dict, animation_path
    """
    cb = progress_cb or _noop_cb
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    results: dict = {}

    def _run(name: str, fn):
        t = time.time()
        cb(name, 0.0, f"Starting {name}...")
        out = fn()
        cb(name, 1.0, f"{name} done in {time.time() - t:.1f}s")
        return out

    results["flattened"] = _run(
        "flatten", lambda: step_flatten(
            config, shared["hfr_interp_data"], shared["swot_features"], cb, use_cache,
        )
    )

    train_out = _run(
        "train", lambda: step_train(config, results["flattened"], cb, use_cache)
    )
    if train_out:
        results["rf_u"], results["rf_v"] = train_out

    results["metrics"] = _run(
        "evaluate", lambda: step_evaluate(
            config, results["rf_u"], results["rf_v"], results["flattened"], cb,
        )
    )

    dicts_out = _run(
        "inference", lambda: step_inference(
            config, results["rf_u"], results["rf_v"],
            shared["swot_features"], shared["hfr_interp_data"],
            results["flattened"], cb, use_cache,
        )
    )
    if dicts_out:
        results["swot_dict"], results["hfr_dict"] = dicts_out

    results["animation_paths"] = _run(
        "animate", lambda: step_animate(
            config, results["swot_dict"], results["hfr_dict"],
            shared.get("cycle_data", {}), shared.get("swot_regridded", {}),
            shared.get("hfr_interp_data", {}),
            results["metrics"], cb,
        )
    )

    return results


# ---------------------------------------------------------------------------
# Main orchestrator (thin wrapper — public signature unchanged)
# ---------------------------------------------------------------------------

def run_pipeline(
    config: SWOTConfig,
    steps: list[str] | None = None,
    progress_cb: ProgressCb | None = None,
    use_cache: bool = True,
) -> dict:
    """
    Run the full SWOTxAI pipeline from a config object.

    Parameters
    ----------
    config : SWOTConfig
    steps : list[str] | None
        Ignored (kept for backward compatibility). All steps always run.
    progress_cb : ProgressCb | None
        Called as (step_name, fraction, message) for each step.
    use_cache : bool
        If True, load cached pickle files when available.

    Returns
    -------
    dict with keys: cycle_data, swot_regridded, era5, swot_features,
                    hfr, hfr_interp_data, flattened, rf_u, rf_v,
                    metrics, swot_dict, hfr_dict, animation_path
    """
    cb = progress_cb or _noop_cb
    t0 = time.time()
    shared = run_shared_steps(config, progress_cb=cb, use_cache=use_cache)
    try:
        per_job = run_per_job_steps(config, shared, progress_cb=cb, use_cache=use_cache)
    finally:
        _cleanup_shared_cache(config)
        shared.clear()
    cb("done", 1.0, f"Pipeline complete in {time.time() - t0:.1f}s")
    return per_job
