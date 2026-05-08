from __future__ import annotations

import pandas as pd
import xarray as xr
from pathlib import Path
from typing import Callable

from swotxai.config import SWOTConfig
from swotxai.pipeline.io_utils import _save, _load, _cached, _load_s3_pkl, _save_s3_pkl

ProgressCb = Callable[[str, float, str], None]


def step_load_preset_swot(config: SWOTConfig, cb: ProgressCb) -> tuple[dict, dict]:
    cb("load_swot", 0.0, "Streaming preset SWOT pkl from S3...")
    swot_regridded = _load_s3_pkl(config.swot_pkl_path)
    cb("load_swot", 1.0, f"Loaded {len(swot_regridded)} cycles.")
    cb("regrid", 1.0, "Skipped — using preset pkl.")
    return {}, swot_regridded


def step_load_preset_hfr(config: SWOTConfig, cb: ProgressCb) -> dict:
    cb("load_hfr", 0.0, "Streaming preset HFR pkl from S3...")
    hfr_interp = _load_s3_pkl(config.hfr_pkl_path)
    cb("load_hfr", 1.0, "Loaded preset HFR.")
    cb("interp_hfr", 1.0, "Skipped — using preset pkl.")
    return hfr_interp


def step_load_swot(config: SWOTConfig, cb: ProgressCb, use_cache: bool) -> dict:
    cache_path = config.cache_path("cycle_data")
    if _cached(cache_path, use_cache):
        cb("load_swot", 0.0, "Loading from cache...")
        result = _load(cache_path)
        cb("load_swot", 1.0, f"Loaded {len(result)} cycles from cache.")
        return result

    from swot import data_loaders
    from swot.download_swaths import find_swaths

    science  = (config.mission == "science")
    sph_path = Path(config.sph_science_path if science else config.sph_calval_path)
    cycles_start, cycles_end = config.cycles_start, config.cycles_end

    if not sph_path.exists():
        raise FileNotFoundError(
            f"Orbit file not found: {sph_path}. Set sph_calval_path / sph_science_path in your config."
        )

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

    from swotxai.data_utils import swot_regrid

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

    pkl_path = config.era5_pkl_path
    if use_cache and pkl_path:
        try:
            cb("load_era5", 0.0, f"Loading ERA5 pkl from {pkl_path}...")
            era5 = _load_s3_pkl(pkl_path) if pkl_path.startswith("s3://") else _load(Path(pkl_path))
            cb("load_era5", 1.0, "ERA5 loaded from pkl.")
            return era5
        except FileNotFoundError:
            cb("load_era5", 0.0, f"ERA5 pkl not found at {pkl_path} — falling back to source.")

    cache_path = config.cache_path("era5")
    if _cached(cache_path, use_cache):
        result = _load(cache_path)
        if "era5_u" in result:
            cb("load_era5", 0.0, "Loading ERA5 from cache...")
            cb("load_era5", 1.0, "Loaded from cache.")
            return result
        cb("load_era5", 0.0, "ERA5 cache missing era5_u — reloading from source...")
        cache_path.unlink(missing_ok=True)

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

    _save(era5, cache_path)

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
        cb("load_goes", 1.0, f"GOES SST loaded from S3.")
        return ds_g

    goes_path = Path(goes_path_str)

    if goes_path.is_file() and goes_path.suffix in (".nc", ".nc4"):
        cb("load_goes", 0.0, f"Loading GOES SST from {goes_path.name}...")
        ds_g = xr.open_dataset(goes_path, engine="netcdf4").load()
        sst_var = next((v for v in ds_g.data_vars if v.upper() == "SST"), None)
        if sst_var is None:
            cb("load_goes", 1.0, f"No SST variable in {goes_path.name}. Available: {list(ds_g.data_vars)}")
            return None
        if sst_var != "SST":
            ds_g = ds_g.rename({sst_var: "SST"})
        _save(ds_g, cache_path)
        cb("load_goes", 1.0, f"GOES SST loaded from {goes_path.name}.")
        return ds_g

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
        cb("interp_sources", 0.0,
           f"GOES coords detected: lat='{lat_key}' lon='{lon_key}', vars={list(goes_ds.data_vars)}, dims={dict(goes_ds.dims)}")
        sources.append({"ds": goes_ds, "vars": ["SST"],
                        "lat": lat_key, "lon": lon_key, "time": "time",
                        "max_dist_deg": 0.05, "save_time_as": "goes_time"})

    if not sources:
        cb("interp_sources", 1.0, "No external sources configured — using raw SWOT features.")
        return swot_regridded

    cb("interp_sources", 0.0, f"Interpolating {[s['vars'] for s in sources]} onto SWOT grid...")
    from swotxai.data_utils import interp_to_swot
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

    from swotxai.data_utils import hfr_on_swot

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
