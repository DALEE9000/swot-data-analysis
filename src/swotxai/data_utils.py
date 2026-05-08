import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import xarray as xr
import numpy as np
from scipy.interpolate import griddata
import xrft
from glob import glob
import os
from tqdm import tqdm
import pickle
from pathlib import Path
from dotenv import load_dotenv

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.geometry import box, LineString, Point
import cmocean as cm

import swot.swot_utils as swot_utils
import swot.data_loaders as data_loaders
import swot.download_swaths as download_swaths

import warnings
warnings.filterwarnings('ignore')

env_path = Path(__file__).parent.parent
load_dotenv(env_path / ".env")

science_path = (env_path.parent / os.getenv("SCIENCE")) if os.getenv("SCIENCE") else None
calval_path  = (env_path.parent / os.getenv("CALVAL"))  if os.getenv("CALVAL")  else None
dict_path    = (env_path.parent / os.getenv("DICT"))    if os.getenv("DICT")    else None


def load_swot_data(
    path: str,
    sw_corner: list,
    ne_corner: list,
    fields: list = None,
    science: bool = False,
):
    lat_lims = [sw_corner[1], ne_corner[1]]

    if science:
        cycles_start, cycles_end = 1, 17
        path_to_sph_file = science_path
        print(f"Using science path: {path_to_sph_file}")
    else:
        cycles_start, cycles_end = 474, 579
        path_to_sph_file = calval_path
        print(f"Using calval path: {path_to_sph_file}")

    cycles = [str(c_num).zfill(3) for c_num in range(cycles_start, cycles_end)]
    pass_IDs_list = download_swaths.find_swaths(
        sw_corner=sw_corner,
        ne_corner=ne_corner,
        path_to_sph_file=path_to_sph_file,
    )

    cycle_data = {}
    for cycle in cycles:
        cycle_data[cycle] = data_loaders.load_cycle(
            path=path,
            fields=fields,
            cycle=cycle,
            pass_ids=pass_IDs_list,
            subset=True,
            lats=lat_lims,
        )

    return cycle_data


def save_dict(data, filename: str, directory: str | Path | None = None):
    base = Path(directory) if directory is not None else dict_path
    path = base / f"{filename}.pkl"
    if path.exists():
        raise FileExistsError(f"{path} already exists; refusing to overwrite.")
    with path.open("wb") as f:
        pickle.dump(data, f)
    print(f"Saved {path}")


def load_dict(filename: str, directory: str | Path | None = None):
    base = Path(directory) if directory is not None else dict_path
    path = base / f"{filename}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    with path.open("rb") as f:
        return pickle.load(f)


def swot_regrid(swot_data: xr.Dataset, features: list | None = None):
    lon2d, lat2d = swot_data['longitude'].values, swot_data['latitude'].values

    lat_flat = swot_data['latitude'].values.flatten()
    lon_flat = swot_data['longitude'].values.flatten()
    points = np.column_stack((lat_flat, lon_flat))

    interpolated_data = {}
    grid_dims = ("num_lines", "num_pixels")
    features = [
        var for var, da in swot_data.data_vars.items()
        if da.dims == grid_dims
    ]

    for f in features:
        values_flat = swot_data[f].values.flatten()
        interpolated_values = griddata(points, values_flat, (lat2d, lon2d), method='linear')
        interpolated_data[f] = (('lat', 'lon'), interpolated_values)

    swot_regridded = xr.Dataset(
        data_vars=interpolated_data,
        coords={
            'lat': (('lat', 'lon'), lat2d),
            'lon': (('lat', 'lon'), lon2d),
        },
    )
    swot_regridded['lon'] = (swot_regridded['lon'] + 180) % 360 - 180

    if 'time' in swot_data:
        try:
            t_flat = swot_data['time'].values.ravel()
            valid = t_flat[~np.isnat(t_flat.astype('datetime64[ns]'))]
            if len(valid):
                swot_regridded = swot_regridded.assign_coords(time=valid[len(valid) // 2])
        except Exception:
            pass

    return swot_regridded


def apply_regrid(swot_dict: dict, features: list | None = None, coarsen_factor: int | None = None):
    regridded_data = {}
    for t, ds_list in tqdm(swot_dict.items(), desc="Timesteps", unit="step"):
        regridded_data[t] = [
            swot_regrid(
                ds.coarsen(num_lines=coarsen_factor, num_pixels=coarsen_factor, boundary="trim").mean()
                if coarsen_factor is not None else ds,
                features,
            )
            for ds in tqdm(ds_list, desc=f"Regridding {t}", leave=False, unit="xarray")
            if ds is not None
        ]
    return regridded_data


def hfr_on_swot(hfr_data: xr.Dataset, swot_regridded: xr.Dataset):
    return hfr_data.interp(
        lat=swot_regridded['lat'],
        lon=swot_regridded['lon'],
        method='linear',
    )


def hfr_interp(hfr: xr.Dataset, cycle_data: dict, swot_regridded: dict):
    hfr_interp_data = {}
    for t in tqdm(swot_regridded.keys(), desc="Timesteps", unit="step"):
        orig_ds_list = cycle_data.get(t, [])
        regrid_ds_list = swot_regridded.get(t, [])
        if not regrid_ds_list:
            continue
        interp_list = []
        for orig_ds, regrid_ds in zip(orig_ds_list, regrid_ds_list):
            if orig_ds is None or regrid_ds is None:
                continue
            first_time = orig_ds["time"].values[0]
            hfr_at_time = hfr.sel(time=first_time, method="nearest")
            interp_list.append(hfr_on_swot(hfr_at_time, regrid_ds))
        hfr_interp_data[t] = interp_list
    return hfr_interp_data


def interp_to_swot(cycle_data: dict, swot_regridded: dict, sources: list) -> dict:
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree

    def _interp_source(source_ds, vars, lat_key, lon_key, swot_ds, max_dist_deg):
        lat_coord = source_ds[lat_key]
        is_2d = lat_coord.ndim > 1

        if is_2d:
            src_lat = lat_coord.values.ravel()
            src_lon = source_ds[lon_key].values.ravel()
            swot_lat = swot_ds["lat"].values
            swot_lon = swot_ds["lon"].values
            swot_points = np.column_stack([swot_lat.ravel(), swot_lon.ravel()])

            valid_coords = np.isfinite(src_lat) & np.isfinite(src_lon)
            if valid_coords.sum() < 2:
                return xr.Dataset({
                    v: (list(swot_ds["lat"].dims), np.full(swot_lat.shape, np.nan, dtype=np.float32))
                    for v in vars
                })

            tree = cKDTree(np.column_stack([src_lat[valid_coords], src_lon[valid_coords]]))
            dist, idx = tree.query(swot_points)
            too_far = dist.reshape(swot_lat.shape) > max_dist_deg

            result_vars = {}
            for v in vars:
                src_vals = source_ds[v].values.ravel()
                interped = src_vals[valid_coords][idx].reshape(swot_lat.shape).astype(np.float32)
                interped[too_far] = np.nan
                result_vars[v] = (list(swot_ds["lat"].dims), interped)
            return xr.Dataset(result_vars)
        else:
            return source_ds[vars].interp(
                {lat_key: swot_ds["lat"], lon_key: swot_ds["lon"]},
                method="linear",
            )

    result = {}
    for t in tqdm(swot_regridded.keys(), desc="interp_to_swot", unit="step"):
        orig_ds_list   = cycle_data.get(t, [])
        regrid_ds_list = swot_regridded.get(t, [])
        if not regrid_ds_list:
            continue

        merged_list = []
        for idx, regrid_ds in enumerate(regrid_ds_list):
            if regrid_ds is None:
                continue
            orig_ds = orig_ds_list[idx] if idx < len(orig_ds_list) else None
            if orig_ds is not None and "time" in orig_ds:
                first_time = orig_ds["time"].values.flat[0]
            elif "time" in regrid_ds.coords:
                first_time = regrid_ds.coords["time"].values
            else:
                merged_list.append(regrid_ds)
                continue

            merged = regrid_ds
            for src in sources:
                ds           = src["ds"]
                vars         = src["vars"]
                lat_key      = src.get("lat",          "lat")
                lon_key      = src.get("lon",          "lon")
                time_key     = src.get("time",         "time")
                max_dist_deg = src.get("max_dist_deg", 0.1)
                save_time_as = src.get("save_time_as", None)

                ds_at_time  = ds.sel({time_key: first_time}, method="nearest")
                actual_time = ds_at_time[time_key].values
                interped    = _interp_source(ds_at_time, vars, lat_key, lon_key, merged, max_dist_deg)
                if "time" in interped.coords:
                    interped = interped.drop_vars("time")
                if save_time_as is not None:
                    interped = interped.assign_coords({save_time_as: actual_time})
                merged = xr.merge([merged, interped])

            merged_list.append(merged)
        result[t] = merged_list

    return result


def rf_flattening(swot_regridded: xr.Dataset, hfr_on_swot: xr.Dataset, target_u, target_v, features: list):
    data_dict = {f: swot_regridded[f].values.flatten() for f in features}
    target_u = target_u.values.flatten()
    target_v = target_v.values.flatten()

    df = pd.DataFrame(data_dict)
    y_u = pd.Series(target_u, name='u')
    y_v = pd.Series(target_v, name='v')

    mask_u = ~y_u.isna()
    mask_v = ~y_v.isna()
    return {
        "df": df,
        "X_u": df[mask_u].reset_index(drop=True),
        "X_v": df[mask_v].reset_index(drop=True),
        "y_u": y_u[mask_u].reset_index(drop=True),
        "y_v": y_v[mask_v].reset_index(drop=True),
    }


def rf_flattening_stencil(
    swot_regridded: xr.Dataset,
    target_u: xr.DataArray,
    target_v: xr.DataArray,
    features: list,
    k: int = 3,
):
    from numpy.lib.stride_tricks import sliding_window_view

    pad = k // 2
    feature_stack = np.stack([swot_regridded[f].values for f in features], axis=0)
    feature_stack = np.pad(
        feature_stack,
        ((0, 0), (pad, pad), (pad, pad)),
        mode='constant', constant_values=np.nan,
    )

    windows = sliding_window_view(feature_stack, (k, k), axis=(1, 2))
    n_features, lat, lon = windows.shape[:3]
    X = windows.transpose(1, 2, 0, 3, 4).reshape(lat * lon, n_features * k * k)

    y_u = np.asarray(target_u.compute(scheduler='threads').values).reshape(-1)
    y_v = np.asarray(target_v.compute(scheduler='threads').values).reshape(-1)

    mask_u = ~np.isnan(y_u)
    mask_v = ~np.isnan(y_v)

    col_names = [f"{f}_d{di}_{dj}" for f in features for di in range(k) for dj in range(k)]
    df = pd.DataFrame(X, columns=col_names)
    y_u = pd.Series(y_u, name="u")
    y_v = pd.Series(y_v, name="v")

    return {
        "df": df,
        "X_u": df[mask_u],
        "X_v": df[mask_v],
        "y_u": y_u[mask_u],
        "y_v": y_v[mask_v],
    }


def flattening(
    hfr_interp_data: dict,
    swot_regridded: dict,
    features: list,
    stenciling: bool = True,
    k: int = 3,
    n_jobs: int = -1,
):
    flattened_data = {}
    for t in tqdm(swot_regridded.keys(), desc="Timesteps", unit="step"):
        hfr_ds_list  = hfr_interp_data.get(t, [])
        swot_ds_list = swot_regridded.get(t, [])
        if not swot_ds_list or not hfr_ds_list:
            continue
        flatten_list = []
        for hfr_ds, swot_ds in zip(hfr_ds_list, swot_ds_list):
            hfr_ds = hfr_ds.compute(scheduler='threads')
            if hfr_ds is None or swot_ds is None:
                continue
            if stenciling:
                flatten_list.append(rf_flattening_stencil(swot_ds, hfr_ds['u'], hfr_ds['v'], features, k))
            else:
                flatten_list.append(rf_flattening(swot_ds, hfr_ds, hfr_ds['u'], hfr_ds['v'], features))
        flattened_data[t] = flatten_list
    return flattened_data


def concat_flattened(flattened_data: dict, training_percentage: float = 0.8, held_out: bool = False):
    X_u_list, X_v_list, y_u_list, y_v_list = [], [], [], []

    keys = list(flattened_data.keys())
    n = max(1, int(len(keys) * training_percentage))
    selected_keys = keys[n:] if held_out else keys[:n]

    for t in selected_keys:
        for entry in flattened_data[t]:
            X_u_list.append(entry["X_u"])
            X_v_list.append(entry["X_v"])
            y_u_list.append(entry["y_u"])
            y_v_list.append(entry["y_v"])

    return (
        np.concatenate(X_u_list, axis=0),
        np.concatenate(X_v_list, axis=0),
        np.concatenate(y_u_list, axis=0),
        np.concatenate(y_v_list, axis=0),
    )


def reshaping(pred: np.ndarray, X: int, Y: int) -> np.ndarray:
    return pred.reshape(X, Y)


def reshaping_to_xarray(pred: np.ndarray, swot_regridded: xr.Dataset, name: str) -> xr.DataArray:
    return xr.DataArray(
        pred,
        dims=next(iter(swot_regridded.data_vars.values())).dims,
        coords={'lat': swot_regridded['lat'], 'lon': swot_regridded['lon']},
        name=name,
    )


def plotter(rf_u, rf_v, swot_regridded: dict, hfr_interp_data: dict, flattened_data: dict,
            cycle: int, element: int, predictions: list[str]):
    from swotxai.training import predict as _predict

    key = str(cycle).zfill(3)
    df = flattened_data[key][element]['df']
    swot_regridded_cycle = swot_regridded[key][element]

    valid_mask = ~df.isna().any(axis=1)
    df_valid = df[valid_mask]

    ssv_pred_u = np.full(len(df), np.nan)
    ssv_pred_v = np.full(len(df), np.nan)

    ssv_pred_u[valid_mask] = _predict(rf_u, df_valid)
    ssv_pred_v[valid_mask] = _predict(rf_v, df_valid)

    pred_full_u = reshaping(ssv_pred_u, swot_regridded_cycle.dims['lat'], swot_regridded_cycle.dims['lon'])
    pred_full_v = reshaping(ssv_pred_v, swot_regridded_cycle.dims['lat'], swot_regridded_cycle.dims['lon'])

    pred_da_u = reshaping_to_xarray(pred_full_u, swot_regridded_cycle, predictions[0])
    pred_da_v = reshaping_to_xarray(pred_full_v, swot_regridded_cycle, predictions[1])

    swot_regridded_cycle[predictions[0]] = pred_da_u
    swot_regridded_cycle[predictions[1]] = pred_da_v
    swot_regridded_cycle[predictions[2]] = np.sqrt(pred_da_u**2 + pred_da_v**2)

    swot_regridded_cycle['gos_filtered'] = np.sqrt(
        swot_regridded_cycle['ugos_filtered']**2 + swot_regridded_cycle['vgos_filtered']**2
    )

    if 'era5_u' in swot_regridded_cycle and 'era5_v' in swot_regridded_cycle:
        swot_regridded_cycle['era5_ssv'] = np.sqrt(
            swot_regridded_cycle['era5_u']**2 + swot_regridded_cycle['era5_v']**2
        )

    hfr_interp_data_cycle = hfr_interp_data[key][element]
    if hfr_interp_data_cycle is None:
        raise ValueError(f"hfr_interp_data[{key}][{element}] is None")
    hfr_interp_data_cycle['ssv'] = np.sqrt(
        hfr_interp_data_cycle['u']**2 + hfr_interp_data_cycle['v']**2
    )

    return [swot_regridded_cycle, hfr_interp_data_cycle]


def plot_dict_assemble(start: int, end: int, u_model, v_model, predictions: list[str],
                       swot_regridded: dict, hfr_interp_data: dict, flattened_data: dict):
    ssv_plots = {}
    for data in range(start, end + 1):
        cycle_list = []
        for dataset in range(0, 2):
            try:
                cycle_list.append(plotter(
                    u_model, v_model, swot_regridded,
                    hfr_interp_data, flattened_data,
                    data, dataset, predictions,
                )[0])
                print(data, dataset, 'is done!')
            except (KeyError, ValueError, FileNotFoundError, IndexError) as e:
                print(f"Skipping {data}, dataset {dataset} -> {e}")
                continue
        ssv_plots[str(data)] = cycle_list
    return ssv_plots


def build_frame_dicts(
    u_model,
    v_model,
    swot_regridded: dict,
    hfr_interp_data: dict,
    flattened_data: dict,
    frames: list[int],
    predictions: list[str],
    passes: list[int] = None,
) -> tuple[dict, dict]:
    if passes is None:
        passes = [0, 1]

    swot_dict = {}
    hfr_dict  = {}

    for frame in frames:
        swot_dict[str(frame)] = [None] * len(passes)
        hfr_dict[str(frame)]  = [None] * len(passes)

        for idx, j in enumerate(passes):
            try:
                swot_ds, hfr_ds = plotter(
                    u_model, v_model, swot_regridded,
                    hfr_interp_data, flattened_data,
                    frame, j, predictions,
                )
                swot_dict[str(frame)][idx] = swot_ds
                hfr_dict[str(frame)][idx]  = hfr_ds
                print(f"Frame {frame}, pass {j} done.")
            except Exception as e:
                print(f"Skipping cycle {frame}, pass {j}: {type(e).__name__}: {e}", flush=True)

    return swot_dict, hfr_dict
