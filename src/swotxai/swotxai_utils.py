# Normal Python packages
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

# Plotting packages
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.geometry import box, LineString, Point
import cmocean as cm

# importing vanilla random forest packages
import sklearn
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import KNNImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score, mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, train_test_split

# Add the path to the swot library
import swot.swot_utils as swot_utils
import swot.data_loaders as data_loaders
import swot.download_swaths as download_swaths

# Turn off warnings
import warnings
warnings.filterwarnings('ignore')

# Cached FIL kwargs that work on this machine; None = not yet probed
_cuml_predict_kw: dict | None = None

# .env paths
# Path to .env is always relative to this script
env_path = Path(__file__).parent.parent
load_dotenv(env_path / ".env")

science_path = (env_path.parent / os.getenv("SCIENCE")) if os.getenv("SCIENCE") else None
calval_path  = (env_path.parent / os.getenv("CALVAL"))  if os.getenv("CALVAL")  else None
dict_path    = (env_path.parent / os.getenv("DICT"))    if os.getenv("DICT")    else None

'''
Function for loading SWOT data.

Parameters:
    path: path to the subsetted SWOT data (str),
    sw_corner: latitude and longitude of thesouthwestern corner of the domain (list),
    ne_corner: latitude and longitude of the northeastern corner of the domain (list),
    cycles_start: start cycle (int),
    cycles_end: end cycle (int),
    path_to_sph_file: path to the sph file (str),
    science: True for 21-day science phase, False for 1-day calibration repeat phase (bool),
'''
def load_swot_data(
    path: str,
    sw_corner: list,
    ne_corner: list,
    fields: list = None,
    science: bool = False):
    # Define domain
    lat_lims = [sw_corner[1],ne_corner[1]]

    # Define mission phase (1-day repeat vs science) and cycles we are interested in
    # Use sph_calval_swath for the 1-day repeats
    # Cycles 474 - 578 are from the 1-day repeat 
    # Use sph_science_swath for the 21-day repeat
    # Cycles 1 - 16 are from the science phase
    if science:
        cycles_start, cycles_end = 1, 17
        path_to_sph_file = science_path
        print(f"Using science path: {path_to_sph_file}")
    else:
        cycles_start, cycles_end = 474, 579
        path_to_sph_file = calval_path
        print(f"Using calval path: {path_to_sph_file}")

    cycles = [str(c_num).zfill(3) for c_num in range(cycles_start,cycles_end)] 
    pass_IDs_list = download_swaths.find_swaths(
                                            sw_corner=sw_corner,
                                            ne_corner=ne_corner,
                                            path_to_sph_file=path_to_sph_file)

    # Be sure to set subset as True when taking a subset of the data!
    cycle_data = {}
    for cycle in cycles:
        cycle_data[cycle] = data_loaders.load_cycle(
                                                path=path,
                                                fields=fields,
                                                cycle=cycle,
                                                pass_ids=pass_IDs_list,
                                                subset=True,lats=lat_lims)

    return cycle_data

'''
    Function for saving dictionary to disk.

    Parameters: data (dict), filename (str), directory (str | Path | None)
'''
def save_dict(
    data, 
    filename: str,
    directory: str | Path | None = None,
):
    base = Path(directory) if directory is not None else dict_path
    path = base / f"{filename}.pkl"
    if path.exists():
        raise FileExistsError(f"{path} already exists; refusing to overwrite.")
    with path.open("wb") as f:
        pickle.dump(data, f)
    print(f"Saved {path}")

'''
    Load the precomputed HFR interp rolling SWOT expert data from a pickle file.
    Parameters
    ----------
    filename : str
        Name of the pickle file (default: "hfr_interp_rolling_swot_expert.pkl").
    directory : str | Path | None
        Directory containing the file. If None, uses the current working directory.
    Returns
    -------
    Any
        The object stored in the pickle file.
'''
def load_dict(
    filename: str,
    directory: str | Path | None = None,
):
    base = Path(directory) if directory is not None else dict_path
    path = base / f"{filename}.pkl"
    if not path.exists():
        raise FileNotFoundError(f"{path} does not exist")
    with path.open("rb") as f:
        return pickle.load(f)

''' 
    Function for regridding SWOT data onto latitude/longitude coordinates.
    NOTE: For the time being, this code only seems to rename the num_lines and num_pixels dimensions to lat and lon.
    Consider refactoring to actually interpolate the SWOT data onto a freshly-constructed latitude/longitude grid.

    Parameters: swot_data (xarray dataset), features (list of variables to interpolate)
'''
def swot_regrid(swot_data: xr.Dataset, features: list | None = None):
    # Create 2D meshgrid for target
    lon2d, lat2d = swot_data['longitude'].values, swot_data['latitude'].values

    # Flatten the along-track coordinates
    lat_flat = swot_data['latitude'].values.flatten()
    lon_flat = swot_data['longitude'].values.flatten()
    points = np.column_stack((lat_flat, lon_flat))

    # Dictionary to store interpolated data
    interpolated_data = {}

    # Keep only variables whose dimensions are in num_lines and num_pixels
    grid_dims = ("num_lines", "num_pixels")

    # Features that have all dimensions
    features = [
        var for var, da in swot_data.data_vars.items()
        if da.dims == grid_dims
    ]

    for f in features:
        # Flatten variable values
        values_flat = swot_data[f].values.flatten()
        
        # Interpolate onto regular grid
        interpolated_values = griddata(points, values_flat, (lat2d, lon2d), method='linear')
        
        # Store in dictionary
        interpolated_data[f] = (('lat','lon'), interpolated_values)

    swot_regridded = xr.Dataset(
        data_vars=interpolated_data,
        coords={
            'lat': (('lat','lon'), lat2d),
            'lon': (('lat','lon'), lon2d)
        }
    )

    swot_regridded['lon'] = (swot_regridded['lon'] + 180) % 360 - 180

    # Preserve a representative overpass time as a scalar coordinate
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
    '''
    Apply swot_regrid to each xarray.Dataset in a dict of time steps.

    Parameters
    ----------
    swot_dict : dict
        Keys = timesteps (str or datetime), values = list of xarray.Dataset objects
    features : list | None
        Variables to interpolate. If None, all 2-D grid variables are used.
    coarsen_factor : int | None
        If provided, coarsen each dataset by this factor before regridding.

    Returns
    -------
    dict
        Same structure, but with regridded xarray.Datasets.
    '''
    regridded_data = {}
    
    # Outer loop over timesteps with progress bar
    for t, ds_list in tqdm(swot_dict.items(), desc="Timesteps", unit="step"):
        # Inner loop over datasets in that timestep
        regridded_data[t] = [
            swot_regrid(
                ds.coarsen(num_lines=coarsen_factor, num_pixels=coarsen_factor, boundary="trim").mean() if coarsen_factor is not None else ds,
                features
            ) 
            for ds in tqdm(
                ds_list, 
                desc=f"Regridding {t}", 
                leave=False,   # prevents too many bars staying on screen
                unit="xarray"
            ) if ds is not None
        ]
    
    return regridded_data

'''
Function for interpolating HFR data onto SWOT grid.

Parameters: hfr_data (xarray dataset), swot_regridded (xarray dataset), both in lat and lon coordinates
'''

def hfr_on_swot(hfr_data: xr.Dataset, swot_regridded: xr.Dataset):

    return hfr_data.interp(
        lat=swot_regridded['lat'],
        lon=swot_regridded['lon'],
        method='linear'
    )

'''
Take HFR data and interpolate onto each regridded SWOT xarray.Dataset.

Parameters: hfr (xarray dataset), cycle_data (dict), swot_regridded (dict)
'''

def hfr_interp(hfr: xr.Dataset, cycle_data: dict, swot_regridded: dict):
    hfr_interp_data = {}

    # Loop over keys (assuming cycle_data and swot_regridded have same keys)
    for t in tqdm(swot_regridded.keys(), desc="Timesteps", unit="step"):
        orig_ds_list = cycle_data.get(t, [])
        regrid_ds_list = swot_regridded.get(t, [])

        if not regrid_ds_list:  # skip empty entries
            continue

        interp_list = []
        for orig_ds, regrid_ds in zip(orig_ds_list, regrid_ds_list):
            if orig_ds is None or regrid_ds is None:
                continue

            # Each original dataset’s first time
            first_time = orig_ds["time"].values[0]

            # Slice HFR individually
            hfr_at_time = hfr.sel(time=first_time, method="nearest")

            # Interpolate HFR onto SWOT grid
            interp_list.append(hfr_on_swot(hfr_at_time, regrid_ds))

        hfr_interp_data[t] = interp_list

    return hfr_interp_data

'''
    Interpolate one or more external datasets onto the SWOT grid and merge
    all new variables into each SWOT snapshot.

    Parameters
    ----------
    cycle_data : dict
        Original SWOT cycle data (used for timestamps).
    swot_regridded : dict
        Regridded SWOT datasets to merge into.
    sources : list[dict]
        Each element describes one source dataset with the following keys:
          - "ds"           : xr.Dataset  (pre-processed: level selected, renamed, lon normalised)
          - "vars"         : list[str]   variables to interpolate and merge
          - "lat"          : str         lat coordinate/dimension name  (default "lat")
          - "lon"          : str         lon coordinate/dimension name  (default "lon")
          - "time"         : str         time dimension name            (default "time")
        If the lat/lon coordinates are 2-D (e.g. GOES geostationary data), scipy
        griddata is used automatically; otherwise xarray interp is used.

    Returns
    -------
    dict
        Same structure as swot_regridded with all source variables merged in.
'''
def interp_to_swot(
    cycle_data: dict,
    swot_regridded: dict,
    sources: list,
) -> dict:
    from scipy.interpolate import griddata
    from scipy.spatial import cKDTree

    def _interp_source(source_ds, vars, lat_key, lon_key, swot_ds, max_dist_deg):
        lat_coord = source_ds[lat_key]
        is_2d = lat_coord.ndim > 1

        if is_2d:
            # 2-D coords (e.g. GOES): nearest neighbour on coordinate grid,
            # preserving NaNs from the source (clouds stay NaN)
            src_lat = lat_coord.values.ravel()
            src_lon = source_ds[lon_key].values.ravel()
            swot_lat = swot_ds["lat"].values
            swot_lon = swot_ds["lon"].values
            swot_points = np.column_stack([swot_lat.ravel(), swot_lon.ravel()])

            # Build tree on all points with valid *coordinates* (not SST values)
            valid_coords = np.isfinite(src_lat) & np.isfinite(src_lon)
            if valid_coords.sum() < 2:
                return xr.Dataset({v: (list(swot_ds["lat"].dims),
                                       np.full(swot_lat.shape, np.nan, dtype=np.float32))
                                   for v in vars})

            tree = cKDTree(np.column_stack([src_lat[valid_coords], src_lon[valid_coords]]))
            dist, idx = tree.query(swot_points)
            too_far = dist.reshape(swot_lat.shape) > max_dist_deg

            result_vars = {}
            for v in vars:
                src_vals = source_ds[v].values.ravel()
                # Look up nearest value — NaN SST (cloud) propagates naturally
                interped = src_vals[valid_coords][idx].reshape(swot_lat.shape).astype(np.float32)
                interped[too_far] = np.nan
                result_vars[v] = (list(swot_ds["lat"].dims), interped)
            return xr.Dataset(result_vars)
        else:
            # 1-D coords (e.g. ERA5, HFR): use xarray interp
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

            # Get timestamp for time-slicing sources.
            # Preset runs have empty cycle_data, so fall back to the scalar
            # time coord saved by swot_regrid.
            orig_ds = orig_ds_list[idx] if idx < len(orig_ds_list) else None
            if orig_ds is not None and "time" in orig_ds:
                first_time = orig_ds["time"].values.flat[0]
            elif "time" in regrid_ds.coords:
                first_time = regrid_ds.coords["time"].values
            else:
                # No time available — skip source interpolation for this snapshot
                merged_list.append(regrid_ds)
                continue

            merged = regrid_ds
            for src in sources:
                ds            = src["ds"]
                vars          = src["vars"]
                lat_key       = src.get("lat",          "lat")
                lon_key       = src.get("lon",          "lon")
                time_key      = src.get("time",         "time")
                max_dist_deg  = src.get("max_dist_deg", 0.1)
                save_time_as  = src.get("save_time_as", None)

                ds_at_time  = ds.sel({time_key: first_time}, method="nearest")
                actual_time = ds_at_time[time_key].values  # scalar nearest-match time
                interped    = _interp_source(ds_at_time, vars, lat_key, lon_key, merged, max_dist_deg)
                # Drop time coord to avoid conflicting with SWOT's scalar time.
                if "time" in interped.coords:
                    interped = interped.drop_vars("time")
                # Optionally re-save the source's nearest-match time under a custom name.
                if save_time_as is not None:
                    interped = interped.assign_coords({save_time_as: actual_time})
                merged = xr.merge([merged, interped])

            merged_list.append(merged)
        result[t] = merged_list

    return result

'''
    Function for flattening the SWOT and HFR data for a single snapshot for Random Forest training.

    Parameters: 
        swot_regridded (xarray dataset), 
        hfr_on_swot (xarray dataset), 
        target_u (xarray dataarray), 
        target_v (xarray dataarray), 
        features (list of variables to feed into random forest model)
'''

def rf_flattening(swot_regridded: xr.Dataset, hfr_on_swot: xr.Dataset, target_u: xr.Dataset, target_v: xr.Dataset, features: list):
    # Flatten each feature
    data_dict = {f: swot_regridded[f].values.flatten() for f in features}

    # Also flatten the target variable
    target_u = target_u.values.flatten()
    target_v = target_v.values.flatten()

    # Convert to pandas DataFrame
    df = pd.DataFrame(data_dict)
    y_u = pd.Series(target_u, name='u')
    y_v = pd.Series(target_v, name='v')

    # 3. Drop NaNs in the target
    mask_u = ~y_u.isna()
    mask_v = ~y_v.isna()
    return {
        "df": df,
        "X_u": df[mask_u].reset_index(drop=True),
        "X_v": df[mask_v].reset_index(drop=True),
        "y_u": y_u[mask_u].reset_index(drop=True),
        "y_v": y_v[mask_v].reset_index(drop=True)
    }

'''
    Function for flattening the SWOT and HFR data for a single snapshot forRandom Forest training with a stencil.
    
    Parameters: 
        swot_regridded (xarray dataset), 
        target_u (xarray dataarray), 
        target_v (xarray dataarray), 
        features (list),
        k (int)
'''
def rf_flattening_stencil(
    swot_regridded: xr.Dataset,
    target_u: xr.DataArray,
    target_v: xr.DataArray,
    features: list,
    k: int = 3,
):
    from numpy.lib.stride_tricks import sliding_window_view

    pad = k // 2

    # Stack features → (n_features, lat, lon), pad edges to preserve shape
    feature_stack = np.stack(
        [swot_regridded[f].values for f in features], axis=0
    )
    feature_stack = np.pad(
        feature_stack,
        ((0, 0), (pad, pad), (pad, pad)),
        mode='constant', constant_values=np.nan
    )

    # sliding_window_view: (n_features, lat, lon, k, k) — zero-copy view
    windows = sliding_window_view(feature_stack, (k, k), axis=(1, 2))

    # Reshape to (lat*lon, n_features*k*k)
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
        "y_v": y_v[mask_v]
    }

'''
    Flatten the HFR data and the regridded SWOT data in assembling the entire flattened data dictionary.

    Parameters: hfr_interp_data (dict), swot_regridded (dict), features (list), n_jobs (int)
'''
def flattening(
    hfr_interp_data: dict,
    swot_regridded: dict,
    features: list,
    stenciling: bool = True,
    k: int = 3,
    n_jobs: int = -1
):
    flattened_data = {}

    for t in tqdm(swot_regridded.keys(), desc="Timesteps", unit="step"):
        hfr_ds_list = hfr_interp_data.get(t, [])
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

'''
    Concatenate X_u, X_v, y_u, y_v arrays from the first 80% of entries
    in each timestep in flattened_data.

    Parameters
    ----------
    flattened_data : dict
        Output from flattening(), dict of timesteps -> list of dicts.

    Returns
    -------
    X_u_all, X_v_all, y_u_all, y_v_all : np.ndarray
        Concatenated arrays along axis 0.
'''
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

    # Concatenate along axis 0
    X_u_all = np.concatenate(X_u_list, axis=0)
    X_v_all = np.concatenate(X_v_list, axis=0)
    y_u_all = np.concatenate(y_u_list, axis=0)
    y_v_all = np.concatenate(y_v_list, axis=0)

    return X_u_all, X_v_all, y_u_all, y_v_all

'''
    Function for training the Random Forest model.

    Parameters: 
        X (pandas dataframe), 
        y (pandas series), 
        n_estimators (int), 
        max_depth (int), 
        random_state (int), 
        n_jobs (int)
'''
def random_forest(X: pd.DataFrame, y: pd.Series, n_estimators: int, max_depth: int, random_state: int, n_jobs: int, use_gpu: bool = True, use_lgbm: bool = False):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=random_state)

    if use_lgbm:
        import lightgbm as lgb
        rf = lgb.LGBMRegressor(
            device="cuda", n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state, n_jobs=n_jobs, verbose=-1,
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")
        return rf

    _use_cuml = False
    if use_gpu:
        try:
            from cuml.ensemble import RandomForestRegressor as cuRF
            _use_cuml = True
        except ImportError:
            print("cuML not available — falling back to sklearn.")

    if _use_cuml:
        X_train_f32 = np.asarray(X_train, dtype="float32")
        y_train_f32 = np.asarray(y_train, dtype="float32")
        rf = cuRF(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state,
                  max_features=1.0, n_bins=256)
        rf.fit(X_train_f32, y_train_f32)
        # Capture importances now — cuML loses them after joblib/pickle serialization
        try:
            rf._feature_importances_saved = np.asarray(rf.feature_importances_)
        except Exception:
            rf._feature_importances_saved = None
        y_pred = np.asarray(rf.predict(np.asarray(X_test, dtype="float32")))
    else:
        rf = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state, n_jobs=n_jobs)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    print(f"RMSE: {rmse:.4f}, R²: {r2:.4f}")
    return rf

'''
    Function for reshaping flattened data.

    Parameters: 
        pred (pandas series), 
        X (int), 
        Y (int), where X and Y are the dimensions of the original data
'''

def reshaping(pred: pd.Series, X: int, Y: int):
    return pred.reshape(X, Y)

'''
    Function to create xarray dataset from reshaped data.

    Parameters: pred (np.ndarray), swot_regridded: xr.Dataset, name: str
'''

def reshaping_to_xarray(pred: np.ndarray, swot_regridded: xr.Dataset, name: str):
    return xr.DataArray(
        pred,
        dims=next(iter(swot_regridded.data_vars.values())).dims,
        coords={
            'lat': swot_regridded['lat'],
            'lon': swot_regridded['lon']
        },
        name=name
    )

'''
    Function for plotting the random forest predictions.

    Parameters: 
        rf_u (sklearn.ensemble._forest.RandomForestRegressor), 
        rf_v (sklearn.ensemble._forest.RandomForestRegressor), 
        swot_regridded (dict), 
        hfr_interp_data (dict), 
        flattened_data (dict), 
        cycle (int), 
        element (int), 
        predictions (list[str])
'''
def plotter(
        rf_u: sklearn.ensemble._forest.RandomForestRegressor, 
        rf_v: sklearn.ensemble._forest.RandomForestRegressor, 
        swot_regridded: dict, 
        hfr_interp_data: dict, 
        flattened_data: dict, 
        cycle: int, 
        element: int, 
        predictions: list[str]
    ):
    key = str(cycle).zfill(3)
    df = flattened_data[key][element]['df']
    swot_regridded_cycle = swot_regridded[key][element]

    valid_mask = ~df.isna().any(axis=1)
    df_valid = df[valid_mask]

    ssv_pred_u = np.full(len(df), np.nan)  # initialize with NaNs
    ssv_pred_v = np.full(len(df), np.nan)

    try:
        from cuml.ensemble import RandomForestRegressor as cuRF
        _is_cuml = isinstance(rf_u, cuRF)
    except ImportError:
        _is_cuml = False

    if _is_cuml:
        global _cuml_predict_kw
        X_valid = np.asarray(df_valid, dtype="float32")
        # Probe FIL layouts once; cache the first one that works.
        # sparse8 / sparse use a different CUDA kernel that avoids the
        # block-size limit hit by large forests with the default dense layout.
        _attempts = (
            [_cuml_predict_kw] if _cuml_predict_kw is not None
            else [{}, {"layout": "sparse8"}, {"layout": "sparse"}, {"default_chunk_size": 1}]
        )
        pred_u = pred_v = None
        for _kw in _attempts:
            try:
                pred_u = np.asarray(rf_u.predict(X_valid, **_kw))
                pred_v = np.asarray(rf_v.predict(X_valid, **_kw))
                _cuml_predict_kw = _kw
                break
            except (RuntimeError, TypeError):
                continue
        if pred_u is None:
            raise RuntimeError(
                "All cuML FIL configurations failed (invalid configuration argument). "
                "Retrain with sklearn or a compatible cuML version."
            )
        ssv_pred_u[valid_mask] = pred_u
        ssv_pred_v[valid_mask] = pred_v
    else:
        ssv_pred_u[valid_mask] = rf_u.predict(df_valid)
        ssv_pred_v[valid_mask] = rf_v.predict(df_valid)

    # Figure out shape of everything here
    pred_full_u = reshaping(ssv_pred_u, swot_regridded_cycle.dims['lat'], swot_regridded_cycle.dims['lon'])
    pred_full_v = reshaping(ssv_pred_v, swot_regridded_cycle.dims['lat'], swot_regridded_cycle.dims['lon'])

    pred_da_u = reshaping_to_xarray(pred_full_u, swot_regridded_cycle, predictions[0])
    pred_da_v = reshaping_to_xarray(pred_full_v, swot_regridded_cycle, predictions[1])

    swot_regridded_cycle[predictions[0]] = pred_da_u
    swot_regridded_cycle[predictions[1]] = pred_da_v
    swot_regridded_cycle[predictions[2]] = np.sqrt(pred_da_u**2 + pred_da_v**2)

    # Calculate geostrophic velocity magnitude for SWOT
    swot_regridded_cycle['gos_filtered'] = np.sqrt(swot_regridded_cycle['ugos_filtered']**2 + swot_regridded_cycle['vgos_filtered']**2)

    # Calculate ERA5 wind speed magnitude
    if 'era5_u' in swot_regridded_cycle and 'era5_v' in swot_regridded_cycle:
        swot_regridded_cycle['era5_ssv'] = np.sqrt(swot_regridded_cycle['era5_u']**2 + swot_regridded_cycle['era5_v']**2)

    # Calculate velocity magnitude for HFR
    hfr_interp_data_cycle = hfr_interp_data[key][element]
    if hfr_interp_data_cycle is None:
        raise ValueError(f"hfr_interp_data[{key}][{element}] is None")
    hfr_interp_data_cycle['ssv'] = np.sqrt(hfr_interp_data_cycle['u']**2 + hfr_interp_data_cycle['v']**2)

    return [swot_regridded_cycle, hfr_interp_data_cycle]

'''
    Function for assembling the dictionary of plots.
    
    Parameters: 
        start (int), 
        end (int), 
        u_model (sklearn.ensemble._forest.RandomForestRegressor), 
        v_model (sklearn.ensemble._forest.RandomForestRegressor), 
        predictions (list[str]),
        swot_regridded (dict),
        hfr_interp_data (dict),
        flattened_data (dict)
'''
def plot_dict_assemble(
    start: int, 
    end: int, 
    u_model: sklearn.ensemble._forest.RandomForestRegressor, 
    v_model: sklearn.ensemble._forest.RandomForestRegressor, 
    predictions: list[str],
    swot_regridded: dict,
    hfr_interp_data: dict,
    flattened_data: dict
):
    ssv_plots = {}
    for data in range(start, end + 1):
        cycle_list = []
        for dataset in range(0, 2):
            try:
                cycle_list.append(plotter(
                    u_model, v_model, swot_regridded,
                    hfr_interp_data, flattened_data,
                    data, dataset, predictions
                )[0])
                print(data, dataset, 'is done!')
            except (KeyError, ValueError, FileNotFoundError, IndexError) as e:
                # skip if data/dataset not found or incompatible
                print(f"Skipping {data}, dataset {dataset} -> {e}")
                continue
        ssv_plots[str(data)] = cycle_list
    return ssv_plots

def build_frame_dicts(
    u_model: sklearn.ensemble._forest.RandomForestRegressor,
    v_model: sklearn.ensemble._forest.RandomForestRegressor,
    swot_regridded: dict,
    hfr_interp_data: dict,
    flattened_data: dict,
    frames: list[int],
    predictions: list[str],
    passes: list[int] = None,
) -> tuple[dict, dict]:
    '''
    Build swot_dict and hfr_dict for a list of cycle frames and passes.

    Returns (swot_dict, hfr_dict), each keyed by str(cycle) with a list
    of datasets per pass (None where data is unavailable).
    '''
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