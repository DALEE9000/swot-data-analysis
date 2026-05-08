from __future__ import annotations

import numpy as np
import pandas as pd
from typing import Callable

from swotxai.config import SWOTConfig

ProgressCb = Callable[[str, float, str], None]


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
        try:
            ds_list = cycle_data.get(key, [])
            if j < len(ds_list) and ds_list[j] is not None:
                t = ds_list[j]["time"].values
                return pd.Timestamp(t.flat[0] if hasattr(t, "flat") else t)
        except Exception:
            pass
        return _hfr_time(cycle, j)

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
                cb("animate", 0.5, "(further skip messages suppressed...)")
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
