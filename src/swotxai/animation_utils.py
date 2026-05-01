from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd
import xarray as xr
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from tqdm import tqdm

DARK_BG = "#2e2e2e"
DARK_FG = "#e0e0e0"
COAST_COLOR = "#aaaaaa"
GRID_COLOR = "#666666"

WHITE_BG = "#ffffff"
WHITE_FG = "#111111"
WHITE_COAST = "#333333"
WHITE_GRID = "#aaaaaa"


def build_goes_index(goes_dir: str | Path) -> tuple[dict, list]:
    """Build a time -> filepath index for GOES SST files."""
    goes_dir = Path(goes_dir)
    all_files = sorted(goes_dir.glob("**/*.nc"))

    def _parse_time(filepath):
        s_part = [p for p in Path(filepath).stem.split("_") if p.startswith("s")][0]
        return datetime.strptime(s_part[1:14], "%Y%j%H%M%S")

    index = {_parse_time(f): f for f in all_files}
    return index, sorted(index.keys())


def build_goes_projection(goes_file: str | Path) -> tuple:
    """Extract geostationary projection info from a GOES file."""
    ds = xr.open_dataset(goes_file, engine="netcdf4")
    h    = float(ds["goes_imager_projection"].perspective_point_height)
    lon0 = float(ds["goes_imager_projection"].longitude_of_projection_origin)
    extent_geos = [
        float(ds.x.min()) * h, float(ds.x.max()) * h,
        float(ds.y.min()) * h, float(ds.y.max()) * h,
    ]
    geos_crs = ccrs.Geostationary(central_longitude=lon0, satellite_height=h, sweep_axis="x")
    ds.close()
    return h, lon0, extent_geos, geos_crs


def from_cycle_dict(data_dict: dict, var: str, n_passes: int = 2) -> Callable:
    """Build a data_fn from a cycle-keyed dict (e.g. swot_dict, hfr_dict)."""
    def data_fn(cycle: int, pass_idx: int):
        entries = data_dict.get(str(cycle), [None] * n_passes)
        if pass_idx >= len(entries) or entries[pass_idx] is None:
            return None
        ds = entries[pass_idx]
        return ds[var] if var in ds else None
    return data_fn


def _is_empty(data) -> bool:
    try:
        vals = data.values if hasattr(data, "values") else np.asarray(data)
        return vals.size == 0 or bool(np.all(np.isnan(vals)))
    except Exception:
        return True


def _has_data_in_bounds(data, lon_bounds, lat_bounds, x_coord="lon", y_coord="lat", min_valid=30) -> bool:
    try:
        vals = np.asarray(data.values if hasattr(data, "values") else data, dtype=float)
        lon_c = getattr(data, x_coord, None)
        lat_c = getattr(data, y_coord, None)
        if lon_c is None or lat_c is None:
            return int(np.sum(~np.isnan(vals))) >= min_valid
        lon = np.asarray(lon_c.values if hasattr(lon_c, "values") else lon_c, dtype=float)
        lat = np.asarray(lat_c.values if hasattr(lat_c, "values") else lat_c, dtype=float)
        mask = (
            (lon >= lon_bounds[0]) & (lon <= lon_bounds[1]) &
            (lat >= lat_bounds[0]) & (lat <= lat_bounds[1]) &
            ~np.isnan(vals)
        )
        return int(mask.sum()) >= min_valid
    except Exception:
        return False


def generate_frames(
    panels: list[dict],
    frames: list[int],
    frame_dir: str | Path,
    lon_bounds: list[float],
    lat_bounds: list[float],
    passes: list[int] = None,
    title_fn: Callable[[int, int], str] | None = None,
    figsize: tuple = (14, 4.5),
    dpi: int = 150,
    wspace: float = 0.03,
    log_fn: Callable[[str], None] | None = None,
) -> list[str]:
    """
    Generate animation frames as PNG files on a dark grey background.

    Parameters
    ----------
    panels : list[dict]
        Each dict describes one subplot. Required fields:
        "title", "data_fn", "cmap", "vmin", "vmax", "colorbar_label".
        Optional: "plot_type" ("pcolormesh" | "imshow"), "x", "y", "transform",
        "extent", "projection".

    frames : list[int]
        Cycle numbers to iterate over.

    frame_dir : str | Path
        Directory to save PNG frames.

    lon_bounds, lat_bounds : list[float]
        Map extent applied to all panels.

    passes : list[int] | None
        Which pass indices to generate per cycle. Defaults to [0, 1].

    title_fn : Callable[[int, int], str] | None
        Called as title_fn(cycle, pass_idx) to get the full suptitle string.
        If None, falls back to "Cycle {cycle}, Pass {pass_idx}".

    figsize, dpi, wspace : layout options.

    Returns
    -------
    list[str]  — paths to generated PNG files.
    """
    if passes is None:
        passes = [0, 1]
    _log = log_fn if log_fn is not None else print

    frame_dir = Path(frame_dir)
    if frame_dir.exists():
        for old in frame_dir.glob("*.png"):
            old.unlink()
    frame_dir.mkdir(parents=True, exist_ok=True)
    frame_files = []
    first_white_done: set[int] = set()  # tracks which pass indices have their white preview

    def _render_fig(panel_data, frame, j, bg, fg, coast, grid):
        fig = plt.figure(figsize=figsize, facecolor=bg)
        n_panels = len(panels)
        axes = [
            fig.add_subplot(
                1, n_panels, k + 1,
                projection=panel.get("projection", ccrs.PlateCarree()),
            )
            for k, panel in enumerate(panels)
        ]

        for k, (ax, panel, data) in enumerate(zip(axes, panels, panel_data)):
            ax.set_facecolor(bg)
            plot_type = panel.get("plot_type", "pcolormesh")
            cmap      = panel.get("cmap", "viridis")
            vmin      = panel.get("vmin", None)
            vmax      = panel.get("vmax", None)
            cb_label  = panel.get("colorbar_label", "")
            is_left   = (k == 0)

            ax.set_extent(
                [lon_bounds[0], lon_bounds[1], lat_bounds[0], lat_bounds[1]],
                crs=ccrs.PlateCarree(),
            )

            if data is None:
                im = None
            elif plot_type == "pcolormesh":
                xc = panel.get("x", "lon")
                yc = panel.get("y", "lat")
                x_flat = getattr(data, xc).values.ravel()
                y_flat = getattr(data, yc).values.ravel()
                z_flat = data.values.ravel().astype(float)
                valid  = ~np.isnan(z_flat) & ~np.isnan(x_flat) & ~np.isnan(y_flat)
                n_valid = int(valid.sum())
                _log(
                    f"  [cycle {frame} pass {j} panel {k}] valid={n_valid}/{len(valid)}, "
                    f"x=[{x_flat[valid].min():.2f},{x_flat[valid].max():.2f}]"
                    if n_valid > 0 else
                    f"  [cycle {frame} pass {j} panel {k}] ALL NaN"
                )
                im = ax.scatter(
                    x_flat[valid], y_flat[valid], c=z_flat[valid],
                    s=2, cmap=cmap,
                    transform=panel.get("transform", ccrs.PlateCarree()),
                    vmin=vmin, vmax=vmax,
                    linewidths=0,
                )
            elif plot_type == "imshow":
                values = data if isinstance(data, np.ndarray) else data.values
                im = ax.imshow(
                    values,
                    origin="upper",
                    extent=panel["extent"],
                    transform=panel["transform"],
                    cmap=cmap, vmin=vmin, vmax=vmax,
                    interpolation="bilinear",
                )
            else:
                raise ValueError(f"Unknown plot_type: {plot_type!r}. Use 'pcolormesh' or 'imshow'.")

            if im is not None:
                cbar = fig.colorbar(im, ax=ax, orientation="vertical", label=cb_label,
                                    shrink=0.6, pad=0.02, aspect=25)
                cbar.ax.yaxis.label.set_color(fg)
                cbar.ax.tick_params(colors=fg)
                cbar.outline.set_edgecolor(fg)

            panel_time_fn = panel.get("time_fn")
            panel_title = panel["title"]
            if panel_time_fn is not None:
                try:
                    ts = panel_time_fn(frame, j)
                    if ts is not None:
                        panel_title = f"{panel_title}\n{ts.strftime('%Y-%m-%d %H:%M:%S UTC')}"
                except Exception:
                    pass
            ax.set_title(panel_title, fontsize=8, color=fg)
            ax.add_feature(cfeature.COASTLINE, edgecolor=coast)
            ax.add_feature(cfeature.BORDERS, linestyle=":", edgecolor=coast)
            gl = ax.gridlines(
                draw_labels=is_left, linewidth=0.5,
                color=grid, alpha=0.7, linestyle="--",
            )
            gl.top_labels   = False
            gl.right_labels = False
            if is_left:
                gl.xlabel_style = {"color": fg, "size": 7}
                gl.ylabel_style = {"color": fg, "size": 7}

        suptitle = title_fn(frame, j) if title_fn is not None else f"Cycle {frame}, Pass {j}"
        fig.subplots_adjust(top=0.88, wspace=wspace)
        fig.suptitle(suptitle, fontsize=10, color=fg)
        return fig

    for frame in tqdm(frames, desc="Frames"):
        for j in passes:
            try:
                panel_data_raw = [panel["data_fn"](frame, j) for panel in panels]
                panel_data = []
                for d in panel_data_raw:
                    if d is None:
                        panel_data.append(None)
                    elif hasattr(d, "load"):
                        panel_data.append(d.load())
                    elif hasattr(d, "compute"):
                        panel_data.append(d.compute())
                    else:
                        panel_data.append(d)

                primary_data  = panel_data[0]
                primary_panel = panels[0]
                x_coord = primary_panel.get("x", "lon")
                y_coord = primary_panel.get("y", "lat")
                if primary_data is None:
                    _log(f"Skipping cycle {frame}, pass {j}: primary data_fn returned None.")
                    continue
                if _is_empty(primary_data):
                    _log(f"Skipping cycle {frame}, pass {j}: primary data is empty/all-NaN.")
                    continue
                if not _has_data_in_bounds(primary_data, lon_bounds, lat_bounds, x_coord, y_coord):
                    try:
                        vals = np.asarray(primary_data.values if hasattr(primary_data, "values") else primary_data, dtype=float)
                        n_valid = int(np.sum(~np.isnan(vals)))
                        lon_c = getattr(primary_data, x_coord, None)
                        lat_c = getattr(primary_data, y_coord, None)
                        lon_range = f"[{float(np.nanmin(lon_c.values)):.2f}, {float(np.nanmax(lon_c.values)):.2f}]" if lon_c is not None else "N/A"
                        lat_range = f"[{float(np.nanmin(lat_c.values)):.2f}, {float(np.nanmax(lat_c.values)):.2f}]" if lat_c is not None else "N/A"
                        _log(f"Skipping cycle {frame}, pass {j}: bounds check failed. "
                             f"n_valid={n_valid}, lon={lon_range} (want {lon_bounds}), lat={lat_range} (want {lat_bounds})")
                    except Exception as diag_e:
                        _log(f"Skipping cycle {frame}, pass {j}: bounds check failed (diag error: {diag_e})")
                    continue

                frame_file = str(frame_dir / f"cycle_{frame:03d}_pass_{j}.png")

                # Dark frame (always)
                fig = _render_fig(panel_data, frame, j, DARK_BG, DARK_FG, COAST_COLOR, GRID_COLOR)
                fig.savefig(frame_file, dpi=dpi, facecolor=DARK_BG)
                plt.close(fig)
                frame_files.append(frame_file)

                # White preview (first valid frame per pass only)
                if j not in first_white_done:
                    first_white_done.add(j)
                    fig_w = _render_fig(panel_data, frame, j, WHITE_BG, WHITE_FG, WHITE_COAST, WHITE_GRID)
                    white_file = str(frame_dir / f"cycle_{frame:03d}_pass_{j}_white.png")
                    fig_w.savefig(white_file, dpi=dpi, facecolor=WHITE_BG)
                    plt.close(fig_w)
                    _log(f"  White preview saved: {white_file}")

            except (KeyError, ValueError, FileNotFoundError, IndexError, AttributeError) as e:
                _log(f"Skipping cycle {frame}, pass {j}: {type(e).__name__}: {e}")
                plt.close("all")
                continue

    _log(f"Frame generation complete. {len(frame_files)} frames saved to {frame_dir}.")
    return frame_files


def assemble_animations_by_pass(
    frame_files: list[str],
    output_stem: str | Path,
    fps: int = 4,
) -> dict[str, Path]:
    """
    Split frames by pass number and assemble one MP4 per pass.

    Looks for 'pass_N' in each filename to group frames. Returns a dict
    keyed by pass label (e.g. "pass_0", "pass_1") mapping to output Path.
    """
    by_pass: dict[str, list[str]] = defaultdict(list)
    for f in frame_files:
        m = re.search(r"pass_(\d+)", Path(f).stem)
        key = f"pass_{m.group(1)}" if m else "pass_0"
        by_pass[key].append(f)

    output_stem = Path(output_stem)
    output_stem.parent.mkdir(parents=True, exist_ok=True)

    paths: dict[str, Path] = {}
    for pass_key, files in sorted(by_pass.items()):
        out = Path(f"{output_stem}_{pass_key}.mp4")
        assemble_animation(sorted(files), out, fps=fps)
        paths[pass_key] = out

    return paths


def assemble_animation(
    frame_files: list[str],
    output_path: str | Path,
    fps: int = 4,
) -> Path:
    """
    Assemble PNG frames into a GIF or MP4.

    Requires: pip install imageio[ffmpeg]
    """
    try:
        import imageio
    except ImportError:
        raise ImportError("Run: pip install imageio[ffmpeg]")

    output_path = Path(output_path)
    writer_kwargs = {"fps": fps}
    if output_path.suffix == ".mp4":
        writer_kwargs["codec"]   = "libx264"
        writer_kwargs["quality"] = 8

    from PIL import Image as _PILImage
    target = imageio.imread(frame_files[0])
    h, w = target.shape[:2]

    with imageio.get_writer(output_path, **writer_kwargs) as writer:
        for f in tqdm(frame_files, desc="Assembling animation"):
            img = imageio.imread(f)
            if img.shape[:2] != (h, w):
                img = np.array(_PILImage.fromarray(img).resize((w, h), _PILImage.LANCZOS))
            writer.append_data(img)

    print(f"Animation saved to {output_path}")
    return output_path
