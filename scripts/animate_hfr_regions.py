#!/usr/bin/env python
"""Daily-mean surface-current animation for every HFR region in the archive.

Reads the hourly archives under HFR_DATA_ROOT (US HFRNet 6 km, European node
and IMOS ACORN folders written by scripts/hfr_convert_networks.py), averages
each calendar day, and renders one frame per day with data: speed as a
sequential colour field, arrows subsampled to ~30 across the grid, coastline
for orientation. Frames are PNGs under archive/hfr_anim_frames/<region>/
(gitignored, skipped when present, so the job survives hibernation); the
finished MP4 goes to docs/figures/hfr_animations/<region>_daily.mp4.

Usage:
    python scripts/animate_hfr_regions.py                      # every region
    python scripts/animate_hfr_regions.py --regions uswc,au_rot
    python scripts/animate_hfr_regions.py --assemble-only      # frames already rendered
    python scripts/animate_hfr_regions.py --coverage           # markdown table of day coverage per region from the daily cache (HFR_ANIM_ROOT)
"""
from __future__ import annotations

import argparse
import os
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import swotxai  # noqa: F401  (torch/BLAS import order)
import numpy as np
import xarray as xr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cmocean
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

from swotxai.presets import REGION_META

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("HFR_DATA_ROOT", r"C:\Users\david\Documents\columbiaocean\HFR\Code\Data"))
FRAMES = REPO / "archive" / "hfr_anim_frames"
OUT = REPO / "docs" / "figures" / "hfr_animations"
VMAX = 1.0        # m/s; colour scale shared by every region so they compare
DPI = 200         # 1600 px wide
plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10", "Computer Modern Roman", "STIXGeneral"],
                     "mathtext.fontset": "cm", "axes.unicode_minus": False, "font.size": 11})
HOURLY_FILE = re.compile(r"(\d{8})T\d{6}\.nc4?$")   # one-file-per-hour archives (ushi, glna, prvi)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def archive_folder(rid: str) -> Path | None:
    label, path, *_ = REGION_META[rid]
    folder = DATA_ROOT / (path if "/" not in path else f"{rid}_6km_Resolution_hourly")
    return folder if folder.is_dir() else None


def daily_fields(folder: Path, files: list[Path] | None = None):
    """Yield (day, u, v, lat, lon) daily means in file order. Multi-day files are resampled;
    one-file-per-hour archives are grouped by the date in the filename. `files` restricts
    the archive to a subset (animate_hfr_swot.py builds its daily cache one year at a time)."""
    if files is None:
        files = sorted(p for p in folder.iterdir() if p.suffix in (".nc", ".nc4"))
    groups: dict[str, list[Path]] = defaultdict(list)
    for p in files:
        m = HOURLY_FILE.search(p.name)
        groups[m.group(1) if m else p.name].append(p)
    for key, fs in groups.items():
        try:
            if len(fs) == 1 and not HOURLY_FILE.search(fs[0].name):
                # multi-day file: read one day at a time (a gak file is 9,116 h x 223 x 413, 3 GB per variable)
                with xr.open_dataset(fs[0]) as ds:
                    days = np.unique(ds["time"].values.astype("datetime64[D]"))
                    for d in days:
                        sub = ds[["u", "v"]].sel(time=slice(d, d + np.timedelta64(1, "D") - np.timedelta64(1, "s"))).load()
                        u = sub["u"].mean("time", skipna=True).values
                        if np.isfinite(u).any():
                            yield str(d), u, sub["v"].mean("time", skipna=True).values, ds["lat"].values, ds["lon"].values
            else:
                ds = xr.concat([xr.open_dataset(f)[["u", "v"]].load() for f in fs], "time")
                u = ds["u"].mean("time", skipna=True).values
                if np.isfinite(u).any():
                    yield f"{key[:4]}-{key[4:6]}-{key[6:]}", u, ds["v"].mean("time", skipna=True).values, ds["lat"].values, ds["lon"].values
        except Exception as e:  # noqa: BLE001
            log(f"  skip {key}: {type(e).__name__}: {e}")


class Renderer:
    """One figure per region; frames update the artists instead of redrawing."""

    def __init__(self, rid: str, lat: np.ndarray, lon: np.ndarray):
        self.label = REGION_META[rid][0]
        aspect = np.ptp(lat) / np.ptp(lon)   # PlateCarree: degrees are square
        h = min(14.0, 0.78 * 8 * aspect / 0.86 + 0.5)       # map fills the axes box
        h = int(np.ceil(h * DPI / 16)) * 16 / DPI             # pixel height divisible by 16 (ffmpeg macro block)
        self.fig = plt.figure(figsize=(8, h))
        self.ax = self.fig.add_axes([0.07, 0.06, 0.78, 0.86], projection=ccrs.PlateCarree())
        self.ax.set_extent([lon.min(), lon.max(), lat.min(), lat.max()])
        self.ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e8e4dc", zorder=0)
        self.ax.coastlines("50m", linewidth=0.5, color="#555")
        gl = self.ax.gridlines(draw_labels=True, linewidth=0.3, color="#bbb")
        gl.top_labels = gl.right_labels = False
        deg = r"$^{\circ}$"                                  # cmr10 has no degree glyph; mathtext does
        gl.xformatter = LongitudeFormatter(degree_symbol=deg)
        gl.yformatter = LatitudeFormatter(degree_symbol=deg)
        z = np.full((lat.size, lon.size), np.nan, "float32")
        self.qm = self.ax.pcolormesh(lon, lat, z, cmap=cmocean.cm.speed, vmin=0, vmax=VMAX,
                                     transform=ccrs.PlateCarree(), zorder=1)
        cax = self.fig.add_axes([0.87, 0.15, 0.025, 0.68])
        self.fig.colorbar(self.qm, cax=cax, label="daily-mean speed (m/s)")
        s = max(1, int(np.ceil(max(lat.size, lon.size) / 30)))
        self.s = s
        LO, LA = np.meshgrid(lon[::s], lat[::s])
        self.Q = self.ax.quiver(LO, LA, z[::s, ::s], z[::s, ::s], scale=6, scale_units="height",
                                width=0.003, color="#111", transform=ccrs.PlateCarree(), zorder=2)
        self.title = self.ax.set_title("", loc="left", fontsize=11)

    def frame(self, day: str, u: np.ndarray, v: np.ndarray, out: Path) -> None:
        spd = np.hypot(u, v)
        self.qm.set_array(spd.ravel())
        s = self.s
        self.Q.set_UVC(u[::s, ::s], v[::s, ::s])
        n = int(np.isfinite(spd).sum())
        self.title.set_text(f"{self.label}, {day}, {n:,} cells")
        self.fig.savefig(out, dpi=DPI)

    def close(self) -> None:
        plt.close(self.fig)


def render(rid: str) -> list[Path]:
    folder = archive_folder(rid)
    if folder is None:
        log(f"{rid}: no archive folder, skipped")
        return []
    fdir = FRAMES / rid
    fdir.mkdir(parents=True, exist_ok=True)
    have = {p.stem for p in fdir.glob("*.png")}
    rend = None
    n_new = 0
    t0 = time.time()
    for day, u, v, lat, lon in daily_fields(folder):
        if day in have:
            continue
        if rend is None:
            rend = Renderer(rid, lat, lon)
        rend.frame(day, u, v, fdir / f"{day}.png")
        n_new += 1
        if n_new % 200 == 0:
            log(f"{rid}: {n_new} new frames, last {day}, {(time.time() - t0) / n_new:.2f} s/frame")
    if rend:
        rend.close()
    frames = sorted(fdir.glob("*.png"))
    log(f"{rid}: {n_new} new, {len(frames)} frames total ({frames[0].stem} .. {frames[-1].stem})" if frames else f"{rid}: no data")
    return frames


def assemble(frames: list[Path], out: Path, fps: int) -> None:
    import imageio.v2 as imageio
    # quality=10 would be lossless x264, which uses the High 4:4:4 profile that Windows players refuse;
    # CRF 18 in the High profile is visually lossless and plays everywhere
    with imageio.get_writer(out, fps=fps, codec="libx264", quality=None, pixelformat="yuv420p",
                            output_params=["-crf", "18", "-profile:v", "high"]) as w:
        for f in frames:
            w.append_data(imageio.imread(f))
    log(f"{out.name}: {len(frames)} frames, {out.stat().st_size / 1e6:.0f} MB")


def coverage() -> None:
    """Days with data per region, from the daily-mean cache that animate_hfr_swot.py keeps under
    HFR_ANIM_ROOT/hfr_daily/<region>/<year>.nc (one time step per day with any finite cell)."""
    import datetime as dt
    import itertools
    daily = Path(os.environ.get("HFR_ANIM_ROOT", REPO / "archive" / "hfr_anim")) / "hfr_daily"
    print("| Region | First | Last | Days in span | Days with data | Missing | Gaps of 7 days or more |")
    print("|---|---|---|---|---|---|---|")
    for rid in REGION_META:
        days = []
        for f in sorted((daily / rid).glob("*.nc")):
            with xr.open_dataset(f) as ds:
                days += [dt.date.fromisoformat(str(t)) for t in ds["time"].values.astype("datetime64[D]")]
        days.sort()
        if not days:
            continue
        span = (days[-1] - days[0]).days + 1
        have = set(days)
        miss = [days[0] + dt.timedelta(i) for i in range(span) if days[0] + dt.timedelta(i) not in have]
        gaps = []
        for _, g in itertools.groupby(enumerate(miss), lambda x: x[1] - dt.timedelta(x[0])):
            g = [x[1] for x in g]
            if len(g) >= 7:
                gaps.append(f"{g[0]} to {g[-1]} ({len(g)})")
        print(f"| {rid} | {days[0]} | {days[-1]} | {span:,} | {len(days):,} | {len(miss):,} | {'; '.join(gaps)} |")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regions", default="all", help="comma list of REGION_META ids, or all")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--coverage", action="store_true", help="print the day-coverage table and exit")
    a = ap.parse_args()
    if a.coverage:
        coverage()
        return
    regions = [r for r in REGION_META if archive_folder(r)] if a.regions == "all" else a.regions.split(",")
    OUT.mkdir(parents=True, exist_ok=True)
    for rid in regions:
        frames = sorted((FRAMES / rid).glob("*.png")) if a.assemble_only else render(rid)
        if frames:
            assemble(frames, OUT / f"{rid}_daily.mp4", a.fps)
    log("done")


if __name__ == "__main__":
    main()
