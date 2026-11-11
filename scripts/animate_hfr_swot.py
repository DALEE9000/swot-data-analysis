#!/usr/bin/env python
"""HFR and SWOT side by side, one frame per calendar day, for every radar region
and for each continent.

Left panel: daily-mean radar surface current (speed colour, arrows). Right panel:
SWOT L3 geostrophic current on the swath(s) that crossed the box that day; when
no pass fell on the day the most recent one up to SWOT_MAX_AGE days old is shown
and the title says how old it is. Same colour scale and arrow scale in both.

Data: hourly radar archives under HFR_DATA_ROOT, reduced once to a daily-mean
cache (HFR_ANIM_ROOT/hfr_daily/<region>/<year>.nc, rebuilt when the archive
gets newer files, which also drops that year's frames so they re-render); SWOT
from the regridded pkls under experiments/<region>/swot_cycles/ (science and
calval). Frames: HFR_ANIM_ROOT/frames/<region or continent>/<day>.png
(skipped when present). MP4s: docs/figures/hfr_animations/<name>_hfr_swot.mp4.

Usage:
    python scripts/animate_hfr_swot.py --build-daily                 # refresh the daily cache, all regions
    python scripts/animate_hfr_swot.py --regions au_rot,au_turq       # region movies
    python scripts/animate_hfr_swot.py --continents europe,australia  # continent movies
    python scripts/animate_hfr_swot.py --regions uswc --redo-from 2024-07-01   # re-render after new SWOT cycles
    python scripts/animate_hfr_swot.py --assemble-only --regions all --continents all
    python scripts/animate_hfr_swot.py --assemble-only --regions all --continents all --since 2023-04-01   # plus SWOT-era cuts
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import pickle
import re
import sys
import time
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

sys.path.insert(0, str(Path(__file__).resolve().parent))
from animate_hfr_regions import DPI, VMAX, HOURLY_FILE, archive_folder, assemble, daily_fields, log  # noqa: E402  (also sets the Computer Modern rcParams)
from swotxai.presets import REGION_META  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
ROOT = Path(os.environ.get("HFR_ANIM_ROOT", REPO / "archive" / "hfr_anim"))
DAILY, FRAMES = ROOT / "hfr_daily", ROOT / "frames"
OUT = REPO / "docs" / "figures" / "hfr_animations"
CONTINENTS = {
    "north_america": ("North America", ["uswc", "usegc", "gak", "glna", "prvi", "ushi"]),
    "europe":        ("Europe", ["eu_biscay", "eu_galicia", "eu_wmed", "eu_tirlig", "eu_gbight", "eu_norway"]),
    "australia":     ("Australia", ["au_rot", "au_turq", "au_cof", "au_newc", "au_sag", "au_nwa", "au_corl"]),
}
RANGE_FILE = re.compile(r"(\d{8})_(\d{8})\.nc4?$")
SWOT_MAX_AGE = 10   # days
FIG_W = 12.0        # inches; 2400 px at DPI 200


def continent_of(rid: str) -> str | None:
    return next((c for c, (_, rids) in CONTINENTS.items() if rid in rids), None)


# ----------------------------------------------------------------- daily-mean radar cache
def file_years(p: Path) -> set[int]:
    m = HOURLY_FILE.search(p.name) or RANGE_FILE.search(p.name)
    if not m:
        return set()
    y0 = int(m.group(1)[:4])
    y1 = int(m.group(2)[:4]) if m.lastindex == 2 else y0
    return set(range(y0, y1 + 1))


def build_daily(rid: str) -> None:
    folder = archive_folder(rid)
    if folder is None:
        log(f"{rid}: no archive folder")
        return
    out_dir = DAILY / rid
    out_dir.mkdir(parents=True, exist_ok=True)
    by_year: dict[int, list[Path]] = {}
    for p in folder.iterdir():
        if p.suffix in (".nc", ".nc4"):
            for y in file_years(p):
                by_year.setdefault(y, []).append(p)
    for y, fs in sorted(by_year.items()):
        fs.sort()
        out = out_dir / f"{y}.nc"
        if out.exists() and out.stat().st_mtime > max(f.stat().st_mtime for f in fs):
            continue
        t0 = time.time()
        days: dict[str, tuple] = {}
        lat = lon = None
        for day, u, v, lat, lon in daily_fields(folder, fs):
            if day[:4] == str(y) and day not in days:
                days[day] = (u.astype("float32"), v.astype("float32"))
        if not days:
            continue
        keys = sorted(days)
        ds = xr.Dataset({"u": (("time", "lat", "lon"), np.stack([days[k][0] for k in keys])),
                         "v": (("time", "lat", "lon"), np.stack([days[k][1] for k in keys]))},
                        coords={"time": np.array(keys, dtype="datetime64[ns]"), "lat": lat, "lon": lon})
        tmp = out.with_suffix(".part")
        ds.to_netcdf(tmp, encoding={k: {"zlib": True, "complevel": 4} for k in ("u", "v")})
        tmp.replace(out)
        # frames of that year are stale now (for the region and its continent)
        for name in (rid, continent_of(rid)):
            if name:
                for f in (FRAMES / name).glob(f"{y}-*.png"):
                    f.unlink()
        log(f"{rid} {y}: {len(keys)} days -> {out.stat().st_size / 1e6:.0f} MB ({time.time() - t0:.0f} s)")


class DailyHFR:
    """Lazy reader of the daily cache: day -> (u, v) or None."""

    def __init__(self, rid: str):
        self.files = {int(p.stem): p for p in (DAILY / rid).glob("*.nc")}
        self.open: dict[int, xr.Dataset] = {}
        self.days: set[str] = set()
        self.lat = self.lon = None
        for y, p in sorted(self.files.items()):
            with xr.open_dataset(p) as ds:
                self.days.update(str(t) for t in ds["time"].values.astype("datetime64[D]"))
                if self.lat is None:
                    self.lat, self.lon = ds["lat"].values, ds["lon"].values

    def get(self, day: str):
        if day not in self.days:
            return None
        y = int(day[:4])
        if y not in self.open:
            self.open[y] = xr.open_dataset(self.files[y])
        sub = self.open[y].sel(time=np.datetime64(day))
        return sub["u"].values, sub["v"].values


# ----------------------------------------------------------------- SWOT passes by day
def swot_index(rid: str) -> dict[str, list[xr.Dataset]]:
    """day -> regridded passes (geostrophic components only) from every pkl of the region."""
    idx: dict[str, list[xr.Dataset]] = {}
    for pk in sorted((REPO / "experiments" / rid / "swot_cycles").glob("*.pkl")):
        with open(pk, "rb") as fh:
            data = pickle.load(fh)
        n = 0
        for entries in data.values():
            for e in entries:
                if "time" in e.coords and not np.isnat(e["time"].values) and "ugos_filtered" in e:
                    idx.setdefault(str(e["time"].values.astype("datetime64[D]")), []).append(
                        e[["ugos_filtered", "vgos_filtered"]].load())
                    n += 1
        log(f"{rid}: {n} SWOT passes from {pk.name}")
    return idx


def swot_for_day(idx: dict, day: str):
    """(passes, age in days) for the frame's day: same-day passes, else the most recent within SWOT_MAX_AGE."""
    d0 = dt.date.fromisoformat(day)
    for age in range(SWOT_MAX_AGE + 1):
        k = str(d0 - dt.timedelta(age))
        if k in idx:
            return idx[k], age
    return [], None


# ----------------------------------------------------------------- figure
class Panel:
    def __init__(self, fig, rect, extent, arrow_px: float = 170):
        self.arrow_px = arrow_px     # pixels per m/s
        self.arrow_w = 0.003 if arrow_px >= 170 else 0.0012   # shaft width (axes fraction): thin on continent maps
        self.ax = fig.add_axes(rect, projection=ccrs.PlateCarree())
        self.ax.set_extent(extent)
        self.ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="#e8e4dc", zorder=0)
        self.ax.coastlines("50m", linewidth=0.5, color="#555")
        gl = self.ax.gridlines(draw_labels=True, linewidth=0.3, color="#bbb")
        gl.top_labels = gl.right_labels = False
        deg = r"$^{\circ}$"
        gl.xformatter = LongitudeFormatter(degree_symbol=deg)
        gl.yformatter = LatitudeFormatter(degree_symbol=deg)
        self.title = self.ax.set_title("", loc="left", fontsize=11)
        self.fixed: dict[str, tuple] = {}
        self.temp: list = []
        self.deg_px = (extent[1] - extent[0]) / (fig.get_figwidth() * rect[2] * DPI)   # degrees per pixel
        self.qm = None

    def _step(self, dlon: float) -> int:
        """Subsample so arrows sit about 30 px apart whatever the grid or the map scale;
        arrow length is fixed too (1 m/s = 170 px, what a region panel gets at 1000 px tall)
        (a 6 km grid on a continent map would otherwise be a black blob)."""
        return max(1, int(round(30 * self.deg_px / max(dlon, 1e-6))))

    def add_grid(self, key: str, lat: np.ndarray, lon: np.ndarray) -> None:
        z = np.full((lat.size, lon.size), np.nan, "float32")
        qm = self.ax.pcolormesh(lon, lat, z, cmap=cmocean.cm.speed, vmin=0, vmax=VMAX,
                                transform=ccrs.PlateCarree(), zorder=1)
        s = self._step(float(abs(lon[1] - lon[0])))
        LO, LA = np.meshgrid(lon[::s], lat[::s])
        Q = self.ax.quiver(LO, LA, z[::s, ::s], z[::s, ::s], scale=1 / self.arrow_px, scale_units="dots",
                           width=self.arrow_w, color="#111", transform=ccrs.PlateCarree(), zorder=2)
        self.fixed[key] = (qm, Q, s, z)
        self.qm = qm

    def set_grid(self, key: str, uv) -> int:
        qm, Q, s, z = self.fixed[key]
        if uv is None:
            qm.set_array(z.ravel())
            Q.set_UVC(z[::s, ::s], z[::s, ::s])
            return 0
        u, v = uv
        spd = np.hypot(u, v)
        qm.set_array(spd.ravel())
        Q.set_UVC(u[::s, ::s], v[::s, ::s])
        return int(np.isfinite(spd).sum())

    def draw_swath(self, e: xr.Dataset) -> None:
        lat, lon = e["lat"].values, e["lon"].values
        u, v = e["ugos_filtered"].values, e["vgos_filtered"].values
        spd = np.ma.masked_invalid(np.hypot(u, v))
        self.temp.append(self.ax.pcolormesh(lon, lat, spd, cmap=cmocean.cm.speed, vmin=0, vmax=VMAX,
                                            transform=ccrs.PlateCarree(), zorder=1, shading="auto"))
        s = self._step(float(np.nanmedian(np.abs(np.diff(lon, axis=1)))))
        self.temp.append(self.ax.quiver(lon[::s, ::s], lat[::s, ::s], u[::s, ::s], v[::s, ::s], scale=1 / self.arrow_px,
                                        scale_units="dots", width=self.arrow_w, color="#111",
                                        transform=ccrs.PlateCarree(), zorder=2))

    def clear_temp(self) -> None:
        for a in self.temp:
            a.remove()
        self.temp = []


class Figure:
    """Two map panels (HFR left/top, SWOT right/bottom) over one extent."""

    def __init__(self, extent: list[float], arrow_px: float = 170):
        aspect = (extent[3] - extent[2]) / (extent[1] - extent[0])
        if aspect > 0.7:                       # side by side
            pw = 0.40
            ph = pw * FIG_W * aspect
            h = int(np.ceil((ph + 1.0) * DPI / 16)) * 16 / DPI      # 0.55 in axis labels below, title above
            self.fig = plt.figure(figsize=(FIG_W, h))
            rects = [[0.05, 0.55 / h, pw, ph / h], [0.50, 0.55 / h, pw, ph / h]]
        else:                                   # stacked
            pw = 0.84
            ph = pw * FIG_W * aspect
            h = int(np.ceil((2 * ph + 1.9) * DPI / 16)) * 16 / DPI
            self.fig = plt.figure(figsize=(FIG_W, h))
            rects = [[0.06, (ph + 1.4) / h, pw, ph / h], [0.06, 0.55 / h, pw, ph / h]]
        self.hfr = Panel(self.fig, rects[0], extent, arrow_px)
        self.swot = Panel(self.fig, rects[1], extent, arrow_px)
        self.cax = self.fig.add_axes([0.92, 0.2, 0.015, 0.6])
        self._cb = None

    def colorbar(self) -> None:
        if self._cb is None and self.hfr.qm is not None:
            self._cb = self.fig.colorbar(self.hfr.qm, cax=self.cax, label="speed (m/s)")

    def save(self, out: Path) -> None:
        self.fig.savefig(out, dpi=DPI)

    def close(self) -> None:
        plt.close(self.fig)


def swot_title(n: int, age) -> str:
    if age is None:
        return f"SWOT L3 geostrophic, no pass within {SWOT_MAX_AGE} days"
    if age == 0:
        return f"SWOT L3 geostrophic, {n} pass{'es' if n != 1 else ''} this day"
    return f"SWOT L3 geostrophic, last pass {age} day{'s' if age != 1 else ''} earlier"


def render(name: str, label: str, rids: list[str], extent: list[float], redo_from: str | None, limit: int | None) -> list[Path]:
    fdir = FRAMES / name
    fdir.mkdir(parents=True, exist_ok=True)
    hfr = {r: DailyHFR(r) for r in rids}
    hfr = {r: h for r, h in hfr.items() if h.lat is not None}
    swot = {r: swot_index(r) for r in rids}
    days_all = set().union(*(h.days for h in hfr.values()), *(set(s) for s in swot.values()))
    if not days_all:
        log(f"{name}: no data")
        return []
    d0, d1 = dt.date.fromisoformat(min(days_all)), dt.date.fromisoformat(max(days_all))
    days = [str(d0 + dt.timedelta(i)) for i in range((d1 - d0).days + 1)]
    todo = [d for d in days if not (fdir / f"{d}.png").exists() or (redo_from and d >= redo_from)]
    if limit:
        todo = todo[:limit]
    log(f"{name}: {len(days)} days {days[0]}..{days[-1]}, {len(todo)} frames to render")
    if todo:
        fig = Figure(extent, arrow_px=170 if len(rids) == 1 else 60)   # continent maps: shorter arrows
        for r, h in hfr.items():
            fig.hfr.add_grid(r, h.lat, h.lon)
        fig.colorbar()
        t0 = time.time()
        for i, day in enumerate(todo, 1):
            n = sum(fig.hfr.set_grid(r, h.get(day)) for r, h in hfr.items())
            fig.hfr.title.set_text(f"{label}, HFR daily mean, {day}, {n:,} cells")
            fig.swot.clear_temp()
            best_age, n_shown = None, 0
            for r in rids:
                passes, age = swot_for_day(swot[r], day)
                if age is not None and (best_age is None or age < best_age):
                    best_age = age
                for e in passes:
                    fig.swot.draw_swath(e)
                    n_shown += 1
            fig.swot.title.set_text(swot_title(n_shown, best_age))
            fig.save(fdir / f"{day}.png")
            if i % 200 == 0:
                log(f"{name}: {i}/{len(todo)} frames, last {day}, {(time.time() - t0) / i:.2f} s/frame")
        fig.close()
    return sorted(fdir.glob("*.png"))


def extent_of(rids: list[str]) -> list[float]:
    boxes = [REGION_META[r][2:4] for r in rids]
    pad = 0.3 if len(rids) == 1 else 1.0
    return [min(b[0][0] for b in boxes) - pad, max(b[1][0] for b in boxes) + pad,
            min(b[0][1] for b in boxes) - pad, max(b[1][1] for b in boxes) + pad]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build-daily", action="store_true", help="build/refresh the daily-mean radar cache for --regions (default all)")
    ap.add_argument("--regions", default="", help="comma list of REGION_META ids, or all")
    ap.add_argument("--continents", default="", help=f"comma list of {list(CONTINENTS)}, or all")
    ap.add_argument("--fps", type=int, default=24)
    ap.add_argument("--assemble-only", action="store_true")
    ap.add_argument("--redo-from", default=None, help="re-render frames from this day (YYYY-MM-DD) even if present")
    ap.add_argument("--limit", type=int, default=None, help="render at most this many frames (testing)")
    ap.add_argument("--since", default=None, help="also write <name>_hfr_swot_since<YYYY>.mp4 from frames on or after this day (the SWOT era)")
    a = ap.parse_args()
    all_rids = [r for r in REGION_META if archive_folder(r)]
    regions = all_rids if a.regions == "all" else [r for r in a.regions.split(",") if r]
    conts = list(CONTINENTS) if a.continents == "all" else [c for c in a.continents.split(",") if c]
    if a.build_daily:
        for r in regions or all_rids:
            build_daily(r)
    OUT.mkdir(parents=True, exist_ok=True)
    jobs = [(r, REGION_META[r][0], [r]) for r in regions] + [(c, CONTINENTS[c][0], CONTINENTS[c][1]) for c in conts]
    for name, label, rids in jobs:
        rids = [r for r in rids if r in all_rids]
        if a.assemble_only:
            frames = sorted((FRAMES / name).glob("*.png"))
        else:
            frames = render(name, label, rids, extent_of(rids), a.redo_from, a.limit)
        if frames and not a.limit:
            main = OUT / f"{name}_hfr_swot.mp4"
            if not (a.assemble_only and a.since and main.exists()):   # --assemble-only --since: cuts only
                assemble(frames, main, a.fps)
            if a.since:
                sub = [f for f in frames if f.stem >= a.since]
                if sub:
                    assemble(sub, OUT / f"{name}_hfr_swot_since{a.since[:4]}.mp4", a.fps)
    log("done")


if __name__ == "__main__":
    main()
