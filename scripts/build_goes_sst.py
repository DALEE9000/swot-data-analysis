#!/usr/bin/env python
"""Build per-region GOES SST datasets at SWOT pass hours from NOAA's public S3.

Source: s3://noaa-goes18 / s3://noaa-goes16 ABI-L2-SSTF (hourly full-disk SST,
anonymous). Pass hours are parsed from the SWOT granule names already on
s3://swot-ai-ssv — SST is fetched ONLY for those hours (interp_to_swot matches
sources by nearest time, so pass-hour snapshots are exactly sufficient).

Each (region, mission) yields one NetCDF with dims (time, y, x), 2-D lat/lon
coords and an SST variable (Kelvin, masked to DQF<=1) — the same shape as the
legacy goes_sst_calval_uswc.nc that interp_to_swot's KD-tree path consumes.

akns is not buildable: 70N+ is outside geostationary full-disk view.
glna rides goes16; SST quality over the Great Lakes is whatever DQF says.

Usage:
    python scripts/build_goes_sst.py --mission science --regions all
    python scripts/build_goes_sst.py --mission calval --regions uswc,gak,ushi
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import os

os.chdir(REPO)

import numpy as np
import s3fs
import xarray as xr

from swotxai.pipeline.io_utils import _save

BUCKET = "swot-ai-ssv"

# [lon, lat] corners, matching build_science_colocations.py
REGIONS = {
    "uswc":  ([-126.8, 31.2], [-116.6, 49.9]),
    "usegc": ([-97.7, 22.8],  [-68.0, 44.4]),
    "gak":   ([-131.2, 53.6], [-129.8, 54.8]),
    "glna":  ([-85.4, 45.3],  [-84.2, 46.4]),
    "prvi":  ([-68.4, 15.8],  [-63.8, 19.6]),
    "ushi":  ([-159.4, 19.3], [-154.3, 22.1]),
}

SATELLITE = {  # region -> GOES bucket
    "uswc": "noaa-goes18", "gak": "noaa-goes18", "ushi": "noaa-goes18",
    "usegc": "noaa-goes16", "glna": "noaa-goes16", "prvi": "noaa-goes16",
}

MISSION_REGIONS = {
    "science": list(REGIONS),
    "calval":  ["uswc", "usegc", "gak", "ushi"],  # glna/prvi: no calval passes
}

MARGIN_DEG = 0.3
_GRAN_TS = re.compile(r"_(\d{8}T\d{6})_")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def pass_hours(fs: s3fs.S3FileSystem, region: str, mission: str) -> list[np.datetime64]:
    """Unique nearest-hours of every SWOT pass, from granule names on S3."""
    prefix = f"{BUCKET}/SWOT_L3/{mission}/Expert_reproc_v3_{region}_{mission}"
    hours = set()
    for key in fs.find(prefix):
        m = _GRAN_TS.search(key.rsplit("/", 1)[-1])
        if not m:
            continue
        t = np.datetime64(
            f"{m.group(1)[:4]}-{m.group(1)[4:6]}-{m.group(1)[6:8]}"
            f"T{m.group(1)[9:11]}:{m.group(1)[11:13]}:{m.group(1)[13:15]}")
        hours.add((t + np.timedelta64(30, "m")).astype("datetime64[h]"))
    return sorted(hours)


def latlon_from_fixed_grid(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    """Full-disk 2-D lat/lon (degrees) from the ABI fixed-grid projection."""
    proj = ds["goes_imager_projection"]
    r_eq = float(proj.semi_major_axis)
    r_pol = float(proj.semi_minor_axis)
    H = float(proj.perspective_point_height) + r_eq
    lon0 = np.deg2rad(float(proj.longitude_of_projection_origin))
    x = ds["x"].values.astype("float64")  # radians
    y = ds["y"].values.astype("float64")
    xx, yy = np.meshgrid(x, y)
    sinx, cosx = np.sin(xx), np.cos(xx)
    siny, cosy = np.sin(yy), np.cos(yy)
    with np.errstate(invalid="ignore"):
        a = sinx**2 + cosx**2 * (cosy**2 + (r_eq**2 / r_pol**2) * siny**2)
        b = -2.0 * H * cosx * cosy
        c = H**2 - r_eq**2
        r_s = (-b - np.sqrt(b**2 - 4 * a * c)) / (2 * a)
        s_x = r_s * cosx * cosy
        s_y = -r_s * sinx
        s_z = r_s * cosx * siny
        lat = np.rad2deg(np.arctan((r_eq**2 / r_pol**2)
                                   * s_z / np.sqrt((H - s_x)**2 + s_y**2)))
        lon = np.rad2deg(lon0 - np.arctan(s_y / (H - s_x)))
    return lat, lon


def region_box(lat: np.ndarray, lon: np.ndarray, sw, ne) -> tuple[slice, slice]:
    """(y_slice, x_slice) covering the region bbox on the fixed grid."""
    m = ((lat >= sw[1] - MARGIN_DEG) & (lat <= ne[1] + MARGIN_DEG)
         & (lon >= sw[0] - MARGIN_DEG) & (lon <= ne[0] + MARGIN_DEG))
    if not m.any():
        return None
    ys, xs = np.where(m)
    return slice(ys.min(), ys.max() + 1), slice(xs.min(), xs.max() + 1)


def sst_file_for_hour(fs: s3fs.S3FileSystem, sat: str, t: np.datetime64) -> str | None:
    dt = t.astype("datetime64[s]").item()
    doy = dt.timetuple().tm_yday
    try:
        ls = fs.ls(f"{sat}/ABI-L2-SSTF/{dt.year}/{doy:03d}/{dt.hour:02d}")
    except FileNotFoundError:
        return None
    ncs = [p for p in ls if p.endswith(".nc")]
    return ncs[0] if ncs else None


def build(region: str, mission: str, force: bool) -> None:
    sat = SATELLITE[region]
    out = Path(f"experiments/{region}/goes/"
               f"goes_sst_{mission}_{region}.nc")
    if out.exists() and not force:
        log(f"{region}/{mission}: {out.name} exists — skipping")
        return
    out.parent.mkdir(parents=True, exist_ok=True)

    fs = s3fs.S3FileSystem(anon=True)
    hours = pass_hours(fs, region, mission)
    if not hours:
        log(f"{region}/{mission}: no granules found — skipping")
        return
    log(f"{region}/{mission}: {len(hours)} pass hours "
        f"({hours[0]} .. {hours[-1]}) from {sat}")

    # grid + region box from the first available file
    probe = None
    for t in hours:
        probe = sst_file_for_hour(fs, sat, t)
        if probe:
            break
    if probe is None:
        log(f"{region}/{mission}: no SSTF files found at any pass hour — skipping")
        return
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / "probe.nc"
        fs.get(probe, str(local))
        with xr.open_dataset(local) as ds0:
            lat_full, lon_full = latlon_from_fixed_grid(ds0)
    box = region_box(lat_full, lon_full, *REGIONS[region])
    if box is None:
        log(f"{region}/{mission}: bbox outside {sat} disk — skipping")
        return
    ysl, xsl = box
    lat2d = lat_full[ysl, xsl].astype("float64")
    lon2d = lon_full[ysl, xsl].astype("float64")
    del lat_full, lon_full
    log(f"{region}/{mission}: fixed-grid box y[{ysl.start}:{ysl.stop}] "
        f"x[{xsl.start}:{xsl.stop}] -> {lat2d.shape}")

    partial = out.with_suffix(".partial.pkl")
    done: dict = {}
    if partial.exists() and not force:
        with open(partial, "rb") as f:
            done = pickle.load(f)
        log(f"{region}/{mission}: resuming ({len(done)} hours done)")

    n_missing = 0
    with tempfile.TemporaryDirectory() as td:
        local = Path(td) / "sst.nc"
        for i, t in enumerate(hours):
            key = str(t)
            if key in done:
                continue
            path = sst_file_for_hour(fs, sat, t)
            if path is None:
                done[key] = None
                n_missing += 1
                continue
            try:
                fs.get(path, str(local))
                with xr.open_dataset(local) as ds:
                    sst = ds["SST"].isel(y=ysl, x=xsl).values.astype("float32")
                    dqf = ds["DQF"].isel(y=ysl, x=xsl).values
                    tval = ds["t"].values
                sst[~np.isin(dqf, (0, 1))] = np.nan
                done[key] = (tval, sst)
            except Exception as e:
                log(f"{region}/{mission}: {key} failed ({type(e).__name__}: {e}) — skipped")
                done[key] = None
                n_missing += 1
            if (len(done) % 50) == 0:
                _save(done, partial)
                log(f"{region}/{mission}: {len(done)}/{len(hours)} hours "
                    f"({n_missing} missing)")
    _save(done, partial)

    frames = [(v[0], v[1]) for v in done.values() if v is not None]
    if not frames:
        log(f"{region}/{mission}: nothing retrieved — leaving partial only")
        return
    frames.sort(key=lambda p: p[0])
    times = np.array([f[0] for f in frames], dtype="datetime64[ns]")
    cube = np.stack([f[1] for f in frames])
    ds_out = xr.Dataset(
        {"SST": (("time", "y", "x"), cube)},
        coords={"time": times,
                "lat": (("y", "x"), lat2d),
                "lon": (("y", "x"), lon2d)},
        attrs={"source": f"{sat} ABI-L2-SSTF, DQF<=1, at SWOT {mission} pass hours",
               "units": "K"},
    )
    enc = {"SST": {"zlib": True, "complevel": 4}}
    ds_out.to_netcdf(out, encoding=enc)
    partial.unlink(missing_ok=True)
    log(f"{region}/{mission}: saved -> {out} ({out.stat().st_size / 1e6:.0f} MB, "
        f"{len(times)} snapshots, {n_missing} hours unavailable)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mission", default="science", choices=list(MISSION_REGIONS))
    ap.add_argument("--regions", default="all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    covered = MISSION_REGIONS[args.mission]
    regions = covered if args.regions == "all" else [
        r.strip() for r in args.regions.split(",")
    ]
    bad = [r for r in regions if r not in REGIONS]
    if bad:
        ap.error(f"unknown/unbuildable region(s) {bad}; choose from {list(REGIONS)}")
    regions = [r for r in regions if r in covered]
    log(f"mission: {args.mission}  regions: {regions}")
    for r in regions:
        build(r, args.mission, args.force)
    log("done.")


if __name__ == "__main__":
    main()
