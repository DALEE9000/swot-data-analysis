#!/usr/bin/env python
"""Build per-region ERA5 10 m wind pkls from the public NCAR mirror on AWS.

Source: s3://nsf-ncar-era5/e5.oper.an.sfc/YYYYMM/ (anonymous), monthly global
0.25-degree hourly 10u/10v NetCDF. Each (region, mission) gets one pkl with
vars era5_u / era5_v and coords latitude / longitude / time — exactly what
step_load_era5's pkl fast-path and interp_to_swot expect.

Outputs experiments/{region}/era5/era5wind_{mission}_{region}_10m.pkl
(upload to s3://swot-ai-ssv/experiments/... is the caller's job — AWS CLI).

Usage:
    python scripts/build_era5_winds.py --mission science --regions all
    python scripts/build_era5_winds.py --mission calval --regions uswc,usegc
"""
from __future__ import annotations

import argparse
import sys
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

NCAR = "nsf-ncar-era5/e5.oper.an.sfc"

# [lon, lat] corners, matching build_science_colocations.py
REGIONS = {
    "uswc":  ([-126.8, 31.2], [-116.6, 49.9]),
    "usegc": ([-97.7, 22.8],  [-68.0, 44.4]),
    "gak":   ([-131.2, 53.6], [-129.8, 54.8]),
    "akns":  ([-162.7, 70.0], [-154.7, 73.1]),
    "glna":  ([-85.4, 45.3],  [-84.2, 46.4]),
    "prvi":  ([-68.4, 15.8],  [-63.8, 19.6]),
    "ushi":  ([-159.4, 19.3], [-154.3, 22.1]),
}

MISSIONS = {
    # (months spanned, regions with any coverage)
    "science": (["2023%02d" % m for m in range(7, 13)]
                + ["2024%02d" % m for m in range(1, 8)], list(REGIONS)),
    "calval":  (["2023%02d" % m for m in range(3, 9)],
                ["uswc", "usegc", "gak", "akns", "ushi"]),
}

MARGIN_DEG = 0.5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def month_files(fs: s3fs.S3FileSystem, yyyymm: str) -> tuple[str, str]:
    ls = fs.ls(f"{NCAR}/{yyyymm}")
    u = next(p for p in ls if "_10u." in p)
    v = next(p for p in ls if "_10v." in p)
    return u, v


def subset_region(ds: xr.Dataset, sw, ne) -> xr.Dataset:
    lon0 = (sw[0] - MARGIN_DEG) % 360
    lon1 = (ne[0] + MARGIN_DEG) % 360
    lat_sl = slice(ne[1] + MARGIN_DEG, sw[1] - MARGIN_DEG)  # ERA5 lat descends
    return ds.sel(latitude=lat_sl, longitude=slice(lon0, lon1))


def build(mission: str, regions: list[str], force: bool) -> None:
    months, covered = MISSIONS[mission]
    regions = [r for r in regions if r in covered]
    outs = {
        r: Path(f"experiments/{r}/era5/era5wind_{mission}_{r}_10m.pkl")
        for r in regions
    }
    todo = [r for r in regions if force or not outs[r].exists()]
    for r in regions:
        if r not in todo:
            log(f"{r}/{mission}: {outs[r].name} exists — skipping")
    if not todo:
        return

    fs = s3fs.S3FileSystem(anon=True)
    parts: dict[str, list[xr.Dataset]] = {r: [] for r in todo}
    for mm in months:
        pu, pv = month_files(fs, mm)
        log(f"{mission}: subsetting {mm} ({len(todo)} regions)...")
        with fs.open(pu) as fu, fs.open(pv) as fv:
            dsu = xr.open_dataset(fu, engine="h5netcdf")
            dsv = xr.open_dataset(fv, engine="h5netcdf")
            uvar = next(v for v in dsu.data_vars if "10U" in v.upper())
            vvar = next(v for v in dsv.data_vars if "10V" in v.upper())
            for r in todo:
                sw, ne = REGIONS[r]
                sub_u = subset_region(dsu[[uvar]], sw, ne).load()
                sub_v = subset_region(dsv[[vvar]], sw, ne).load()
                month = xr.merge([sub_u.rename({uvar: "era5_u"}),
                                  sub_v.rename({vvar: "era5_v"})])
                parts[r].append(month.astype("float32"))

    for r in todo:
        ds = xr.concat(parts[r], dim="time").sortby("time")
        ds["longitude"] = (ds["longitude"] + 180) % 360 - 180
        ds = ds.sortby("longitude")
        out = outs[r]
        out.parent.mkdir(parents=True, exist_ok=True)
        _save(ds, out)
        log(f"{r}/{mission}: saved -> {out} ({out.stat().st_size / 1e6:.0f} MB, "
            f"{ds.sizes['time']} hours, {ds.sizes['latitude']}x{ds.sizes['longitude']})")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mission", default="science", choices=list(MISSIONS))
    ap.add_argument("--regions", default="all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    regions = list(REGIONS) if args.regions == "all" else [
        r.strip() for r in args.regions.split(",")
    ]
    bad = [r for r in regions if r not in REGIONS]
    if bad:
        ap.error(f"unknown region(s) {bad}")
    build(args.mission, regions, args.force)
    log("done.")


if __name__ == "__main__":
    main()
