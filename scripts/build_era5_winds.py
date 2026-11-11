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
    python scripts/build_era5_winds.py --months 2024-08 2026-08    # append months to existing pkls (all 19 regions)

Append mode: when a region's pkl exists, only the requested months missing from
its time axis are fetched and concatenated onto it (resumable per month).
"""
from __future__ import annotations

import argparse
import io
import pickle
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
    # European HFR node and IMOS regions (2026-11-05), same boxes as the colocation script
    "eu_biscay":  ([-4.2, 43.1], [-1.3, 44.4]),
    "eu_galicia": ([-10.2, 41.4], [-8.0, 43.9]),
    "eu_wmed":    ([-0.6, 38.2], [3.6, 42.3]),
    "eu_tirlig":  ([7.0, 38.4], [12.6, 44.6]),
    "eu_gbight":  ([5.8, 53.4], [9.0, 55.6]),
    "eu_norway":  ([3.8, 59.4], [6.2, 62.2]),
    "au_rot":     ([114.4, -32.7], [115.9, -31.2]),
    "au_turq":    ([113.4, -31.6], [115.6, -29.4]),
    "au_cof":     ([152.9, -31.1], [154.6, -29.4]),
    "au_newc":    ([151.4, -33.6], [153.6, -31.9]),
    "au_sag":     ([134.4, -36.6], [138.6, -33.9]),
    "au_nwa":     ([113.4, -22.1], [116.1, -19.4]),
    "au_corl":    ([111.3, -32.5], [116.4, -27.6]),
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
    if lon0 > lon1:  # box crosses the prime meridian (eu_wmed): two slices on the 0-360 axis
        return xr.concat([ds.sel(latitude=lat_sl, longitude=slice(lon0, 360)),
                          ds.sel(latitude=lat_sl, longitude=slice(0, lon1))], dim="longitude")
    return ds.sel(latitude=lat_sl, longitude=slice(lon0, lon1))


def have_months(path: Path) -> set[str]:
    with open(path, "rb") as fh:
        t = pickle.load(fh)["time"].values.astype("datetime64[M]")
    return {str(x).replace("-", "") for x in np.unique(t)}


def build(mission: str, regions: list[str], force: bool, months: list[str] | None = None) -> None:
    default_months, covered = MISSIONS[mission]
    months = months or default_months
    regions = [r for r in regions if r in covered]
    outs = {
        r: Path(f"experiments/{r}/era5/era5wind_{mission}_{r}_10m.pkl")
        for r in regions
    }
    # months each region still needs: all of them for a new/forced pkl, else the
    # requested months absent from the existing pkl's time axis (append mode)
    need = {r: [m for m in months if force or not outs[r].exists() or m not in have_months(outs[r])]
            for r in regions}
    todo = [r for r in regions if need[r]]
    for r in regions:
        if r not in todo:
            log(f"{r}/{mission}: {outs[r].name} has every requested month — skipping")
    if not todo:
        return

    # each (region, month) subset is pickled to experiments/<r>/era5/_months/ as soon as
    # it is cut, so a crash (or a MemoryError while a render job holds the RAM) costs one
    # month, not the whole stream; the final concat reads them back one region at a time
    def month_cache(r: str, mm: str) -> Path:
        return outs[r].parent / "_months" / f"{mission}_{mm}.pkl"

    fs = s3fs.S3FileSystem(anon=True)
    for mm in sorted({m for r in todo for m in need[r]}):
        month_regions = [r for r in todo if mm in need[r] and not month_cache(r, mm).exists()]
        if not month_regions:
            log(f"{mission}: {mm} already cut for every region, skipping download")
            continue
        pu, pv = month_files(fs, mm)
        log(f"{mission}: subsetting {mm} ({len(month_regions)} regions)...")
        # One GET per file (~1.5 GB gzip chunks) into memory, then subset locally: with
        # fs.open() every region re-fetched its own chunks (13 regions = 20+ min/month).
        with io.BytesIO(fs.cat(pu)) as fu, io.BytesIO(fs.cat(pv)) as fv:
            dsu = xr.open_dataset(fu, engine="h5netcdf")
            dsv = xr.open_dataset(fv, engine="h5netcdf")
            uvar = next(v for v in dsu.data_vars if "10U" in v.upper())
            vvar = next(v for v in dsv.data_vars if "10V" in v.upper())
            for r in month_regions:
                sw, ne = REGIONS[r]
                sub_u = subset_region(dsu[[uvar]], sw, ne).load()
                sub_v = subset_region(dsv[[vvar]], sw, ne).load()
                month = xr.merge([sub_u.rename({uvar: "era5_u"}),
                                  sub_v.rename({vvar: "era5_v"})]).astype("float32")
                month_cache(r, mm).parent.mkdir(parents=True, exist_ok=True)
                with open(month_cache(r, mm), "wb") as fh:
                    pickle.dump(month, fh, protocol=pickle.HIGHEST_PROTOCOL)

    for r in todo:
        parts = []
        for mm in need[r]:
            with open(month_cache(r, mm), "rb") as fh:
                parts.append(pickle.load(fh))
        ds = xr.concat(parts, dim="time").sortby("time")
        del parts
        ds["longitude"] = (ds["longitude"] + 180) % 360 - 180
        ds = ds.sortby("longitude")
        out = outs[r]
        if out.exists() and not force:  # append to the existing record
            with open(out, "rb") as fh:
                old = pickle.load(fh)
            ds = xr.concat([old, ds], dim="time").sortby("time")
        out.parent.mkdir(parents=True, exist_ok=True)
        _save(ds, out)
        log(f"{r}/{mission}: saved -> {out} ({out.stat().st_size / 1e6:.0f} MB, "
            f"{ds.sizes['time']} hours, {ds.sizes['latitude']}x{ds.sizes['longitude']})")
        for mm in need[r]:
            month_cache(r, mm).unlink()
        del ds


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mission", default="science", choices=list(MISSIONS))
    ap.add_argument("--regions", default="all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--months", nargs=2, metavar=("START", "END"),
                    help="YYYY-MM YYYY-MM inclusive; overrides the mission's default window")
    args = ap.parse_args()
    months = None
    if args.months:
        a, b = (np.datetime64(m, "M") for m in args.months)
        months = [str(m).replace("-", "") for m in np.arange(a, b + 1)]
    regions = list(REGIONS) if args.regions == "all" else [
        r.strip() for r in args.regions.split(",")
    ]
    bad = [r for r in regions if r not in REGIONS]
    if bad:
        ap.error(f"unknown region(s) {bad}")
    build(args.mission, regions, args.force, months)
    log("done.")


if __name__ == "__main__":
    main()
