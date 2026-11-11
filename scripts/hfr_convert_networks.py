#!/usr/bin/env python
"""Convert European HFR node and IMOS ACORN radar files into the HFRNet-style
hourly archive layout that scripts/build_hfr_multires.py reads.

Input (from fetch_hfr_eu.py / fetch_hfr_imos.py under HFR_DATA_ROOT):
  eu_<net>_hourly/<Net>_YYYYMM.nc          TIME,DEPTH,LATITUDE,LONGITUDE; EWCT,NSCT,QCflag
  imos_<site>_hourly/<YYYY>/*.nc            one file per hour; UCUR,VCUR,*_quality_control;
                                            lat/lon 1-D (CODAR sites) or 2-D curvilinear (WERA sites)
Output:
  <region>_<res>_Resolution_hourly/<region>_<res>_Resolution_hourly_YYYYMMDD_YYYYMMDD.nc
    dims (time, lat, lon), float32 u, v in m/s, one file per month, regular lat/lon grid
    at the finest native spacing of the region's networks; several networks in one
    region are nearest-cell binned onto that grid and averaged where they overlap.
QC: keep flags 1 (good) and 2 (probably good); everything else becomes NaN.
Resumable: existing month files are skipped.

Usage:
    python scripts/hfr_convert_networks.py                   # all regions
    python scripts/hfr_convert_networks.py --regions eu_tirlig,au_corl
    python scripts/hfr_convert_networks.py --start 2009-01 --end 2026-10   # full records after the 2026-11 fetch
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.spatial import cKDTree

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "scripts"))
from build_science_colocations import REGIONS  # noqa: E402  (bounding boxes)

DATA_ROOT = Path(os.environ.get("HFR_DATA_ROOT", r"C:\Users\david\Documents\columbiaocean\HFR\Code\Data"))

# region -> (kind, sources, resolution label). Spacing measured 2026-11-05 from the files.
SOURCES = {
    "eu_biscay":  ("eu", ["EUSKOOS"], "5km"),
    "eu_galicia": ("eu", ["Galicia"], "6km"),
    "eu_wmed":    ("eu", ["Ibiza", "DeltaEbro", "ICATMAR"], "3km"),
    "eu_tirlig":  ("eu", ["TirLig", "ARPAS"], "2km"),
    "eu_gbight":  ("eu", ["COSYNA"], "2km"),
    "eu_norway":  ("eu", ["Vestlandet"], "5km"),
    "au_rot":     ("imos", ["ROT"], "4km"),
    "au_turq":    ("imos", ["TURQ"], "6km"),
    "au_cof":     ("imos", ["COF"], "1km"),
    "au_newc":    ("imos", ["NEWC"], "6km"),
    "au_sag":     ("imos", ["SAG"], "4km"),
    "au_nwa":     ("imos", ["NWA"], "7km"),
    "au_corl":    ("imos", ["CORL"], "6km"),
}
GOOD = (1, 2)


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def months(start="2023-07", end="2024-07"):
    y, m = map(int, start.split("-")); ye, me = map(int, end.split("-"))
    while (y, m) <= (ye, me):
        y2, m2 = (y + 1, 1) if m == 12 else (y, m + 1)
        yield f"{y:04d}{m:02d}", np.datetime64(f"{y:04d}-{m:02d}-01"), np.datetime64(f"{y2:04d}-{m2:02d}-01")
        y, m = y2, m2


# ----------------------------------------------------------------- readers
def read_eu(net: str, tag: str) -> xr.Dataset | None:
    f = DATA_ROOT / f"eu_{net.lower()}_hourly" / f"{net}_{tag}.nc"
    if not f.exists():
        return None
    d = xr.open_dataset(f).squeeze("DEPTH", drop=True)
    ok = d["QCflag"].isin(GOOD)
    out = xr.Dataset({"u": d["EWCT"].where(ok).astype("float32"),
                      "v": d["NSCT"].where(ok).astype("float32")})
    out = out.rename({"TIME": "time", "LATITUDE": "lat", "LONGITUDE": "lon"})
    d.close()
    return out.sortby("time").sortby("lat").sortby("lon")


def read_imos(site: str, t0: np.datetime64, t1: np.datetime64) -> xr.Dataset | None:
    """One month of hourly files -> Dataset(time, [lat, lon] or [I, J] with 2-D coords)."""
    y0 = str(t0.astype("datetime64[Y]")); y1 = str((t1 - np.timedelta64(1, "D")).astype("datetime64[Y]"))
    files = []
    for y in sorted({y0, y1}):
        files += glob.glob(str(DATA_ROOT / f"imos_{site.lower()}_hourly" / y / f"IMOS_ACORN_V_{str(t0)[:7].replace('-', '')}*.nc"))
    files = sorted(set(files))
    if not files:
        return None
    parts = []
    for f in files:
        with xr.open_dataset(f) as d:
            ok = d["UCUR_quality_control"].isin(GOOD) & d["VCUR_quality_control"].isin(GOOD)
            p = xr.Dataset({"u": d["UCUR"].where(ok).astype("float32"),
                            "v": d["VCUR"].where(ok).astype("float32")})
            p = p.assign_coords(LATITUDE=d["LATITUDE"], LONGITUDE=d["LONGITUDE"]).load()
        parts.append(p)
    ds = xr.concat(parts, dim="TIME").sortby("TIME").rename({"TIME": "time"})
    if ds["LATITUDE"].ndim == 1:
        ds = ds.rename({"LATITUDE": "lat", "LONGITUDE": "lon"}).sortby("lat").sortby("lon")
    return ds


# ----------------------------------------------------------------- regridding
def target_grid(region: str, dlat: float, dlon: float):
    (lo0, la0), (lo1, la1) = REGIONS[region]
    return (np.arange(la0, la1 + dlat / 2, dlat).astype("float32"),
            np.arange(lo0, lo1 + dlon / 2, dlon).astype("float32"))


def native_spacing(ds: xr.Dataset) -> tuple[float, float]:
    if "lat" in ds.dims:
        return float(np.median(np.diff(ds["lat"]))), float(np.median(np.diff(ds["lon"])))
    la, lo = ds["LATITUDE"].values, ds["LONGITUDE"].values
    return (float(np.median(np.abs(np.diff(la, axis=0)))), float(np.median(np.abs(np.diff(lo, axis=1)))))


def to_grid(ds: xr.Dataset, lat: np.ndarray, lon: np.ndarray, dlat: float, dlon: float) -> xr.Dataset:
    """Nearest-cell binning onto the regular (lat, lon) grid; NaN where no source cell within ~0.7 cell."""
    if "lat" in ds.dims:  # regular 1-D source: exact nearest reindex
        return ds.reindex(lat=lat, lon=lon, method="nearest",
                          tolerance=max(dlat, dlon) * 0.7)
    # curvilinear source: KD-tree from source points to every target cell, once per month
    la, lo = ds["LATITUDE"].values, ds["LONGITUDE"].values
    src = np.column_stack([la.ravel(), lo.ravel() * np.cos(np.deg2rad(la.ravel()))])
    good = np.isfinite(src).all(1)
    tree = cKDTree(src[good])
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    q = np.column_stack([LA.ravel(), LO.ravel() * np.cos(np.deg2rad(LA.ravel()))])
    dist, idx = tree.query(q, distance_upper_bound=max(dlat, dlon) * 0.7)
    hit = np.isfinite(dist)
    src_idx = np.flatnonzero(good)[idx[hit]]
    nt = ds.sizes["time"]
    out = {}
    for k in ("u", "v"):
        flat = ds[k].values.reshape(nt, -1)
        arr = np.full((nt, lat.size * lon.size), np.nan, "float32")
        arr[:, hit] = flat[:, src_idx]
        out[k] = (("time", "lat", "lon"), arr.reshape(nt, lat.size, lon.size))
    return xr.Dataset(out, coords={"time": ds["time"].values, "lat": lat, "lon": lon})


def convert(region: str, force: bool = False, start: str = "2023-07", end: str = "2024-07") -> None:
    kind, nets, res = SOURCES[region]
    folder = DATA_ROOT / f"{region}_{res}_Resolution_hourly"
    folder.mkdir(parents=True, exist_ok=True)
    grid = None
    n_done = n_skip = 0
    for tag, t0, t1 in months(start, end):
        last = str((t1 - np.timedelta64(1, "D")).astype("datetime64[D]")).replace("-", "")
        out = folder / f"{region}_{res}_Resolution_hourly_{str(t0).replace('-', '')}_{last}.nc"
        if out.exists() and not force:
            n_skip += 1
            continue
        parts = []
        for net in nets:
            ds = read_eu(net, tag) if kind == "eu" else read_imos(net, t0, t1)
            if ds is not None:  # the node once returned the whole record for one month's request
                ds = ds.sel(time=slice(t0, t1 - np.timedelta64(1, "s")))
            if ds is None or ds.sizes.get("time", 0) == 0:
                continue
            if grid is None:
                dl = [native_spacing(ds)]
                grid = (min(d[0] for d in dl), min(d[1] for d in dl))
            dlat, dlon = grid
            lat, lon = target_grid(region, dlat, dlon)
            parts.append(to_grid(ds, lat, lon, dlat, dlon))
        if not parts:
            log(f"{region} {tag}: no source data")
            continue
        if len(parts) == 1:
            merged = parts[0]
        else:
            times = np.unique(np.concatenate([p["time"].values for p in parts]))
            parts = [p.reindex(time=times) for p in parts]
            merged = xr.Dataset({k: xr.concat([p[k] for p in parts], dim="src").mean("src", skipna=True)
                                 for k in ("u", "v")})
        merged = merged.transpose("time", "lat", "lon")
        for k in ("u", "v"):
            merged[k] = merged[k].astype("float32")
            merged[k].attrs["units"] = "m s-1"
        merged.attrs.update(title=f"{region} hourly surface current, {', '.join(nets)}, QC flags {GOOD} kept, "
                                  f"nearest-cell binned to {dlat:.4f} x {dlon:.4f} deg", source=kind)
        enc = {k: {"zlib": True, "complevel": 4, "dtype": "float32"} for k in ("u", "v")}
        tmp = out.with_suffix(".part")
        merged.to_netcdf(tmp, encoding=enc)
        tmp.rename(out)
        n_done += 1
        cov = float(np.isfinite(merged["u"].values).mean())
        log(f"{region} {tag}: {merged.sizes['time']} h, grid {merged.sizes['lat']}x{merged.sizes['lon']}, "
            f"finite {cov:.1%}, {out.stat().st_size / 1e6:.1f} MB")
    log(f"{region} done: {n_done} months written, {n_skip} already present -> {folder}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regions", default="all", help=f"comma list of {list(SOURCES)} or all")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--start", default="2023-07", help="first month YYYY-MM")
    ap.add_argument("--end", default="2024-07", help="last month YYYY-MM (inclusive)")
    a = ap.parse_args()
    regions = list(SOURCES) if a.regions == "all" else a.regions.split(",")
    for r in regions:
        convert(r, force=a.force, start=a.start, end=a.end)


if __name__ == "__main__":
    main()
