#!/usr/bin/env python
"""Build 25-h-mean HFR colocation targets at EVERY available spatial resolution.

For each (region, resolution) beyond the default targets (which
build_science_colocations.py produces as hfr_science_{region}.pkl), this
reads the LOCAL raw hourly archives in HFR/Code/Data — kept as originals —
and colocates each SWOT science pass with the 25-hour centered mean HFR
field at that resolution. Memory stays flat: one ±31 h slab per pass.

Outputs experiments/{region}/hfr_target/hfr_science_{region}_{res}.pkl
then uploads via AWS CLI conventions are handled by the caller.

Usage:
    python scripts/build_hfr_multires.py --targets prvi_2km
    python scripts/build_hfr_multires.py --targets all
"""
from __future__ import annotations

import argparse
import pickle
import re
import sys
import time
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
import os

os.chdir(REPO)

import numpy as np
import xarray as xr

from swotxai.data_utils import hfr_on_swot
from swotxai.pipeline.io_utils import _save

RAW_ROOT = Path(r"C:\Users\david\Documents\columbiaocean\HFR\Code\Data")

# (region, res) -> raw hourly folder. Defaults (6km, gak 2km) are built by
# build_science_colocations.py; these are the additional resolutions.
TARGETS = {
    "uswc_500m":  ("uswc", "uswc_500m_Resolution_hourly"),
    "uswc_1km":   ("uswc", "uswc_1km_Resolution_hourly"),
    "uswc_2km":   ("uswc", "uswc_2km_Resolution_hourly"),
    "usegc_1km":  ("usegc", "usegc_1km_Resolution_hourly"),
    "usegc_2km":  ("usegc", "usegc_2km_Resolution_hourly"),
    "gak_1km":    ("gak", "gak_1km_Resolution_hourly"),
    "gak_6km":    ("gak", "gak_6km_Resolution_hourly"),
    "glna_500m":  ("glna", "glna_500m_Resolution_hourly"),
    "glna_1km":   ("glna", "glna_1km_Resolution_hourly"),
    "glna_2km":   ("glna", "glna_2km_Resolution_hourly"),
    "prvi_2km":   ("prvi", "prvi_2km_Resolution_hourly"),
    "ushi_1km":   ("ushi", "ushi_1km_Resolution_hourly"),
    "ushi_2km":   ("ushi", "ushi_2km_Resolution_hourly"),
}

MARGIN = np.timedelta64(31, "h")
WINDOW = np.timedelta64(12, "h")  # 25-h centered mean = nearest sample ± 12 h
MAX_GAP_H = 36


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def swot_pkl(region: str):
    """Load the region's science SWOT pkl (local mirror, else S3)."""
    local = Path(f"experiments/{region}/swot_cycles/"
                 f"swot_expert_reproc_v3_{region}_science.pkl")
    if local.exists():
        with open(local, "rb") as f:
            return pickle.load(f)
    import s3fs
    fs = s3fs.S3FileSystem(anon=True)
    key = (f"swot-ai-ssv/experiments/{region}/swot_cycles/"
           f"swot_expert_reproc_v3_{region}_science.pkl")
    log(f"{region}: streaming SWOT pkl from s3://{key}")
    with fs.open(key) as f:
        return pickle.load(f)


_TS_RE = re.compile(r"(\d{8}T\d{6})")
_RANGE_RE = re.compile(r"(\d{8})_(\d{8})\.nc4?$")


def index_folder(folder: Path):
    """[(start, end_or_None, path)] sorted — hourly singles or date-range chunks."""
    out = []
    for fp in sorted(folder.glob("*.nc*")):
        m = _TS_RE.search(fp.name)
        if m:
            t = np.datetime64(datetime.strptime(m.group(1), "%Y%m%dT%H%M%S"))
            out.append((t, None, fp))
            continue
        m = _RANGE_RE.search(fp.name)
        if m:
            t0 = np.datetime64(datetime.strptime(m.group(1), "%Y%m%d"))
            t1 = np.datetime64(datetime.strptime(m.group(2), "%Y%m%d")) + np.timedelta64(2, "D")
            out.append((t0, t1, fp))
    out.sort(key=lambda x: x[0])
    return out


def slab_mean(files_idx, t0: np.datetime64):
    """25-h centered mean u/v around t0, loading only the needed 25 hours.

    Two passes over the covering files: read time coordinates alone to find
    the sample nearest t0 (36 h gap guard), then load u/v strictly within
    nearest ± 12 h and mean over time. On an hourly archive this equals
    rolling(time=25, center=True, min_periods=1).mean().sel(time=nearest),
    but peak memory is ~25 grid-hours instead of the ~60 full-grid
    temporaries xarray rolling allocates (which OOM'd usegc_1km), and it
    windows by real time rather than sample count across archive gaps.
    """
    lo, hi = t0 - MARGIN, t0 + MARGIN
    covering = [(s, e, fp) for s, e, fp in files_idx
                if s <= hi and (e or s) >= lo]
    tvals = []
    for _s, _e, fp in covering:
        try:
            with xr.open_dataset(fp) as ds:
                tv = ds["time"].sel(time=slice(lo, hi)).values
                if len(tv):
                    tvals.append(tv)
        except OSError:
            continue
    if not tvals:
        return None
    tall = np.unique(np.concatenate(tvals))
    nearest = tall[np.argmin(np.abs(tall - t0))]
    if abs((nearest - t0) / np.timedelta64(1, "h")) > MAX_GAP_H:
        return None

    wlo, whi = nearest - WINDOW, nearest + WINDOW
    das_u, das_v = [], []
    grid = None
    for start, end, fp in covering:
        if not (start <= whi and (end or start) >= wlo):
            continue
        try:
            with xr.open_dataset(fp) as ds:
                sub = ds[["u", "v"]].sel(time=slice(wlo, whi))
                if sub.sizes.get("time", 0) == 0:
                    continue
                shape = (sub.sizes["lat"], sub.sizes["lon"])
                if grid is None:
                    grid = shape
                elif shape != grid:
                    continue  # cross-era grid change; keep the window's era
                loaded = sub.load()
                das_u.append(loaded["u"].astype("float32"))
                das_v.append(loaded["v"].astype("float32"))
        except OSError:
            continue
    if not das_u:
        return None
    u = xr.concat(das_u, dim="time").sortby("time")
    v = xr.concat(das_v, dim="time").sortby("time")
    del das_u, das_v
    mean = xr.Dataset({"u": u.mean("time"), "v": v.mean("time")})
    return mean.assign_coords(time=nearest)


def build(target: str, force: bool = False) -> None:
    region, folder_name = TARGETS[target]
    res = target.split("_", 1)[1]
    out = Path(f"experiments/{region}/hfr_target/"
               f"hfr_science_{region}_{res}.pkl")
    if out.exists() and not force:
        log(f"{target}: {out.name} exists — skipping")
        return
    folder = RAW_ROOT / folder_name
    files_idx = index_folder(folder)
    if not files_idx:
        log(f"{target}: no raw files in {folder} — skipping")
        return

    swot = swot_pkl(region)
    n_passes = sum(len(v or []) for v in swot.values())
    log(f"{target}: colocating {n_passes} passes from {len(files_idx)} raw files...")

    import gc

    # per-cycle checkpoint: a crash on a 30-hour target resumes instead of
    # restarting (the usegc_1km giant made this non-optional)
    partial = out.parent / (out.stem + ".partial.pkl")
    result: dict[str, list] = {}
    if partial.exists() and not force:
        with open(partial, "rb") as f:
            result = pickle.load(f)
        log(f"{target}: resuming from checkpoint ({len(result)} cycles done)")

    n_ok = n_skip = 0
    t_start = time.time()
    for cycle in sorted(swot):
        if cycle in result:
            n_ok += len(result[cycle])
            continue
        interp_list = []
        for ds in swot[cycle] or []:
            if ds is None or "time" not in ds.coords:
                n_skip += 1
                continue
            t = np.atleast_1d(ds.coords["time"].values).astype("datetime64[ns]")
            t = t[~np.isnat(t)]
            if not len(t):
                n_skip += 1
                continue
            field = slab_mean(files_idx, t[0])
            if field is None:
                n_skip += 1
                continue
            res_ds = hfr_on_swot(field, ds)
            if res_ds is not None:
                interp_list.append(res_ds.load() if hasattr(res_ds, "load") else res_ds)
                n_ok += 1
            del field
            gc.collect()  # big-grid slabs fragment the heap without this
        result[cycle] = interp_list
        _save(result, partial)  # checkpoint after every cycle
        log(f"{target}/cycle_{cycle}: {len(interp_list)} colocated "
            f"({time.time() - t_start:.0f}s elapsed)")
    _save(result, out)
    partial.unlink(missing_ok=True)
    log(f"{target}: saved -> {out} ({out.stat().st_size / 1e6:.0f} MB, "
        f"{n_ok} colocated, {n_skip} skipped)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", default="all")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()
    targets = list(TARGETS) if args.targets == "all" else [
        t.strip() for t in args.targets.split(",")
    ]
    bad = [t for t in targets if t not in TARGETS]
    if bad:
        ap.error(f"unknown target(s) {bad}; choose from {list(TARGETS)}")
    log(f"targets: {targets}")
    for t in targets:
        build(t, force=args.force)
    log("done.")


if __name__ == "__main__":
    main()
