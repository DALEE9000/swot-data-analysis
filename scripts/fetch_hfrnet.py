#!/usr/bin/env python
"""Fill and extend the US HFRNet 6 km hourly archives from NOAA NCEI's HF radar
RTV archive (free, HTTPS directory listings, one NetCDF per hour since 2006).

The Scripps HFRNet THREDDS server (hfrnet-tds.ucsd.edu) was retired in 2026;
NDBC and the CoastWatch ERDDAPs only keep the last three months. NCEI holds the
whole record at
  https://www.ncei.noaa.gov/data/oceans/ioos/hfradar/rtv/<YYYY>/<YYYYMM>/<REGION>/
with files `<YYYYMMDDHHMM>_hfr_<rid>_6km_rtv_uwls_<SIO|NDBC>.nc` (both producers
for most hours up to mid-2025; SIO is the HFRNet product and is preferred) and,
from Aug 2025, `rtv-<rid>-6km-uwls_v1r0_hfr_s<YYYYMMDDHHMM>...nc`. 25-hour
averages are skipped.

For each region the script compares the months present in the local archive
folder (HFR_DATA_ROOT/<rid>_6km_Resolution_hourly) with the months the server
holds and fetches (a) calendar months with no local hours at all and (b) every
month after the last local hour. Months partly present locally are left alone so
no hour is ever duplicated when the readers concatenate the folder. Files are
saved under the archive's one-file-per-hour naming,
<rid>_6km_<YYYYMMDD>T<HHMMSS>.nc, so every reader (animations, multires
targets) picks them up unchanged. Resumable: existing files are skipped.

Usage:
    python scripts/fetch_hfrnet.py                         # all six regions, prints the plan
    python scripts/fetch_hfrnet.py --go                    # fetch
    python scripts/fetch_hfrnet.py --regions prvi,ushi --go
"""
from __future__ import annotations

import argparse
import os
import re
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import xarray as xr

DATA_ROOT = Path(os.environ.get("HFR_DATA_ROOT", r"C:\Users\david\Documents\columbiaocean\HFR\Code\Data"))
BASE = "https://www.ncei.noaa.gov/data/oceans/ioos/hfradar/rtv"
CODES = {"uswc": "USWC", "usegc": "USEGC", "gak": "GAK", "glna": "GLNA", "prvi": "PRVI", "ushi": "USHI"}
HOURLY_FILE = re.compile(r"(\d{8})T\d{6}\.nc4?$")
OLD_NAME = re.compile(r'href="((\d{12})_hfr_(\w+?)_6km_rtv_uwls_(SIO|NDBC)\.nc)"')
NEW_NAME = re.compile(r'href="(rtv-(\w+?)-6km-uwls_v\d+r\d+_hfr_s(\d{12})\d*_e\d+_c\d+\.nc)"')
WORKERS = 8


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def get(url: str, attempts: int = 4, **kw) -> requests.Response | None:
    for i in range(attempts):
        try:
            r = requests.get(url, timeout=120, **kw)
            if r.status_code == 404:
                return None
            r.raise_for_status()
            return r
        except Exception as e:  # noqa: BLE001
            log(f"  {url.rsplit('/', 1)[-1]} attempt {i + 1}: {type(e).__name__}")
            time.sleep(15 * (i + 1))
    return None


def local_hours(folder: Path) -> Counter:
    """Hours per calendar month YYYYMM in the local archive (filename date for per-hour
    files, time axis for multi-day files)."""
    c: Counter = Counter()
    for p in sorted(folder.iterdir()):
        if p.suffix not in (".nc", ".nc4") or p.name.endswith(".part"):
            continue
        m = HOURLY_FILE.search(p.name)
        if m:
            c[m.group(1)[:6]] += 1
        else:
            with xr.open_dataset(p) as ds:
                t = ds["time"].values.astype("datetime64[M]")
            c.update(str(x).replace("-", "") for x in t)
    return c


def server_months() -> list[str]:
    """Every YYYYMM folder on the server, from its year listings."""
    out = []
    r = get(f"{BASE}/")
    for y in sorted(set(re.findall(r'href="(\d{4})/"', r.text))):
        ry = get(f"{BASE}/{y}/")
        if ry:
            out += sorted(set(re.findall(r'href="(\d{6})/"', ry.text)))
    return out


def plan(rid: str, months: list[str]) -> list[str]:
    have = local_hours(DATA_ROOT / f"{rid}_6km_Resolution_hourly")
    first, last = min(have), max(have)
    return [m for m in months if m > last or (first <= m <= last and have.get(m, 0) == 0)]


def month_files(rid: str, month: str) -> dict[str, str]:
    """stamp YYYYMMDDHHMM -> filename for one month; SIO preferred over NDBC."""
    r = get(f"{BASE}/{month[:4]}/{month}/{CODES[rid]}/")
    if r is None:
        return {}
    files: dict[str, str] = {}
    for fname, stamp, reg, prod in OLD_NAME.findall(r.text):
        if reg == rid and (stamp not in files or prod == "SIO"):
            files[stamp] = fname
    for fname, reg, stamp in NEW_NAME.findall(r.text):
        if reg == rid:
            files.setdefault(stamp, fname)
    return files


def download(url: str, out: Path) -> bool:
    if out.exists() and out.stat().st_size > 0:
        return True
    part = out.with_suffix(".part")
    for attempt in range(4):
        r = get(url, stream=True)
        if r is None:
            return False
        try:
            with open(part, "wb") as f:
                for chunk in r.iter_content(1 << 18):
                    f.write(chunk)
            part.rename(out)
            return True
        except Exception as e:  # noqa: BLE001
            log(f"  {out.name} body attempt {attempt + 1}: {type(e).__name__}")
            time.sleep(15 * (attempt + 1))
    return False


def fetch_month(rid: str, month: str, folder: Path) -> tuple[int, int]:
    files = month_files(rid, month)
    jobs = []
    for stamp, fname in sorted(files.items()):
        out = folder / f"{rid}_6km_{stamp[:8]}T{stamp[8:]}00.nc"
        jobs.append((f"{BASE}/{month[:4]}/{month}/{CODES[rid]}/{fname}", out))
    todo = [j for j in jobs if not (j[1].exists() and j[1].stat().st_size > 0)]
    with ThreadPoolExecutor(WORKERS) as pool:
        n_ok = sum(pool.map(lambda j: download(*j), todo))
    return len(jobs), n_ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regions", default="all", help=f"comma list of {list(CODES)} or all")
    ap.add_argument("--go", action="store_true", help="fetch (default: print the plan only)")
    a = ap.parse_args()
    regions = list(CODES) if a.regions == "all" else a.regions.split(",")
    months = server_months()
    log(f"NCEI holds {months[0]}..{months[-1]} ({len(months)} months)")
    for rid in regions:
        folder = DATA_ROOT / f"{rid}_6km_Resolution_hourly"
        todo = plan(rid, months)
        log(f"{rid}: {len(todo)} months to fetch: {todo[:4]}{' ...' if len(todo) > 4 else ''}")
        if not a.go:
            continue
        for m in todo:
            n, ok = fetch_month(rid, m, folder)
            log(f"{rid} {m}: {ok}/{n} hourly files")
    log("done")


if __name__ == "__main__":
    main()
