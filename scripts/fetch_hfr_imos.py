#!/usr/bin/env python
"""Fetch IMOS ACORN hourly gridded HFR current maps (Australia) from the AODN
THREDDS server (free, no login). One ~125 KB file per site per hour.

Output: {HFR_DATA_ROOT}/imos_{site}_hourly/{YYYY}/{original filename}
Resumable: existing non-empty files are skipped; missing days (404) are
recorded in a `_missing_days.txt` beside the year folders so they are not
re-listed.

Usage:
    python scripts/fetch_hfr_imos.py                     # all sites, 2023-07-01 .. 2024-07-31
    python scripts/fetch_hfr_imos.py --sites ROT,COF --start 2023-07-01 --end 2024-07-31
"""
from __future__ import annotations

import argparse
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta
from pathlib import Path

import requests

DATA_ROOT = Path(os.environ.get("HFR_DATA_ROOT", r"C:\Users\david\Documents\columbiaocean\HFR\Code\Data"))
BASE = "https://thredds.aodn.org.au/thredds"
PRODUCT = "IMOS/ACORN/gridded_1h-avg-current-map_QC"
# site -> colocation region in build_science_colocations.REGIONS
SITES = {"ROT": "au_rot", "TURQ": "au_turq", "COF": "au_cof", "NEWC": "au_newc",
         "CBG": "au_cbg", "SAG": "au_sag", "BONC": "au_bonc", "NWA": "au_nwa",
         "CORL": "au_corl"}  # CBG and BONC end in 2017 on the server; CORL (Coral Coast, WA) runs 2019-
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
            log(f"  {url.rsplit('/', 1)[-1]} attempt {i + 1}: {type(e).__name__}: {e}")
            time.sleep(15 * (i + 1))
    return None


def download(url_path: str, out: Path) -> bool:
    if out.exists() and out.stat().st_size > 0:
        return True
    part = out.with_suffix(".part")
    for attempt in range(4):   # the body stream can be reset mid-file (seen 2026-11-07), not only the request
        r = get(f"{BASE}/fileServer/{url_path}", stream=True)
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


def fetch_site(site: str, d0: date, d1: date) -> None:
    root = DATA_ROOT / f"imos_{site.lower()}_hourly"
    root.mkdir(parents=True, exist_ok=True)
    missing_f = root / "_missing_days.txt"
    missing = set(missing_f.read_text().split()) if missing_f.exists() else set()
    n_new = n_have = 0
    d = d0
    with ThreadPoolExecutor(WORKERS) as pool:
        while d <= d1:
            tag = d.strftime("%Y-%m-%d")
            if tag not in missing:
                day = f"{PRODUCT}/{site}/{d:%Y}/{d:%m}/{d:%d}"
                r = get(f"{BASE}/catalog/{day}/catalog.xml")
                if r is None:
                    missing.add(tag)
                    missing_f.write_text("\n".join(sorted(missing)))
                else:
                    names = re.findall(r'urlPath="([^"]+\.nc)"', r.text)
                    jobs = [(n, root / d.strftime("%Y") / n.rsplit("/", 1)[-1]) for n in names]
                    (root / d.strftime("%Y")).mkdir(exist_ok=True)
                    todo = [j for j in jobs if not (j[1].exists() and j[1].stat().st_size > 0)]
                    n_have += len(jobs) - len(todo)
                    n_new += sum(pool.map(lambda j: download(*j), todo))
            d += timedelta(days=1)
            if d.day == 1:
                log(f"{site}: through {tag}: {n_new} fetched, {n_have} already present, {len(missing)} missing days")
    log(f"{site} done: {n_new} fetched, {n_have} already present, {len(missing)} missing days")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--sites", default="all", help=f"comma list of {list(SITES)} or all")
    ap.add_argument("--start", default="2023-07-01")
    ap.add_argument("--end", default="2024-07-31")
    a = ap.parse_args()
    sites = list(SITES) if a.sites == "all" else a.sites.split(",")
    d0, d1 = date.fromisoformat(a.start), date.fromisoformat(a.end)
    log(f"IMOS ACORN: {sites} {d0}..{d1} -> {DATA_ROOT}")
    for s in sites:
        fetch_site(s, d0, d1)
    log("done")


if __name__ == "__main__":
    main()
