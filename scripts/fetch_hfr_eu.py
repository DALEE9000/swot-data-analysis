#!/usr/bin/env python
"""Fetch hourly gridded HFR totals from the European HFR node, one NetCDF per
network per month, via the THREDDS NetCDF Subset Service (free, no login).

Output: {HFR_DATA_ROOT}/eu_{net}_hourly/{net}_YYYYMM.nc  (EWCT, NSCT, QCflag)
Resumable: existing non-empty files are skipped; partial downloads go to a
.part file and are renamed only when the response is a valid NetCDF.

Usage:
    python scripts/fetch_hfr_eu.py                      # all networks, Jul 2023 - Jul 2024
    python scripts/fetch_hfr_eu.py --nets TirLig,Ibiza --start 2023-07 --end 2024-07
"""
from __future__ import annotations

import argparse
import os
import re
import time
from pathlib import Path

import requests

DATA_ROOT = Path(os.environ.get("HFR_DATA_ROOT", r"C:\Users\david\Documents\columbiaocean\HFR\Code\Data"))
NCSS = "https://thredds.hfrnode.eu/thredds/ncss/grid/EUHFR_NRTcurrent_HFR-{net}-Total_{ver}"
# network -> colocation region in build_science_colocations.REGIONS
NETS = {
    "EUSKOOS": "eu_biscay", "Galicia": "eu_galicia",
    "Ibiza": "eu_wmed", "DeltaEbro": "eu_wmed", "ICATMAR": "eu_wmed",
    "TirLig": "eu_tirlig", "LaMMA": "eu_tirlig", "ARPAS": "eu_tirlig",
    "COSYNA": "eu_gbight", "Vestlandet": "eu_norway",
}
VARS = ["EWCT", "NSCT", "QCflag"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def months(start: str, end: str):
    y, m = map(int, start.split("-"))
    ye, me = map(int, end.split("-"))
    while (y, m) <= (ye, me):
        y2, m2 = (y + 1, 1) if m == 12 else (y, m + 1)
        yield f"{y:04d}{m:02d}", f"{y:04d}-{m:02d}-01T00:00:00Z", f"{y2:04d}-{m2:02d}-01T00:00:00Z"
        y, m = y2, m2


def span(net: str, ver: str) -> tuple[str, str] | None:
    """(first, last) month YYYY-MM of the dataset from its NCSS description, None if absent."""
    try:
        r = requests.get(NCSS.format(net=net, ver=ver) + "/dataset.xml", timeout=120)
        m = re.search(r"<begin>(\S+)</begin>\s*<end>(\S+)</end>", r.text)
        return (m.group(1)[:7], m.group(2)[:7]) if m else None
    except Exception:  # noqa: BLE001
        return None


def fetch(net: str, tag: str, t0: str, t1: str, spans: dict | None = None) -> str:
    out = DATA_ROOT / f"eu_{net.lower()}_hourly" / f"{net}_{tag}.nc"
    if out.exists() and out.stat().st_size > 0:
        return "skip"
    out.parent.mkdir(parents=True, exist_ok=True)
    # NCSS grid rejects accept=netcdf4 on this server; netcdf3 is fine (a month is ~5-40 MB)
    params = [("var", v) for v in VARS] + [("time_start", t0), ("time_end", t1), ("accept", "netcdf")]
    month = f"{tag[:4]}-{tag[4:]}"
    vers = [v for v in ("v3", "v2.2") if spans is None or (spans.get(v) and spans[v][0] <= month <= spans[v][1])]
    if not vers:
        return "outside record"   # months the node does not hold would otherwise cost 4 x 500-error retries
    for ver in vers:  # some networks only publish v2.2
        for attempt in range(4):
            try:
                r = requests.get(NCSS.format(net=net, ver=ver), params=params, stream=True, timeout=600)
                if r.status_code in (400, 401, 403, 404):  # bad window, restricted or absent: do not retry
                    break
                r.raise_for_status()
                part = out.with_suffix(".part")
                with open(part, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                with open(part, "rb") as f:
                    head = f.read(8)
                if not (head.startswith(b"\x89HDF") or head.startswith(b"CDF")):
                    part.unlink()
                    raise RuntimeError(f"not NetCDF (status {r.status_code}, head {head!r})")
                part.rename(out)
                return f"{ver} {out.stat().st_size / 1e6:.1f} MB"
            except Exception as e:  # noqa: BLE001
                log(f"  {net} {tag} {ver} attempt {attempt + 1}: {type(e).__name__}: {e}")
                time.sleep(30 * (attempt + 1))
    return "FAILED"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--nets", default="all", help=f"comma list of {list(NETS)} or all")
    ap.add_argument("--start", default="2023-07", help="first month YYYY-MM")
    ap.add_argument("--end", default="2024-07", help="last month YYYY-MM (inclusive)")
    a = ap.parse_args()
    nets = list(NETS) if a.nets == "all" else a.nets.split(",")
    log(f"EU HFR node: {nets} {a.start}..{a.end} -> {DATA_ROOT}")
    for net in nets:
        spans = {v: span(net, v) for v in ("v3", "v2.2")}
        log(f"{net}: record v3 {spans['v3']}, v2.2 {spans['v2.2']}")
        for tag, t0, t1 in months(a.start, a.end):
            log(f"{net} {tag}: {fetch(net, tag, t0, t1, spans)}")
    log("done")


if __name__ == "__main__":
    main()
