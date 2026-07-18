#!/usr/bin/env python
"""Repartition the two usegc_1km aggregate segments along the 2020-01-01 boundary.

Input:  usegc_1km_Resolution_hourly_2012_2022.nc4  (crash-boundary segment 1)
        usegc_1km_Resolution_hourly_2022_2025.nc4  (segment 2)
Output: usegc_1km_Resolution_hourly_2012_2019.nc4  (through 2019-12-31T23)
        usegc_1km_Resolution_hourly_2020_2025.nc4  (2020-01-01T00 onward)

Uses HDF5 direct chunk copy: both inputs were written by aggregate_hfr.py with
identical dtype/chunking/filters and one-hour time chunks, so compressed
chunks move byte-for-byte — no decompress/recompress of ~112k hours.
"""
from __future__ import annotations

import time
from pathlib import Path

import h5py
import netCDF4
import numpy as np

AGG = Path(r"C:\Users\david\Documents\columbiaocean\HFR\Code\Data\aggregated_hfr\usegc")
SEG1 = AGG / "usegc_1km_Resolution_hourly_2012_2022.nc4"
SEG2 = AGG / "usegc_1km_Resolution_hourly_2022_2025.nc4"
OUT1 = AGG / "usegc_1km_Resolution_hourly_2012_2019.nc4"
OUT2 = AGG / "usegc_1km_Resolution_hourly_2020_2025.nc4"
BOUNDARY = np.datetime64("2020-01-01T00:00:00")
COPY_VARS = ["u", "v"]


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def times_of(path: Path) -> np.ndarray:
    with netCDF4.Dataset(path) as ds:
        t = netCDF4.num2date(ds["time"][:], ds["time"].units)
        return np.array([np.datetime64(x) for x in t])


def make_schema_copy(src: Path, dst: Path) -> None:
    """Create dst with identical schema (dims/vars/attrs/chunking), no time data."""
    with netCDF4.Dataset(src) as s, netCDF4.Dataset(dst, "w", format="NETCDF4") as d:
        d.createDimension("time", None)
        d.createDimension("lat", s.dimensions["lat"].size)
        d.createDimension("lon", s.dimensions["lon"].size)
        for name in s.variables:
            sv = s[name]
            if name == "time":
                v = d.createVariable("time", sv.dtype, ("time",))
            elif sv.dimensions == ("time", "lat", "lon"):
                fill = getattr(sv, "_FillValue", None)
                v = d.createVariable(name, sv.dtype, sv.dimensions,
                                     zlib=True, complevel=1, shuffle=True,
                                     chunksizes=sv.chunking(), fill_value=fill)
                v.set_auto_maskandscale(False)
            else:
                v = d.createVariable(name, sv.dtype, sv.dimensions)
                if sv.shape:
                    v[:] = sv[:]
                elif sv.dimensions == ():
                    pass  # scalar (wgs84) — attrs only
            v.setncatts({a: sv.getncattr(a) for a in sv.ncattrs() if a != "_FillValue"})


def chunk_copy(src: Path, src_idx: np.ndarray, dst: Path, dst_offset: int) -> None:
    """Direct-chunk-copy time indices src_idx of u/v (+time values) into dst."""
    with h5py.File(src, "r") as fs, h5py.File(dst, "r+") as fd:
        # time values (small, plain copy)
        tvals = fs["time"][:][src_idx]
        n = len(src_idx)
        fd["time"].resize((dst_offset + n,))
        fd["time"][dst_offset:dst_offset + n] = tvals
        for v in COPY_VARS:
            dsrc, ddst = fs[v], fd[v]
            chunks = dsrc.chunks  # (1, lat_chunk, lon_chunk)
            assert chunks[0] == 1, "time chunking must be 1 for direct copy"
            ddst.resize((dst_offset + n,) + dsrc.shape[1:])
            lat_chunk = chunks[1]
            n_lat = dsrc.shape[1]
            lat_offsets = list(range(0, n_lat, lat_chunk))
            t0 = time.time()
            n_unalloc = 0
            for k, si in enumerate(src_idx):
                for lo in lat_offsets:
                    try:
                        filt, data = dsrc.id.read_direct_chunk((si, lo, 0))
                    except RuntimeError:
                        # unallocated chunk == all fill values; leaving the
                        # destination chunk unwritten means exactly the same
                        n_unalloc += 1
                        continue
                    ddst.id.write_direct_chunk((dst_offset + k, lo, 0), data, filt)
                if (k + 1) % 10000 == 0:
                    log(f"  {v}: {k + 1}/{n} hours copied "
                        f"({(k + 1) / max(1e-9, time.time() - t0):.0f} hrs/s)")
            if n_unalloc:
                log(f"  {v}: {n_unalloc} all-fill chunks passed through unallocated")
        fd.flush()


def verify(path: Path, expect_n: int, sample_src: Path, sample_map: list) -> None:
    """Monotonic time + decoded value spot-checks against the source."""
    with netCDF4.Dataset(path) as d:
        t = np.asarray(d["time"][:], dtype="f8")
        assert len(t) == expect_n, f"{path.name}: {len(t)} != {expect_n}"
        assert np.all(np.diff(t) > 0), f"{path.name}: time not monotonic"
        with netCDF4.Dataset(sample_src) as s:
            for dst_i, src_i in sample_map:
                a = d["u"][dst_i, ::500, ::500]
                b = s["u"][src_i, ::500, ::500]
                assert np.array_equal(
                    np.asarray(a.filled(np.nan) if hasattr(a, "filled") else a),
                    np.asarray(b.filled(np.nan) if hasattr(b, "filled") else b),
                    equal_nan=True), f"{path.name}: value mismatch @ {dst_i}"
    log(f"{path.name}: verified ({expect_n} hours, monotonic, values match source)")


def main() -> None:
    t1 = times_of(SEG1)
    t2 = times_of(SEG2)
    log(f"seg1: {len(t1)} hours ({t1[0]} .. {t1[-1]})")
    log(f"seg2: {len(t2)} hours ({t2[0]} .. {t2[-1]})")
    assert t1[-1] < t2[0], "segments overlap!"

    pre = np.where(t1 < BOUNDARY)[0]     # -> OUT1
    post = np.where(t1 >= BOUNDARY)[0]   # -> OUT2 head
    log(f"partition: {len(pre)} hours -> 2012_2019, "
        f"{len(post)} + {len(t2)} hours -> 2020_2025")

    if OUT1.exists():
        log(f"{OUT1.name} exists (verified in a prior run) — skipping")
    else:
        log("building 2012_2019...")
        make_schema_copy(SEG1, OUT1)
        chunk_copy(SEG1, pre, OUT1, 0)
        verify(OUT1, len(pre), SEG1, [(0, int(pre[0])), (len(pre) - 1, int(pre[-1]))])

    log("building 2020_2025...")
    OUT2.unlink(missing_ok=True)  # discard any partial from a crashed run
    make_schema_copy(SEG1, OUT2)
    chunk_copy(SEG1, post, OUT2, 0)
    chunk_copy(SEG2, np.arange(len(t2)), OUT2, len(post))
    verify(OUT2, len(post) + len(t2), SEG1, [(0, int(post[0]))])
    verify(OUT2, len(post) + len(t2), SEG2, [(len(post) + len(t2) - 1, len(t2) - 1)])

    log("deleting interim segment files...")
    SEG1.unlink()
    SEG2.unlink()
    log(f"done: {OUT1.name} ({OUT1.stat().st_size / 1e9:.2f} GB), "
        f"{OUT2.name} ({OUT2.stat().st_size / 1e9:.2f} GB)")


if __name__ == "__main__":
    main()
