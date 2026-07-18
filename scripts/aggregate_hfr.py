#!/usr/bin/env python
"""Aggregate raw hourly HFR files into single per-(region, resolution) NetCDFs
and upload them to S3, matching the conventions of the existing aggregates
(e.g. uswc_6km_Resolution_hourly_2012_2024.nc4).

Handles both input shapes found in HFR/Code/Data: one-hour-per-file dumps
(prvi, ushi, glna, gak_1km — up to ~120k files/folder) and multi-hour chunk
files (gak_6km, usegc_1km/2km). Streams into an unlimited time dimension with
the SOURCE variable encoding preserved (packed int16 stays int16), so memory
stays flat and the big grids don't pay a float re-encoding tax.

Resume-safe: finished aggregates are skipped; uploads are size-verified and
the local aggregate is deleted afterward (raw hourly sources are kept).

Usage:
    python scripts/aggregate_hfr.py --targets glna_500m,gak_6km
    python scripts/aggregate_hfr.py --targets all --phase build
    python scripts/aggregate_hfr.py --targets all --phase upload
"""
from __future__ import annotations

import argparse
import re
import time as _time
from pathlib import Path

import netCDF4
import numpy as np

DATA_ROOT = Path(r"C:\Users\david\Documents\columbiaocean\HFR\Code\Data")
AGG_ROOT = DATA_ROOT / "aggregated_hfr"
S3_BUCKET = "swot-ai-ssv"

# Folders with no single-file aggregate (surveyed 2026-07-13).
TARGETS = [
    "prvi_2km", "prvi_6km",
    "ushi_1km", "ushi_2km", "ushi_6km",
    "gak_1km", "gak_6km",
    "glna_500m", "glna_1km", "glna_2km",
    "usegc_1km", "usegc_2km",
]

COPY_VARS = ["u", "v"]
TIME_UNITS = "seconds since 1970-01-01 00:00:00"


def log(msg: str) -> None:
    print(f"[{_time.strftime('%H:%M:%S')}] {msg}", flush=True)


def folder_for(target: str) -> Path:
    return DATA_ROOT / f"{target}_Resolution_hourly"


def region_of(target: str) -> str:
    return target.split("_")[0]


def _chunksizes(ny: int, nx: int, itemsize: int) -> tuple[int, int, int]:
    """Time-major chunks capped near 8 MB so huge grids stay writable."""
    per_slab = ny * nx * itemsize
    if per_slab <= 8e6:
        return (max(1, int(8e6 // per_slab)), ny, nx)
    split = int(np.ceil(per_slab / 8e6))
    return (1, max(1, ny // split), nx)


_RANGE_RE = re.compile(r"(\d{8})_(\d{8})\.nc4?$")
_TS_RE = re.compile(r"(\d{8}T\d{6})")


def _file_start_epoch(fp: Path) -> float | None:
    """First timestamp encoded in the filename, as epoch seconds."""
    from datetime import datetime, timezone
    m = _TS_RE.search(fp.name)
    if m:
        dt = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    m = _RANGE_RE.search(fp.name)
    if m:
        dt = datetime.strptime(m.group(1), "%Y%m%d").replace(tzinfo=timezone.utc)
        return dt.timestamp()
    return None


class _GridWriter:
    """One output aggregate per distinct source grid (domains get regridded
    over the years — e.g. gak_1km — and eras must not be silently dropped)."""

    def __init__(self, target: str, tmp_dir: Path, grid_no: int, src: netCDF4.Dataset):
        self.target = target
        self.grid_no = grid_no
        self.idx = 0
        self.last_t = -np.inf
        self.t_first = None
        lat, lon = src["lat"][:], src["lon"][:]
        self.ny, self.nx = len(lat), len(lon)
        self.tmp = tmp_dir / f"{target}_grid{grid_no}.nc4.tmp"
        self.tmp.unlink(missing_ok=True)
        self.ds = netCDF4.Dataset(self.tmp, "w", format="NETCDF4")
        self.ds.createDimension("time", None)
        self.ds.createDimension("lat", self.ny)
        self.ds.createDimension("lon", self.nx)
        vt = self.ds.createVariable("time", "f8", ("time",))
        vt.units = TIME_UNITS
        vt.standard_name = "time"
        self.ds.createVariable("lat", "f8", ("lat",))[:] = lat
        self.ds.createVariable("lon", "f8", ("lon",))[:] = lon
        if "wgs84" in src.variables:
            vw = self.ds.createVariable("wgs84", "i4", ())
            vw.setncatts({a: src["wgs84"].getncattr(a) for a in src["wgs84"].ncattrs()})
        for v in COPY_VARS:
            attrs = {a: src[v].getncattr(a) for a in src[v].ncattrs()
                     if a not in ("_ChunkSizes",)}
            fill = attrs.pop("_FillValue", None)
            var = self.ds.createVariable(
                v, src[v].dtype, ("time", "lat", "lon"),
                zlib=True, complevel=1, shuffle=True,
                chunksizes=_chunksizes(self.ny, self.nx, src[v].dtype.itemsize),
                fill_value=fill,
            )
            var.setncatts(attrs)
            var.set_auto_maskandscale(False)

    def finalize(self, out_dir: Path) -> Path | None:
        self.ds.close()
        if self.idx == 0:
            self.tmp.unlink(missing_ok=True)
            return None
        with netCDF4.Dataset(self.tmp) as chk:
            tv = chk["time"][:]
            assert len(tv) == self.idx and np.all(np.diff(tv) > 0), "verification failed"
            y0 = netCDF4.num2date(tv[0], TIME_UNITS).year
            y1 = netCDF4.num2date(tv[-1], TIME_UNITS).year
        suffix = f"_grid{self.grid_no}" if self.grid_no > 0 else ""
        out = out_dir / f"{self.target}_Resolution_hourly_{y0}_{y1}{suffix}.nc4"
        self.tmp.rename(out)
        return out


def build(target: str, force: bool = False, resume: bool = False,
          files_after: float | None = None) -> Path | None:
    src_dir = folder_for(target)
    files = sorted(list(src_dir.glob("*.nc4")) + list(src_dir.glob("*.nc")))
    if files_after is not None:
        files = [fp for fp in files
                 if (_file_start_epoch(fp) or 0) > files_after]
        log(f"{target}: --files-after filter keeps {len(files)} source files")
    if not files:
        log(f"{target}: no source files in {src_dir} — skipping")
        return None

    out_dir = AGG_ROOT / region_of(target)
    existing = sorted(out_dir.glob(f"{target}_Resolution_hourly_*.nc4"))
    if existing and not force:
        log(f"{target}: {[p.name for p in existing]} exist — skipping build")
        return existing[0]
    out_dir.mkdir(parents=True, exist_ok=True)

    writers: dict[tuple[int, int], _GridWriter] = {}
    n_dup = 0

    # --resume: attach to a crashed run's grid0 tmp, rewind to the boundary
    # of the input file that was in flight, and continue (single-grid only).
    start_i = 0
    if resume:
        old_tmp = out_dir / f"{target}_grid0.nc4.tmp"
        if old_tmp.exists():
            ds_r = netCDF4.Dataset(old_tmp, "a")
            t_arr = np.asarray(ds_r["time"][:], dtype="f8")
            if len(t_arr):
                t_last = float(t_arr[-1])
                # boundary file = last file starting at/before the last
                # written hour; rewrite it fully to flush any partial tail
                starts = [(_file_start_epoch(fp), i) for i, fp in enumerate(files)]
                cand = [i for s, i in starts if s is not None and s <= t_last]
                start_i = max(cand) if cand else 0
                boundary_epoch = starts[start_i][0]
                idx0 = int(np.searchsorted(t_arr, boundary_epoch))
                w = _GridWriter.__new__(_GridWriter)
                w.target, w.grid_no = target, 0
                w.tmp, w.ds = old_tmp, ds_r
                w.ny = ds_r.dimensions["lat"].size
                w.nx = ds_r.dimensions["lon"].size
                w.idx = idx0
                w.last_t = float(t_arr[idx0 - 1]) if idx0 > 0 else -np.inf
                for v in COPY_VARS:
                    ds_r[v].set_auto_maskandscale(False)
                writers[(w.ny, w.nx)] = w
                log(f"{target}: RESUMING from file {start_i + 1}/{len(files)} "
                    f"(rewound to hour index {idx0} of {len(t_arr)})")
            else:
                ds_r.close()

    t_start = _time.time()
    files = files[start_i:]
    for i, fp in enumerate(files):
        try:
            with netCDF4.Dataset(fp) as src:
                shape = (src.dimensions["lat"].size, src.dimensions["lon"].size)
                w = writers.get(shape)
                if w is None:
                    w = _GridWriter(target, out_dir, len(writers), src)
                    writers[shape] = w
                    if len(writers) > 1:
                        log(f"{target}: new grid era detected "
                            f"({shape[0]}x{shape[1]}) — writing separate aggregate")
                t_raw = src["time"]
                t_epoch = netCDF4.date2num(
                    netCDF4.num2date(t_raw[:], t_raw.units), TIME_UNITS)
                t_epoch = np.atleast_1d(np.asarray(t_epoch, dtype="f8"))
                keep = t_epoch > w.last_t  # sources sorted; drops overlap/dupes
                if not keep.any():
                    n_dup += len(t_epoch)
                    continue
                n_dup += int((~keep).sum())
                sel = np.where(keep)[0]
                n = len(sel)
                # sorted times make the keep-mask a contiguous suffix
                k0 = int(sel[0])
                assert n == len(t_epoch) - k0, "non-contiguous dedupe mask"
                w.ds["time"][w.idx:w.idx + n] = t_epoch[sel]
                for v in COPY_VARS:
                    sv = src[v]
                    sv.set_auto_maskandscale(False)
                    if getattr(sv, "ndim", 3) == 3 and len(t_epoch) > 1:
                        # stream day-sized slices — weekly usegc_1km chunks
                        # decode to 7+ GB per variable if read whole (OOM)
                        step = 24
                        for j in range(k0, k0 + n, step):
                            jn = min(k0 + n, j + step)
                            w.ds[v][w.idx + (j - k0): w.idx + (jn - k0)] = sv[j:jn]
                    else:
                        data = sv[:]
                        data = data[sel] if data.ndim == 3 else data[np.newaxis][sel]
                        w.ds[v][w.idx:w.idx + n] = data
                w.idx += n
                w.last_t = float(t_epoch[sel][-1])
        except OSError as e:
            log(f"{target}: unreadable file {fp.name} ({e}) — skipped")
        if (i + 1) % 2000 == 0 or (i + 1) == len(files):
            rate = (i + 1) / max(1e-9, _time.time() - t_start)
            total = sum(w.idx for w in writers.values())
            log(f"{target}: {i + 1}/{len(files)} files, {total} hours written "
                f"({rate:.0f} files/s, {len(writers)} grid era(s))")

    outputs = []
    for w in writers.values():
        out = w.finalize(out_dir)
        if out is not None:
            outputs.append(out)
            log(f"{target}: aggregated -> {out.name} "
                f"({out.stat().st_size / 1e9:.2f} GB, {w.idx} hours)")
    log(f"{target}: {len(outputs)} aggregate(s), {n_dup} duplicate hours dropped")
    return outputs[0] if outputs else None


def upload(target: str, keep_local: bool = False) -> None:
    import s3fs
    region = region_of(target)
    out_dir = AGG_ROOT / region
    cands = sorted(out_dir.glob(f"{target}_Resolution_hourly_*.nc4"))
    if not cands:
        log(f"{target}: no aggregate to upload")
        return
    fs = s3fs.S3FileSystem(anon=False)
    for src in cands:  # multi-grid targets produce one aggregate per era
        dst = f"{S3_BUCKET}/HFR/{region}/{src.name}"

        def _ok() -> bool:
            try:
                return fs.exists(dst) and fs.info(dst).get("size") == src.stat().st_size
            except Exception:
                return False

        try:
            if not _ok():
                log(f"{target}: uploading {src.stat().st_size / 1e9:.2f} GB -> s3://{dst}")
                fs.put(str(src), dst)
            if _ok():
                log(f"{target}: verified on S3 ({src.name}).")
                if not keep_local:
                    src.unlink()
                    log(f"{target}: local aggregate deleted (raw hourly files kept).")
            else:
                log(f"{target}: upload NOT verified — local kept.")
        except Exception as e:
            log(f"{target}: UPLOAD FAILED ({type(e).__name__}: {e}) — "
                "refresh AWS creds (`aws login` + export) and rerun --phase upload.")
            return


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--targets", default="all",
                    help=f"comma-separated subset of {TARGETS} or 'all'")
    ap.add_argument("--phase", default="all", choices=["all", "build", "upload"])
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--keep-local", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="resume a crashed build from its grid0 .tmp file")
    ap.add_argument("--files-after", default=None,
                    help="only process source files starting after this UTC "
                         "time (YYYY-MM-DDTHH) — for building a follow-on segment")
    args = ap.parse_args()

    targets = TARGETS if args.targets == "all" else [t.strip() for t in args.targets.split(",")]
    bad = [t for t in targets if t not in TARGETS]
    if bad:
        ap.error(f"unknown target(s) {bad}; choose from {TARGETS}")

    files_after = None
    if args.files_after:
        from datetime import datetime, timezone
        files_after = datetime.strptime(args.files_after, "%Y-%m-%dT%H").replace(
            tzinfo=timezone.utc).timestamp()

    log(f"targets: {targets}  phase: {args.phase}")
    for t in targets:
        if args.phase in ("all", "build"):
            build(t, force=args.force, resume=args.resume, files_after=files_after)
        if args.phase in ("all", "upload"):
            upload(t, keep_local=args.keep_local)
    log("done.")


if __name__ == "__main__":
    main()
