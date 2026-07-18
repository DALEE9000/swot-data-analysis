#!/usr/bin/env python
"""Colocate every HFR network with SWOT science-phase data.

For each HFR region, finds the science passes crossing that network's actual
data footprint, downloads cycles 001-016 from AVISO+ (each granule fetched
once and cut into per-region segments), regrids the segments into the same
pkl format the app's presets consume (with per-pass time coords), and uploads
the pkls to the project S3 bucket.

Resume-safe at every phase: existing granule segments and pkls are skipped,
and passes with no data in a region are recorded in a manifest so they are
never re-downloaded.

Usage:
    python scripts/build_science_colocations.py --regions gak
    python scripts/build_science_colocations.py --regions all --phase download
    python scripts/build_science_colocations.py --regions all --phase upload
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
os.chdir(REPO)

import numpy as np
import paramiko
import xarray as xr

from swot import data_loaders, swot_utils
from swot.download_swaths import find_swaths
from swotxai.data_utils import swot_regrid
from swotxai.pipeline.io_utils import _save

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
AVISO_HOST = "ftp-access.aviso.altimetry.fr"
AVISO_PORT = 2221
AVISO_USER = os.environ.get("AVISO_USER", "tdmonkman@uchicago.edu")
AVISO_PW   = os.environ.get("AVISO_PW", "2prSvl")
REMOTE_ROOT = "swot_products/l3_karin_nadir/l3_lr_ssh/v3_0/Expert/reproc"

S3_BUCKET = "swot-ai-ssv"

# Mission profile — set by --mission (default science). Cal/val redo
# (user request 2026-07-14) supersedes the legacy hfr_calval_* pkls.
MISSIONS = {
    "science": {
        "sph":       "orbit_data/sph_science_swath.zip",
        "cycles":    [f"{c:03d}" for c in range(1, 17)],
        "local":     Path("SWOT_L3/science"),
        "dir_tmpl":  "Expert_reproc_v3_{region}_science",
        "pkl_tmpl":  "swot_expert_reproc_v3_{region}_science.pkl",
        "hfr_tmpl":  "hfr_science_{region}.pkl",
        "s3_gran":   "SWOT_L3/science",
    },
    "calval": {
        "sph":       "orbit_data/sph_calval_swath.zip",
        "cycles":    [f"{c:03d}" for c in range(474, 579)],
        "local":     Path("SWOT_L3/calval"),
        "dir_tmpl":  "Expert_reproc_v3_{region}_calval",
        "pkl_tmpl":  "swot_expert_reproc_v3_calval_{region}_474_578.pkl",
        "hfr_tmpl":  "hfr_calval_{region}.pkl",
        "s3_gran":   "SWOT_L3/calval",
    },
}
MISSION = "science"  # overridden in main()


def _m() -> dict:
    return MISSIONS[MISSION]


SPH_SCIENCE = MISSIONS["science"]["sph"]  # legacy alias
LOCAL_ROOT = MISSIONS["science"]["local"]
MANIFEST = LOCAL_ROOT / "colocation_manifest.json"
CYCLES = MISSIONS["science"]["cycles"]

# Bounding boxes = each HFR network's measured *data* footprint (not its file
# grid, which is mostly empty ocean) + 0.5 deg pad. Measured 2026-07-10 from
# the last month of data in each archive on S3.
REGIONS: dict[str, tuple[list[float], list[float]]] = {
    "uswc":  ([-126.8, 31.2], [-116.6, 49.9]),
    "usegc": ([-97.7, 22.8],  [-68.0, 44.4]),
    "gak":   ([-131.2, 53.6], [-129.8, 54.8]),
    "akns":  ([-162.7, 70.0], [-154.7, 73.1]),
    "glna":  ([-85.4, 45.3],  [-84.2, 46.4]),
    # measured 2026-07-13 from the newly aggregated archives
    "prvi":  ([-68.4, 15.8],  [-63.8, 19.6]),
    "ushi":  ([-159.4, 19.3], [-154.3, 22.1]),
}

# The SWOT variables the pipeline can use as features — everything else is
# dropped before regridding (3x faster) and the pkls stay small.
KEEP_VARS = ["mdt", "ssha_filtered", "ugos_filtered", "vgos_filtered",
             "ugosa_filtered", "vgosa_filtered"]
CARRY_VARS = ["latitude", "longitude", "time", "quality_flag"]

# Full-network HFR archives on S3, one per region (the colocation target).
HFR_SOURCES = {
    "uswc":  "s3://swot-ai-ssv/HFR/uswc/uswc_6km_Resolution_hourly_2012_2024.nc4",
    "usegc": "s3://swot-ai-ssv/HFR/usegc/usegc_6km_Resolution_hourly_2012_2025.nc4",
    "gak":   "s3://swot-ai-ssv/HFR/gak/gak_2km_Resolution_hourly_2017_2025.nc4",
    "akns":  "s3://swot-ai-ssv/HFR/akns/akns_6km_Resolution_Hourly_RTV_best_2010_2024.nc4",
    "glna":  "s3://swot-ai-ssv/HFR/glna/glna_6km_Resolution_hourly_2022_2025.nc4",
    "prvi":  "s3://swot-ai-ssv/HFR/prvi/prvi_6km_Resolution_hourly_2010_2025.nc4",
    "ushi":  "s3://swot-ai-ssv/HFR/ushi/ushi_6km_Resolution_hourly_2010_2025.nc4",
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def region_dir(region: str) -> Path:
    return LOCAL_ROOT / _m()["dir_tmpl"].format(region=region)


def pkl_path(region: str) -> Path:
    return Path(f"experiments/{region}/swot_cycles/"
                + _m()["pkl_tmpl"].format(region=region))


def hfr_pkl_path(region: str) -> Path:
    return Path(f"experiments/{region}/hfr_target/"
                + _m()["hfr_tmpl"].format(region=region))


def s3_key(region: str) -> str:
    return (f"{S3_BUCKET}/experiments/{region}/swot_cycles/"
            + _m()["pkl_tmpl"].format(region=region))


def hfr_s3_key(region: str) -> str:
    return (f"{S3_BUCKET}/experiments/{region}/hfr_target/"
            + _m()["hfr_tmpl"].format(region=region))


def region_passes(region: str) -> list[str]:
    sw, ne = REGIONS[region]
    return find_swaths(sw_corner=sw, ne_corner=ne, path_to_sph_file=SPH_SCIENCE)


def load_manifest() -> dict:
    if MANIFEST.exists():
        return json.loads(MANIFEST.read_text())
    return {}


def save_manifest(m: dict) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(m, indent=0, sort_keys=True))


# ---------------------------------------------------------------------------
# AVISO SFTP with reconnect
# ---------------------------------------------------------------------------
class Aviso:
    def __init__(self):
        self._ssh = None
        self._sftp = None
        self._connect()

    def _connect(self):
        if self._ssh is not None:
            try:
                self._ssh.close()
            except Exception:
                pass
        self._ssh = paramiko.SSHClient()
        self._ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        self._ssh.connect(AVISO_HOST, AVISO_PORT, AVISO_USER, AVISO_PW, timeout=60)
        self._sftp = self._ssh.open_sftp()

    def _retry(self, op, attempts=6):
        # op is a closure that reads self._sftp at call time, so after a
        # reconnect the retry hits the NEW connection (a bound method of the
        # old client would keep failing on the closed socket forever).
        for i in range(attempts):
            try:
                return op()
            except FileNotFoundError:
                raise  # missing remote path is a real answer, not a flake
            except Exception as e:
                if i == attempts - 1:
                    raise
                log(f"  SFTP error ({type(e).__name__}: {e}) — reconnecting "
                    f"(attempt {i + 2}/{attempts})...")
                time.sleep(10 * (i + 1))
                try:
                    self._connect()
                except Exception as ce:
                    log(f"  reconnect failed ({type(ce).__name__}: {ce}) — will retry")

    def listdir(self, path: str) -> list[str]:
        return self._retry(lambda: self._sftp.listdir(path))

    def get(self, remote: str, local: str) -> None:
        self._retry(lambda: self._sftp.get(remote, local))

    def close(self):
        try:
            self._ssh.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Phase 1 — download granules, one fetch per (cycle, pass), cut per region
# ---------------------------------------------------------------------------
def phase_download(regions: list[str]) -> None:
    pass_map: dict[str, list[str]] = {}
    for r in regions:
        ps = region_passes(r)
        log(f"{r}: {len(ps)} science passes over HFR footprint")
        for p in ps:
            pass_map.setdefault(p, []).append(r)
    log(f"{len(pass_map)} unique passes to fetch across {len(regions)} regions")

    manifest = load_manifest()
    av = Aviso()
    tmp = LOCAL_ROOT / "_tmp_granule.nc"
    LOCAL_ROOT.mkdir(parents=True, exist_ok=True)
    n_fetched = n_skipped = 0

    try:
        for cycle in CYCLES:
            remote_dir = f"{REMOTE_ROOT}/cycle_{cycle}"
            try:
                files = av.listdir(remote_dir)
            except FileNotFoundError:
                log(f"cycle_{cycle}: not on server — skipping")
                continue
            by_pass = {f.split("_")[6]: f for f in files if f.endswith(".nc")}

            for pass_id in sorted(pass_map):
                fname = by_pass.get(pass_id)
                if fname is None:
                    continue  # pass missing from this cycle (normal)
                need = []
                for r in pass_map[pass_id]:
                    key = f"{r}/{cycle}/{pass_id}"
                    out = region_dir(r) / f"cycle_{cycle}" / fname
                    # "uploaded" = segment is on S3 and the local copy was
                    # deliberately deleted — do not re-fetch.
                    if manifest.get(key) in ("empty", "uploaded") or out.exists():
                        n_skipped += 1
                        continue
                    need.append((r, key, out))
                if not need:
                    continue

                av.get(f"{remote_dir}/{fname}", str(tmp))
                with xr.open_dataset(tmp) as raw:
                    swath = raw.load()
                for r, key, out in need:
                    lat_b = [REGIONS[r][0][1], REGIONS[r][1][1]]
                    seg = swot_utils.subset(swath, lat_b)
                    if seg is None:
                        manifest[key] = "empty"
                    else:
                        out.parent.mkdir(parents=True, exist_ok=True)
                        seg.to_netcdf(out)
                        manifest[key] = "saved"
                swath.close()
                tmp.unlink(missing_ok=True)
                n_fetched += 1
                if n_fetched % 10 == 0:
                    save_manifest(manifest)
                    log(f"cycle_{cycle}: {n_fetched} granules fetched so far "
                        f"({n_skipped} segment(s) already present)")
            save_manifest(manifest)
            log(f"cycle_{cycle} complete.")
    finally:
        save_manifest(manifest)
        av.close()
        tmp.unlink(missing_ok=True)
    log(f"Download phase done: {n_fetched} granules fetched, {n_skipped} segments already present.")


# ---------------------------------------------------------------------------
# Phase 2 — regrid segments into per-region pkls (slim, float32, time coords)
# ---------------------------------------------------------------------------
def phase_pkl(regions: list[str], force: bool = False) -> None:
    for r in regions:
        out = pkl_path(r)
        if out.exists() and not force:
            log(f"{r}: {out} exists — skipping (use --force-pkl to rebuild)")
            continue
        sw, ne = REGIONS[r]
        data: dict[str, list] = {}
        n_entries = 0
        t0 = time.time()
        for cycle in CYCLES:
            cdir = region_dir(r) / f"cycle_{cycle}"
            if not cdir.exists():
                data[cycle] = []
                continue
            swaths = data_loaders.load_cycle(
                path=str(region_dir(r)), cycle=cycle, pass_ids=None, subset=False,
            )
            entries = []
            for swath in swaths:
                if swath is None:
                    continue
                keep = [v for v in KEEP_VARS + CARRY_VARS if v in swath.data_vars]
                regridded = swot_regrid(swath[keep])
                for v in regridded.data_vars:
                    regridded[v] = regridded[v].astype("float32")
                if "time" not in regridded.coords:
                    log(f"  WARNING {r}/cycle_{cycle}: regridded pass lacks time coord")
                entries.append(regridded)
            data[cycle] = entries
            n_entries += len(entries)
            log(f"{r}/cycle_{cycle}: {len(entries)} passes regridded "
                f"({time.time() - t0:.0f}s elapsed)")
        _save(data, out)
        log(f"{r}: pkl saved -> {out} ({out.stat().st_size / 1e6:.0f} MB, "
            f"{n_entries} pass entries)")


# ---------------------------------------------------------------------------
# Phase 3 — HFR colocation: 24-h-mean HFR at each pass time, on the SWOT grid.
# Loaded one cycle window (~21 days) at a time, so the big-network archives
# (usegc's year would be ~19 GB whole) never blow up memory.
# ---------------------------------------------------------------------------
def phase_hfr(regions: list[str], force: bool = False) -> None:
    import s3fs

    from swotxai.data_utils import hfr_on_swot
    from swotxai.pipeline.io_utils import _load
    from swotxai.pipeline.steps_data import _coord_slice

    for r in regions:
        out = hfr_pkl_path(r)
        if out.exists() and not force:
            log(f"{r}: {out} exists — skipping (use --force-pkl to rebuild)")
            continue
        src_pkl = pkl_path(r)
        if src_pkl.exists():
            swot_regridded = _load(src_pkl)
        else:
            # local copy was deleted after verified upload — stream from S3
            import pickle

            import s3fs as _s3fs
            _fs = _s3fs.S3FileSystem(anon=True)
            if not _fs.exists(s3_key(r)):
                log(f"{r}: no SWOT pkl locally or at s3://{s3_key(r)} — run the pkl phase first")
                continue
            log(f"{r}: streaming SWOT pkl from s3://{s3_key(r)}...")
            with _fs.open(s3_key(r)) as f:
                swot_regridded = pickle.load(f)
        sw, ne = REGIONS[r]

        def _open_sub():
            # transient S3 errors (endpoint drops, spurious Forbidden) are
            # survivable — retry the open itself, not just the reads
            last_err = None
            for i in range(5):
                try:
                    _fs2 = s3fs.S3FileSystem(anon=True, skip_instance_cache=True)
                    _f2 = _fs2.open(HFR_SOURCES[r].replace("s3://", ""))
                    _src2 = xr.open_dataset(_f2, engine="h5netcdf")[["u", "v"]]
                    _sub2 = _src2.sel(
                        lat=_coord_slice(_src2, "lat", sw[1] - 0.5, ne[1] + 0.5),
                        lon=_coord_slice(_src2, "lon", sw[0] - 0.5, ne[0] + 0.5),
                    )
                    return _f2, _src2, _sub2
                except Exception as e:
                    last_err = e
                    log(f"{r}: HFR source open failed ({type(e).__name__}: {e}) "
                        f"— retry {i + 2}/5 in {60 * (i + 1)}s")
                    time.sleep(60 * (i + 1))
            raise last_err

        f, src, sub = _open_sub()
        log(f"{r}: HFR grid {sub.sizes['lat']}x{sub.sizes['lon']}, colocating "
            f"{sum(len(v) for v in swot_regridded.values())} passes...")

        pad = np.timedelta64(36, "h")
        # Per-cycle checkpoint: a network crash resumes from the last
        # completed cycle instead of losing hours of colocation work.
        partial = out.parent / (out.stem + ".partial.pkl")
        result: dict[str, list] = {}
        if partial.exists() and not force:
            result = _load(partial)
            log(f"{r}: resuming from checkpoint ({len(result)} cycles already done)")
        n_total = n_no_time = 0
        try:
            for cycle in sorted(swot_regridded):
                if cycle in result:
                    n_total += len(result[cycle])
                    continue
                ds_list = swot_regridded[cycle] or []
                times = []
                for ds in ds_list:
                    if ds is None or "time" not in ds.coords:
                        continue
                    t = np.atleast_1d(ds.coords["time"].values).astype("datetime64[ns]")
                    t = t[~np.isnat(t)]
                    if len(t):
                        times.append(t[0])
                if not times:
                    result[cycle] = []
                    continue
                win = None
                for attempt in range(4):
                    try:
                        win = sub.sel(time=slice(min(times) - pad, max(times) + pad)).load()
                        break
                    except Exception as e:
                        if attempt == 3:
                            raise
                        log(f"{r}/cycle_{cycle}: window load failed "
                            f"({type(e).__name__}) — reconnecting "
                            f"(attempt {attempt + 2}/4)...")
                        time.sleep(30 * (attempt + 1))
                        try:
                            src.close(); f.close()
                        except Exception:
                            pass
                        f, src, sub = _open_sub()
                # Seasonal networks (e.g. akns under sea ice) can have zero
                # HFR samples in a cycle's window — that's a valid answer.
                if win.sizes.get("time", 0) == 0:
                    result[cycle] = []
                    log(f"{r}/cycle_{cycle}: no HFR data in window — 0 passes colocated")
                    continue
                interp_list = []
                for ds in ds_list:
                    if ds is None or "time" not in ds.coords:
                        n_no_time += 1
                        continue
                    t = np.atleast_1d(ds.coords["time"].values).astype("datetime64[ns]")
                    t = t[~np.isnat(t)]
                    if not len(t):
                        n_no_time += 1
                        continue
                    t0 = win["time"].sel(time=t[0], method="nearest").values
                    if abs((t0 - t[0]) / np.timedelta64(1, "h")) > 36:
                        continue  # nearest HFR sample too far from pass time
                    # 25-h centered window: the standard HFR detiding filter
                    # (kills diurnal + semidiurnal tides symmetrically).
                    margin = np.timedelta64(31, "h")
                    roll = (win.sel(time=slice(t0 - margin, t0 + margin))
                               .rolling(time=25, center=True, min_periods=1)
                               .mean().sel(time=t0))
                    res = hfr_on_swot(roll, ds)
                    if res is not None:
                        interp_list.append(res)
                result[cycle] = interp_list
                n_total += len(interp_list)
                del win
                _save(result, partial)  # checkpoint after every cycle
                log(f"{r}/cycle_{cycle}: {len(interp_list)} passes colocated")
        finally:
            src.close()
            f.close()
        if n_no_time:
            log(f"{r}: WARNING {n_no_time} passes had no timestamp")
        _save(result, out)
        partial.unlink(missing_ok=True)
        log(f"{r}: HFR target saved -> {out} "
            f"({out.stat().st_size / 1e6:.0f} MB, {n_total} pass entries)")


# ---------------------------------------------------------------------------
# Phase 4 — upload pkls (SWOT + HFR target) to S3; delete local after verify
# ---------------------------------------------------------------------------
def _verified_on_s3(fs, key: str, local: Path) -> bool:
    try:
        return fs.exists(key) and fs.info(key).get("size") == local.stat().st_size
    except Exception:
        return False


def phase_upload(regions: list[str], keep_local: bool = False) -> None:
    import s3fs
    fs = s3fs.S3FileSystem(anon=False)
    for r in regions:
        for src, dst in ((pkl_path(r), s3_key(r)),
                         (hfr_pkl_path(r), hfr_s3_key(r))):
            if not src.exists():
                log(f"{r}: no {src} — skipping")
                continue
            try:
                if not _verified_on_s3(fs, dst, src):
                    log(f"{r}: uploading {src.stat().st_size / 1e6:.0f} MB -> s3://{dst}")
                    fs.put(str(src), dst)
                if _verified_on_s3(fs, dst, src):
                    log(f"{r}: verified on S3 ({dst.rsplit('/', 1)[-1]}).")
                    if not keep_local:
                        src.unlink()
                        log(f"{r}: local copy deleted ({src}).")
                else:
                    log(f"{r}: upload NOT verified (size mismatch) — keeping local copy.")
            except Exception as e:
                log(f"{r}: UPLOAD FAILED ({type(e).__name__}: {e}). "
                    "If credentials expired, run `aws login` and rerun "
                    "--phase upload. Local copy kept.")


# ---------------------------------------------------------------------------
# Phase 5 — upload raw granule segments to SWOT_L3/science/ (mirrors local
# layout); each segment's local copy is deleted once verified on S3.
# ---------------------------------------------------------------------------
def phase_granules(regions: list[str], keep_local: bool = False) -> None:
    import s3fs
    fs = s3fs.S3FileSystem(anon=False)
    manifest = load_manifest()
    for r in regions:
        root = region_dir(r)
        if not root.exists():
            log(f"{r}: no granules at {root} — run the download phase first")
            continue
        files = sorted(root.rglob("*.nc"))
        n_up = n_del = 0
        for fp in files:
            key = f"{S3_BUCKET}/{_m()['s3_gran']}/{fp.relative_to(LOCAL_ROOT).as_posix()}"
            try:
                if not _verified_on_s3(fs, key, fp):
                    fs.put(str(fp), key)
                    n_up += 1
                if _verified_on_s3(fs, key, fp):
                    cycle = fp.parent.name.replace("cycle_", "")
                    pass_id = fp.name.split("_")[6]
                    manifest[f"{r}/{cycle}/{pass_id}"] = "uploaded"
                    if not keep_local:
                        fp.unlink()
                        n_del += 1
                if (n_up + n_del) % 100 == 0 and (n_up or n_del):
                    log(f"{r}: {n_up} uploaded, {n_del} local deleted...")
                    save_manifest(manifest)
            except Exception as e:
                save_manifest(manifest)
                log(f"{r}: granule upload failed for {fp.name} "
                    f"({type(e).__name__}: {e}) — rerun --phase granules to resume.")
                return
        save_manifest(manifest)
        if not keep_local:
            # tidy now-empty cycle dirs
            for d in sorted(root.rglob("cycle_*"), reverse=True):
                if d.is_dir() and not any(d.iterdir()):
                    d.rmdir()
        log(f"{r}: granules synced ({n_up} uploaded, {n_del} local copies deleted).")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--regions", default="all",
                    help=f"comma-separated subset of {list(REGIONS)} or 'all'")
    ap.add_argument("--phase", default="all",
                    help="comma-separated subset of "
                         "[download, pkl, hfr, upload, granules] or 'all'")
    ap.add_argument("--force-pkl", action="store_true",
                    help="rebuild pkls even if they exist")
    ap.add_argument("--keep-local", action="store_true",
                    help="keep local copies after verified S3 upload "
                         "(default: delete them)")
    ap.add_argument("--mission", default="science", choices=list(MISSIONS),
                    help="orbit phase: science (cycles 1-16) or calval (474-578)")
    args = ap.parse_args()

    global MISSION, SPH_SCIENCE, LOCAL_ROOT, MANIFEST, CYCLES
    MISSION = args.mission
    SPH_SCIENCE = _m()["sph"]
    LOCAL_ROOT = _m()["local"]
    MANIFEST = LOCAL_ROOT / "colocation_manifest.json"
    CYCLES = _m()["cycles"]

    regions = list(REGIONS) if args.regions == "all" else [
        r.strip() for r in args.regions.split(",")
    ]
    unknown = [r for r in regions if r not in REGIONS]
    if unknown:
        ap.error(f"unknown region(s) {unknown}; choose from {list(REGIONS)}")

    all_phases = ["download", "pkl", "hfr", "upload", "granules"]
    phases = all_phases if args.phase == "all" else [
        p.strip() for p in args.phase.split(",")
    ]
    bad = [p for p in phases if p not in all_phases]
    if bad:
        ap.error(f"unknown phase(s) {bad}; choose from {all_phases}")

    log(f"mission: {MISSION}  regions: {regions}  phases: {phases}")
    if "download" in phases:
        phase_download(regions)
    if "pkl" in phases:
        phase_pkl(regions, force=args.force_pkl)
    if "hfr" in phases:
        phase_hfr(regions, force=args.force_pkl)
    if "upload" in phases:
        phase_upload(regions, keep_local=args.keep_local)
    if "granules" in phases:
        phase_granules(regions, keep_local=args.keep_local)
    log("done.")


if __name__ == "__main__":
    main()
