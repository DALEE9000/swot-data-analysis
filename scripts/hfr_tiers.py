"""Build alternative HFR truth tiers for the held-out rows from the raw hourly record.

The project's targets are 25-hour centred running means (see README, "HFR
colocation targets"). This script builds, for every held-out row of a
region (as replayed by scripts/baselines_eval.py into baselines/meta_<rid>.npz):

    raw     the hourly field at the hour nearest the SWOT pass (|dt| <= 1 h)
    m25     the 25-h centred mean at that hour  (should reproduce the target)
    bw18    Butterworth order 2, 18-h cutoff, zero-phase (filtfilt), at that hour
    bw48    Butterworth order 2, 48-h cutoff, zero-phase, at that hour

The hourly record is read from the external-drive mirror of HFRNet files
(half-monthly aggregates for uswc/usegc, one file per hour elsewhere) for
the held-out window plus 10 days of padding on each side; the filter runs
along time on the native 6-km grid, and the result is linearly interpolated
to each row's (lat, lon), as the colocation build does. Cells whose hourly
coverage within +-2 cutoff periods of the pass is below 80 % are NaN
(short gaps <= 6 h are linearly interpolated before filtering).

Output: experiments/evolve/<run>/baselines/tiers_<rid>.npz with arrays
y_raw, y_m25, y_bw18, y_bw48 (n_rows, 2) float32, plus a verification
summary (correlation and RMS difference between y_m25 and the stored target).

Run (project env):
    python scripts/hfr_tiers.py --regions uswc,usegc,prvi,ushi
"""
from __future__ import annotations

import argparse
import glob
import os
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt

ROOT = Path(__file__).resolve().parents[1]
HFR_ROOT = Path(os.environ.get("HFR_DATA_ROOT", "Z:/data_repository/columbia_oceanography/HFR/Code/Data"))
RES = {"uswc": "6km", "usegc": "6km", "prvi": "6km", "ushi": "6km", "gak": "2km", "glna": "6km"}
PAD_H = 240          # hours of record each side of the held-out window
TIERS = {"bw18": 18.0, "bw48": 48.0}


def hourly_files(rid: str, t0: np.datetime64, t1: np.datetime64) -> list[str]:
    """Files overlapping [t0, t1]; handles half-monthly aggregates and per-hour files."""
    d = HFR_ROOT / f"{rid}_{RES[rid]}_Resolution_hourly"
    out = []
    for p in sorted(glob.glob(str(d / "*.nc4"))):
        name = Path(p).name
        m = re.search(r"(\d{8})_(\d{8})\.nc4$", name)
        if m:                                           # aggregate: start_end
            a = np.datetime64(f"{m[1][:4]}-{m[1][4:6]}-{m[1][6:]}")
            b = np.datetime64(f"{m[2][:4]}-{m[2][4:6]}-{m[2][6:]}") + np.timedelta64(1, "D")
        else:
            m = re.search(r"(\d{8})T(\d{2})\d{4}\.nc4$", name)
            if not m:
                continue
            a = np.datetime64(f"{m[1][:4]}-{m[1][4:6]}-{m[1][6:]}T{m[2]}:00")
            b = a + np.timedelta64(1, "h")
        if b >= t0 and a <= t1:
            out.append(p)
    return out


def butter_lowpass(arr: np.ndarray, cutoff_h: float) -> np.ndarray:
    """Zero-phase Butterworth (order 2) along axis 0 of an hourly (t, n) array with no NaN."""
    b, a = butter(2, 2.0 / cutoff_h)             # Wn = (1/cutoff) / (fs/2), fs = 1 per hour
    return filtfilt(b, a, arr, axis=0)


def build(rid: str, run: str) -> dict:
    out_dir = ROOT / "experiments" / "evolve" / run / "baselines"
    meta = dict(np.load(out_dir / f"meta_{rid}.npz", allow_pickle=True))
    tp = meta["swot_time"].astype("datetime64[s]")
    t0 = (tp.min() - np.timedelta64(PAD_H, "h")).astype("datetime64[h]")
    t1 = (tp.max() + np.timedelta64(PAD_H, "h")).astype("datetime64[h]")
    files = hourly_files(rid, t0, t1)
    if not files:
        raise FileNotFoundError(f"{rid}: no hourly files in {t0}..{t1}")
    tic = time.time()
    ds = xr.open_mfdataset(files, combine="by_coords", data_vars=["u", "v"], coords="minimal",
                           compat="override", parallel=False)[["u", "v"]]
    ds = ds.sel(time=slice(t0, t1))
    lat0, lat1 = meta["lat"].min() - 0.3, meta["lat"].max() + 0.3
    lon0, lon1 = meta["lon"].min() - 0.3, meta["lon"].max() + 0.3
    ds = ds.sel(lat=slice(lat0, lat1), lon=slice(lon0, lon1))
    # regular hourly axis (gaps -> NaN), drop duplicate hours
    ds = ds.drop_duplicates("time")
    hours = np.arange(t0, t1 + np.timedelta64(1, "h"), np.timedelta64(1, "h"))
    ds = ds.reindex(time=hours).load()
    lat, lon = ds["lat"].values.astype(float), ds["lon"].values.astype(float)
    U = np.stack([ds["u"].values, ds["v"].values], axis=-1)        # (t, lat, lon, 2)
    print(f"  {rid}: {len(files)} files, {U.shape[0]} h x {U.shape[1]}x{U.shape[2]} cells, "
          f"{np.isfinite(U[..., 0]).mean():.1%} finite, loaded in {time.time()-tic:.0f}s")

    T, NY, NX, _ = U.shape
    full = U.reshape(T, NY * NX * 2)
    keep = np.where(np.isfinite(full).any(0))[0]          # cells that ever carry data
    flat = full[:, keep].astype(np.float32)
    del U, full
    finite = np.isfinite(flat)
    print(f"  {rid}: {len(keep):,} of {NY*NX*2:,} cell-components ever finite; working arrays "
          f"{flat.nbytes/1e9:.2f} GB each")
    # 25-h centred mean with min_periods=1 (the colocation build's definition)
    s = np.where(finite, flat, np.float32(0)); c = finite.astype(np.float32)
    def run_mean(x):
        return uniform_filter1d(x, 25, axis=0, mode="constant") * 25.0
    cnt = run_mean(c)
    m25 = run_mean(s) / np.maximum(cnt, 1e-9)
    m25[cnt < 0.5] = np.nan

    # gap-filled copy for the Butterworth tiers: linear interpolation of gaps <= 6 h
    filled = flat.copy()
    idx = np.arange(T)
    cols = np.where(finite.mean(0) > 0.5)[0]
    for j in cols:
        f = finite[:, j]
        if f.all():
            continue
        filled[:, j] = np.interp(idx, idx[f], flat[f, j])
    bw = {}
    for name, cut in TIERS.items():
        arr = np.full_like(flat, np.nan)
        arr[:, cols] = butter_lowpass(filled[:, cols], cut)
        # coverage mask: fraction of finite hours within +-3 cutoff periods
        w = int(2 * cut)
        cov = run_window_mean(c, 2 * w + 1)
        arr[cov < 0.8] = np.nan
        bw[name] = arr

    # sample every held-out pass at its nearest hour and interpolate to the rows
    out = {k: np.full((len(tp), 2), np.nan, np.float32) for k in ["y_raw", "y_m25", *[f"y_{n}" for n in TIERS]]}
    blocks = meta["cycle"] * 1000 + meta["pass_idx"]
    for bid in np.unique(blocks):
        rows = np.where(blocks == bid)[0]
        t = tp[rows[0]].astype("datetime64[h]")
        dt = (tp[rows[0]] - t.astype("datetime64[s]")) / np.timedelta64(1, "h")
        ti = int((t + (np.timedelta64(1, "h") if dt > 0.5 else np.timedelta64(0, "h")) - t0) / np.timedelta64(1, "h"))
        if not (0 <= ti < T):
            continue
        pts = np.column_stack([meta["lat"][rows], meta["lon"][rows]])
        for key, src in [("y_raw", flat), ("y_m25", m25), *[(f"y_{n}", bw[n]) for n in TIERS]]:
            field = np.full(NY * NX * 2, np.nan, np.float32)
            field[keep] = src[ti]
            field = field.reshape(NY, NX, 2)
            for comp in range(2):
                interp = RegularGridInterpolator((lat, lon), field[..., comp], method="linear",
                                                 bounds_error=False, fill_value=np.nan)
                out[key][rows, comp] = interp(pts)

    # verification against the stored target (the 25-h mean the pipeline trained on)
    ver = {}
    for comp, k in enumerate(["y_u", "y_v"]):
        a, b = out["y_m25"][:, comp], meta[k]
        ok = np.isfinite(a) & np.isfinite(b)
        ver[k] = {"n_both": int(ok.sum()), "n_target": int(np.isfinite(b).sum()),
                  "corr": float(np.corrcoef(a[ok], b[ok])[0, 1]), "rms_diff": float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2))),
                  "target_std": float(b[np.isfinite(b)].std())}
    for key in out:
        ver[key + "_finite_frac"] = float(np.isfinite(out[key][:, 0]).mean())
    np.savez(out_dir / f"tiers_{rid}.npz", **out, verification=json.dumps(ver))
    print(f"  {rid}: verification vs stored target: " + ", ".join(
        f"{k}: r={v['corr']:.4f} rms={v['rms_diff']:.4f} (std {v['target_std']:.3f}, n={v['n_both']:,}/{v['n_target']:,})"
        for k, v in ver.items() if isinstance(v, dict)))
    print("  finite fractions: " + ", ".join(f"{k[:-12]}={v:.2f}" for k, v in ver.items() if k.endswith("_finite_frac")))
    return ver


def run_window_mean(x: np.ndarray, width: int) -> np.ndarray:
    return uniform_filter1d(x, width, axis=0, mode="nearest")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="evolve_all_science_v1")
    ap.add_argument("--regions", default="uswc,usegc,prvi,ushi")
    a = ap.parse_args()
    for rid in a.regions.split(","):
        print(f"== {rid}")
        build(rid, a.run)
