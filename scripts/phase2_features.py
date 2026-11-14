"""Phase 2 features on top of the frozen holdout rows: wind history, Coriolis, season.

For every region frozen by scripts/holdout_eval.py (experiments/holdout/<rid>.npz) this
writes experiments/holdout/<rid>_p2.npz with one float32 matrix E (n_rows, 19) and its
column names:

    era5_u_lag{3,6,12,18,24,48,96,192}h, era5_v_lag{...}h   ERA5 10 m wind at the SWOT
        observation time minus the lag, nearest hour and nearest 0.25 deg cell (16 columns;
        lag 0 is already in the stencil block of the frozen X); NaN where the ERA5 record
        does not reach back that far
    f          Coriolis parameter 2 Omega sin(lat), 1/s (times 1e4 so it is order one)
    sin_doy, cos_doy   season as the phase of the day of year

Usage:
    python scripts/phase2_features.py            # every frozen region
    python scripts/phase2_features.py uswc usegc

Reads the per-region ERA5 pickles (experiments/<rid>/era5/era5wind_science_<rid>_10m.pkl).
Nothing is downloaded. Output lands in experiments/holdout/, a junction to Z:.
"""
from __future__ import annotations

import pickle
import sys
import time
from pathlib import Path

import swotxai  # noqa: F401  (torch-before-numpy DLL ordering)
import numpy as np

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments" / "holdout"
LAGS_H = [3, 6, 12, 18, 24, 48, 96, 192]
NAMES = [f"era5_{c}_lag{h}h" for c in "uv" for h in LAGS_H] + ["f_1e4", "sin_doy", "cos_doy"]
OMEGA = 7.2921e-5


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def build(rid: str) -> Path:
    out = OUT / f"{rid}_p2.npz"
    if out.exists():
        log(f"{rid}: {out.name} exists, skipping"); return out
    d = np.load(OUT / f"{rid}.npz")
    t, lat, lon = d["time"], d["lat"], d["lon"]
    n = len(t)
    era = pickle.load(open(REPO / "experiments" / rid / "era5" / f"era5wind_science_{rid}_10m.pkl", "rb"))
    et = era["time"].values.astype("datetime64[s]").astype(np.int64)
    assert np.all(np.diff(et) == 3600), f"{rid}: ERA5 not hourly"
    elat, elon = era["latitude"].values, era["longitude"].values
    lon_r = np.where(lon > elon.max() + 0.25, lon - 360, lon)          # match the pickle's convention
    lon_r = np.where(lon_r < elon.min() - 0.25, lon_r + 360, lon_r)
    yi = np.clip(np.round((lat - elat[0]) / (elat[1] - elat[0])).astype(np.int32), 0, len(elat) - 1)
    xi = np.clip(np.round((lon_r - elon[0]) / (elon[1] - elon[0])).astype(np.int32), 0, len(elon) - 1)
    assert np.abs(elat[yi] - lat).max() < 0.26 and np.abs(elon[xi] - lon_r).max() < 0.26, f"{rid}: rows outside the ERA5 box"
    E = np.full((n, len(NAMES)), np.nan, np.float32)
    for c, comp in enumerate("uv"):
        A = era[f"era5_{comp}"].values.astype(np.float32)             # (time, lat, lon)
        for j, h in enumerate(LAGS_H):
            ti = np.round((t - h * 3600 - et[0]) / 3600).astype(np.int64)
            ok = (ti >= 0) & (ti < len(et))
            col = np.full(n, np.nan, np.float32)
            col[ok] = A[ti[ok], yi[ok], xi[ok]]
            E[:, c * len(LAGS_H) + j] = col
        del A
    E[:, 16] = 2 * OMEGA * np.sin(np.deg2rad(lat)) * 1e4
    doy = (t.astype("datetime64[s]") - t.astype("datetime64[s]").astype("datetime64[Y]")).astype("timedelta64[D]").astype(float)
    E[:, 17] = np.sin(2 * np.pi * doy / 365.25); E[:, 18] = np.cos(2 * np.pi * doy / 365.25)
    np.savez(out, E=E, names=np.array(NAMES))
    fin = np.isfinite(E[:, :16]).all(1).mean()
    log(f"{rid}: {n:,} rows, lag winds finite on {100 * fin:.1f} %, f range {E[:, 16].min():.2f}..{E[:, 16].max():.2f} -> {out.name}")
    return out


if __name__ == "__main__":
    rids = sys.argv[1:] or sorted(p.stem for p in OUT.glob("*.npz") if not p.stem.endswith("_p2") and "/" not in p.stem)
    for rid in rids:
        build(rid)
