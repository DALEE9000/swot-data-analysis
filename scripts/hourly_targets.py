"""Hourly, detided radar targets for the holdout rows (the "25 rows per pass" experiment).

The holdout target in experiments/holdout/<rid>.npz is the 25-hour centred mean radar
current, which averages away the inertial response that carries wind history (research
log 2026-11-14 01:10). This script keeps, for every radar cell a holdout row touches, the
raw hourly current over the whole holdout span and a tidal prediction fitted on the
training year only, so the builder in hourly_eval.py can take the detided current at any
hour offset from the pass without touching the raw archive again.

Output experiments/holdout/<rid>_hourly.npz (on Z: through the junction):
    t0          int64, seconds since epoch of hour index 0 (regular hourly axis)
    n_hours     number of hours on the axis
    cell_lat, cell_lon   (n_cells,) the touched radar cells, nearest-cell to each row
    row_cell    (n_rows,) int32 index into the cells, aligned with <rid>.npz rows
    raw         (n_hours, n_cells, 2) float32 hourly u, v from the archive, NaN where absent
    tide        (n_hours, n_cells, 2) float32 harmonic prediction (M2 S2 N2 K1 O1, no mean),
                NaN for cells with fewer than MIN_HOURS valid training hours
    n_fit       (n_cells,) valid training hours per cell
    coef        (n_cells, 2, 11) harmonic coefficients [mean, cos/sin x 5]

Usage:
    python scripts/hourly_targets.py uswc usegc
Reads HFR_DATA_ROOT/<rid>_6km_Resolution_hourly (Z:). Nothing is downloaded.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import swotxai  # noqa: F401  (torch-before-numpy DLL ordering)
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_hfr_multires import index_folder  # noqa: E402

REPO = Path(__file__).resolve().parents[1]
OUT = REPO / "experiments" / "holdout"
RAW_ROOT = Path(os.environ.get("HFR_DATA_ROOT", r"Z:\data_repository\columbia_oceanography\HFR\Code\Data"))
TRAIN_LAST_CYCLE = 16
MIN_HOURS = 2000
# periods in hours: M2, S2, N2, K1, O1
PERIODS = np.array([12.4206012, 12.0, 12.6583475, 23.9344696, 25.8193387])
H = np.timedelta64(1, "h")


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def design(hours: np.ndarray) -> np.ndarray:
    """(n, 11): mean, then cos/sin of each constituent, hours = float hours since t0."""
    w = 2 * np.pi / PERIODS
    ph = hours[:, None] * w[None, :]
    return np.column_stack([np.ones(len(hours)), np.cos(ph), np.sin(ph)]).astype(np.float64)


def fit_harmonics(A: np.ndarray, Y: np.ndarray):
    """Masked least squares per cell. A (n, 11); Y (n, n_cells) with NaN gaps.
    Returns coef (n_cells, 11) and the valid-hour count. Normal equations with the
    mask as weights, batched over cells in chunks."""
    n_c = Y.shape[1]
    coef = np.full((n_c, A.shape[1]), np.nan)
    nfit = np.zeros(n_c, np.int32)
    for s in range(0, n_c, 2000):
        y = Y[:, s:s + 2000]
        m = np.isfinite(y)
        y0 = np.where(m, y, 0.0)
        AtA = np.einsum("ti,tc,tj->cij", A, m.astype(np.float64), A)
        Atb = np.einsum("ti,tc->ci", A, y0)
        cnt = m.sum(0)
        ok = cnt >= MIN_HOURS
        if ok.any():
            coef[s:s + 2000][ok] = np.linalg.solve(AtA[ok], Atb[ok][..., None])[..., 0]
        nfit[s:s + 2000] = cnt
    return coef, nfit


def build(rid: str) -> Path:
    out = OUT / f"{rid}_hourly.npz"
    if out.exists():
        log(f"{rid}: {out.name} exists, skipping"); return out
    d = np.load(OUT / f"{rid}.npz")
    t, lat, lon, cyc = d["time"], d["lat"], d["lon"], d["cycle"]
    files = index_folder(RAW_ROOT / f"{rid}_6km_Resolution_hourly")
    with xr.open_dataset(files[-1][2]) as ds:
        glat, glon = ds["lat"].values, ds["lon"].values
    yi = np.clip(np.round((lat - glat[0]) / (glat[1] - glat[0])).astype(np.int32), 0, len(glat) - 1)
    xi = np.clip(np.round((lon - glon[0]) / (glon[1] - glon[0])).astype(np.int32), 0, len(glon) - 1)
    flat = yi.astype(np.int64) * len(glon) + xi
    cells, row_cell = np.unique(flat, return_inverse=True)
    cy, cx = cells // len(glon), cells % len(glon)
    log(f"{rid}: {len(t):,} rows touch {len(cells):,} of {len(glat) * len(glon):,} radar cells")

    t0 = (np.datetime64(int(t.min()), "s") - np.timedelta64(13, "h")).astype("datetime64[h]")
    t1 = (np.datetime64(int(t.max()), "s") + np.timedelta64(14, "h")).astype("datetime64[h]")
    n_hours = int((t1 - t0) / H) + 1
    raw = np.full((n_hours, len(cells), 2), np.nan, np.float32)
    lo, hi = t0.astype("datetime64[ns]"), t1.astype("datetime64[ns]")
    sel = [(s, e, fp) for s, e, fp in files if s <= hi and (e or s) >= lo]
    log(f"{rid}: hourly axis {t0} .. {t1} ({n_hours:,} h) from {len(sel):,} archive files")
    filled = 0
    for k, (_s, _e, fp) in enumerate(sel):
        try:   # full read then numpy pointwise index: xarray's vectorised isel on a chunk file took 123 s, this 1 s
            with xr.open_dataset(fp) as ds:
                sub = ds[["u", "v"]].sel(time=slice(lo, hi))
                tv = sub["time"].values.astype("datetime64[h]")
                if not len(tv):
                    continue
                hi_idx = ((tv - t0) / H).astype(np.int64)
                u = sub["u"].values[:, cy, cx].astype(np.float32); v = sub["v"].values[:, cy, cx].astype(np.float32)
        except OSError as e:
            log(f"  skip {fp.name}: {e}"); continue
        raw[hi_idx, :, 0] = u; raw[hi_idx, :, 1] = v
        filled += len(tv)
        if k % 2000 == 0:
            log(f"  {k:,}/{len(sel):,} files, {filled:,} grid-hours")
    log(f"{rid}: archive read, {np.isfinite(raw[..., 0]).mean():.1%} of cell-hours present")

    # tides: fit on the training year only (hours up to the end of cycle TRAIN_LAST_CYCLE)
    train_end = np.datetime64(int(t[cyc <= TRAIN_LAST_CYCLE].max()), "s").astype("datetime64[h]")
    n_tr = int((train_end - t0) / H) + 1
    hours = np.arange(n_hours, dtype=np.float64)
    A = design(hours)
    coef = np.full((len(cells), 2, A.shape[1]), np.nan)
    nfit = np.zeros(len(cells), np.int32)
    for c in range(2):
        coef[:, c, :], nfit = fit_harmonics(A[:n_tr], raw[:n_tr, :, c].astype(np.float64))
    tide = np.full_like(raw, np.nan)
    for c in range(2):
        pred = A[:, 1:] @ np.nan_to_num(coef[:, c, 1:]).T      # (n_hours, n_cells), no mean term
        pred[:, ~np.isfinite(coef[:, c, 0])] = np.nan
        tide[:, :, c] = pred.astype(np.float32)
    ok = np.isfinite(coef[:, 0, 0])
    amp = np.hypot(coef[ok, 0, 1], coef[ok, 0, 6])
    log(f"{rid}: tides fitted on {n_tr:,} h for {ok.sum():,}/{len(cells):,} cells; "
        f"median M2 u-amplitude {np.nanmedian(amp):.3f} m/s")
    np.savez(out, t0=np.int64(t0.astype("datetime64[s]").astype(np.int64)), n_hours=n_hours,
             cell_lat=glat[cy].astype(np.float32), cell_lon=glon[cx].astype(np.float32),
             row_cell=row_cell.astype(np.int32), raw=raw, tide=tide, n_fit=nfit, coef=coef.astype(np.float32))
    log(f"{rid}: wrote {out} ({out.stat().st_size / 1e9:.1f} GB)")
    return out


if __name__ == "__main__":
    for rid in sys.argv[1:] or ["uswc", "usegc"]:
        build(rid)
