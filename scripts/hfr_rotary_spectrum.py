"""Rotary power spectrum of hourly HFR currents with the target-filter responses overlaid.

Why: the project's target is the 25-h running mean of the radar current at the SWOT
pass time, and the truth-tier study re-scores against 18-h and 48-h Butterworth
low-passes. This figure shows, in one panel, where the energy the altimeter cannot
see lives (semidiurnal and diurnal tides on both rotary sides, inertial motion on
the clockwise side) and how much of it each filter lets through, against the
sub-inertial continuum they all keep.

Rotary: w = u + i v; negative frequencies rotate clockwise (CW), positive
counterclockwise (CCW). Welch, 30-day Hann segments, 50 % overlap, averaged over
the best-covered cells of each region. Filter power responses: boxcar 25-point
|H|^2 (applied once); Butterworth order 2 |H|^4 (filtfilt applies it twice).

Real data needs the hourly archive (HFR_DATA_ROOT or Z:, see hfr_tiers.py):
    python scripts/hfr_rotary_spectrum.py --regions usegc,uswc --start 2024-03-01 --end 2024-06-30
Self-check on a synthetic series (no data):
    python scripts/hfr_rotary_spectrum.py --selftest
Writes docs/figures/hfr_target/hfr_rotary_spectrum.{png,svg}.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))
import swotxai  # noqa: F401  (torch/DLL import order)
import numpy as np
import xarray as xr
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, filtfilt, freqz, welch

import hfr_tiers
import baselines_figures as bf
from baselines_figures import C_U, GRID, INK, MUTED, REGION_NAME, save, style
bf.OUT = ROOT / "docs" / "figures" / "hfr_target"      # target-definition figure: neither science nor calval
import matplotlib.pyplot as plt

FS = 24.0                                   # samples per day
OMEGA = 7.2921e-5                           # rad/s
TIDES = {"M2": 12.4206, "K1": 23.9345, "O1": 25.8193}   # hours
REGION_COLOR = {"usegc": INK, "uswc": "#be185d", "prvi": "#0f766e", "ushi": "#b45309"}
# filters: the target in teal, the two Butterworth tiers amber / violet (same in both panels)
FILT_COLOR = {"25-h running mean": "#0e7490", "Butterworth 18-h (zero-phase)": "#d97706",
              "Butterworth 48-h (zero-phase)": "#6d28d9"}


def inertial_period_h(lat_deg: float) -> float:
    return 2 * np.pi / (2 * OMEGA * np.sin(np.radians(abs(lat_deg)))) / 3600.0


def fill_gaps(x: np.ndarray) -> np.ndarray:
    """Linear interpolation over NaN along axis 0 (columns independent)."""
    t = np.arange(x.shape[0])
    out = x.copy()
    for j in range(x.shape[1]):
        ok = np.isfinite(x[:, j])
        out[:, j] = np.interp(t, t[ok], x[ok, j])
    return out


def rotary_psd(u: np.ndarray, v: np.ndarray, nperseg: int = 30 * 24):
    """Mean rotary PSD over cells. u, v: (t, cells) gap-free. Returns f (cpd, signed), psd (m^2 s^-2 / cpd)."""
    w = (u - u.mean(0)) + 1j * (v - v.mean(0))
    f, p = welch(w, fs=FS, window="hann", nperseg=nperseg, noverlap=nperseg // 2, detrend="linear",
                 return_onesided=False, scaling="density", axis=0)
    order = np.argsort(f)
    return f[order], p[order].mean(1).real


def filter_power_responses(f_cpd: np.ndarray) -> dict[str, np.ndarray]:
    """Fraction of variance passed at each frequency by the three target filters."""
    f_cph = f_cpd / 24.0                                   # cycles per hour (fs = 1 per hour)
    out = {}
    n = 25
    num = np.sin(np.pi * f_cph * n); den = n * np.sin(np.pi * f_cph)
    H = np.where(np.abs(den) < 1e-12, 1.0, num / np.where(np.abs(den) < 1e-12, 1.0, den))
    out["25-h running mean"] = H ** 2
    for cut in (18.0, 48.0):
        b, a = butter(2, 2.0 / cut)
        _, h = freqz(b, a, worN=2 * np.pi * f_cph)
        out[f"Butterworth {cut:.0f}-h (zero-phase)"] = np.abs(h) ** 4
    return out


def load_region(rid: str, t0: np.datetime64, t1: np.datetime64, ncells: int, min_cov: float):
    """Best-covered cells of a region over [t0, t1]: (u, v) gap-filled (t, n), mean lat, hours."""
    meta = dict(np.load(ROOT / "experiments/evolve/evolve_all_science_v1/baselines" / f"meta_{rid}.npz",
                        allow_pickle=True))
    files = hfr_tiers.hourly_files(rid, t0, t1)
    if not files:
        raise FileNotFoundError(f"{rid}: no hourly files under {hfr_tiers.HFR_ROOT} for {t0}..{t1}")
    tic = time.time()
    ds = xr.open_mfdataset(files, combine="by_coords", data_vars=["u", "v"], coords="minimal",
                           compat="override", parallel=False)[["u", "v"]].sel(time=slice(t0, t1))
    ds = ds.sel(lat=slice(meta["lat"].min() - 0.3, meta["lat"].max() + 0.3),
                lon=slice(meta["lon"].min() - 0.3, meta["lon"].max() + 0.3)).drop_duplicates("time")
    hours = np.arange(t0, t1 + np.timedelta64(1, "h"), np.timedelta64(1, "h"))
    ds = ds.reindex(time=hours).load()
    U, V = ds["u"].values, ds["v"].values                           # (t, lat, lon)
    T = U.shape[0]
    u2, v2 = U.reshape(T, -1), V.reshape(T, -1)
    cov = np.isfinite(u2).mean(0)
    idx = np.argsort(cov)[::-1][:ncells]
    idx = idx[cov[idx] >= min_cov]
    if len(idx) < 3:
        raise RuntimeError(f"{rid}: only {len(idx)} cells with coverage >= {min_cov:.0%}")
    lat2 = np.broadcast_to(ds["lat"].values[:, None], U.shape[1:]).reshape(-1)
    print(f"  {rid}: {len(files)} files, {T} h, {len(idx)} cells, coverage {cov[idx].min():.0%}-{cov[idx].max():.0%}, "
          f"mean lat {lat2[idx].mean():.1f}, loaded in {time.time()-tic:.0f}s")
    return fill_gaps(u2[:, idx]), fill_gaps(v2[:, idx]), float(lat2[idx].mean()), hours


def synthetic(lat=35.0, days=120, seed=0):
    """Red noise + M2 ellipse (CW-dominant) + K1 + CW inertial + white noise, hourly."""
    rng = np.random.default_rng(seed)
    T = days * 24
    t = np.arange(T, dtype=float)
    def ar1(n, phi=0.995, s=0.012):
        x = np.zeros(n); e = rng.normal(0, s, n)
        for i in range(1, n): x[i] = phi * x[i - 1] + e[i]
        return x
    w = ar1(T) + 1j * ar1(T)
    om = lambda P: 2 * np.pi / P
    w += 0.18 * np.exp(-1j * om(TIDES["M2"]) * t) + 0.06 * np.exp(1j * om(TIDES["M2"]) * t)   # elliptical M2
    w += 0.05 * np.exp(-1j * om(TIDES["K1"]) * t) + 0.04 * np.exp(1j * om(TIDES["K1"]) * t)
    w += 0.08 * np.exp(-1j * om(inertial_period_h(lat)) * t) * (1 + 0.5 * np.sin(om(24 * 7) * t))  # CW inertial
    w += rng.normal(0, 0.01, T) + 1j * rng.normal(0, 0.01, T)
    u = np.column_stack([w.real + rng.normal(0, 0.003, T) for _ in range(5)])
    v = np.column_stack([w.imag + rng.normal(0, 0.003, T) for _ in range(5)])
    hours = np.arange(np.datetime64("2024-03-01T00"), np.datetime64("2024-03-01T00") + np.timedelta64(T, "h"),
                      np.timedelta64(1, "h"))
    return u, v, lat, hours


def figure(regions: dict, out_dir: Path | None, name="hfr_rotary_spectrum"):
    """regions: rid -> (u, v, mean_lat, hours). Three stacked panels: spectrum, filter responses
    (same frequency axis, so vertical alignment does the reading), ten-day time strip."""
    fig = plt.figure(figsize=(11.5, 9.6))
    gs = fig.add_gridspec(3, 1, height_ratios=[2.2, 1.0, 1.0], hspace=0.12)
    ax = fig.add_subplot(gs[0]); axf = fig.add_subplot(gs[1], sharex=ax); ax2 = fig.add_subplot(gs[2])
    xlim = (1 / 60, 12.0)

    def mark_lines(a, label: bool):
        for nm, P in TIDES.items():
            a.axvline(24 / P, color=MUTED, lw=0.7, alpha=0.6)
            if label:
                ha = {"O1": "right", "K1": "left"}.get(nm, "center")
                a.text(24 / P, 0.975, nm, transform=a.get_xaxis_transform(), ha=ha, va="top", fontsize=8,
                       color=MUTED, bbox=dict(facecolor="white", edgecolor="none", pad=1))
        for rid, (u, v, lat, hours) in regions.items():
            a.axvline(24 / inertial_period_h(lat), color=REGION_COLOR.get(rid, INK), lw=0.8, ls=":", alpha=0.9)
        if label:
            lat_mean = np.mean([r[2] for r in regions.values()])
            a.text(24 / inertial_period_h(lat_mean), 0.925, "inertial", transform=a.get_xaxis_transform(),
                   ha="center", va="top", fontsize=8, color=INK, bbox=dict(facecolor="white", edgecolor="none", pad=1))

    # (1) rotary spectrum
    for rid, (u, v, lat, hours) in regions.items():
        f, pxx = rotary_psd(u, v)
        neg, pos = f < 0, f > 0
        c = REGION_COLOR.get(rid, INK)
        ax.loglog(-f[neg], pxx[neg], color=c, lw=1.5, label=f"{REGION_NAME.get(rid, rid)}, clockwise")
        ax.loglog(f[pos], pxx[pos], color=c, lw=1.2, ls="--", alpha=0.85,
                  label=f"{REGION_NAME.get(rid, rid)}, counterclockwise")
    mark_lines(ax, label=True)
    ax.set_xlim(*xlim)
    ax.set_ylabel("rotary power density\n(m$^2$ s$^{-2}$ per cpd)")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    ax.set_title("Where the radar's energy sits (top) and what each target filter keeps (middle)", fontsize=11, pad=10)
    ax.tick_params(labelbottom=False)
    style(ax)

    # (2) filter power responses, same frequency axis
    ff = np.logspace(np.log10(xlim[0]), np.log10(xlim[1]), 2000)
    for (nm, r), ls in zip(filter_power_responses(ff).items(), ("-", "--", "-.")):
        axf.semilogx(ff, r, color=FILT_COLOR[nm], lw=1.8 if nm.startswith("25") else 1.3, ls=ls, label=nm)
    mark_lines(axf, label=False)
    axf.set_ylim(0, 1.05); axf.set_yticks([0, 0.5, 1.0])
    axf.set_ylabel("fraction of variance\npassed by the filter")
    axf.set_xlabel("frequency (cycles per day): slow multi-day flow on the left, fast sub-daily motion on the right; "
                   "dotted: local inertial frequency")
    axf.legend(loc="lower left", fontsize=8, frameon=False)
    style(axf)

    # (3) time strip: one cell, u, raw / 25-h / 18-h / 48-h
    rid0 = list(regions)[0]; u, v, lat, hours = regions[rid0]
    j = 0; n = min(len(hours), 10 * 24); i0 = len(hours) // 3; sl = slice(i0, i0 + n)
    raw = u[:, j]
    m25 = uniform_filter1d(raw, 25, mode="nearest")
    b, a = butter(2, 2.0 / 48.0); bw48 = filtfilt(b, a, raw)
    b, a = butter(2, 2.0 / 18.0); bw18 = filtfilt(b, a, raw)
    tt = hours[sl].astype("datetime64[h]").astype("datetime64[ns]")
    ax2.plot(tt, raw[sl], color=MUTED, lw=0.7, alpha=0.6, label="hourly $u$")
    ax2.plot(tt, bw18[sl], color=FILT_COLOR["Butterworth 18-h (zero-phase)"], lw=1.1, ls="--", label="Butterworth 18-h")
    ax2.plot(tt, m25[sl], color=FILT_COLOR["25-h running mean"], lw=1.8, label="25-h running mean (the target)")
    ax2.plot(tt, bw48[sl], color=FILT_COLOR["Butterworth 48-h (zero-phase)"], lw=1.2, ls="-.", label="Butterworth 48-h")
    ax2.axhline(0, color=GRID, lw=0.8)
    ax2.set_ylabel("$u$ (m s$^{-1}$)")
    ax2.set_title(f"{REGION_NAME.get(rid0, rid0)}, one cell, ten days: the same filters in time", fontsize=10, pad=6)
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.28), fontsize=8, ncol=4, frameon=False)  # below the strip, never over the data
    style(ax2)

    fig.text(0.5, -0.015, "Top: rotary spectrum of hourly HFR currents (Welch, 30-day Hann segments, mean over the best-covered "
             "cells); solid clockwise, dashed counterclockwise.\nMiddle: fraction of variance each target filter passes "
             "(25-point boxcar once; order-2 Butterworth forward and backward). Tidal lines M2, K1, O1.",
             ha="center", fontsize=8, color=MUTED)
    fig.subplots_adjust(left=0.09, right=0.98, top=0.95, bottom=0.09)
    gs.update(hspace=0.12)
    # the time strip needs its own gap below the frequency panels
    pos = ax2.get_position(); ax2.set_position([pos.x0, pos.y0 - 0.01, pos.width, pos.height - 0.03])
    if out_dir is None:
        save(fig, name)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)
        fig.savefig(out_dir / f"{name}.png", dpi=200, bbox_inches="tight", facecolor="white")
        plt.close(fig)
        print("wrote", out_dir / f"{name}.png")


def selftest():
    u, v, lat, hours = synthetic()
    f, p = rotary_psd(u, v)
    fi = 24 / inertial_period_h(lat)
    def peak(side, f0, tol=0.08):
        m = (np.sign(f) == side) & (np.abs(np.abs(f) - f0) < tol * f0)
        return p[m].max()
    assert peak(-1, fi) / peak(+1, fi) > 5, "inertial peak must be clockwise-only"
    fm2 = 24 / TIDES["M2"]
    assert peak(-1, fm2) > 3 * peak(+1, fm2), "M2 ellipse built CW-dominant"
    r = filter_power_responses(np.array([24 / TIDES["M2"], 24 / TIDES["K1"], 24 / 48.0 / 4]))
    assert r["25-h running mean"][0] < 0.01 and r["25-h running mean"][1] < 0.01, "boxcar must kill M2, K1"
    assert r["Butterworth 48-h (zero-phase)"][0] < 0.001, "bw48 must kill M2"
    assert r["Butterworth 18-h (zero-phase)"][1] > r["Butterworth 48-h (zero-phase)"][1], "bw18 leaks more K1 than bw48"
    assert all(v_[2] > 0.9 for v_ in r.values()), "all filters pass >90 % of 8-day variability (boxcar: 94.5 %)"
    print("selftest ok: inertial CW-only, M2 CW-dominant, boxcar/Butterworth responses as expected")
    out = Path(sys.argv[0]).resolve().parents[1] / "archive" / "selftest"
    figure({"usegc": (u, v, lat, hours)}, out_dir=out, name="hfr_rotary_spectrum_selftest")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--regions", default="usegc,uswc")
    ap.add_argument("--start", default="2024-03-01")
    ap.add_argument("--end", default="2024-06-30")
    ap.add_argument("--ncells", type=int, default=30)
    ap.add_argument("--min-cov", type=float, default=0.9)
    ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        selftest(); sys.exit(0)
    t0, t1 = np.datetime64(a.start + "T00"), np.datetime64(a.end + "T23")
    regions = {rid: load_region(rid, t0, t1, a.ncells, a.min_cov) for rid in a.regions.split(",")}
    figure(regions, out_dir=None)
