"""The hourly experiment: 25 rows per pass on a detided hourly radar target.

Same SWOT stencil rows as scripts/holdout_eval.py, but the target is the radar current
at an hour offset from the pass (-12 .. +12 h, every 3 h) with the training-year tidal
fit removed (scripts/hourly_targets.py), and the wind inputs are recomputed relative to
that hour: lags 0 .. 48 h, a slab-model inertial current (Pollard & Millard 1970)
integrated on the ERA5 grid, the offset itself, f and season. Train on uswc + usegc year
one (cycles 001-016), score cycles 017-058 by offset and feature set with a block
bootstrap over (cycle, pass). Research log 2026-11-14 02:05 has the design and caveats.

Feature sets (extra columns after the 72 stencil columns, routed to the champion copy's
dense path through n_stencil_cols):
    o     offset_h                                          snapshot decay alone
    ow    offset_h + wind lags at the target hour + slab    does wind history matter hourly?
    ofsw  ow + f, season                                    everything

Usage:
    python scripts/hourly_eval.py --train ofsw [--target raw] [--pert]
    python scripts/hourly_eval.py --score
    python scripts/hourly_eval.py --figures
Folders: experiments/holdout/hourly_<featset>[_raw]/ (Z: junction). Nothing is downloaded.
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import swotxai  # noqa: F401
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import holdout_eval as he  # noqa: E402

REPO, OUT, FIG = he.REPO, he.OUT, he.FIG
REGIONS = ["uswc", "usegc"]
OFFSETS = list(range(-12, 13, 3))
LAGS = [0, 3, 6, 9, 12, 15, 18, 21, 24, 36, 48]
WIND = [f"w_{c}_lag{h}h" for c in "uv" for h in LAGS]
SLAB = ["slab_u", "slab_v"]
FEATSETS = {"o": ["offset_h"], "ow": ["offset_h"] + WIND + SLAB,
            "ofsw": ["offset_h"] + WIND + SLAB + ["f_1e4", "sin_doy", "cos_doy"]}
SUB = 8                  # rows at a non-zero offset: a fixed 1/SUB sample (same rows at every offset)
PERT_ROWS = 150_000
PERT_DELTA = 2.0
# slab model: mixed-layer depth, damping time, densities, drag
SLAB_H, SLAB_R, RHO_W, RHO_A, CD = 25.0, 1 / (4 * 86400), 1025.0, 1.22, 1.2e-3
OMEGA = 7.2921e-5
log = he.log


# --------------------------------------------------------------------------
# 1. Inputs at an arbitrary hour: ERA5 lookup, slab model
# --------------------------------------------------------------------------

class Era5:
    """Hourly ERA5 10 m wind of a region plus the slab-model inertial current on the same grid."""

    def __init__(self, rid: str):
        era = pickle.load(open(REPO / "experiments" / rid / "era5" / f"era5wind_science_{rid}_10m.pkl", "rb"))
        self.t0 = int(era["time"].values.astype("datetime64[s]").astype(np.int64)[0])
        self.lat, self.lon = era["latitude"].values, era["longitude"].values
        self.u = era["era5_u"].values.astype(np.float32)
        self.v = era["era5_v"].values.astype(np.float32)
        self.n = len(self.u)
        self.slab = self._slab()

    def _slab(self) -> np.ndarray:
        """Z = u + i v, dZ/dt + (r + i f) Z = tau / (rho H); exact hourly step with tau held
        over the hour. Returns complex64 (time, lat, lon) in m/s."""
        f = 2 * OMEGA * np.sin(np.deg2rad(self.lat))[:, None].astype(np.complex128)
        a = SLAB_R + 1j * f
        dt = 3600.0
        decay = np.exp(-a * dt)
        gain = (1 - decay) / a / (RHO_W * SLAB_H)
        W = self.u.astype(np.float64) + 1j * self.v.astype(np.float64)
        Z = np.zeros((self.n,) + W.shape[1:], np.complex64)
        z = np.zeros(W.shape[1:], np.complex128)
        for k in range(self.n):
            tau = RHO_A * CD * np.abs(W[k]) * W[k]
            z = z * decay + tau * gain
            Z[k] = z
        return Z

    def index(self, lat, lon):
        lon_r = np.where(lon > self.lon.max() + 0.25, lon - 360, lon)
        lon_r = np.where(lon_r < self.lon.min() - 0.25, lon_r + 360, lon_r)
        yi = np.clip(np.round((lat - self.lat[0]) / (self.lat[1] - self.lat[0])).astype(np.int32), 0, len(self.lat) - 1)
        xi = np.clip(np.round((lon_r - self.lon[0]) / (self.lon[1] - self.lon[0])).astype(np.int32), 0, len(self.lon) - 1)
        return yi, xi

    def at(self, A, t_s, yi, xi):
        ti = np.round((t_s - self.t0) / 3600).astype(np.int64)
        ok = (ti >= 0) & (ti < self.n)
        out = np.full(len(t_s), np.nan, np.float32)
        out[ok] = A[ti[ok], yi[ok], xi[ok]]
        return out


def extras(era: Era5, t_s, lat, lon, offset_h, featset: str) -> np.ndarray:
    """(n, len(FEATSETS[featset])) extra columns for rows whose target hour is t_s + offset."""
    cols = FEATSETS[featset]
    E = np.full((len(t_s), len(cols)), np.nan, np.float32)
    yi, xi = era.index(lat, lon)
    th = t_s + offset_h * 3600
    for j, c in enumerate(cols):
        if c == "offset_h":
            E[:, j] = offset_h
        elif c.startswith("w_"):
            comp, lag = c[2], int(c.split("lag")[1][:-1])
            E[:, j] = era.at(era.u if comp == "u" else era.v, th - lag * 3600, yi, xi)
        elif c == "slab_u":
            E[:, j] = era.at(era.slab.real, th, yi, xi)
        elif c == "slab_v":
            E[:, j] = era.at(era.slab.imag, th, yi, xi)
        elif c == "f_1e4":
            E[:, j] = 2 * OMEGA * np.sin(np.deg2rad(lat)) * 1e4
        elif c in ("sin_doy", "cos_doy"):
            ts = th.astype("datetime64[s]")
            doy = (ts - ts.astype("datetime64[Y]")).astype("timedelta64[D]").astype(float)
            E[:, j] = (np.sin if c == "sin_doy" else np.cos)(2 * np.pi * doy / 365.25)
    return E


# --------------------------------------------------------------------------
# 2. Rows: stencil + extras, hourly target
# --------------------------------------------------------------------------

def region_rows(rid: str, featset: str, target: str, era: Era5 | None = None):
    """All (row, offset) pairs of a region with a finite hourly target. Returns dict of arrays:
    X (with extras), Y, time (pass), cycle, pass_idx, offset, base (index into the frozen rows),
    plus the frozen flags needed for the scorable mask."""
    d = he.load_region(rid)
    h = np.load(OUT / f"{rid}_hourly.npz")
    raw, tide, row_cell, t0 = h["raw"], h["tide"], h["row_cell"], int(h["t0"])
    era = era or Era5(rid)
    n = len(d["time"])
    keep_sub = np.random.default_rng(1).random(n) < 1 / SUB
    parts = []
    for k in OFFSETS:
        sel = np.ones(n, bool) if k == 0 else keep_sub
        idx = np.flatnonzero(sel)
        hi = np.round((d["time"][idx] - t0) / 3600).astype(np.int64) + k
        hok = (hi >= 0) & (hi < len(raw))
        idx, hi = idx[hok], hi[hok]
        y = raw[hi, row_cell[idx]]
        if target == "detided":
            y = y - tide[hi, row_cell[idx]]
        fin = np.isfinite(y).any(1)
        idx, y = idx[fin], y[fin]
        E = extras(era, d["time"][idx], d["lat"][idx], d["lon"][idx], k, featset)
        parts.append(dict(X=np.hstack([d["X"][idx], E]), Y=y.astype(np.float32), base=idx,
                          offset=np.full(len(idx), k, np.int8)))
    out = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    for key in ("time", "cycle", "pass_idx", "lat", "lon", "grid_ok", "dt_h", "era5_ok"):
        out[key] = d[key][out["base"]]
    return out


def scorable(r: dict) -> np.ndarray:
    X = r["X"]
    return (r["grid_ok"] & (np.abs(r["dt_h"]) <= he.MAX_DT_H) & r["era5_ok"]
            & np.isfinite(X[:, he.col("ugos_filtered")]) & np.isfinite(X[:, he.col("vgos_filtered")])
            & np.isfinite(X[:, he.col("era5_u")]) & np.isfinite(X[:, he.col("era5_v")]))


def vdir_of(featset: str, target: str) -> Path:
    return OUT / (f"hourly_{featset}" + ("_raw" if target == "raw" else ""))


def build_data(featset: str, target: str, pert: bool) -> None:
    vdir = vdir_of(featset, target); vdir.mkdir(parents=True, exist_ok=True)
    Xtr, Ytr, Ttr, Xte, spans, pos = [], [], [], [], {}, 0
    for rid in REGIONS:
        r = region_rows(rid, featset, target)
        te = r["cycle"] > he.LAST_TRAIN_CYCLE
        Xtr.append(r["X"][~te]); Ytr.append(r["Y"][~te]); Ttr.append(r["time"][~te])
        Xte.append(r["X"][te]); spans[rid] = [pos, pos + int(te.sum())]; pos += int(te.sum())
        np.savez(vdir / f"rows_{rid}.npz", **{k: r[k][te] for k in ("base", "offset", "cycle", "pass_idx")})
        log(f"{vdir.name}/{rid}: {int((~te).sum()):,} train rows, {int(te.sum()):,} test rows "
            f"({len(r['Y']) / len(np.unique(r['base'])):.1f} per base row)")
        if pert and rid in he.PERT_REGIONS and "w" in featset:
            Xp, meta = pert_rows(r, te, featset)
            Xte.append(Xp); spans[f"pert_{rid}"] = [pos, pos + len(Xp)]; pos += len(Xp)
            (vdir / f"pert_meta_{rid}.json").write_text(json.dumps(meta), encoding="utf-8")
        del r
    Xtr, Ytr, Ttr = (np.concatenate(a) for a in (Xtr, Ytr, Ttr))
    order = np.argsort(Ttr, kind="stable")     # chronological by pass time: the recipe's last-10 % validation
    Xtr, Ytr = Xtr[order], Ytr[order]           # split then lands at (or inside one) pass boundary
    Xte = np.concatenate(Xte)
    np.savez(vdir / "data.npz", X_train=Xtr, Y_train=Ytr, X_test=Xte, Y_test=np.zeros((len(Xte), 2), np.float32))
    (vdir / "data_meta.json").write_text(json.dumps({
        "featset": featset, "target": target, "extra_columns": FEATSETS[featset], "offsets": OFFSETS, "sub": SUB,
        "train_regions": REGIONS, "train_cycles": [1, he.LAST_TRAIN_CYCLE], "n_train": int(len(Xtr)),
        "n_test": int(len(Xte)), "test_spans": spans, "features": he.FEATURES, "stencil_k": he.K,
        "slab": dict(H=SLAB_H, r_per_s=SLAB_R, cd=CD)}, indent=2), encoding="utf-8")
    log(f"{vdir.name}: froze {len(Xtr):,} training rows, {len(Xte):,} test rows")


def pert_rows(r: dict, te: np.ndarray, featset: str):
    """Clean offset-0 test rows with +PERT_DELTA on one wind lag column (and on the 9 stencil
    wind cells for lag 0, which the stencil also carries), one block per (component, lag)."""
    cols = FEATSETS[featset]
    n_st = len(he.FEATURES) * he.K * he.K
    ok = np.flatnonzero(te & (r["offset"] == 0) & scorable(r))
    ok = ok[np.isfinite(r["X"][ok][:, n_st:]).all(1)]
    pick = np.sort(np.random.default_rng(0).choice(ok, min(PERT_ROWS, len(ok)), replace=False))
    base = r["X"][pick]
    blocks, meta = [base], {"delta": PERT_DELTA, "n_per_block": int(len(pick)), "blocks": ["base"]}
    for comp in "uv":
        for h in LAGS:
            Xp = base.copy()
            Xp[:, n_st + cols.index(f"w_{comp}_lag{h}h")] += PERT_DELTA
            if h == 0:
                for cell in range(he.K * he.K):
                    Xp[:, he.FEATURES.index(f"era5_{comp}") * he.K * he.K + cell] += PERT_DELTA
            blocks.append(Xp); meta["blocks"].append(f"{comp}_lag{h}")
    return np.concatenate(blocks), meta


def train(featset: str, target: str = "detided", seed: int = 42, pert: bool = False) -> Path:
    vdir = vdir_of(featset, target)
    pred_path = vdir / f"pred_seed{seed}.npz"
    if pred_path.exists():
        log(f"{vdir.name}: {pred_path.name} exists, skipping"); return pred_path
    if not (vdir / "data.npz").exists():
        subprocess.run([he.PY, __file__, "--_build_one", featset, target, "1" if pert else "0"],
                       cwd=REPO, check=True, env=he.ENV)
    params = dict(json.loads((he.RUN / "candidates" / he.CHAMPION / "params.json").read_text()), seed=seed,
                  n_stencil_cols=len(he.FEATURES) * he.K * he.K)
    (vdir / f"params_seed{seed}.json").write_text(json.dumps(params), encoding="utf-8")
    log(f"{vdir.name}: training {he.CHAMPION} copy seed {seed} (budget {params['time_budget_s']:.0f} s)...")
    t0 = time.time()
    r = subprocess.run([he.PY, "-m", "swotxai.evolve.run_candidate", str(REPO / "scripts" / "phase2_candidate.py"),
                        str(vdir / "data.npz"), str(pred_path), str(vdir / f"meta_seed{seed}.json"),
                        str(vdir / f"params_seed{seed}.json")],
                       cwd=REPO, timeout=params["time_budget_s"] * 2 + 600, check=False, env=he.ENV)
    if not (vdir / f"meta_seed{seed}.json").exists():
        raise RuntimeError(f"{vdir.name}: training died without meta (exit {r.returncode})")
    meta = json.loads((vdir / f"meta_seed{seed}.json").read_text())
    if meta.get("status") != "ok":
        raise RuntimeError(f"{vdir.name}: training failed: {meta.get('error')}")
    log(f"{vdir.name}: done in {time.time() - t0:.0f} s -> {pred_path}")
    return pred_path


# --------------------------------------------------------------------------
# 3. Score by offset
# --------------------------------------------------------------------------

def _align(vd: Path, rid: str, r: dict, seed: int | None = None) -> np.ndarray:
    """Predictions of a trained variant placed on the rows of region_rows (NaN where absent):
    one seed, or the mean over every pred_seed*.npz present (the ensemble the tables show)."""
    meta = json.loads((vd / "data_meta.json").read_text())
    s = meta["test_spans"][rid]
    files = [vd / f"pred_seed{seed}.npz"] if seed is not None else sorted(vd.glob("pred_seed*.npz"))
    P = np.mean([np.load(f)["pred"][s[0]:s[1]] for f in files], axis=0)
    rows = np.load(vd / f"rows_{rid}.npz")
    full = np.full((len(r["Y"]), 2), np.nan, np.float32)
    key_r = r["base"].astype(np.int64) * 100 + (r["offset"].astype(np.int64) + 50)
    key_p = rows["base"].astype(np.int64) * 100 + (rows["offset"].astype(np.int64) + 50)
    order = np.argsort(key_r)
    pos = np.searchsorted(key_r[order], key_p)
    assert np.array_equal(key_r[order][pos], key_p), f"{vd.name}/{rid}: test rows do not align"
    full[order[pos]] = P
    return full


DAILY_REF = "us_y1"   # the daily-target champion the averaged hourly predictions are measured against


def _seeds(vd: Path) -> list[int]:
    return sorted(int(f.stem[len("pred_seed"):]) for f in vd.glob("pred_seed*.npz"))


def _r2(y: np.ndarray, p: np.ndarray) -> float:
    ok = np.isfinite(y) & np.isfinite(p)
    return float(1 - np.sum((y[ok] - p[ok]) ** 2) / np.sum((y[ok] - y[ok].mean()) ** 2))


def daily_from_hourly(B: int = 1000) -> dict:
    """Does the hourly route improve the daily number? Average each pass's hourly predictions over
    the nine offsets and score the average against the frozen 25-hour-mean target, on the base rows
    that carry all nine offsets, next to the daily champion (DAILY_REF) on the same rows."""
    import baselines_eval as be
    be.FEATURES, be.K, be.CENTER = he.FEATURES, he.K, he.K * he.K // 2
    res = {"daily_ref": DAILY_REF, "offsets": OFFSETS, "regions": {}}
    for rid in REGIONS:
        d = he.load_region(rid)
        sp = json.loads((OUT / DAILY_REF / "data_meta.json").read_text())["test_spans"][rid]
        daily = np.full((len(d["Y"]), 2), np.nan, np.float32)
        # seed ensemble of the daily reference, so it is compared like for like with the hourly ensembles
        daily[np.where(he.test_mask(d))[0]] = np.mean([np.load(f)["pred"][sp[0]:sp[1]] for f in sorted((OUT / DAILY_REF).glob("pred_seed*.npz"))], axis=0)
        r = region_rows(rid, "o", "detided")
        preds = {fs: _align(vdir_of(fs, "detided"), rid, r) for fs in ("o", "ow", "ofsw")
                 if (vdir_of(fs, "detided") / "pred_seed42.npz").exists()}
        good = scorable(r) & (r["cycle"] > he.LAST_TRAIN_CYCLE)
        for P in preds.values():
            good &= np.isfinite(P).all(1)
        ub, inv, cnt = np.unique(r["base"][good], return_inverse=True, return_counts=True)

        def avg(A):
            S = np.zeros((len(ub), 2)); np.add.at(S, inv, A[good]); return S / cnt[:, None]

        keep = (cnt == len(OFFSETS)) & np.isfinite(d["Y"][ub]).all(1) & np.isfinite(daily[ub]).all(1)
        Yd, Yh = d["Y"][ub], avg(r["Y"])
        at0 = good & (r["offset"] == 0)
        models = {DAILY_REF: daily[ub]}
        for fs, P in preds.items():
            models[f"avg9_{fs}"] = avg(P)
            snap = np.full((len(ub), 2), np.nan); snap[np.searchsorted(ub, r["base"][at0])] = P[at0]
            models[f"pass_hour_{fs}"] = snap
        blocks = d["cycle"][ub].astype(np.int64) * 1000 + d["pass_idx"][ub]
        reg = {"n_bases": int(keep.sum()), "n_passes": int(len(np.unique(blocks[keep]))), "vs": {}}
        for tname, Y in (("daily_25h_mean", Yd), ("mean_of_9_hourly", Yh)):
            reg["vs"][tname] = {name: {c: be.score(Y[keep, ci], P[keep, ci], blocks[keep], B=B)
                                       for ci, c in enumerate("uv")} for name, P in models.items()}
        res["regions"][rid] = reg
        log(f"{rid}: {keep.sum():,} bases with all {len(OFFSETS)} offsets, {reg['n_passes']} passes")
    (OUT / "hourly_daily.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    lines = [f"# Hourly route scored as a daily product (average of the {len(OFFSETS)} hourly predictions, "
             f"cycles 017-058, bases with all offsets; daily reference `{DAILY_REF}`)", ""]
    for rid, reg in res["regions"].items():
        lines += [f"## {rid} ({reg['n_bases']:,} bases, {reg['n_passes']} passes)", "",
                  "| model | vs 25 h mean target (u / v) | vs mean of 9 hourly targets (u / v) |", "|---|---|---|"]
        for name in reg["vs"]["daily_25h_mean"]:
            cells = [f"{reg['vs'][t][name]['u']['r2']:.3f} / {reg['vs'][t][name]['v']['r2']:.3f}" for t in reg["vs"]]
            lines.append(f"| {name} | " + " | ".join(cells) + " |")
        lines.append("")
    (OUT / "hourly_daily.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return res


def score_all(B: int = 1000) -> dict:
    import baselines_eval as be
    be.FEATURES, be.K, be.CENTER = he.FEATURES, he.K, he.K * he.K // 2
    res = {"offsets": OFFSETS, "sub": SUB, "train_regions": REGIONS, "regions": {}}
    vdirs = sorted(p.parent for p in OUT.glob("hourly_*/data_meta.json") if (p.parent / "pred_seed42.npz").exists())
    for rid in REGIONS:
        reg = {"targets": {}}
        for target in ("detided", "raw"):
            r = region_rows(rid, "ofsw", target)      # the widest extras: baselines read wind at the target hour
            te = r["cycle"] > he.LAST_TRAIN_CYCLE
            ok_all = scorable(r)
            tr = ~te & ok_all
            G = r["X"][:, [he.col("ugos_filtered"), he.col("vgos_filtered")]]
            n_st = len(he.FEATURES) * he.K * he.K
            W = r["X"][:, [n_st + FEATSETS["ofsw"].index("w_u_lag0h"), n_st + FEATSETS["ofsw"].index("w_v_lag0h")]]
            seeds: dict[str, dict] = {}
            models = {"train_mean": np.tile(np.nanmean(r["Y"][tr], 0), (len(r["Y"]), 1)),
                      "geos_calibrated": np.column_stack(
                          [be.fit_linear(G[tr][:, [c]], r["Y"][tr][:, [c]])[0](G[:, [c]])[:, 0] for c in range(2)]),
                      "geos_cal+wind_lsq": be.fit_linear(np.column_stack([G[tr], W[tr]]), r["Y"][tr])[0](np.column_stack([G, W]))}
            # champion runs on this target: align predictions through the saved test rows
            for vd in vdirs:
                meta = json.loads((vd / "data_meta.json").read_text())
                if meta["target"] != target:
                    continue
                models[f"champion_{meta['featset']}"] = _align(vd, rid, r)
                seeds[f"champion_{meta['featset']}"] = {sd: _align(vd, rid, r, sd) for sd in _seeds(vd)} if len(_seeds(vd)) > 1 else {}
            blocks = r["cycle"].astype(np.int64) * 1000 + r["pass_idx"]
            tgt = {"n_rows": int((te & ok_all).sum()), "by_offset": {}, "models": list(models)}
            for k in OFFSETS:
                m = te & ok_all & (r["offset"] == k)
                entry = {"n": int(m.sum()), "n_passes": int(len(np.unique(blocks[m])))}
                for name, P in models.items():
                    mm = m & np.isfinite(P).all(1)
                    entry[name] = {c: be.score(r["Y"][mm, ci], P[mm, ci], blocks[mm], B=B) for ci, c in enumerate("uv")}
                    if seeds.get(name):
                        entry[name]["per_seed_r2"] = {c: {str(sd): _r2(r["Y"][mm, ci], Ps[mm, ci]) for sd, Ps in seeds[name].items()}
                                                      for ci, c in enumerate("uv")}
                tgt["by_offset"][str(k)] = entry
            reg["targets"][target] = tgt
            log(f"{rid}/{target}: scored {len(models)} models at {len(OFFSETS)} offsets")
        res["regions"][rid] = reg
    (OUT / "hourly_results.json").write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    write_md(res)
    return res


def write_md(res: dict) -> None:
    lines = ["# Hourly experiment: R² by offset from the pass (cycles 017-058, train uswc+usegc year one)", ""]
    for rid, reg in res["regions"].items():
        for target, tgt in reg["targets"].items():
            lines += [f"## {rid}, target {target}", "", "| model | " + " | ".join(f"{k:+d} h" for k in OFFSETS) + " |",
                      "|---|" + "---|" * len(OFFSETS)]
            for name in tgt["models"]:
                cells = []
                for k in OFFSETS:
                    e = tgt["by_offset"][str(k)].get(name)
                    cells.append(f"{e['u']['r2']:.2f} / {e['v']['r2']:.2f}" if e else "")
                lines.append(f"| {name} | " + " | ".join(cells) + " |")
            for name in tgt["models"]:
                ps = tgt["by_offset"]["0"].get(name, {}).get("per_seed_r2")
                if ps:
                    lines.append("")
                    lines.append(f"{name}: single-seed R² u / v at " + "; ".join(
                        f"{k:+d} h: " + ", ".join(f"{tgt['by_offset'][str(k)][name]['per_seed_r2']['u'][sd]:.3f}/"
                                                  f"{tgt['by_offset'][str(k)][name]['per_seed_r2']['v'][sd]:.3f}" for sd in ps["u"])
                        for k in (-12, 0, 12)) + " (table row is the seed-mean prediction)")
            lines.append("")
    (OUT / "hourly_results.md").write_text("\n".join(lines), encoding="utf-8")


# --------------------------------------------------------------------------
# 4. Figures
# --------------------------------------------------------------------------

LABELS = {"geos_calibrated": "SWOT geostrophy calibrated", "geos_cal+wind_lsq": "+ wind at that hour, linear",
          "champion_o": "champion + offset only", "champion_ow": "+ wind lags, slab", "champion_ofsw": "+ f, season"}
COLORS = {"geos_calibrated": "0.55", "geos_cal+wind_lsq": "0.75", "champion_o": "#0072B2", "champion_ow": "#CC79A7",
          "champion_ofsw": "#D55E00"}


def fig_offset(res: dict) -> None:
    plt = he._mpl()
    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), sharex=True)
    for i, rid in enumerate(REGIONS):
        tgt = res["regions"][rid]["targets"]["detided"]
        for j, c in enumerate("uv"):
            ax = axes[j, i]
            for name in LABELS:
                if name not in tgt["models"]:
                    continue
                ys = [tgt["by_offset"][str(k)][name][c] for k in OFFSETS]
                r2 = np.array([e["r2"] for e in ys])
                lo = np.array([e.get("r2_ci", [np.nan])[0] for e in ys]); hi = np.array([e.get("r2_ci", [np.nan, np.nan])[1] for e in ys])
                ax.plot(OFFSETS, r2, marker="o", ms=4, color=COLORS[name], label=LABELS[name])
                if np.isfinite(lo).all():
                    ax.fill_between(OFFSETS, lo, hi, color=COLORS[name], alpha=0.15, lw=0)
            ax.axhline(0, color="k", lw=0.8); ax.axvline(0, color="0.8", lw=0.8); ax.grid(color="0.92")
            ax.set_title(f"{rid}, {c} component (detided hourly target)", loc="left", fontsize=9)
            if i == 0:
                ax.set_ylabel("R$^2$ on cycles 017-058")
            if j == 1:
                ax.set_xlabel("target hour relative to the SWOT pass (h)")
    fig.suptitle("Skill against time offset: the snapshot ages, the wind inputs carry the hour", fontsize=10)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), fontsize=7, frameon=False, ncol=5,
               loc="upper center", bbox_to_anchor=(0.5, 0.955))
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    he._save(fig, "hourly_skill_offset")


def fig_tide(res: dict) -> None:
    """Raw against detided at offset 0: how much of the hourly variance the tide holds."""
    plt = he._mpl()
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    for i, rid in enumerate(REGIONS):
        ax = axes[i]; reg = res["regions"][rid]["targets"]
        names = [n for n in LABELS if all(n in reg[t]["models"] for t in reg)]
        x = np.arange(len(names)); w = 0.2
        for jt, target in enumerate(t for t in ("raw", "detided") if t in reg):
            for jc, c in enumerate("uv"):
                vals = [reg[target]["by_offset"]["0"][n][c]["r2"] for n in names]
                ax.bar(x + (2 * jt + jc - 1.5) * w, np.clip(vals, -0.1, None), w,
                       color=["0.7", "0.45", "#56B4E9", "#0072B2"][2 * jt + jc], label=f"{target}, {c}")
        ax.set_xticks(x); ax.set_xticklabels([LABELS[n] for n in names], rotation=25, ha="right", fontsize=7)
        ax.axhline(0, color="k", lw=0.8); ax.set_title(f"{rid}, offset 0 h", loc="left", fontsize=9)
        ax.set_ylabel("R$^2$")
    axes[0].legend(fontsize=7, frameon=False)
    fig.suptitle("Hourly target with and without the tidal fit removed", fontsize=10)
    fig.tight_layout()
    he._save(fig, "hourly_tide_effect")


def fig_response() -> None:
    """Wind response function on the hourly target from the perturbation blocks of hourly_ofsw."""
    vd = vdir_of("ofsw", "detided")
    metas = {rid: vd / f"pert_meta_{rid}.json" for rid in he.PERT_REGIONS}
    if not all(p.exists() for p in metas.values()) or not (vd / "pred_seed42.npz").exists():
        log("no perturbation rows for hourly_ofsw; skipping response figure"); return
    plt = he._mpl()
    spans = json.loads((vd / "data_meta.json").read_text())["test_spans"]
    P = np.load(vd / "pred_seed42.npz")["pred"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), sharex=True)
    for i, rid in enumerate(he.PERT_REGIONS):
        meta = json.loads(metas[rid].read_text()); n = meta["n_per_block"]; s = spans[f"pert_{rid}"][0]
        base = P[s:s + n]
        # wind at the target hour: lag-0 columns of the base rows
        rows = np.load(vd / "data.npz", mmap_mode="r")["X_test"][s:s + n]
        n_st = len(he.FEATURES) * he.K * he.K
        wu, wv = rows[:, n_st + FEATSETS["ofsw"].index("w_u_lag0h")], rows[:, n_st + FEATSETS["ofsw"].index("w_v_lag0h")]
        spd = np.hypot(wu, wv); ang = np.arctan2(wv, wu)
        along, cross = [], []
        for h in LAGS:
            du = (P[s + n * meta["blocks"].index(f"u_lag{h}"):][:n] - base) / PERT_DELTA
            dv = (P[s + n * meta["blocks"].index(f"v_lag{h}"):][:n] - base) / PERT_DELTA
            # response to a unit wind along the current wind direction: rotate the (du, dv) Jacobian
            ru = np.cos(ang)[:, None] * du + np.sin(ang)[:, None] * dv      # d(u,v) per +1 m/s along the wind
            al = ru[:, 0] * np.cos(ang) + ru[:, 1] * np.sin(ang)
            cr = -ru[:, 0] * np.sin(ang) + ru[:, 1] * np.cos(ang)
            along.append(100 * np.nanmean(al)); cross.append(100 * np.nanmean(cr))
        for j, (vals, lab) in enumerate(((along, "along-wind"), (cross, "cross-wind, + = left"))):
            ax = axes[j, i]
            ax.plot(LAGS, vals, marker="o", color="k")
            ax.axhline(0, color="0.7", lw=0.8); ax.grid(color="0.92")
            ax.set_title(f"{rid}: {lab} response (% of wind speed per +1 m/s at one lag)", loc="left", fontsize=8.5)
            if j == 1:
                ax.set_xlabel("lag of the perturbed wind before the target hour (h)")
        mag, a = np.hypot(sum(along), sum(cross)), np.degrees(np.arctan2(sum(cross), sum(along)))
        axes[0, i].text(0.98, 0.95, f"sum over lags: {mag:.2f} % of wind speed, {abs(a):.0f}$^\\circ$ {'left' if a > 0 else 'right'}",
                        transform=axes[0, i].transAxes, ha="right", va="top", fontsize=7, bbox=dict(fc="white", ec="none", alpha=0.8))
    fig.suptitle("Wind response of the hourly-target model (detided, offset 0 rows)", fontsize=10)
    fig.tight_layout()
    he._save(fig, "hourly_wind_response")


def figures() -> None:
    res = json.loads((OUT / "hourly_results.json").read_text())
    for f in (fig_offset, fig_tide):
        try:
            f(res)
        except Exception as e:  # noqa: BLE001
            log(f"{f.__name__} failed: {e!r}")
    try:
        fig_response()
    except Exception as e:  # noqa: BLE001
        log(f"fig_response failed: {e!r}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train", choices=list(FEATSETS))
    ap.add_argument("--target", choices=["detided", "raw"], default="detided")
    ap.add_argument("--pert", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--daily", action="store_true", help="average the hourly predictions per pass and score them as a daily product")
    ap.add_argument("--B", type=int, default=1000)
    ap.add_argument("--_build_one", nargs=3, help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a._build_one:
        fs, tg, pt = a._build_one
        build_data(fs, tg, pt == "1"); return
    if a.train:
        train(a.train, a.target, a.seed, a.pert)
    if a.score:
        score_all(a.B)
    if a.daily:
        daily_from_hourly(a.B)
    if a.figures:
        figures()


if __name__ == "__main__":
    main()
