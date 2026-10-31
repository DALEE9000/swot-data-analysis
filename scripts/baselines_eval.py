"""Phase-1 evaluation: baselines, temporal-holdout protocols, block bootstrap.

Scores the pooled-science champion (and its solo-trained twin) against the
baselines the field reports: SWOT L3 geostrophic velocity, geostrophic plus
an instantaneous Ekman term (classical and least-squares fitted), a ridge on
the 72 stencil columns, and the training-period mean. Everything is scored
on the SAME held-out rows (the frozen evolve test split, cycles 13-16) under
two protocols: all test rows (0-day buffer) and rows at least 30 days after
the end of training. Confidence intervals come from a block bootstrap over
(cycle, pass) snapshots, which also gives an effective-N estimate.

Nothing in the pipeline is modified. Inputs are the frozen
``experiments/evolve/<run>/data.npz``, the champion ``predictions.npz``, the
``solo/<rid>/`` artifacts, and the cached ``swot_features`` / ``hfr_target``
pickles used to recover per-row cycle/pass/time/lat/lon (the frozen arrays
hold no metadata; the replay is verified by exact equality with Y_test).

Run (project env):
    python scripts/baselines_eval.py --run evolve_all_science_v1
    python scripts/baselines_eval.py --selftest

Outputs under experiments/evolve/<run>/baselines/:
    meta_<rid>.npz   per-region replayed row metadata (checkpoint)
    results.json     every metric, every model, every protocol
    results.md       the paper table
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

FEATURES = ["mdt", "ssha_filtered", "ugos_filtered", "vgos_filtered",
            "ugosa_filtered", "vgosa_filtered", "era5_u", "era5_v"]
K = 3
CENTER = K * K // 2


def col(feature: str) -> int:
    return FEATURES.index(feature) * K * K + CENTER


# --------------------------------------------------------------------------
# 1. Replay per-row metadata from the cached inputs
# --------------------------------------------------------------------------

def _load_pkl(p: Path):
    with open(p, "rb") as f:
        return pickle.load(f)


def replay_meta(rid: str, mission: str, c0: int, c1: int, train_fraction: float, out: Path,
                split: str = "test") -> dict:
    """Rebuild cycle / pass / time / lat / lon / hfr(u,v) for every held-out row,
    following step_flatten + concat_for_ann exactly (dict order, list order,
    row-major grid order, rows kept when either target is finite)."""
    if out.exists():
        d = dict(np.load(out, allow_pickle=True))
        return {k: (v.item() if v.ndim == 0 else v) for k, v in d.items()}

    t0 = time.time()
    swot = _load_pkl(ROOT / "cache" / f"swot_features_{mission}_{rid}_{c0}_{c1}.pkl")
    hfr = _load_pkl(ROOT / "experiments" / rid / "hfr_target" / f"hfr_{mission}_{rid}.pkl")
    keys = [t for t in swot if swot[t] and hfr.get(t, [])]          # flattened.keys()
    n_tr = max(1, int(len(keys) * train_fraction))
    train_keys, test_keys = keys[:n_tr], keys[n_tr:]

    train_end = max(np.datetime64(ds["time"].values) for t in train_keys for ds in swot[t] if ds is not None)

    rec = {k: [] for k in ["cycle", "pass_idx", "swot_time", "hfr_time", "lat", "lon",
                           "y_u", "y_v", "grid_ok", "dt_h"]}
    from swotxai.pipeline.steps_ml import _pair_by_time   # the pipeline's own pairing (time-matched, <= 1 h)
    for t in (test_keys if split == "test" else train_keys):
        for j, (hfr_ds, swot_ds) in enumerate(_pair_by_time(hfr[t], swot[t], lambda *a: None)):
            if hfr_ds is None or swot_ds is None:
                continue
            n_rows = swot_ds["lat"].size
            yu = pd.Series(np.asarray(hfr_ds["u"].values, np.float32).reshape(-1)).reindex(range(n_rows))
            yv = pd.Series(np.asarray(hfr_ds["v"].values, np.float32).reshape(-1)).reindex(range(n_rows))
            valid = yu.notna().to_numpy() | yv.notna().to_numpy()
            if not valid.any():
                continue
            st = np.datetime64(swot_ds["time"].values)
            ht = np.datetime64(hfr_ds["time"].values) if "time" in hfr_ds.coords else np.datetime64("NaT")
            same_grid = (hfr_ds["lat"].shape == swot_ds["lat"].shape) and \
                np.allclose(hfr_ds["lat"].values, swot_ds["lat"].values, equal_nan=True) and \
                np.allclose(hfr_ds["lon"].values, swot_ds["lon"].values, equal_nan=True)
            m = int(valid.sum())
            rec["cycle"].append(np.full(m, int(t)))
            rec["pass_idx"].append(np.full(m, j))
            rec["swot_time"].append(np.full(m, st.astype("datetime64[s]").astype("int64")))
            rec["hfr_time"].append(np.full(m, ht.astype("datetime64[s]").astype("int64")))
            rec["lat"].append(swot_ds["lat"].values.reshape(-1)[valid])
            rec["lon"].append(swot_ds["lon"].values.reshape(-1)[valid])
            rec["y_u"].append(yu.to_numpy(np.float32)[valid])
            rec["y_v"].append(yv.to_numpy(np.float32)[valid])
            rec["grid_ok"].append(np.full(m, same_grid))
            rec["dt_h"].append(np.full(m, (ht - st) / np.timedelta64(1, "h")))
    meta = {k: np.concatenate(v) for k, v in rec.items()}
    meta["train_end"] = train_end.astype("datetime64[s]").astype("int64")
    meta["train_keys"] = np.array(train_keys)
    meta["test_keys"] = np.array(test_keys)
    np.savez(out, **meta)
    print(f"  replayed {rid}: {len(meta['y_u']):,} rows, {len(test_keys)} test cycles, {time.time()-t0:.0f}s")
    del swot, hfr
    return meta


# --------------------------------------------------------------------------
# 2. Baselines
# --------------------------------------------------------------------------

def sample_duacs(rid: str, t_sec: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """DUACS L4 daily ugos/vgos linearly interpolated in (time, lat, lon) to the rows.
    Returns (n, 2) with NaN where DUACS is land-masked. None if the file is absent."""
    import xarray as xr
    from scipy.interpolate import RegularGridInterpolator
    files = sorted((ROOT / "experiments" / rid / "duacs").glob(f"duacs_l4_{rid}_*.nc"))
    if not files:
        return None
    ds = xr.concat([xr.open_dataset(f)[["ugos", "vgos"]] for f in files], dim="time").sortby("time")
    ds = ds.isel(time=np.unique(ds["time"].values, return_index=True)[1])
    t_days = (ds["time"].values.astype("datetime64[s]").astype("int64")) / 86400.0
    inside = (t_sec / 86400.0 >= t_days.min()) & (t_sec / 86400.0 <= t_days.max())
    if inside.sum() < 50:
        return None
    td = (ds["time"].values.astype("datetime64[s]").astype("int64")) / 86400.0
    pts = np.column_stack([t_sec / 86400.0, lat, lon])
    out = np.full((len(t_sec), 2), np.nan, np.float32)
    for c, v in enumerate(["ugos", "vgos"]):
        f = RegularGridInterpolator((td, ds["latitude"].values, ds["longitude"].values), ds[v].values,
                                    method="linear", bounds_error=False, fill_value=np.nan)
        out[:, c] = f(pts)
    return out


def cell_climatology(lat_tr, lon_tr, Y_tr, lat, lon):
    """Training-period mean (u, v) at each radar grid point, looked up for the rows at (lat, lon).
    Grid points never seen in training fall back to the global training mean. The
    baseline a model must beat before it can claim to have learned anything but
    where it is: with repeat tracks a network can read location off the static
    inputs (mdt, stencil pattern) and memorise the local mean current."""
    key_tr = pd.MultiIndex.from_arrays([np.round(lat_tr, 3), np.round(lon_tr, 3)])
    tab = pd.DataFrame(Y_tr, index=key_tr).groupby(level=[0, 1]).mean()
    key = pd.MultiIndex.from_arrays([np.round(lat, 3), np.round(lon, 3)])
    out = tab.reindex(key).to_numpy(np.float32)
    g = np.nanmean(Y_tr, 0)
    for c in range(2):
        out[np.isnan(out[:, c]), c] = g[c]
    return out


def cell_linear(lat_tr, lon_tr, X_tr5, Y_tr, lat, lon, X5, min_n=5, prior_rows=5.0):
    """Per-grid-point linear model y ~ 1 + x at each radar cell (x = g_u, g_v, W_u, W_v),
    partially pooled: each cell's coefficients shrink toward the global fit with a prior
    worth ``prior_rows`` training rows (predictors standardised so the shrinkage is even).
    With ~170 rows per cell (calval) the local fit dominates; with ~10 (science, 12
    cycles) the prior does. Cells with fewer than ``min_n`` complete rows use the global fit.
    This is the cell-specific calibration slope and wind response a network could learn
    from repeat tracks once it knows where it is."""
    ok_tr = np.isfinite(X_tr5).all(1) & np.isfinite(Y_tr).all(1)
    mu, sd = X_tr5[ok_tr].mean(0), X_tr5[ok_tr].std(0) + 1e-9
    Z_tr, Z = (X_tr5 - mu) / sd, (X5 - mu) / sd
    fg, bg = fit_linear(Z_tr, Y_tr)                       # global coefficients (P, 2)
    out = fg(Z).astype(np.float32)
    k_tr = pd.MultiIndex.from_arrays([np.round(lat_tr, 3), np.round(lon_tr, 3)])
    k_te = pd.MultiIndex.from_arrays([np.round(lat, 3), np.round(lon, 3)])
    codes, uniq = pd.factorize(k_tr.append(k_te))
    c_tr, c_te = codes[:len(k_tr)], codes[len(k_tr):]
    order_tr = np.argsort(c_tr[ok_tr], kind="stable"); idx_tr = np.flatnonzero(ok_tr)[order_tr]
    bounds_tr = np.searchsorted(c_tr[idx_tr], np.arange(len(uniq) + 1))
    order_te = np.argsort(c_te, kind="stable")
    bounds_te = np.searchsorted(c_te[order_te], np.arange(len(uniq) + 1))
    P = Z.shape[1] + 1
    lam = prior_rows * np.eye(P)
    n_local, n_rows = 0, []
    for c in range(len(uniq)):
        rows = idx_tr[bounds_tr[c]:bounds_tr[c + 1]]
        te = order_te[bounds_te[c]:bounds_te[c + 1]]
        if len(rows) < min_n or len(te) == 0:
            continue
        A = np.column_stack([np.ones(len(rows)), Z_tr[rows]])
        beta = np.linalg.solve(A.T @ A + lam, A.T @ Y_tr[rows] + lam @ bg)
        out[te] = np.column_stack([np.ones(len(te)), Z[te]]) @ beta
        n_local += len(te); n_rows.append(len(rows))
    print(f"  cell_linear: {n_local / max(1, len(X5)):.0%} of rows from a cell fit "
          f"(median {int(np.median(n_rows)) if n_rows else 0} training rows per cell, prior worth {prior_rows:g})")
    return out


def ekman_classical(wu, wv, frac=0.03, deg=45.0):
    """Surface Ekman drift: ``frac`` of wind speed, rotated ``deg`` to the right
    of the wind (northern hemisphere). Rule-of-thumb baseline."""
    th = np.deg2rad(deg)
    return frac * (wu * np.cos(th) + wv * np.sin(th)), frac * (-wu * np.sin(th) + wv * np.cos(th))


def fit_linear(X, Y):
    """Least squares with intercept, NaN rows dropped; returns predict(X)."""
    ok = np.isfinite(X).all(1) & np.isfinite(Y).all(1)
    A = np.column_stack([np.ones(ok.sum()), X[ok]])
    coef, *_ = np.linalg.lstsq(A, Y[ok], rcond=None)
    return lambda Xn: np.column_stack([np.ones(len(Xn)), Xn]) @ coef, coef


def fit_ridge72(X, Y, alpha=1.0):
    """Ridge on all 72 stencil columns; NaN -> train column mean; standardized.
    The linear-skill reference (what a linear model gets from the same inputs)."""
    from sklearn.linear_model import Ridge
    mu = np.nanmean(X, 0); sd = np.nanstd(X, 0) + 1e-6
    def prep(A):
        A = (A - mu) / sd
        return np.where(np.isfinite(A), A, 0.0)
    models = []
    for c in range(2):
        ok = np.isfinite(Y[:, c])
        models.append(Ridge(alpha=alpha).fit(prep(X[ok]), Y[ok, c]))
    return lambda Xn: np.column_stack([m.predict(prep(Xn)) for m in models])


# --------------------------------------------------------------------------
# 3. Metrics + block bootstrap
# --------------------------------------------------------------------------

def _block_sums(y, p, blocks):
    """Per-block sufficient statistics so any block resample is a sum."""
    ok = np.isfinite(y) & np.isfinite(p)
    y, p, b = y[ok], p[ok], blocks[ok]
    e2 = (y - p) ** 2
    ub, inv = np.unique(b, return_inverse=True)
    n = np.bincount(inv).astype(float)
    S = {"n": n,
         "sy": np.bincount(inv, y), "sy2": np.bincount(inv, y * y),
         "se2": np.bincount(inv, e2), "se": np.bincount(inv, y - p)}
    return S


def _r2_rmse(S, w=None):
    w = np.ones_like(S["n"]) if w is None else w
    n = (w * S["n"]).sum(); sy = (w * S["sy"]).sum(); sy2 = (w * S["sy2"]).sum()
    sse = (w * S["se2"]).sum()
    sst = sy2 - sy * sy / n
    return 1.0 - sse / sst, np.sqrt(sse / n), (w * S["se"]).sum() / n


def score(y, p, blocks, B=1000, seed=0):
    S = _block_sums(y, p, blocks)
    r2, rmse, _ = _r2_rmse(S)
    ok = np.isfinite(y) & np.isfinite(p)
    pear = float(np.corrcoef(y[ok], p[ok])[0, 1]) if ok.sum() > 2 and p[ok].std() > 0 else float("nan")
    std_ratio = float(p[ok].std() / y[ok].std()) if ok.sum() > 2 else float("nan")
    nb = len(S["n"]); N = S["n"].sum()
    rng = np.random.default_rng(seed)
    cnt = rng.multinomial(nb, np.full(nb, 1 / nb), size=B)       # block resamples
    boot = np.array([_r2_rmse(S, c) for c in cnt])
    # effective N: iid variance of the mean error / block-bootstrap variance of the mean error
    e_var = ((S["se2"].sum() / N) - (S["se"].sum() / N) ** 2)
    var_mean_block = boot[:, 2].var()
    n_eff = float(e_var / var_mean_block) if var_mean_block > 0 else float("nan")
    return {"n": int(N), "n_blocks": int(nb), "r2": float(r2), "rmse": float(rmse),
            "pearson_r": pear, "std_ratio_pred_over_obs": std_ratio,
            "r2_ci": [float(np.percentile(boot[:, 0], 2.5)), float(np.percentile(boot[:, 0], 97.5))],
            "rmse_ci": [float(np.percentile(boot[:, 1], 2.5)), float(np.percentile(boot[:, 1], 97.5))],
            "n_eff": n_eff}


def resid_r2(y, p, base):
    """R² on the residual HFR - geostrophic: 1 - SSE(model) / Var(y - base)."""
    ok = np.isfinite(y) & np.isfinite(p) & np.isfinite(base)
    r = y[ok] - base[ok]
    return float(1.0 - ((y[ok] - p[ok]) ** 2).sum() / ((r - r.mean()) ** 2).sum())


# --------------------------------------------------------------------------
# 4. Driver
# --------------------------------------------------------------------------

def evaluate(run: str, regions: list[str], B: int, buffer_days: float, max_dt_h: float):
    run_dir = ROOT / "experiments" / "evolve" / run
    out_dir = run_dir / "baselines"; out_dir.mkdir(exist_ok=True)
    meta_all = json.load(open(run_dir / "data_meta.json"))
    global FEATURES, K, CENTER
    FEATURES, K = list(meta_all["features"]), int(meta_all["stencil_k"]); CENTER = K * K // 2
    mission = meta_all["mission"]
    pooled = bool(meta_all.get("pooled", False))
    c0, c1 = int(meta_all.get("cycles_start", 1)), int(meta_all.get("cycles_end", 16))
    if not pooled:                       # single-region run (e.g. the calval campaign)
        regions = [meta_all["region"]]
    data = np.load(run_dir / "data.npz")
    X_test, Y_test = data["X_test"], data["Y_test"]
    champ_dir = max((d for d in (run_dir / "candidates").iterdir() if (d / "predictions.npz").exists()),
                    key=lambda d: _champion_fitness(run_dir, d.name))
    pred_pooled = np.load(champ_dir / "predictions.npz")["pred"]
    print(f"champion: {champ_dir.name}")

    results = {"run": run, "champion": champ_dir.name, "mission": mission,
               "protocols": {"all_test": "all held-out rows (cycles after the 80% split; 0-day buffer)",
                             f"buffer{int(buffer_days)}d": f"held-out rows >= {buffer_days} days after the last training pass",
                             "all_rows": "every clean-paired held-out row (models defined without SWOT only)",
                             "swot_valid": "rows whose centre-cell SWOT geostrophy and ERA5 are finite (all models)"},
               "regions": {}}
    for rid in regions:
        print(f"== {rid}")
        if pooled or rid == meta_all.get("region"):
            spans = meta_all["region_test_slices"][rid] if pooled else [[0, len(X_test)]]
            idx = np.concatenate([np.arange(s, e) for s, e in spans])
            Xr, Yr, Pr = X_test[idx], Y_test[idx], pred_pooled[idx]
        else:   # sister region of a single-region run: rows come from its solo freeze; no campaign predictions
            sd0 = np.load(run_dir / "solo" / rid / "data.npz")
            Xr, Yr, Pr = sd0["X_test"], sd0["Y_test"], None
            idx = np.arange(len(Yr))
        meta = replay_meta(rid, mission, c0, c1, meta_all["train_fraction"], out_dir / f"meta_{rid}.npz")
        # --- verification: replay must reproduce Y_test exactly
        assert len(meta["y_u"]) == len(idx), f"{rid}: replay rows {len(meta['y_u'])} != test rows {len(idx)}"
        for c, k in enumerate(["y_u", "y_v"]):
            same = np.array_equal(meta[k], Yr[:, c], equal_nan=True)
            assert same, f"{rid}: replayed {k} differs from Y_test"
        print("  replay verified: Y_test reproduced exactly")

        # --- solo twin (same candidate code, trained on this region only)
        solo = run_dir / "solo" / rid
        P_solo = None
        if (solo / "data.npz").exists():
            sd = np.load(solo / "data.npz")
            assert np.array_equal(sd["Y_test"], Yr, equal_nan=True), f"{rid}: solo test rows differ from pooled spans"
            X_tr, Y_tr = sd["X_train"], sd["Y_train"]
            seeds = sorted(solo.glob("seed*/pred.npz"))
            if seeds:
                P_solo = np.mean([np.load(p)["pred"] for p in seeds], axis=0)
                print(f"  solo twin: {len(seeds)} seed(s) averaged")
        elif not pooled:
            X_tr, Y_tr = data["X_train"], data["Y_train"]
        else:
            raise FileNotFoundError(f"{solo}/data.npz needed for training-period baselines")

        # --- row masks
        clean = meta["grid_ok"] & (np.abs(meta["dt_h"]) <= max_dt_h)
        swot_ok = np.isfinite(Xr[:, col("ugos_filtered")]) & np.isfinite(Xr[:, col("vgos_filtered")]) \
            & np.isfinite(Xr[:, col("era5_u")]) & np.isfinite(Xr[:, col("era5_v")])
        t = meta["swot_time"]
        buffered = t >= meta["train_end"] + int(buffer_days * 86400)
        blocks = meta["cycle"] * 1000 + meta["pass_idx"]
        print(f"  rows {len(idx):,}: clean pairing {clean.mean():.1%}, SWOT+ERA5 valid {swot_ok.mean():.1%}, "
              f"buffered {buffered.mean():.1%} ({len(np.unique(blocks[buffered]))} passes)")

        # --- training-row metadata (lat/lon/time per training row), verified against Y_train
        mtr = replay_meta(rid, mission, c0, c1, meta_all["train_fraction"], out_dir / f"meta_train_{rid}.npz", split="train")
        assert len(mtr["y_u"]) == len(Y_tr) and np.array_equal(mtr["y_u"], Y_tr[:, 0], equal_nan=True), \
            f"{rid}: train replay does not match Y_train"

        # --- baselines
        ug, vg = Xr[:, col("ugos_filtered")], Xr[:, col("vgos_filtered")]
        wu, wv = Xr[:, col("era5_u")], Xr[:, col("era5_v")]
        geos = np.column_stack([ug, vg])
        eu, ev = ekman_classical(wu, wv)
        ekman = geos + np.column_stack([eu, ev])
        # fitted instantaneous wind response on this region's training rows
        Wtr = X_tr[:, [col("era5_u"), col("era5_v")]]
        Gtr = X_tr[:, [col("ugos_filtered"), col("vgos_filtered")]]
        fwind, coef = fit_linear(Wtr, Y_tr - Gtr)
        windfit = geos + fwind(np.column_stack([wu, wv]))
        ridge = fit_ridge72(X_tr, Y_tr)(Xr)
        # per-grid-point climatology, alone and with a wind response fitted to its residual
        cellmean = cell_climatology(mtr["lat"], mtr["lon"], Y_tr, meta["lat"], meta["lon"])
        cellmean_tr = cell_climatology(mtr["lat"], mtr["lon"], Y_tr, mtr["lat"], mtr["lon"])
        fcw, _ = fit_linear(Wtr, Y_tr - cellmean_tr)
        cellmean_wind = cellmean + fcw(np.column_stack([wu, wv]))
        # per-grid-point linear: cell-specific slopes on (g_u, g_v, W_u, W_v)
        X5tr = np.column_stack([Gtr, Wtr]); X5 = np.column_stack([geos, wu, wv])
        cell_lin = cell_linear(mtr["lat"], mtr["lon"], X5tr, Y_tr, meta["lat"], meta["lon"], X5)
        trmean = np.tile(np.nanmean(Y_tr, 0), (len(Yr), 1))

        fwind_only, _ = fit_linear(Wtr, Y_tr)
        windonly = fwind_only(np.column_stack([wu, wv]))
        # calibrated geostrophy: per-component a + b*g fitted on training rows
        # (raw L3 velocity overshoots the radar; a least-squares rescale is the
        # fair linear-physics baseline, cf. Tchonang's 70-km optimum)
        geos_cal = np.column_stack([fit_linear(Gtr[:, [c]], Y_tr[:, [c]])[0](geos[:, [c]])[:, 0] for c in range(2)])
        fgw, coef_gw = fit_linear(np.column_stack([Gtr, Wtr]), Y_tr)
        geos_cal_wind = fgw(np.column_stack([geos, wu, wv]))
        # DUACS L4 (nadir-altimeter maps): raw, calibrated on training rows, + wind
        duacs = sample_duacs(rid, meta["swot_time"], meta["lat"], meta["lon"])
        duacs_models = {}
        if duacs is not None and np.isfinite(duacs).all(1).sum() < 50:
            print(f"  DUACS finite on only {int(np.isfinite(duacs).all(1).sum())} held-out rows; DUACS baselines skipped")
            duacs = None
        if duacs is not None:
            Dtr = sample_duacs(rid, mtr["swot_time"], mtr["lat"], mtr["lon"])
            duacs_cal = np.column_stack([fit_linear(Dtr[:, [c]], Y_tr[:, [c]])[0](duacs[:, [c]])[:, 0] for c in range(2)])
            fdw, _ = fit_linear(np.column_stack([Dtr, Wtr]), Y_tr)
            duacs_cal_wind = fdw(np.column_stack([duacs, wu, wv]))
            duacs_models = {"duacs_geos": (duacs, False), "duacs_calibrated": (duacs_cal, False),
                            "duacs_cal+wind_lsq": (duacs_cal_wind, False)}
            reg_duacs_frac = float(np.isfinite(duacs[:, 0]).mean())
            print(f"  DUACS finite on {reg_duacs_frac:.1%} of held-out rows")
            del Dtr
        # (prediction, needs finite centre-cell SWOT geostrophy?)
        models = {
            "train_mean": (trmean, False),
            "cell_mean": (cellmean, False),
            "cell_mean+wind_lsq": (cellmean_wind, False),
            "cell_linear_geos_wind": (cell_lin, True),
            "wind_only_lsq": (windonly, False),
            "swot_geos": (geos, True),
            "geos+ekman_3pct45deg": (ekman, True),
            "geos+wind_lsq": (windfit, True),
            "geos_calibrated": (geos_cal, True),
            "geos_cal+wind_lsq": (geos_cal_wind, True),
            **duacs_models,
            "ridge72_solo": (ridge, False),
        }
        if Pr is not None:
            models["champion_pooled"] = (Pr, False)
        if P_solo is not None:
            models["champion_solo"] = (P_solo, False)

        reg = {"n_test_rows": int(len(idx)), "clean_pair_frac": float(clean.mean()),
               "duacs_finite_frac": float(np.isfinite(duacs[:, 0]).mean()) if duacs is not None else None,
               "swot_valid_frac": float(swot_ok.mean()),
               "train_end": str(np.datetime64(int(meta["train_end"]), "s")),
               "test_cycles": [int(c) for c in meta["test_keys"]],
               "wind_lsq_coef": {"u": coef[:, 0].tolist(), "v": coef[:, 1].tolist(),
                                 "note": "[intercept, d/d era5_u, d/d era5_v] on (hfr - geos)"},
               "geos_cal_wind_coef": {"u": coef_gw[:, 0].tolist(), "v": coef_gw[:, 1].tolist(),
                                      "note": "[intercept, ugos, vgos, era5_u, era5_v] -> hfr component"},
               "protocols": {}}
        bd = f"buffer{int(buffer_days)}d"
        protocols = {
            "all_test/all_rows": (clean, False),
            "all_test/swot_valid": (clean & swot_ok, True),
            f"{bd}/all_rows": (clean & buffered, False),
            f"{bd}/swot_valid": (clean & swot_ok & buffered, True),
        }
        # truth tiers: the stored 25-h target always; others from hfr_tiers.py when present
        truths = {"target_25h": Yr}
        tiers_path = out_dir / f"tiers_{rid}.npz"
        if tiers_path.exists():
            tz = np.load(tiers_path)
            for k in ["y_raw", "y_bw18", "y_bw48"]:
                if k in tz:
                    truths[k[2:]] = tz[k]
            reg["tiers_verification"] = json.loads(str(tz["verification"]))
        reg["tiers"] = {}
        # all tiers are scored on the rows where EVERY tier is finite (common subset),
        # so tier-to-tier differences are not coverage differences; the main
        # "protocols" table (target_25h) is scored on all rows separately below.
        common = np.ones(len(Yr), bool)
        for Yt in truths.values():
            common &= np.isfinite(Yt).all(1)
        reg["tiers_common_frac"] = float(common.mean())
        for tname, Yt in truths.items():
            reg["tiers"][tname] = {}
            for pname, (pmask0, with_swot) in protocols.items():
                pmask = pmask0 & (common if len(truths) > 1 else np.isfinite(Yt).any(1))
                if pmask.sum() < 50:
                    reg["tiers"][tname][pname] = {"skipped": f"only {int(pmask.sum())} rows"}
                    continue
                prot = {}
                for mname, (P, needs_swot) in models.items():
                    if needs_swot and not with_swot:
                        continue
                    row = {}
                    for c, cn in enumerate("uv"):
                        s = score(Yt[pmask, c], P[pmask, c], blocks[pmask], B=B)
                        s["r2_resid_vs_geos"] = resid_r2(Yt[pmask, c], P[pmask, c], geos[pmask, c]) if with_swot else float("nan")
                        row[cn] = s
                    prot[mname] = row
                reg["tiers"][tname][pname] = add_pct(prot)
                print(f"  [{tname} | {pname}] n={int(pmask.sum()):,}  " + "  ".join(
                    f"{m}: {prot[m]['u']['r2']:.3f}/{prot[m]['v']['r2']:.3f}" for m in prot if not m.startswith("_")))
        for pname, (pmask, with_swot) in protocols.items():
            if len(truths) == 1:
                reg["protocols"][pname] = reg["tiers"]["target_25h"][pname]
                continue
            if pmask.sum() < 50:
                reg["protocols"][pname] = {"skipped": f"only {int(pmask.sum())} rows"}
                continue
            prot = {}
            for mname, (P, needs_swot) in models.items():
                if needs_swot and not with_swot:
                    continue
                row = {}
                for c, cn in enumerate("uv"):
                    s = score(Yr[pmask, c], P[pmask, c], blocks[pmask], B=B)
                    s["r2_resid_vs_geos"] = resid_r2(Yr[pmask, c], P[pmask, c], geos[pmask, c]) if with_swot else float("nan")
                    row[cn] = s
                prot[mname] = row
            reg["protocols"][pname] = add_pct(prot)
        results["regions"][rid] = reg
        del X_tr, Y_tr

    json.dump(results, open(out_dir / "results.json", "w"), indent=1)
    (out_dir / "results.md").write_text(render_md(results), encoding="utf-8")
    print(f"wrote {out_dir/'results.json'} and results.md")
    return results


def add_pct(prot: dict) -> dict:
    """% RMSE change vs the calibrated geostrophy (raw L3 geostrophy overshoots
    the radar, so it is not a sensible reference); train mean where SWOT is absent."""
    ref = "geos_calibrated" if "geos_calibrated" in prot else "train_mean"
    g = prot[ref]
    for mname, row in prot.items():
        if mname.startswith("_"):
            continue
        for cn in "uv":
            row[cn]["pct_rmse_vs_ref"] = 100.0 * (row[cn]["rmse"] - g[cn]["rmse"]) / g[cn]["rmse"]
    prot["_ref"] = ref
    return prot


def _champion_fitness(run_dir: Path, cid: str) -> float:
    best = -np.inf
    for line in open(run_dir / "database.jsonl", encoding="utf-8"):
        r = json.loads(line)
        if r.get("candidate_id") == cid and r.get("status") == "ok":
            best = max(best, r["metrics"]["fitness"])
    return best


def render_md(res: dict) -> str:
    L = [f"# Baselines — {res['run']} ({res['mission']}), champion {res['champion']}", "",
         f"`champion_pooled` = AlphaEvolve candidate {res['champion']} of run {res['run']} "
         f"(`experiments/evolve/{res['run']}/candidates/{res['champion']}/candidate.py`; ridge-residual MLP), "
         "scored with its campaign predictions; `champion_solo` = the same code retrained per region "
         "(`scripts/solo_vs_pooled.py`). Not the app's production `ann` backend.", ""]
    L.append("R² = 1 − SSE/SST on the held-out rows (Nash–Sutcliffe with the held-out mean); "
             "RMSE in m/s; %ΔRMSE relative to the reference named per block (calibrated geostrophy where SWOT is present, "
             "else the training mean; negative = better); R²_res = 1 − SSE/Var(HFR − raw L3 geos). 95% CIs from a block bootstrap over (cycle, pass) "
             "snapshots; N_eff = iid variance of the mean error / block-bootstrap variance of it. "
             "Rows: clean SWOT↔HFR pairing and finite centre-cell SWOT+ERA5. "
             "What each model regresses against what: docs/figures/baselines/README.md, section \"What is regressed against what\".")
    for rid, reg in res["regions"].items():
        L += ["", f"## {rid}  (test cycles {reg['test_cycles'][0]}–{reg['test_cycles'][-1]}, "
              f"train end {reg['train_end'][:10]}, clean pairs {reg['clean_pair_frac']:.0%}, "
              f"SWOT-valid {reg['swot_valid_frac']:.0%})"]
        tiers = {"MAIN (target_25h, all rows)": reg["protocols"]}
        if len(reg.get("tiers", {})) > 1:
            tiers.update({f"{k} (common rows, {reg['tiers_common_frac']:.0%} of rows)": v for k, v in reg["tiers"].items()})
        if "tiers_verification" in reg:
            v = reg["tiers_verification"]
            L += ["", f"Tier check — rebuilt 25-h mean vs stored target: r = {v['y_u']['corr']:.4f} (u), "
                  f"{v['y_v']['corr']:.4f} (v); RMS diff {v['y_u']['rms_diff']:.4f} / {v['y_v']['rms_diff']:.4f} m/s."]
        for tname, prots in tiers.items():
          L += ["", f"### truth tier: {tname}"]
          for pname, prot in prots.items():
            if "skipped" in prot:
                L += ["", f"**{pname}**: skipped ({prot['skipped']})"]; continue
            any_row = prot["train_mean"]
            ref = prot["_ref"]
            L += ["", f"**{pname}** — n = {any_row['u']['n']:,} rows, {any_row['u']['n_blocks']} passes, "
                  f"N_eff ≈ {any_row['u']['n_eff']:.0f} (u) / {any_row['v']['n_eff']:.0f} (v); %ΔRMSE vs {ref}", "",
                  "| model | R²_u [95% CI] | R²_v [95% CI] | r_u | r_v | RMSE_u | RMSE_v | %ΔRMSE_u | %ΔRMSE_v | R²res_u | R²res_v |",
                  "|---|---|---|---|---|---|---|---|---|---|---|"]
            for m, row in prot.items():
                if m.startswith("_"):
                    continue
                u, v = row["u"], row["v"]
                rr = (f"{u['r2_resid_vs_geos']:.3f} | {v['r2_resid_vs_geos']:.3f}" if ref == "swot_geos" else "– | –")
                L.append(f"| {m} | {u['r2']:.3f} [{u['r2_ci'][0]:.3f}, {u['r2_ci'][1]:.3f}] | "
                         f"{v['r2']:.3f} [{v['r2_ci'][0]:.3f}, {v['r2_ci'][1]:.3f}] | "
                         f"{u['pearson_r']:.2f} | {v['pearson_r']:.2f} | "
                         f"{u['rmse']:.3f} | {v['rmse']:.3f} | {u['pct_rmse_vs_ref']:+.1f} | {v['pct_rmse_vs_ref']:+.1f} | {rr} |")
    return "\n".join(L) + "\n"


# --------------------------------------------------------------------------
# 5. Self-check
# --------------------------------------------------------------------------

def selftest():
    rng = np.random.default_rng(1)
    y = rng.normal(size=5000); p = y + rng.normal(scale=0.5, size=5000)
    blocks = np.repeat(np.arange(50), 100)
    s = score(y, p, blocks, B=200)
    from sklearn.metrics import r2_score, mean_squared_error
    assert abs(s["r2"] - r2_score(y, p)) < 1e-9
    assert abs(s["rmse"] - np.sqrt(mean_squared_error(y, p))) < 1e-9
    assert s["r2_ci"][0] <= s["r2"] <= s["r2_ci"][1]
    # independent blocks -> N_eff close to N (within bootstrap noise)
    assert 2500 < s["n_eff"] < 10000, s["n_eff"]
    # strongly block-correlated errors -> N_eff near the number of blocks
    p2 = y + np.repeat(rng.normal(size=50), 100)
    s2 = score(y, p2, blocks, B=200)
    assert s2["n_eff"] < 200, s2["n_eff"]
    assert abs(resid_r2(y, y, np.zeros_like(y)) - 1.0) < 1e-12
    print("selftest ok", {k: s[k] for k in ("r2", "n_eff")}, "corr N_eff", round(s2["n_eff"]))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="evolve_all_science_v1")
    ap.add_argument("--regions", default="uswc,usegc,gak,glna,prvi,ushi")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--buffer-days", type=float, default=30.0)
    ap.add_argument("--max-dt-h", type=float, default=2.0, help="drop SWOT↔HFR pairs further apart than this")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--render-only", action="store_true", help="re-render results.md from results.json")
    a = ap.parse_args()
    if a.selftest:
        selftest()
    elif a.render_only:
        out_dir = ROOT / "experiments" / "evolve" / a.run / "baselines"
        res = json.load(open(out_dir / "results.json"))
        for reg in res["regions"].values():
            for prots in [reg["protocols"], *reg.get("tiers", {}).values()]:
                for pn, prot in prots.items():
                    if "skipped" not in prot:
                        prots[pn] = add_pct(prot)
        json.dump(res, open(out_dir / "results.json", "w"), indent=1)
        (out_dir / "results.md").write_text(render_md(res), encoding="utf-8")
        print("re-rendered", out_dir / "results.md")
    else:
        evaluate(a.run, a.regions.split(","), a.bootstrap, a.buffer_days, a.max_dt_h)
