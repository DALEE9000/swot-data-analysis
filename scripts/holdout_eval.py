#!/usr/bin/env python
"""Out-of-time and out-of-region evaluation on SWOT science cycles 017-058.

The AlphaEvolve champion (evolve_all_science_v1 / gen031_00) is a training
recipe, not weights. Here it is retrained on year one (cycles 001-016, the
cycles the search was allowed to see) on several region sets, and every model
is scored once on cycles 017-058, which no decision in the chain ever saw:

    variant     training regions (cycles 001-016)
    us_y1       the six US regions            (scored on US = out-of-time only)
    eu_y1       the six European regions
    au_y1       the seven Australian regions
    all_y1      all nineteen
    not_us_y1   Europe + Australia            (scored on US = out-of-time AND out-of-region)
    not_eu_y1   US + Australia
    not_au_y1   US + Europe

Baselines are fitted per region on the same year-one rows (calibrated SWOT
geostrophy, with and without a fitted wind response; DUACS L4 likewise; the
per-cell climatology; the training mean) and scored on the same held-out rows.
Scoring follows scripts/baselines_eval.py: clean pairing (same grid, |dt| <= 2 h),
SWOT-valid rows (finite centre-cell ugos/vgos/era5), block bootstrap over
(cycle, pass) for the confidence intervals. Rows after the end of the ERA5
record are excluded (the pipeline would give them a stale wind).

Stages (each resumable; run from the repo root in the project env):
    python scripts/holdout_eval.py --freeze glna            # one region -> experiments/holdout/<rid>.npz
    python scripts/holdout_eval.py --freeze all             # every region, one subprocess each
    python scripts/holdout_eval.py --train us_y1,not_us_y1  # retrain the champion recipe (GPU, ~30-55 min each)
    python scripts/holdout_eval.py --train all
    python scripts/holdout_eval.py --score                  # results.json / results.md
    python scripts/holdout_eval.py --figures                # docs/figures/holdout/

Everything reads the local mirrors (experiments/<rid>/...); nothing touches S3.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
os.chdir(REPO)

import swotxai  # noqa: F401  (torch-before-numpy DLL ordering)
import numpy as np

ENV = dict(os.environ, PYTHONUTF8="1")   # pipeline log lines carry arrows; Windows consoles default to cp1252

OUT = REPO / "experiments" / "holdout"
FIG = REPO / "docs" / "figures" / "holdout"
RUN = REPO / "experiments" / "evolve" / "evolve_all_science_v1"
CHAMPION = "gen031_00"
CONTINENTS = {
    "us": ["uswc", "usegc", "gak", "glna", "prvi", "ushi"],
    "eu": ["eu_biscay", "eu_galicia", "eu_wmed", "eu_tirlig", "eu_gbight", "eu_norway"],
    "au": ["au_rot", "au_turq", "au_cof", "au_newc", "au_sag", "au_nwa", "au_corl"],
}
REGIONS = [r for c in CONTINENTS.values() for r in c]
CONT_OF = {r: c for c, rs in CONTINENTS.items() for r in rs}
VARIANTS = {
    "us_y1": CONTINENTS["us"], "eu_y1": CONTINENTS["eu"], "au_y1": CONTINENTS["au"],
    "all_y1": REGIONS,
    "not_us_y1": CONTINENTS["eu"] + CONTINENTS["au"],
    "not_eu_y1": CONTINENTS["us"] + CONTINENTS["au"],
    "not_au_y1": CONTINENTS["us"] + CONTINENTS["eu"],
}
# leave-one-region-out within the US: train on the other five, score the held-out one
# (gak and glna have no clean test rows, so holding them out teaches nothing)
LORO = {f"us_wo_{r}_y1": [x for x in CONTINENTS["us"] if x != r] for r in ["uswc", "usegc", "prvi", "ushi"]}
VARIANTS.update(LORO)
FEATURES = ["mdt", "ssha_filtered", "ugos_filtered", "vgos_filtered",
            "ugosa_filtered", "vgosa_filtered", "era5_u", "era5_v"]
K = 3
LAST_TRAIN_CYCLE = 16
LAST_CYCLE = 58
MAX_DT_H = 2.0
PY = sys.executable


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def col(feature: str) -> int:
    return FEATURES.index(feature) * K * K + K * K // 2


# --------------------------------------------------------------------------
# 1. Freeze one region: X, Y and row metadata for every cycle in the pkls
# --------------------------------------------------------------------------

def freeze(rid: str, chunk: int | None = None) -> Path:
    out = OUT / f"{rid}.npz"
    if out.exists():
        log(f"{rid}: {out.name} exists, skipping")
        return out
    import pickle

    from swotxai.config import load_config
    from swotxai.pipeline.orchestrator import run_shared_steps
    from swotxai.pipeline.steps_ml import _pair_by_time, pass_time, step_flatten
    from swotxai.presets import config_overrides

    ov = dict(config_overrides(rid, "science"), model="ann", features=FEATURES, stencil_k=K, cycles_end=LAST_CYCLE)
    cfg = replace(load_config("config.yaml"), **ov)
    for p in (cfg.swot_pkl_path, cfg.hfr_pkl_path, cfg.era5_pkl_path):
        local = Path(p.replace("s3://swot-ai-ssv/", ""))
        assert local.exists(), f"{rid}: local mirror missing: {local} (would stream from S3; refusing)"
    with open(Path(cfg.era5_pkl_path.replace("s3://swot-ai-ssv/", "")), "rb") as fh:
        era5_end = np.datetime64(pickle.load(fh)["time"].values.max(), "s")
    cb = lambda step, frac, msg: (frac in (0.0, 1.0)) and log(f"  [{step}] {msg}")  # noqa: E731
    shared = run_shared_steps(cfg, progress_cb=cb, use_cache=True)
    swot, hfr = shared["swot_features"], shared["hfr_interp_data"]
    del shared
    # Flatten in cycle chunks when the whole record does not fit in commit memory (usegc:
    # 60 passes x 58 cycles blew the 72 GB limit). Each chunk gets its own flatten cache
    # name through cycles_start/cycles_end; shared-step caches stay on the full range.
    cycles = sorted(swot)
    groups = [cycles[i:i + chunk] for i in range(0, len(cycles), chunk)] if chunk else [cycles]

    rec = {k: [] for k in ["X", "Y", "cycle", "pass_idx", "time", "hfr_time", "lat", "lon", "grid_ok", "dt_h"]}
    for g in groups:
        fcfg = replace(cfg, cycles_start=int(g[0]), cycles_end=int(g[-1])) if chunk else cfg
        flat = step_flatten(fcfg, {t: hfr[t] for t in g if t in hfr}, {t: swot[t] for t in g}, cb, True)
        for t in flat:
            pairs = [(h, s) for h, s in _pair_by_time(hfr[t], swot[t], lambda *a: None)
                     if h is not None and s is not None]
            assert len(pairs) == len(flat[t]), f"{rid}/{t}: {len(pairs)} pairs vs {len(flat[t])} entries"
            for j, (entry, (h, s)) in enumerate(zip(flat[t], pairs)):
                df = entry["df"]
                yu, yv = entry["y_u"].reindex(df.index), entry["y_v"].reindex(df.index)
                valid = yu.notna().to_numpy() | yv.notna().to_numpy()
                if not valid.any():
                    continue
                m = int(valid.sum())
                st = np.datetime64(pass_time(s), "s")
                ht = np.datetime64(h["time"].values, "s") if "time" in h.coords else np.datetime64("NaT")
                same_grid = h["lat"].shape == s["lat"].shape and \
                    np.allclose(h["lat"].values, s["lat"].values, equal_nan=True) and \
                    np.allclose(h["lon"].values, s["lon"].values, equal_nan=True)
                rec["X"].append(df.to_numpy(np.float32)[valid])
                rec["Y"].append(np.column_stack([yu.to_numpy(np.float32)[valid], yv.to_numpy(np.float32)[valid]]))
                rec["cycle"].append(np.full(m, int(t), np.int16))
                rec["pass_idx"].append(np.full(m, j, np.int16))
                rec["time"].append(np.full(m, st.astype("int64")))
                rec["hfr_time"].append(np.full(m, ht.astype("int64")))
                rec["lat"].append(s["lat"].values.reshape(-1)[df.index.to_numpy()][valid].astype(np.float32))
                rec["lon"].append(s["lon"].values.reshape(-1)[df.index.to_numpy()][valid].astype(np.float32))
                rec["grid_ok"].append(np.full(m, same_grid))
                rec["dt_h"].append(np.full(m, float((ht - st) / np.timedelta64(1, "h")), np.float32))
        del flat
    del swot, hfr
    arr = {k: np.concatenate(v) for k, v in rec.items()}
    arr["era5_ok"] = arr["time"] <= era5_end.astype("int64")
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(out, **arr)
    cyc = np.unique(arr["cycle"])
    log(f"{rid}: {len(arr['Y']):,} rows, cycles {cyc.min():03d}-{cyc.max():03d} ({len(cyc)}), "
        f"{(arr['cycle'] > LAST_TRAIN_CYCLE).sum():,} rows after cycle {LAST_TRAIN_CYCLE:03d}, "
        f"ERA5 ends {era5_end} -> {out}")
    return out


def load_region(rid: str) -> dict:
    d = np.load(OUT / f"{rid}.npz")
    return {k: d[k] for k in d.files}


# --------------------------------------------------------------------------
# 2. Retrain the champion recipe on a region set (year one) and predict the rest
# --------------------------------------------------------------------------

def test_mask(d: dict) -> np.ndarray:
    return d["cycle"] > LAST_TRAIN_CYCLE


def train(variant: str, seed: int = 42) -> Path:
    vdir = OUT / variant
    pred_path = vdir / f"pred_seed{seed}.npz"
    if pred_path.exists():
        log(f"{variant}: {pred_path.name} exists, skipping")
        return pred_path
    vdir.mkdir(parents=True, exist_ok=True)
    data_path = vdir / "data.npz"
    if not data_path.exists():
        Xtr, Ytr, Ttr = [], [], []
        for rid in VARIANTS[variant]:
            d = load_region(rid)
            m = ~test_mask(d)
            Xtr.append(d["X"][m]); Ytr.append(d["Y"][m]); Ttr.append(d["time"][m])
        Xtr, Ytr, Ttr = (np.concatenate(a) for a in (Xtr, Ytr, Ttr))
        order = np.argsort(Ttr, kind="stable")   # chronological: the recipe validates on its last 10 % of rows
        Xtr, Ytr = Xtr[order], Ytr[order]
        Xte, spans, pos = [], {}, 0
        for rid in REGIONS:                       # every region is scored, in-region or not
            d = load_region(rid)
            m = test_mask(d)
            Xte.append(d["X"][m]); spans[rid] = [pos, pos + int(m.sum())]; pos += int(m.sum())
        Xte = np.concatenate(Xte)
        np.savez(data_path, X_train=Xtr, Y_train=Ytr, X_test=Xte, Y_test=np.zeros((len(Xte), 2), np.float32))
        (vdir / "data_meta.json").write_text(json.dumps({
            "variant": variant, "train_regions": VARIANTS[variant], "train_cycles": [1, LAST_TRAIN_CYCLE],
            "n_train": int(len(Xtr)), "n_test": int(len(Xte)), "test_spans": spans,
            "features": FEATURES, "stencil_k": K, "row_order": "chronological"}, indent=2), encoding="utf-8")
        log(f"{variant}: froze {len(Xtr):,} training rows ({len(VARIANTS[variant])} regions), {len(Xte):,} test rows")
        del Xtr, Ytr, Xte
    cand = RUN / "candidates" / CHAMPION / "candidate.py"
    params = dict(json.loads((RUN / "candidates" / CHAMPION / "params.json").read_text()), seed=seed)
    (vdir / f"params_seed{seed}.json").write_text(json.dumps(params), encoding="utf-8")
    log(f"{variant}: training {CHAMPION} seed {seed} (budget {params['time_budget_s']:.0f} s)...")
    t0 = time.time()
    subprocess.run([PY, "-m", "swotxai.evolve.run_candidate", str(cand), str(data_path), str(pred_path),
                    str(vdir / f"meta_seed{seed}.json"), str(vdir / f"params_seed{seed}.json")],
                   cwd=REPO, timeout=params["time_budget_s"] * 2 + 600, check=False, env=ENV)
    meta = json.loads((vdir / f"meta_seed{seed}.json").read_text())
    if meta.get("status") != "ok":
        raise RuntimeError(f"{variant}: training failed: {meta.get('error')}")
    log(f"{variant}: done in {time.time() - t0:.0f} s (train {meta['train_seconds']:.0f} s) -> {pred_path}")
    return pred_path


# --------------------------------------------------------------------------
# 3. Score
# --------------------------------------------------------------------------

def _variant_preds(rid: str) -> dict[str, np.ndarray]:
    """variant -> list of (n_test_rows_of_region, 2) predictions, one per seed."""
    out = {}
    for v in VARIANTS:
        vdir = OUT / v
        if not (vdir / "data_meta.json").exists():
            continue
        s, e = json.load(open(vdir / "data_meta.json"))["test_spans"][rid]
        preds = [np.load(p)["pred"][s:e] for p in sorted(vdir.glob("pred_seed*.npz"))]
        if preds:
            out[v] = preds
    return out


def score_all(B: int = 1000) -> dict:
    import baselines_eval as be
    be.FEATURES, be.K, be.CENTER = FEATURES, K, K * K // 2
    results = {"champion": f"{RUN.name}/{CHAMPION}", "train_cycles": [1, LAST_TRAIN_CYCLE],
               "test_cycles": [LAST_TRAIN_CYCLE + 1, LAST_CYCLE], "variants": VARIANTS,
               "rows": "clean pairing (same grid, |dt| <= 2 h), finite centre-cell SWOT geostrophy and ERA5, "
                       "inside the ERA5 record", "regions": {}}
    for rid in REGIONS:
        if not (OUT / f"{rid}.npz").exists():
            log(f"{rid}: not frozen, skipped"); continue
        d = load_region(rid)
        tr, te = ~test_mask(d), test_mask(d)
        X, Y = d["X"], d["Y"]
        ok = (te & d["grid_ok"] & (np.abs(d["dt_h"]) <= MAX_DT_H) & d["era5_ok"]
              & np.isfinite(X[:, col("ugos_filtered")]) & np.isfinite(X[:, col("vgos_filtered")])
              & np.isfinite(X[:, col("era5_u")]) & np.isfinite(X[:, col("era5_v")]))
        if ok.sum() < 200:
            log(f"{rid}: only {int(ok.sum())} scorable rows, skipped"); continue
        Xtr, Ytr = X[tr], Y[tr]
        G = X[:, [col("ugos_filtered"), col("vgos_filtered")]]
        W = X[:, [col("era5_u"), col("era5_v")]]
        models = {}
        models["train_mean"] = np.tile(np.nanmean(Ytr, 0), (len(Y), 1))
        models["cell_mean"] = be.cell_climatology(d["lat"][tr], d["lon"][tr], Ytr, d["lat"], d["lon"])
        models["swot_geos"] = G
        models["geos_calibrated"] = np.column_stack(
            [be.fit_linear(G[tr][:, [c]], Ytr[:, [c]])[0](G[:, [c]])[:, 0] for c in range(2)])
        models["geos_cal+wind_lsq"] = be.fit_linear(np.column_stack([G[tr], W[tr]]), Ytr)[0](np.column_stack([G, W]))
        duacs = be.sample_duacs(rid, d["time"], d["lat"], d["lon"])
        if duacs is not None and np.isfinite(duacs[ok]).all(1).sum() >= 200 and np.isfinite(duacs[tr]).all(1).sum() >= 200:
            models["duacs_calibrated"] = np.column_stack(
                [be.fit_linear(duacs[tr][:, [c]], Ytr[:, [c]])[0](duacs[:, [c]])[:, 0] for c in range(2)])
            models["duacs_cal+wind_lsq"] = be.fit_linear(np.column_stack([duacs[tr], W[tr]]), Ytr)[0](
                np.column_stack([duacs, W]))
        preds = _variant_preds(rid)
        te_idx = np.flatnonzero(te)
        per_seed = {}
        for v, ps in preds.items():
            full = np.full((len(Y), 2), np.nan, np.float32); full[te_idx] = np.mean(ps, axis=0)
            models[f"champion_{v}"] = full
            if len(ps) > 1:   # R2 of each single seed (no bootstrap): the run-to-run spread
                sst = np.nansum((Y[ok] - np.nanmean(Y[ok], 0)) ** 2, 0)
                r2s = []
                for p1 in ps:
                    f1 = np.full((len(Y), 2), np.nan, np.float32); f1[te_idx] = p1
                    r2s.append((1 - np.nansum((Y[ok] - f1[ok]) ** 2, 0) / sst).round(4).tolist())
                per_seed[f"champion_{v}"] = r2s
        blocks = d["cycle"].astype(np.int64) * 1000 + d["pass_idx"]
        train_end = np.datetime64(int(d["time"][tr].max()), "s")
        reg = {"n_rows": int(ok.sum()), "n_passes": int(len(np.unique(blocks[ok]))),
               "train_end": str(train_end), "continent": CONT_OF[rid],
               "first_test_day": str(np.datetime64(int(d["time"][ok].min()), "s")),
               "last_test_day": str(np.datetime64(int(d["time"][ok].max()), "s")),
               "overall": {}, "per_cycle": {}, "per_seed_r2": per_seed}
        ref = models["geos_calibrated"]
        for name, P in models.items():
            m = {}
            for c, comp in enumerate("uv"):
                s = be.score(Y[ok, c], P[ok, c], blocks[ok], B=B)
                rmse_ref = np.sqrt(np.nanmean((Y[ok, c] - ref[ok, c]) ** 2))
                s["pct_rmse_vs_geos_cal"] = float(100 * (s["rmse"] / rmse_ref - 1))
                m[comp] = s
            reg["overall"][name] = m
        # per-cycle skill (no bootstrap): RMSE over both components, R2 per component
        for cyc in np.unique(d["cycle"][ok]):
            mc = ok & (d["cycle"] == cyc)
            entry = {"n": int(mc.sum()), "date": str(np.datetime64(int(np.median(d["time"][mc])), "s").astype("datetime64[D]")), "models": {}}
            for name, P in models.items():
                e = Y[mc] - P[mc]
                ssq = np.nansum(e ** 2, 0); sst = np.nansum((Y[mc] - np.nanmean(Y[mc], 0)) ** 2, 0)
                entry["models"][name] = {"rmse": float(np.sqrt(np.nanmean(e ** 2))),
                                         "r2_u": float(1 - ssq[0] / sst[0]), "r2_v": float(1 - ssq[1] / sst[1])}
            reg["per_cycle"][f"{cyc:03d}"] = entry
        # per-pass sums of squared error: pooled RMSE over any window is sqrt(sum sse / sum n),
        # and a bootstrap over passes needs nothing else
        bl = blocks[ok]; ub, inv = np.unique(bl, return_inverse=True)
        pp = {"cycle": (ub // 1000).astype(int).tolist(), "models": {}}
        for name, P in models.items():
            e2 = (Y[ok] - P[ok]) ** 2; fin = np.isfinite(e2)
            sse = np.zeros((len(ub), 2)); n = np.zeros((len(ub), 2))
            for c in range(2):
                np.add.at(sse[:, c], inv[fin[:, c]], e2[fin[:, c], c]); np.add.at(n[:, c], inv, fin[:, c])
            pp["models"][name] = {"sse": np.round(sse, 6).tolist(), "n": n.astype(int).tolist()}
        reg["per_pass"] = pp
        results["regions"][rid] = reg
        ch = {k: v for k, v in reg["overall"].items() if k.startswith("champion_")}
        log(f"{rid}: {reg['n_rows']:,} rows, {reg['n_passes']} passes; geos_cal R2 "
            f"{reg['overall']['geos_calibrated']['u']['r2']:.2f}/{reg['overall']['geos_calibrated']['v']['r2']:.2f}; "
            + "; ".join(f"{k[9:]} {v['u']['r2']:.2f}/{v['v']['r2']:.2f}" for k, v in ch.items()))
    (OUT / "results.json").write_text(json.dumps(results, indent=1), encoding="utf-8")
    write_md(results)
    return results


def write_md(res: dict) -> None:
    L = [f"# Out-of-time evaluation: train cycles 001-{LAST_TRAIN_CYCLE:03d}, test {LAST_TRAIN_CYCLE + 1:03d}-{LAST_CYCLE:03d}",
         "", f"Champion recipe `{res['champion']}` retrained on year one per variant; baselines fitted on the same "
         "year-one rows of each region. Rows: " + res["rows"] + ". R² = Nash-Sutcliffe on the held-out rows "
         "[95% block-bootstrap CI over (cycle, pass)]; %ΔRMSE against calibrated SWOT geostrophy fitted on year one.", ""]
    for rid, reg in res["regions"].items():
        L.append(f"## {rid} ({reg['continent']}): n = {reg['n_rows']:,} rows, {reg['n_passes']} passes, "
                 f"{reg['first_test_day'][:10]} to {reg['last_test_day'][:10]}, train end {reg['train_end'][:10]}")
        L.append("")
        L.append("| model | R²_u [95% CI] | R²_v [95% CI] | RMSE_u | RMSE_v | %ΔRMSE_u | %ΔRMSE_v | N_eff u/v |")
        L.append("|---|---|---|---|---|---|---|---|")
        for name, m in reg["overall"].items():
            u, v = m["u"], m["v"]
            L.append(f"| {name} | {u['r2']:.3f} [{u['r2_ci'][0]:.3f}, {u['r2_ci'][1]:.3f}] | "
                     f"{v['r2']:.3f} [{v['r2_ci'][0]:.3f}, {v['r2_ci'][1]:.3f}] | {u['rmse']:.3f} | {v['rmse']:.3f} | "
                     f"{u['pct_rmse_vs_geos_cal']:+.1f} | {v['pct_rmse_vs_geos_cal']:+.1f} | "
                     f"{u['n_eff']:.0f}/{v['n_eff']:.0f} |")
        for name, r2s in reg.get("per_seed_r2", {}).items():
            L.append(f"\n{name}: single-seed R²_u/R²_v = " + ", ".join(f"{a:.3f}/{b:.3f}" for a, b in r2s)
                     + f" (table row is the {len(r2s)}-seed mean prediction)")
        L.append("")
    (OUT / "results.md").write_text("\n".join(L), encoding="utf-8")
    log(f"wrote {OUT / 'results.md'}")


# --------------------------------------------------------------------------
# 4. Figures
# --------------------------------------------------------------------------

# Okabe-Ito (colour-blind safe), fixed order by role
C_CHAMP, C_ALL, C_AWAY, C_DUACS, C_GEOS, C_CELL = "#0072B2", "#009E73", "#D55E00", "#E69F00", "#999999", "#CC79A7"


def _mpl():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
                         "mathtext.fontset": "cm", "axes.formatter.use_mathtext": True, "font.size": 9,
                         "axes.spines.top": False, "axes.spines.right": False})
    return plt


def _save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=300, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    log(f"wrote {FIG / name}.png/.svg")


def _r2(m: dict) -> float:
    return 0.5 * (m["u"]["r2"] + m["v"]["r2"])


def own(rid: str) -> str:
    return f"champion_{CONT_OF[rid]}_y1"


def away(rid: str) -> str:
    return f"champion_not_{CONT_OF[rid]}_y1"


def _lab(rid: str) -> str:
    return rid.replace("_", r"$\_$")  # cmr10 has no underscore glyph


def fig_time(res: dict) -> None:
    """Per region: % RMSE change vs calibrated geostrophy by cycle, training year shaded."""
    plt = _mpl()
    import matplotlib.dates as mdates
    regs = [r for r in REGIONS if r in res["regions"]]
    ncol = 4; nrow = -(-len(regs) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.3 * nrow), sharex=True, sharey=True)
    for ax, rid in zip(axes.flat, regs):
        reg = res["regions"][rid]
        cyc = sorted(reg["per_cycle"])
        dates = np.array([reg["per_cycle"][c]["date"] for c in cyc], dtype="datetime64[s]")
        xs = dates.astype("datetime64[D]").astype(object)
        ci = np.array([int(c) for c in cyc]); gap = np.r_[False, np.diff(ci) > 1]  # first point after a hole
        ref = np.array([reg["per_cycle"][c]["models"]["geos_calibrated"]["rmse"] for c in cyc])
        lines = [(own(rid), C_CHAMP, "champion, own continent", 1.8), ("champion_all_y1", C_ALL, "champion, all regions", 1.2),
                 (away(rid), C_AWAY, "champion, other continents", 1.2),
                 ("duacs_cal+wind_lsq", C_DUACS, "DUACS calibrated + wind", 1.0), ("cell_mean", C_CELL, "cell climatology", 1.0)]
        for name, color, label, lw in lines:
            if name not in reg["overall"]:
                continue
            y = 100 * (np.array([reg["per_cycle"][c]["models"][name]["rmse"] for c in cyc]) / ref - 1)
            y = np.where(gap, np.nan, y)  # a NaN after each gap breaks the line there
            ax.plot(xs, y, color=color, lw=lw, label=label, marker="o", ms=2)
            ax.plot(xs[gap], (100 * (np.array([reg["per_cycle"][c]["models"][name]["rmse"] for c in cyc]) / ref - 1))[gap],
                    color=color, ls="none", marker="o", ms=2)
        ax.axhline(0, color=C_GEOS, lw=0.8, ls="--")
        t_end = np.datetime64(reg["train_end"][:10])  # datetime64 -> datetime.date for the axis calls
        day = lambda t: t.astype("datetime64[D]").astype(object)  # noqa: E731
        ax.axvspan(day(t_end - np.timedelta64(365, "D")), day(t_end), color="0.92", zorder=0)
        ax.set_xlim(day(t_end - np.timedelta64(380, "D")), day(dates.max() + np.timedelta64(30, "D")))
        ax.set_title(_lab(rid), loc="left", fontsize=9)
        ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes.flat[len(regs):]:
        ax.axis("off")
    axes.flat[0].set_ylim(-60, 60)
    for ax in axes[:, 0]:
        ax.set_ylabel("RMSE change vs\ncalibrated geostrophy (%)")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle(f"Skill through time: models trained on cycles 001-{LAST_TRAIN_CYCLE:03d} (shaded), "
                 f"scored per cycle on {LAST_TRAIN_CYCLE + 1:03d}-{LAST_CYCLE:03d}; below zero is better than calibrated SWOT geostrophy",
                 fontsize=10, y=1.01)
    fig.tight_layout()
    _save(fig, "skill_through_time")


def _pooled_pct(sse, n, sse_ref, n_ref, idx, grp, ngrp):
    """% RMSE change of a model against the reference, pooled over the passes `idx`, per group."""
    def rmse(S, N):
        a = np.zeros(ngrp); b = np.zeros(ngrp)
        np.add.at(a, grp[idx], S[idx].sum(1)); np.add.at(b, grp[idx], N[idx].sum(1))
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.sqrt(a / b)
    with np.errstate(invalid="ignore", divide="ignore"):
        return 100 * (rmse(sse, n) / rmse(sse_ref, n_ref) - 1)


def fig_time_quarterly(res: dict, B: int = 500, seed: int = 0) -> None:
    """Per region: % RMSE change vs calibrated geostrophy pooled by quarter with a 95% bootstrap
    interval over passes, the per-cycle values as faint dots, and a row-weighted linear trend
    (slope in % per year with its bootstrap interval) for the own-continent champion, the
    all-regions champion and DUACS + wind."""
    plt = _mpl()
    import matplotlib.dates as mdates
    rng = np.random.default_rng(seed)
    regs = [r for r in REGIONS if r in res["regions"] and "per_pass" in res["regions"][r]]
    ncol = 4; nrow = -(-len(regs) // ncol)
    fig, axes = plt.subplots(nrow, ncol, figsize=(12, 2.5 * nrow), sharex=True, sharey=True)
    for ax, rid in zip(axes.flat, regs):
        reg = res["regions"][rid]; pp = reg["per_pass"]
        cyc_of_pass = np.array(pp["cycle"]); npass = len(cyc_of_pass)
        cdate = {int(c): np.datetime64(v["date"]) for c, v in reg["per_cycle"].items()}
        pdate = np.array([cdate[c] for c in cyc_of_pass], dtype="datetime64[D]")
        yr = pdate.astype("datetime64[Y]").astype(int) + 1970; mo = pdate.astype("datetime64[M]").astype(int) % 12
        qid = (yr - 2024) * 4 + mo // 3; uq = np.unique(qid); qgrp = np.searchsorted(uq, qid)
        qmid = np.array([np.datetime64(f"{2024 + q // 4}-{3 * (q % 4) + 2:02d}-15") for q in uq]).astype(object)
        ucy = np.unique(cyc_of_pass); cgrp = np.searchsorted(ucy, cyc_of_pass)
        t_end = np.datetime64(reg["train_end"][:10]); day = lambda t: t.astype("datetime64[D]").astype(object)  # noqa: E731
        tcy = np.array([(cdate[c] - t_end) / np.timedelta64(365, "D") for c in ucy], float)  # years after training
        ref_s = np.array(pp["models"]["geos_calibrated"]["sse"]); ref_n = np.array(pp["models"]["geos_calibrated"]["n"])
        draws = [rng.integers(0, npass, npass) for _ in range(B)]; full = np.arange(npass)
        lines = [(own(rid), C_CHAMP, "champion, own continent", True), ("champion_all_y1", C_ALL, "champion, all regions", True),
                 (away(rid), C_AWAY, "champion, other continents", False),
                 ("duacs_cal+wind_lsq", C_DUACS, "DUACS calibrated + wind", True), ("cell_mean", C_CELL, "cell climatology", False)]
        txt = []
        for name, color, label, trend in lines:
            if name not in pp["models"]:
                continue
            S = np.array(pp["models"][name]["sse"]); N = np.array(pp["models"][name]["n"])
            q0 = _pooled_pct(S, N, ref_s, ref_n, full, qgrp, len(uq))
            qb = np.array([_pooled_pct(S, N, ref_s, ref_n, d, qgrp, len(uq)) for d in draws])
            lo, hi = np.nanpercentile(qb, [2.5, 97.5], axis=0)
            c0 = _pooled_pct(S, N, ref_s, ref_n, full, cgrp, len(ucy))
            ax.plot(np.array([day(cdate[c]) for c in ucy]), c0, ls="none", marker="o", ms=1.8, color=color, alpha=0.3)
            ax.errorbar(qmid, q0, yerr=[q0 - lo, hi - q0], color=color, lw=1.4, marker="o", ms=3.5, capsize=2, label=label)
            if trend:
                w = N.sum(1); wc = np.zeros(len(ucy)); np.add.at(wc, cgrp, w)
                def slope(y, wts):
                    m = np.isfinite(y)
                    if m.sum() < 3:
                        return np.nan
                    return np.polyfit(tcy[m], y[m], 1, w=np.sqrt(wts[m]))[0]
                s0 = slope(c0, wc)
                sb = []
                for d in draws:
                    wd = np.zeros(len(ucy)); np.add.at(wd, cgrp[d], w[d])
                    sb.append(slope(_pooled_pct(S, N, ref_s, ref_n, d, cgrp, len(ucy)), wd))
                slo, shi = np.nanpercentile(sb, [2.5, 97.5])
                m = np.isfinite(c0)
                ax.plot([day(cdate[ucy[m][0]]), day(cdate[ucy[m][-1]])],
                        np.polyval(np.polyfit(tcy[m], c0[m], 1, w=np.sqrt(wc[m])), [tcy[m][0], tcy[m][-1]]),
                        color=color, lw=0.9, ls="--", alpha=0.9)
                txt.append((color, f"{s0:+.1f} [{slo:+.1f}, {shi:+.1f}] %/yr"))
        for k, (color, t) in enumerate(txt):
            ax.text(0.02, 0.04 + 0.09 * k, t, transform=ax.transAxes, fontsize=6.5, color=color, ha="left", va="bottom",
                    bbox=dict(fc="white", ec="none", alpha=0.75, pad=1), zorder=6)
        ax.axhline(0, color=C_GEOS, lw=0.8, ls="--")
        ax.axvspan(day(t_end - np.timedelta64(365, "D")), day(t_end), color="0.92", zorder=0)
        ax.set_xlim(day(t_end - np.timedelta64(380, "D")), day(np.datetime64("2026-11-15")))
        ax.set_title(_lab(rid), loc="left", fontsize=9)
        ax.xaxis.set_major_locator(mdates.YearLocator()); ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    for ax in axes.flat[len(regs):]:
        ax.axis("off")
    axes.flat[0].set_ylim(-60, 60)
    for ax in axes[:, 0]:
        ax.set_ylabel("RMSE change vs\ncalibrated geostrophy (%)")
    h, l = axes.flat[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.suptitle("Skill through time, pooled by quarter (95% bootstrap interval over passes; dots = single cycles); "
                 "dashed = row-weighted linear trend, slope printed as % per year [95% interval]", fontsize=10, y=1.01)
    fig.tight_layout()
    _save(fig, "skill_through_time_quarterly")


def fig_loro(res: dict) -> None:
    """US regions: R² per component (u left, v right) for the in-training model, the all-regions model,
    the leave-one-region-out model, the other-continents model and DUACS + wind; dots = single seeds."""
    plt = _mpl()
    regs = [r for r in CONTINENTS["us"] if r in res["regions"] and f"champion_us_wo_{r}_y1" in res["regions"][r]["overall"]]
    bars = [("champion_us_y1", C_CHAMP, "US (own continent)"), ("champion_all_y1", C_ALL, "all 19 regions"),
            ("loro", "#56B4E9", "US without this region"), ("champion_not_us_y1", C_AWAY, "other continents"),
            ("duacs_cal+wind_lsq", C_DUACS, "DUACS calibrated + wind")]
    w = 0.16
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), sharey=True)
    for c, (ax, comp) in enumerate(zip(axes, "uv")):
        for k, (name, color, label) in enumerate(bars):
            x = np.arange(len(regs)) + (k - 2) * w
            vals, seeds = [], []
            for rid in regs:
                reg = res["regions"][rid]; nm = f"champion_us_wo_{rid}_y1" if name == "loro" else name
                vals.append(reg["overall"][nm][comp]["r2"] if nm in reg["overall"] else np.nan)
                seeds.append([r2[c] for r2 in reg.get("per_seed_r2", {}).get(nm, [])])
            vals = np.array(vals)
            ax.bar(x, np.clip(vals, -1, None), w, color=color, label=label, zorder=2)
            for xi, v, sd in zip(x, vals, seeds):
                if v < -1:
                    ax.text(xi, -0.98, f"{v:.1f}", ha="center", va="bottom", fontsize=5.5, rotation=90, color="white", zorder=4)
                if sd:
                    ax.plot([xi] * len(sd), np.clip(sd, -1, None), ls="none", marker="o", ms=2.5, mfc="white", mec="k", mew=0.5, zorder=3)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(range(len(regs))); ax.set_xticklabels([_lab(r) for r in regs])
        ax.set_title(f"R$^2$, {comp} component", loc="left", fontsize=9)
        ax.set_ylim(-1, 0.6); ax.grid(axis="y", color="0.9", zorder=0)
    axes[0].set_ylabel("R$^2$ on cycles 017-058 (clipped at $-$1)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=5, frameon=False, bbox_to_anchor=(0.5, -0.03), fontsize=8)
    fig.suptitle("Leave-one-region-out within the US: skill of the champion recipe by training set "
                 "(bar = seed-averaged prediction scored once, so it sits above the dots where seeds disagree; dots = single seeds)", fontsize=10, y=1.0)
    fig.tight_layout()
    _save(fig, "loro_us")


def fig_map(res: dict) -> None:
    """World map: one marker per region, colour = R² (mean of u, v) of the champion trained
    on the other continents (left) and on the region's own continent (right)."""
    plt = _mpl()
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    from swotxai.presets import REGION_META
    fig = plt.figure(figsize=(12, 6.2))
    panels = [("trained on the other continents (out-of-region and out-of-time)", away),
              ("trained on the region's own continent (out-of-time only)", own)]
    for i, (title, pick) in enumerate(panels):
        ax = fig.add_subplot(2, 1, i + 1, projection=ccrs.PlateCarree(central_longitude=0))
        ax.set_extent([-170, 170, -50, 72], crs=ccrs.PlateCarree())
        ax.add_feature(cfeature.LAND, facecolor="0.93", edgecolor="none")
        ax.coastlines(lw=0.4, color="0.5")
        ax.set_title(title, loc="left", fontsize=9)
        for rid, reg in res["regions"].items():
            name = pick(rid)
            if name not in reg["overall"]:
                continue
            r2 = _r2(reg["overall"][name])
            _, _, sw, ne = REGION_META[rid]
            lon, lat = 0.5 * (sw[0] + ne[0]), 0.5 * (sw[1] + ne[1])
            sc = ax.scatter(lon, lat, c=[max(r2, -0.05)], cmap="Blues", vmin=-0.05, vmax=0.6, s=120,
                            edgecolor="k", lw=0.5, transform=ccrs.PlateCarree(), zorder=5)
        cb = fig.colorbar(sc, ax=ax, shrink=0.8, pad=0.02)
        cb.set_label("R$^2$ (mean of u, v), cycles 017-058")
    fig.tight_layout()
    _save(fig, "transfer_map")


def fig_grid(res: dict) -> None:
    """Heat map: test region (rows, by continent) x training set (columns), R² mean of u, v.
    In-continent cells (out-of-time only) are outlined."""
    plt = _mpl()
    regs = [r for r in REGIONS if r in res["regions"]]
    base = [v for v in VARIANTS if v not in LORO]
    cols = base + ["loro", "geos_cal+wind_lsq", "duacs_cal+wind_lsq", "cell_mean"]
    labels = [c.replace("_y1", "").replace("not_", "not ") for c in base] + ["US w/o own region", "geos cal+wind", "DUACS cal+wind", "cell clim."]
    M = np.full((len(regs), len(cols)), np.nan)
    for i, rid in enumerate(regs):
        for j, c in enumerate(cols):
            name = f"champion_{c}" if c in VARIANTS else f"champion_us_wo_{rid}_y1" if c == "loro" else c
            if name in res["regions"][rid]["overall"]:
                M[i, j] = _r2(res["regions"][rid]["overall"][name])
    fig, ax = plt.subplots(figsize=(0.75 * len(cols) + 2.5, 0.32 * len(regs) + 1.5))
    im = ax.imshow(np.clip(M, 0, None), cmap="Blues", vmin=0, vmax=0.6, aspect="auto")
    for i in range(len(regs)):
        for j in range(len(cols)):
            if np.isfinite(M[i, j]):
                txt = f"{M[i, j]:.2f}" if M[i, j] > -1 else "$<$ -1"
                ax.text(j, i, txt, ha="center", va="center", fontsize=7, color="white" if M[i, j] > 0.4 else "black")
            else:
                ax.text(j, i, "n/a", ha="center", va="center", fontsize=7, color="0.6")
            if cols[j] in ("all_y1", f"{CONT_OF[regs[i]]}_y1"):
                ax.add_patch(plt.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, ec="k", lw=1.2, zorder=3))
    ax.set_xticks(range(len(cols))); ax.set_xticklabels(labels, rotation=40, ha="right")
    ax.set_yticks(range(len(regs))); ax.set_yticklabels([f"{_lab(r)}  ({CONT_OF[r]})" for r in regs])
    ax.set_xlabel("training set (champion recipe, cycles 001-016) or baseline (fitted on the region's year one)")
    ax.set_title(f"R$^2$ on cycles {LAST_TRAIN_CYCLE + 1:03d}-{LAST_CYCLE:03d}; outlined = trained on the region's own continent",
                 loc="left", fontsize=9)
    cb = fig.colorbar(im, ax=ax, shrink=0.6, pad=0.02); cb.set_label("R$^2$ (mean of u, v); negative shown as 0")
    fig.tight_layout()
    _save(fig, "transfer_grid")


def figures() -> None:
    res = json.loads((OUT / "results.json").read_text(encoding="utf-8"))
    fig_time(res); fig_time_quarterly(res); fig_grid(res); fig_loro(res); fig_map(res)


# --------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--freeze", help="region list or all (each region in its own subprocess)")
    ap.add_argument("--train", help="variant list or all")
    ap.add_argument("--seeds", default="42")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--bootstrap", type=int, default=1000)
    ap.add_argument("--figures", action="store_true")
    ap.add_argument("--chunk", type=int, help="freeze: flatten this many cycles at a time (memory relief for big regions)")
    ap.add_argument("--_freeze_one", help=argparse.SUPPRESS)
    a = ap.parse_args()
    if a._freeze_one:
        freeze(a._freeze_one, a.chunk); return
    if a.freeze:
        for rid in (REGIONS if a.freeze == "all" else a.freeze.split(",")):
            if (OUT / f"{rid}.npz").exists():
                log(f"{rid}: frozen already"); continue
            subprocess.run([PY, __file__, "--_freeze_one", rid] + (["--chunk", str(a.chunk)] if a.chunk else []),
                           cwd=REPO, check=True, env=ENV)
    if a.train:
        for v in (list(VARIANTS) if a.train == "all" else a.train.split(",")):
            for seed in (int(s) for s in a.seeds.split(",")):
                train(v, seed)
    if a.score:
        score_all(a.bootstrap)
    if a.figures:
        figures()
    log("done.")


if __name__ == "__main__":
    main()
