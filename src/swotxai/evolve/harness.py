"""Evaluation harness: frozen data arrays and trusted fitness scoring.

Candidates never compute their own scores — they emit predictions on X_test
and this module (running in the parent process) scores them. The test split is
temporally held out (the last 20% of cycles, via concat_for_ann's held_out
mode) rather than the legacy random 80/20 split, so a candidate cannot win by
memorizing rows it trained on.

Fitness = (r2_u + r2_v) / 2, matching swotxai.experiments.scored_experiments.
"""
from __future__ import annotations

import gc
import json
from pathlib import Path
from typing import Callable

import numpy as np

from swotxai.config import SWOTConfig

ProgressCb = Callable[[str, float, str], None]


def prepare_data(config: SWOTConfig, run_dir: Path,
                 progress_cb: ProgressCb | None = None,
                 train_fraction: float = 0.8) -> Path:
    """Build (or reuse) the frozen data.npz for an evolve run.

    Runs the shared pipeline steps + flatten (both cached by the existing
    pipeline machinery — all local reads, never S3), concatenates via the ANN
    dataset builder, and freezes float32 arrays so every candidate in the run
    trains and is scored on byte-identical data.
    """
    data_path = run_dir / "data.npz"
    meta_path = run_dir / "data_meta.json"
    if data_path.exists() and meta_path.exists():
        return data_path

    from swotxai.pipeline.orchestrator import run_shared_steps
    from swotxai.pipeline.steps_ml import step_flatten
    from swotxai.models.ann.dataset import concat_for_ann

    cb = progress_cb or (lambda step, frac, msg: print(f"[{step}] {msg}"))
    shared = run_shared_steps(config, progress_cb=cb, use_cache=True)
    flattened = step_flatten(config, shared["hfr_interp_data"], shared["swot_features"], cb, True)
    shared.clear()

    X_train, Y_train = concat_for_ann(flattened, training_percentage=train_fraction)
    X_test, Y_test = concat_for_ann(flattened, training_percentage=train_fraction, held_out=True)
    del flattened
    gc.collect()

    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(data_path,
             X_train=X_train.astype(np.float32), Y_train=Y_train.astype(np.float32),
             X_test=X_test.astype(np.float32), Y_test=Y_test.astype(np.float32))
    meta_path.write_text(json.dumps({
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "n_inputs": int(X_train.shape[1]),
        "features": list(config.features), "stencil_k": config.stencil_k,
        "region": config.region, "mission": config.mission,
        "cycles_start": config.cycles_start, "cycles_end": config.cycles_end,
        "train_fraction": train_fraction,
    }, indent=2), encoding="utf-8")
    return data_path


def prepare_pooled_data(base_config: SWOTConfig, mission: str, run_dir: Path,
                        progress_cb: ProgressCb | None = None,
                        regions: list[str] | None = None,
                        train_fraction: float = 0.8) -> Path:
    """Build a frozen multi-region data.npz: all trainable regions of a mission.

    Mirrors swotxai.multiregion's pooling, but concatenates per region *before*
    stacking: each region gets its own temporal 80/20 split (first/last cycles),
    so every region is represented in both the train and the held-out test
    portions with its chronological order preserved. Per-region test slices are
    recorded in data_meta.json so candidates can be scored per region too.
    All reads are local (per-region pipeline caches / local mirrors).
    """
    data_path = run_dir / "data.npz"
    meta_path = run_dir / "data_meta.json"
    if data_path.exists() and meta_path.exists():
        return data_path

    from dataclasses import replace

    from swotxai.models.ann.dataset import concat_for_ann
    from swotxai.pipeline.orchestrator import _cleanup_shared_cache, run_shared_steps
    from swotxai.pipeline.steps_ml import step_flatten
    from swotxai.presets import MISSION_REGIONS, config_overrides

    cb = progress_cb or (lambda step, frac, msg: print(f"[{step}] {msg}"))
    regions = regions or MISSION_REGIONS[mission]

    parts: dict[str, tuple] = {}
    n_cols = None
    for i, rid in enumerate(regions):
        rcfg = replace(base_config, **config_overrides(rid, mission))
        rcb = (lambda step, frac, msg, _r=rid, _i=i:
               cb(step, frac, f"[{_r} {_i + 1}/{len(regions)}] {msg}"))
        shared = run_shared_steps(rcfg, progress_cb=rcb, use_cache=True)
        try:
            flat = step_flatten(rcfg, shared["hfr_interp_data"],
                                shared["swot_features"], rcb, True)
        finally:
            _cleanup_shared_cache(rcfg)
            shared.clear()
        try:
            X_tr, Y_tr = concat_for_ann(flat, training_percentage=train_fraction)
            X_te, Y_te = concat_for_ann(flat, training_percentage=train_fraction, held_out=True)
        except ValueError:
            cb("flatten", 1.0, f"WARNING: no valid rows for {rid} — pooling without it")
            del flat
            gc.collect()
            continue
        del flat
        gc.collect()

        if n_cols is None:
            n_cols = X_tr.shape[1]
        elif X_tr.shape[1] != n_cols:
            raise RuntimeError(
                f"Feature-column mismatch: {rid} has {X_tr.shape[1]} cols, "
                f"earlier regions {n_cols}. All regions must expose the same "
                "features (check ERA5/GOES availability)."
            )
        parts[rid] = tuple(a.astype(np.float32) for a in (X_tr, Y_tr, X_te, Y_te))

    if not parts:
        raise RuntimeError("No region produced any training rows.")

    # Time-interleave regions in chunks: chunk i of every region covers the
    # same fraction of the mission window, so "later rows = later in time"
    # holds ACROSS regions — candidates' temporal-tail validation and recency
    # weighting stay meaningful on pooled data. Per-region test rows become
    # lists of [start, end) spans instead of one contiguous block.
    n_chunks = 20
    train_pieces: list[tuple] = []
    test_pieces: list[tuple] = []
    region_test_slices: dict[str, list[list[int]]] = {rid: [] for rid in parts}
    pos = 0
    for i in range(n_chunks):
        for rid, (X_tr, Y_tr, X_te, Y_te) in parts.items():
            s, e = len(X_tr) * i // n_chunks, len(X_tr) * (i + 1) // n_chunks
            if e > s:
                train_pieces.append((X_tr[s:e], Y_tr[s:e]))
            s, e = len(X_te) * i // n_chunks, len(X_te) * (i + 1) // n_chunks
            if e > s:
                test_pieces.append((X_te[s:e], Y_te[s:e]))
                region_test_slices[rid].append([pos, pos + (e - s)])
                pos += e - s

    X_train = np.concatenate([p[0] for p in train_pieces])
    Y_train = np.concatenate([p[1] for p in train_pieces])
    X_test = np.concatenate([p[0] for p in test_pieces])
    Y_test = np.concatenate([p[1] for p in test_pieces])
    parts.clear()
    train_pieces.clear()
    test_pieces.clear()
    gc.collect()

    run_dir.mkdir(parents=True, exist_ok=True)
    np.savez(data_path, X_train=X_train, Y_train=Y_train, X_test=X_test, Y_test=Y_test)
    meta_path.write_text(json.dumps({
        "pooled": True, "mission": mission, "regions": list(region_test_slices),
        "row_order": "time-interleaved across regions in 20 chunks "
                     "(later rows = later in the mission window, all regions mixed)",
        "region_test_slices": region_test_slices,
        "n_train": int(len(X_train)), "n_test": int(len(X_test)),
        "n_inputs": int(X_train.shape[1]),
        "features": list(base_config.features), "stencil_k": base_config.stencil_k,
        "train_fraction": train_fraction,
    }, indent=2), encoding="utf-8")
    return data_path


def score_predictions(pred: np.ndarray, Y_test: np.ndarray) -> dict:
    """Per-component RMSE/R² on the valid (finite-target) test rows."""
    from sklearn.metrics import mean_squared_error, r2_score

    metrics: dict[str, float] = {}
    for i, comp in enumerate(("u", "v")):
        valid = np.isfinite(Y_test[:, i])
        y, p = Y_test[valid, i], pred[valid, i]
        metrics[f"rmse_{comp}"] = float(np.sqrt(mean_squared_error(y, p)))
        metrics[f"r2_{comp}"] = float(r2_score(y, p))
    metrics["fitness"] = (metrics["r2_u"] + metrics["r2_v"]) / 2.0
    return metrics


def diagnostics(pred: np.ndarray, Y_test: np.ndarray, X_test: np.ndarray) -> dict:
    """Rich error breakdown fed back to the mutator LLM (all held-out).

    The single fitness scalar tells the proposer *whether* an idea worked;
    these bins tell it *where* the remaining error lives: over the forecast
    horizon (early vs late test), by flow regime (true current speed), and by
    observation quality (stencil validity fraction = distance to swath edge).
    """
    d: dict[str, float] = {}

    def _r2_pair(sl_pred, sl_y):
        out = {}
        for i, comp in enumerate(("u", "v")):
            valid = np.isfinite(sl_y[:, i])
            if valid.sum() < 100:
                continue
            y, p = sl_y[valid, i], sl_pred[valid, i]
            ss_res = float(np.sum((y - p) ** 2))
            ss_tot = float(np.sum((y - y.mean()) ** 2))
            if ss_tot > 0:
                out[comp] = round(1.0 - ss_res / ss_tot, 4)
        return out

    # Forecast horizon: test rows are in cycle (time) order.
    n = len(Y_test)
    half = n // 2
    for name, sl in (("early_test", slice(0, half)), ("late_test", slice(half, n))):
        for comp, val in _r2_pair(pred[sl], Y_test[sl]).items():
            d[f"r2_{comp}_{name}"] = val

    # Flow regime: bins of true current speed (rows where both components valid).
    both = np.isfinite(Y_test).all(axis=1)
    if both.sum() >= 300:
        spd = np.hypot(Y_test[both, 0], Y_test[both, 1])
        pe, ye = pred[both], Y_test[both]
        q1, q2 = np.quantile(spd, [1 / 3, 2 / 3])
        for label, msk in (("slow", spd <= q1), ("mid", (spd > q1) & (spd <= q2)),
                           ("fast", spd > q2)):
            if msk.sum() >= 100:
                d[f"rmse_speed_{label}"] = round(
                    float(np.sqrt(np.mean((pe[msk] - ye[msk]) ** 2))), 4)

    # Observation quality: stencil validity fraction (1.0 = full 3x3 coverage,
    # lower = nearer a swath edge / more NaN padding).
    vf = np.isfinite(X_test).mean(axis=1)
    for label, msk in (("full_stencil", vf >= 0.999), ("edge_stencil", vf < 0.999)):
        if msk.sum() >= 300:
            for comp, val in _r2_pair(pred[msk], Y_test[msk]).items():
                d[f"r2_{comp}_{label}"] = val

    return d
