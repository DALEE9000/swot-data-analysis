"""Pooled multi-region training: one model over every colocated region.

For each region the shared data steps + flatten run with that region's preset
paths (step outputs cache per region as usual). The per-region flattened
dicts are then merged ROUND-ROBIN — concat_flattened's temporal 80/20 split
takes the first 80% of keys, so interleaving keeps every region represented
in both the train and eval portions while preserving each region's
chronological order. One model trains on the pooled matrix; evaluation
reports pooled metrics plus a per-region breakdown (each computed with the
same convention step_evaluate uses for single-region runs).

Inference/animation are skipped: they are single-grid concepts.
"""
from __future__ import annotations

import time
from dataclasses import replace

from swotxai.config import SWOTConfig
from swotxai.models import get_backend
from swotxai.pipeline.orchestrator import (
    ProgressCb,
    _cleanup_shared_cache,
    _noop_cb,
    run_shared_steps,
)
from swotxai.pipeline.steps_ml import step_flatten
from swotxai.presets import MISSION_REGIONS, config_overrides


def _region_config(base: SWOTConfig, rid: str, mission: str) -> SWOTConfig:
    return replace(base, **config_overrides(rid, mission))


def _column_signature(flat: dict):
    for items in flat.values():
        for entry in items:
            return entry["X_u"].shape[1]
    return None


def _merge_round_robin(flats: dict[str, dict]) -> dict:
    """Interleave regions: r1[0], r2[0], ..., r1[1], r2[1], ..."""
    pooled: dict = {}
    key_lists = {r: list(f.keys()) for r, f in flats.items()}
    depth = max((len(k) for k in key_lists.values()), default=0)
    for i in range(depth):
        for r, keys in key_lists.items():
            if i < len(keys):
                pooled[f"{r}|{keys[i]}"] = flats[r][keys[i]]
    return pooled


def run_multiregion(
    config: SWOTConfig,
    mission: str,
    regions: list[str] | None = None,
    progress_cb: ProgressCb | None = None,
    use_cache: bool = True,
) -> dict:
    cb = progress_cb or _noop_cb
    t0 = time.time()
    regions = regions or MISSION_REGIONS[mission]

    flats: dict[str, dict] = {}
    for i, rid in enumerate(regions):
        rcfg = _region_config(config, rid, mission)

        def rcb(step, frac, msg, _r=rid, _i=i):
            cb(step, frac, f"[{_r} {_i + 1}/{len(regions)}] {msg}")

        shared = run_shared_steps(rcfg, progress_cb=rcb, use_cache=use_cache)
        try:
            flats[rid] = step_flatten(
                rcfg, shared["hfr_interp_data"], shared["swot_features"],
                rcb, use_cache,
            )
        finally:
            _cleanup_shared_cache(rcfg)
            shared.clear()

    sigs = {r: _column_signature(f) for r, f in flats.items()}
    live = {r: s for r, s in sigs.items() if s is not None}
    if not live:
        raise RuntimeError("No region produced any training rows.")
    if len(set(live.values())) > 1:
        raise RuntimeError(
            f"Feature-column mismatch across regions: {live}. All regions "
            "must expose the same features (check ERA5/GOES availability)."
        )
    empty = [r for r, s in sigs.items() if s is None]
    if empty:
        cb("flatten", 1.0, f"WARNING: no rows from {empty} — pooling without them")
        for r in empty:
            flats.pop(r)

    pooled = _merge_round_robin(flats)
    cb("flatten", 1.0,
       f"Pooled {len(pooled)} pass entries from {len(flats)} regions "
       f"({', '.join(flats)})")

    pcfg = replace(
        config,
        mission=mission,
        region=f"all_{mission}",
        swot_pkl_path=None, hfr_pkl_path=None,
    )
    if not pcfg.experiment_id:
        from swotxai.experiments import new_experiment_id
        pcfg.experiment_id = new_experiment_id(pcfg.model)
    pcfg.ensure_output_paths()

    backend = get_backend(pcfg.model)
    results: dict = {}

    t = time.time()
    train_out = backend.step_train(pcfg, pooled, cb, use_cache)
    train_seconds = round(time.time() - t, 1)
    if train_out:
        results["model_u"], results["model_v"] = train_out

    metrics = backend.step_evaluate(
        pcfg, results["model_u"], results["model_v"], pooled, cb,
    )
    per_region = {}
    for rid, flat in flats.items():
        cb("evaluate", 0.9, f"Per-region metrics: {rid}...")
        m = backend.step_evaluate(pcfg, results["model_u"], results["model_v"], flat, cb)
        per_region[rid] = {k: v for k, v in m.items()
                           if k.startswith(("rmse_", "r2_"))}
    metrics["per_region"] = per_region
    results["metrics"] = metrics

    try:
        from swotxai.experiments import record_experiment
        results["experiment"] = record_experiment(
            pcfg, metrics,
            extra={"pooled_regions": list(flats), "mission": mission,
                   "step_seconds": {"train": train_seconds}},
        )
        cb("experiment", 1.0,
           f"Recorded experiment {results['experiment']['experiment_id']}")
    except Exception as exc:
        cb("experiment", 1.0, f"WARNING: could not record experiment: {exc}")

    for name in ("inference", "animate"):
        cb(name, 1.0, "Skipped — not applicable to pooled multi-region runs.")
    cb("done", 1.0,
       f"Pooled {mission} training complete in {time.time() - t0:.1f}s")
    return results
