from __future__ import annotations

import time
from pathlib import Path
from typing import Callable

from swotxai.config import SWOTConfig
from swotxai.pipeline.io_utils import _cached
from swotxai.pipeline.steps_data import (
    step_load_preset_swot,
    step_load_preset_hfr,
    step_load_swot,
    step_regrid,
    step_load_era5,
    step_load_goes,
    step_interp_sources,
    step_load_hfr,
    step_interp_hfr,
)
from swotxai.pipeline.steps_ml import (
    step_flatten,
    step_inference,
)
from swotxai.pipeline.steps_viz import step_animate
from swotxai.models import get_backend

ProgressCb = Callable[[str, float, str], None]

STEPS = [
    "load_swot",
    "regrid",
    "load_era5",
    "load_goes",
    "interp_sources",
    "load_hfr",
    "interp_hfr",
    "flatten",
    "train",
    "evaluate",
    "inference",
    "animate",
]

SHARED_STEPS  = ["load_swot", "regrid", "load_era5", "load_goes", "interp_sources", "load_hfr", "interp_hfr"]
PER_JOB_STEPS = ["flatten", "train", "evaluate", "inference", "animate"]

_SHARED_CACHE_KEYS = ["cycle_data", "swot_regridded", "era5", "goes", "swot_features", "hfr", "hfr_interp"]


def _noop_cb(step: str, frac: float, msg: str) -> None:
    if step == "train_epoch":  # structured event for live monitors; the
        return                 # human-readable "train" line already prints
    print(f"[{step}] {int(frac * 100):3d}%  {msg}")


def _cleanup_shared_cache(config: SWOTConfig) -> None:
    for name in _SHARED_CACHE_KEYS:
        p = config.cache_path(name)
        if p.exists():
            p.unlink()


def run_shared_steps(
    config: SWOTConfig,
    progress_cb: ProgressCb | None = None,
    use_cache: bool = True,
) -> dict:
    cb = progress_cb or _noop_cb
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    results: dict = {}

    def _run(name: str, fn):
        t = time.time()
        cb(name, 0.0, f"Starting {name}...")
        out = fn()
        cb(name, 1.0, f"{name} done in {time.time() - t:.1f}s")
        return out

    swot_loaded = False
    if config.swot_pkl_path:
        try:
            cb("load_swot", 0.0, "Starting load_swot (preset)...")
            cb("regrid", 0.0, "Starting regrid (preset)...")
            results["cycle_data"], results["swot_regridded"] = step_load_preset_swot(config, cb)
            cb("load_swot", 1.0, "load_swot done")
            cb("regrid", 1.0, "regrid done")
            swot_loaded = True
        except (FileNotFoundError, OSError) as e:
            if not config.swot_path:
                raise
            cb("load_swot", 0.0,
               f"Preset SWOT pkl unavailable ({type(e).__name__}) — building from source...")
    if not swot_loaded:
        results["cycle_data"]     = _run("load_swot", lambda: step_load_swot(config, cb, use_cache))
        results["swot_regridded"] = _run("regrid",    lambda: step_regrid(config, results["cycle_data"], cb, use_cache))

    results["era5"] = _run("load_era5", lambda: step_load_era5(config, cb, use_cache))
    results["goes"] = _run("load_goes", lambda: step_load_goes(config, cb, use_cache))
    results["swot_features"] = _run("interp_sources", lambda: step_interp_sources(
        config, results["cycle_data"], results["swot_regridded"],
        results.get("era5"), results.get("goes"), cb, use_cache,
    ))

    hfr_preset = None
    if config.hfr_pkl_path:
        try:
            cb("load_hfr", 0.0, "Starting load_hfr (preset)...")
            cb("interp_hfr", 0.0, "Starting interp_hfr (preset)...")
            hfr_preset = step_load_preset_hfr(config, cb)
            cb("load_hfr", 1.0, "load_hfr done")
            cb("interp_hfr", 1.0, "interp_hfr done")
        except (FileNotFoundError, OSError) as e:
            if not config.hfr_path:
                raise
            cb("load_hfr", 0.0,
               f"Preset HFR pkl unavailable ({type(e).__name__}) — building from source...")
    if hfr_preset is not None:
        results["hfr"]             = None
        results["hfr_interp_data"] = hfr_preset
    else:
        if _cached(config.cache_path("hfr_interp"), use_cache):
            results["hfr"] = None
            cb("load_hfr", 1.0, "Skipped — hfr_interp cache exists.")
        else:
            results["hfr"] = _run("load_hfr", lambda: step_load_hfr(
                config, cb, use_cache, results["swot_regridded"],
            ))
        results["hfr_interp_data"] = _run("interp_hfr", lambda: step_interp_hfr(
            config, results.get("hfr"), results["cycle_data"], results["swot_regridded"], cb, use_cache,
        ))
        # Raw HFR can be multi-GB for full-network sources; only the
        # interpolated per-pass fields are needed downstream.
        results["hfr"] = None

    return results


def run_per_job_steps(
    config: SWOTConfig,
    shared: dict,
    progress_cb: ProgressCb | None = None,
    use_cache: bool = True,
) -> dict:
    cb = progress_cb or _noop_cb
    Path(config.cache_dir).mkdir(parents=True, exist_ok=True)
    results: dict = {}
    step_seconds: dict[str, float] = {}

    # Mint the run's unique id up front so unnamed animation outputs can be
    # named after it; the same id keys the experiment-registry record.
    if not config.experiment_id:
        from swotxai.experiments import new_experiment_id
        config.experiment_id = new_experiment_id(config.model)
    config.ensure_output_paths()
    results["frame_dir"] = config.frame_dir

    def _run(name: str, fn):
        t = time.time()
        cb(name, 0.0, f"Starting {name}...")
        out = fn()
        step_seconds[name] = round(time.time() - t, 1)
        cb(name, 1.0, f"{name} done in {step_seconds[name]:.1f}s")
        return out

    results["flattened"] = _run(
        "flatten", lambda: step_flatten(
            config, shared["hfr_interp_data"], shared["swot_features"], cb, use_cache,
        )
    )

    backend = get_backend(config.model)

    train_out = _run(
        "train", lambda: backend.step_train(config, results["flattened"], cb, use_cache)
    )
    if train_out:
        results["model_u"], results["model_v"] = train_out

    results["metrics"] = _run(
        "evaluate", lambda: backend.step_evaluate(
            config, results["model_u"], results["model_v"], results["flattened"], cb,
        )
    )

    dicts_out = _run(
        "inference", lambda: step_inference(
            config, results["model_u"], results["model_v"],
            shared["swot_features"], shared["hfr_interp_data"],
            results["flattened"], cb, use_cache,
        )
    )
    if dicts_out:
        results["swot_dict"], results["hfr_dict"] = dicts_out

    results["animation_paths"] = _run(
        "animate", lambda: step_animate(
            config, results["swot_dict"], results["hfr_dict"],
            shared.get("cycle_data", {}), shared.get("swot_regridded", {}),
            shared.get("hfr_interp_data", {}),
            results["metrics"], cb,
        )
    )

    # Run completed — auto-record it in the per-model experiment registry.
    try:
        from swotxai.experiments import record_experiment

        extra = {"step_seconds": step_seconds}
        base = getattr(results.get("model_u"), "base", None)
        meta = getattr(base, "meta", None)
        if meta:
            extra["training"] = {
                k: meta.get(k)
                for k in ("activation", "norm", "optimizer", "loss", "lr_scheduler",
                          "best_epoch", "best_val_loss", "train_seconds", "device")
            }
        results["experiment"] = record_experiment(config, results["metrics"], extra=extra)
        cb("experiment", 1.0,
           f"Recorded experiment {results['experiment']['experiment_id']}")
    except Exception as exc:
        cb("experiment", 1.0, f"WARNING: could not record experiment: {exc}")

    return results


def run_pipeline(
    config: SWOTConfig,
    steps: list[str] | None = None,
    progress_cb: ProgressCb | None = None,
    use_cache: bool = True,
) -> dict:
    cb = progress_cb or _noop_cb
    t0 = time.time()
    shared = run_shared_steps(config, progress_cb=cb, use_cache=use_cache)
    try:
        per_job = run_per_job_steps(config, shared, progress_cb=cb, use_cache=use_cache)
    finally:
        _cleanup_shared_cache(config)
        shared.clear()
    cb("done", 1.0, f"Pipeline complete in {time.time() - t0:.1f}s")
    return per_job
