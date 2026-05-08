from __future__ import annotations

import hashlib
import json
import os
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Callable

from swotxai.config import AVAILABLE_FEATURES, DEFAULT_FEATURES, SWOTConfig
from swotxai.pipeline import ProgressCb, run_per_job_steps, run_shared_steps, _cleanup_shared_cache

# Callback signature: (run_id, status, results_or_None)
# status is one of "running" | "done" | "failed"
JobStatusCb = Callable[[str, str, dict | None], None]


@dataclass
class JobSpec:
    run_id: str
    features: list[str]        = field(default_factory=lambda: list(DEFAULT_FEATURES))
    stencil_k: int             = 3
    n_estimators: int          = 50
    max_depth: int             = 15
    random_state: int          = 42
    frame_dir: str             = ""   # auto-derived from run_id if blank
    animation_output: str      = ""   # auto-derived from run_id if blank


@dataclass
class BatchConfig:
    jobs: list[JobSpec]
    max_workers: int           = 2

    # Shared data fields — same domain / data source for all jobs
    swot_path: str             = ""
    hfr_path: str              = ""
    era5_path: str             = ""
    goes_nc_path: str | None       = None
    sw_corner: list[float]     = field(default_factory=lambda: [-127.0, 37.5])
    ne_corner: list[float]     = field(default_factory=lambda: [-123.0, 42.5])
    mission: str               = "calval"
    sph_calval_path: str       = "orbit_data/sph_calval_swath.zip"
    sph_science_path: str      = "orbit_data/sph_science_swath.zip"
    cycles_start: int          = 474
    cycles_end: int            = 578
    cache_dir: str             = "cache"
    fps: int                   = 8
    dpi: int                   = 150
    swot_pkl_path: str | None  = None
    hfr_pkl_path: str | None   = None
    region: str | None         = None  # "uswc" | "usegc" | None


def derive_base_run_id(batch_cfg: BatchConfig) -> str:
    """Stable, deterministic cache key for the shared data steps."""
    key = {
        "swot_path":      batch_cfg.swot_path,
        "hfr_path":       batch_cfg.hfr_path,
        "era5_path":      batch_cfg.era5_path,
        "goes_nc_path":       batch_cfg.goes_nc_path,
        "sw_corner":      batch_cfg.sw_corner,
        "ne_corner":      batch_cfg.ne_corner,
        "mission":        batch_cfg.mission,
        "cycles_start":   batch_cfg.cycles_start,
        "cycles_end":     batch_cfg.cycles_end,
        "swot_pkl_path":  batch_cfg.swot_pkl_path,
        "hfr_pkl_path":   batch_cfg.hfr_pkl_path,
    }
    digest = hashlib.sha1(
        json.dumps(key, sort_keys=True).encode()
    ).hexdigest()[:8]
    return f"shared_{digest}"


def _build_job_config(batch_cfg: BatchConfig, job: JobSpec, sklearn_n_jobs: int) -> SWOTConfig:
    """Overlay job-specific hyperparams onto the shared base config."""
    return SWOTConfig(
        swot_path        = batch_cfg.swot_path,
        hfr_path         = batch_cfg.hfr_path,
        era5_path        = batch_cfg.era5_path,
        goes_nc_path         = batch_cfg.goes_nc_path,
        sw_corner        = batch_cfg.sw_corner,
        ne_corner        = batch_cfg.ne_corner,
        mission          = batch_cfg.mission,
        sph_calval_path  = batch_cfg.sph_calval_path,
        sph_science_path = batch_cfg.sph_science_path,
        cycles_start     = batch_cfg.cycles_start,
        cycles_end       = batch_cfg.cycles_end,
        cache_dir        = batch_cfg.cache_dir,
        fps              = batch_cfg.fps,
        dpi              = batch_cfg.dpi,
        swot_pkl_path    = batch_cfg.swot_pkl_path,
        hfr_pkl_path     = batch_cfg.hfr_pkl_path,
        region           = batch_cfg.region,
        # Per-job fields
        run_id           = job.run_id,
        features         = job.features,
        stencil_k        = job.stencil_k,
        n_estimators     = job.n_estimators,
        max_depth        = job.max_depth,
        random_state     = job.random_state,
        sklearn_n_jobs   = sklearn_n_jobs,
        frame_dir        = job.frame_dir,
        animation_output = job.animation_output,
    )


def run_batch(
    batch_cfg: BatchConfig,
    progress_cb: ProgressCb | None = None,
    job_status_cb: JobStatusCb | None = None,
    use_cache: bool = True,
) -> dict[str, dict]:
    """
    Run shared data steps once, then run per-job ML steps in parallel.

    Parameters
    ----------
    batch_cfg : BatchConfig
    progress_cb : ProgressCb | None
        Receives step-level updates from the shared data steps.
    job_status_cb : JobStatusCb | None
        Called as (run_id, status, results) when each job changes state.
        status is "running", "done", or "failed".
    use_cache : bool

    Returns
    -------
    dict mapping run_id → per-job results dict (keys: flattened, rf_u, rf_v,
    metrics, swot_dict, hfr_dict, animation_path).
    """
    # --- Shared steps (run once) ---
    base_run_id = derive_base_run_id(batch_cfg)
    base_cfg = SWOTConfig(
        swot_path        = batch_cfg.swot_path,
        hfr_path         = batch_cfg.hfr_path,
        era5_path        = batch_cfg.era5_path,
        goes_nc_path         = batch_cfg.goes_nc_path,
        sw_corner        = batch_cfg.sw_corner,
        ne_corner        = batch_cfg.ne_corner,
        mission          = batch_cfg.mission,
        sph_calval_path  = batch_cfg.sph_calval_path,
        sph_science_path = batch_cfg.sph_science_path,
        cycles_start     = batch_cfg.cycles_start,
        cycles_end       = batch_cfg.cycles_end,
        cache_dir        = batch_cfg.cache_dir,
        fps              = batch_cfg.fps,
        dpi              = batch_cfg.dpi,
        swot_pkl_path    = batch_cfg.swot_pkl_path,
        hfr_pkl_path     = batch_cfg.hfr_pkl_path,
        region           = batch_cfg.region,
        run_id           = base_run_id,
    )
    shared = run_shared_steps(base_cfg, progress_cb=progress_cb)

    # Distribute CPU cores evenly across parallel workers
    sklearn_n_jobs = max(1, (os.cpu_count() or 1) // batch_cfg.max_workers)

    # Per-job logs: each thread appends to its own list under a lock
    job_logs: dict[str, list[str]] = {job.run_id: [] for job in batch_cfg.jobs}
    log_lock = threading.Lock()

    def _make_job_cb(run_id: str) -> ProgressCb:
        def cb(step: str, frac: float, msg: str) -> None:
            with log_lock:
                job_logs[run_id].append(f"[{step}] {msg}")
        return cb

    def _run_job(job: JobSpec) -> dict:
        job_cfg = _build_job_config(batch_cfg, job, sklearn_n_jobs)
        if job_status_cb:
            job_status_cb(job.run_id, "running", None)
        try:
            result = run_per_job_steps(
                job_cfg, shared,
                progress_cb=_make_job_cb(job.run_id),
                use_cache=use_cache,
            )
            if job_status_cb:
                job_status_cb(job.run_id, "done", result)
            return result
        except Exception as exc:
            if job_status_cb:
                job_status_cb(job.run_id, "failed", {"error": str(exc)})
            raise

    # --- Per-job steps (parallel) ---
    all_results: dict[str, dict] = {}
    with ThreadPoolExecutor(max_workers=batch_cfg.max_workers) as executor:
        future_to_job = {executor.submit(_run_job, job): job for job in batch_cfg.jobs}
        for future in as_completed(future_to_job):
            job = future_to_job[future]
            try:
                all_results[job.run_id] = future.result()
            except Exception as exc:
                all_results[job.run_id] = {"error": str(exc)}

    # All jobs done — delete shared cache files and free large in-memory objects
    _cleanup_shared_cache(base_cfg)
    shared.clear()

    # Attach per-job logs to results for the UI
    for run_id, logs in job_logs.items():
        if run_id in all_results:
            all_results[run_id]["log"] = logs

    return all_results
