"""
SWOTxAI — Local Streamlit GUI
Run with: streamlit run app.py
"""
from __future__ import annotations

import os
import queue
import threading
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

from swotxai.batch import BatchConfig, JobSpec, run_batch
from swotxai.config import (
    AVAILABLE_FEATURES,
    SWOTConfig,
    load_config,
    save_config,
)
from swotxai.pipeline import SHARED_STEPS, STEPS, run_pipeline

PRESETS = {
    "US West Coast (calval)": {
        "swot_pkl": "s3://swot-ai-ssv/experiments/uswc/swot_cycles/swot_expert_reproc_v3_calval_uswc_474_578.pkl",
        "hfr_pkl":  "s3://swot-ai-ssv/experiments/uswc/hfr_target/hfr_calval_uswc.pkl",
        "sw_corner": [-127.0, 37.5],
        "ne_corner": [-123.0, 42.5],
        "mission": "calval",
        "cycles_start": 474,
        "cycles_end": 578,
        "region": "uswc",
    },
    "US East-Gulf Coast (calval)": {
        "swot_pkl": "s3://swot-ai-ssv/experiments/usegc/swot_cycles/swot_expert_reproc_v3_calval_usegc_474_578.pkl",
        "hfr_pkl":  "s3://swot-ai-ssv/experiments/usegc/hfr_target/hfr_calval_usegc.pkl",
        "sw_corner": [-76.157227, 35.835628],
        "ne_corner": [-68.334961, 42.827639],
        "mission": "calval",
        "cycles_start": 474,
        "cycles_end": 578,
        "region": "usegc",
    },
}

# ---------------------------------------------------------------------------
# Page setup
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SWOTxAI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("🌊 SWOTxAI")
st.caption("Surface water & ocean topography × machine learning")

# ---------------------------------------------------------------------------
# Session state initialisation
# ---------------------------------------------------------------------------
for key, default in [
    ("running", False),
    ("stop_event", None),
    ("results", {}),
    ("log", []),
    ("step_status", {s: "pending" for s in STEPS}),
    ("current_step", None),
    ("current_step_frac", 0.0),
    ("current_step_msg", ""),
    ("msg_queue", queue.Queue()),
    ("pipeline_start_time", None),
    ("pipeline_end_time", None),
    # Batch mode
    ("batch_running", False),
    ("batch_jobs", []),
    ("batch_job_statuses", {}),
    ("batch_job_results", {}),
    ("batch_job_logs", {}),
    ("batch_shared_step_status", {s: "pending" for s in SHARED_STEPS}),
    ("batch_msg_queue", queue.Queue()),
    ("batch_start_time", None),
    ("batch_end_time", None),
]:
    if key not in st.session_state:
        st.session_state[key] = default


# ---------------------------------------------------------------------------
# Sidebar — configuration
# ---------------------------------------------------------------------------
with st.sidebar:
    st.header("Configuration")

    # --- Region preset ---
    preset = st.selectbox(
        "Region preset",
        ["Custom"] + list(PRESETS.keys()),
        help="Choose a pre-processed default region or configure your own.",
    )
    using_preset = preset != "Custom"
    pcfg = PRESETS.get(preset, {})

    # Load / save config file
    with st.expander("Load / save config file", expanded=False):
        config_file = st.file_uploader("Load config.yaml", type=["yaml", "yml"])
        if config_file:
            import yaml, tempfile, os
            with tempfile.NamedTemporaryFile(delete=False, suffix=".yaml") as tmp:
                tmp.write(config_file.read())
                tmp_path = tmp.name
            try:
                loaded = load_config(tmp_path)
                st.session_state["loaded_config"] = loaded
                st.success("Config loaded.")
            except Exception as e:
                st.error(f"Invalid config: {e}")
            finally:
                os.unlink(tmp_path)

    base = st.session_state.get("loaded_config", SWOTConfig())

    # --- Data sources ---
    st.subheader("Data sources")
    if using_preset:
        st.info(f"Pre-processed SWOT and HFR data loaded from S3 for **{preset}**.")
        swot_path    = ""
        hfr_path     = ""
        swot_pkl_path = pcfg["swot_pkl"]
        hfr_pkl_path  = pcfg["hfr_pkl"]
    else:
        st.caption("SWOT and HFR are required. ERA5 and GOES are optional — leave blank to skip.")
        swot_path    = st.text_input("SWOT path (S3 or local) *", value=base.swot_path)
        hfr_path     = st.text_input("HFR path (.nc) *", value=base.hfr_path)
        swot_pkl_path = None
        hfr_pkl_path  = None
    era5_path = st.text_input("ERA5 path (.nc) — optional", value=base.era5_path)
    goes_dir  = st.text_input("GOES SST — optional (path to a .nc file with SST variable, or directory of GOES-16/17/18 .nc scan files)", value=base.goes_dir or "")

    # --- Domain (hidden when preset) ---
    if using_preset:
        sw_lon, sw_lat = pcfg["sw_corner"]
        ne_lon, ne_lat = pcfg["ne_corner"]
        mission = pcfg["mission"]
        sph_calval_path  = base.sph_calval_path
        sph_science_path = base.sph_science_path
    else:
        st.subheader("Domain")
        col1, col2 = st.columns(2)
        sw_lon = col1.number_input("SW lon", value=float(base.sw_corner[0]), step=0.5)
        sw_lat = col2.number_input("SW lat", value=float(base.sw_corner[1]), step=0.5)
        ne_lon = col1.number_input("NE lon", value=float(base.ne_corner[0]), step=0.5)
        ne_lat = col2.number_input("NE lat", value=float(base.ne_corner[1]), step=0.5)
        mission = st.radio("Mission phase", ["calval", "science"],
                           index=0 if base.mission == "calval" else 1, horizontal=True)
        sph_calval_path  = st.text_input("Calval orbit file (.zip)", value=base.sph_calval_path)
        sph_science_path = st.text_input("Science orbit file (.zip)", value=base.sph_science_path)

    # --- Features ---
    st.subheader("Features")
    selected_features = st.multiselect(
        "RF input features",
        options=AVAILABLE_FEATURES,
        default=base.features,
    )

    # --- Model hyperparameters ---
    st.subheader("Model hyperparameters")
    n_estimators = st.slider("n_estimators", 10, 500, base.n_estimators, step=10)
    max_depth    = st.slider("max_depth", 3, 50, base.max_depth)
    stencil_k    = st.select_slider("stencil k (must be odd)", options=[1, 3, 5, 7], value=base.stencil_k)

    # --- Animation ---
    st.subheader("Animation")
    if using_preset:
        cycles_start = pcfg["cycles_start"]
        cycles_end   = pcfg["cycles_end"]
        st.caption(f"Cycles: {cycles_start}–{cycles_end} (locked to preset)")
    else:
        cycle_mode = st.radio(
            "Cycle range",
            ["Full calval (474–578)", "Full science (1–16)", "Custom"],
            horizontal=True,
        )
        if cycle_mode == "Full calval (474–578)":
            cycles_start, cycles_end = 474, 578
        elif cycle_mode == "Full science (1–16)":
            cycles_start, cycles_end = 1, 16
        else:
            col3, col4 = st.columns(2)
            cycles_start = col3.number_input("Cycle start", value=base.cycles_start, step=1)
            cycles_end   = col4.number_input("Cycle end",   value=base.cycles_end,   step=1)

    _anim_region = pcfg.get("region") if using_preset else None
    _anim_base   = f"SWOTxAI/animations/{_anim_region}" if _anim_region else "SWOTxAI/animations"
    anim_name    = st.text_input(
        "Animation name",
        value=base.run_id,
        help="Stem for output files. Saved as SWOTxAI/animations/[region/]{name}_pass_N.mp4",
    )
    anim_output  = f"{_anim_base}/{anim_name}" if anim_name else ""
    if anim_name:
        st.caption(f"Animations → `{_anim_base}/{anim_name}_pass_N.mp4`")
    fps = st.slider("FPS", 1, 30, base.fps)
    dpi = st.slider("DPI", 72, 300, base.dpi, step=10)

    # --- Caching ---
    st.subheader("Caching")
    run_id = st.text_input("Run ID *", value=base.run_id,
                           help="Short unique name for this run (letters, digits, _ or - only).")
    if using_preset:
        cache_dir = base.cache_dir
        _region = pcfg.get("region", "")
        st.caption(
            f"Outputs cached to `SWOTxAI/code/experiments/{_region}/` "
            f"— reuses existing files if Run ID matches."
        )
    else:
        cache_dir = st.text_input("Cache dir", value=base.cache_dir,
                                  help="Relative to repo root. Intermediate results are pickled here so slow steps don't re-run.")
        st.caption(f"Cache will be saved to: `{Path(cache_dir).resolve()}`")
    use_cache = st.checkbox("Use cached steps", value=True,
                            help="Uncheck to force all steps to re-run from scratch.")

    try:
        current_config = SWOTConfig(
            swot_path=swot_path, hfr_path=hfr_path, era5_path=era5_path,
            goes_dir=goes_dir or None,
            sw_corner=[sw_lon, sw_lat], ne_corner=[ne_lon, ne_lat],
            mission=mission,
            sph_calval_path=sph_calval_path, sph_science_path=sph_science_path,
            features=selected_features,
            stencil_k=stencil_k, n_estimators=n_estimators,
            max_depth=max_depth, random_state=42,
            cycles_start=int(cycles_start), cycles_end=int(cycles_end),
            frame_dir="", animation_output=anim_output,
            fps=fps, dpi=dpi, cache_dir=cache_dir, run_id=run_id,
            swot_pkl_path=swot_pkl_path,
            hfr_pkl_path=hfr_pkl_path,
            region=pcfg.get("region") if using_preset else None,
        )
        config_valid = True
    except Exception as e:
        st.warning(f"Config error: {e}")
        config_valid = False

    required_ok = bool((using_preset or (swot_path and hfr_path)) and run_id)
    if not required_ok:
        if not run_id:
            st.warning("Run ID is required.")
        else:
            st.warning("SWOT path and HFR path are required.")
    run_btn = st.button("▶ Run Pipeline",
                        disabled=not config_valid or not required_ok or st.session_state.running,
                        type="primary", width='stretch')
    if st.session_state.running:
        if st.button("⏹ Stop", width='stretch'):
            ev = st.session_state.get("stop_event")
            if ev is not None:
                ev.set()


# ---------------------------------------------------------------------------
# Run pipeline in a background thread so Streamlit stays responsive
# ---------------------------------------------------------------------------
if run_btn and config_valid and not st.session_state.running:
    import threading as _threading
    stop_event = _threading.Event()
    st.session_state.running             = True
    st.session_state.stop_event          = stop_event
    st.session_state.log                 = []
    st.session_state.results             = {}
    st.session_state.step_status         = {s: "pending" for s in STEPS}
    st.session_state.pipeline_start_time = time.time()
    st.session_state.pipeline_end_time   = None

    q = queue.Queue()
    st.session_state.msg_queue = q

    def _run(cfg, use_cache_, q_, stop_):
        def _progress(step_name, frac, msg):
            if stop_.is_set():
                raise RuntimeError("Pipeline stopped by user.")
            q_.put({"step": step_name, "frac": frac, "msg": msg})

        try:
            results = run_pipeline(cfg, progress_cb=_progress, use_cache=use_cache_)
            q_.put({"step": "done", "frac": 1.0, "msg": "__results__", "results": results})
        except RuntimeError as e:
            if "stopped by user" in str(e).lower():
                q_.put({"step": "stopped", "frac": 0.0, "msg": "Pipeline stopped by user."})
            else:
                q_.put({"step": "error", "frac": 0.0, "msg": str(e)})
        except Exception as e:
            q_.put({"step": "error", "frac": 0.0, "msg": str(e)})
        finally:
            q_.put({"step": "__done__", "frac": 1.0, "msg": ""})

    t = threading.Thread(target=_run, args=(current_config, use_cache, q, stop_event), daemon=True)
    t.start()


# ---------------------------------------------------------------------------
# Main panel — three tabs
# ---------------------------------------------------------------------------
tab_pipeline, tab_results, tab_animation, tab_batch = st.tabs(
    ["Pipeline", "Results", "Animation", "Batch"]
)

# ---- Tab 1: Pipeline progress ----
with tab_pipeline:
    # Drain the queue (written by background thread, safe to read here)
    q = st.session_state.get("msg_queue")
    if q:
        while not q.empty():
            item = q.get_nowait()
            sn, frac, msg = item["step"], item["frac"], item["msg"]
            if sn == "__done__":
                st.session_state.running = False
                st.session_state.pipeline_end_time = time.time()
            elif sn == "done" and msg == "__results__":
                st.session_state.results = item.get("results", {})
            elif sn in ("error", "stopped"):
                st.session_state.log.append(f"{'ERROR' if sn == 'error' else 'STOPPED'}: {msg}")
                st.session_state.running = False
            else:
                if sn in STEPS:
                    st.session_state.step_status[sn] = "done" if frac >= 1.0 else "running"
                st.session_state.current_step      = sn
                st.session_state.current_step_frac = frac
                st.session_state.current_step_msg  = msg
                entry = f"[{sn}] {msg}"
                if st.session_state.log and st.session_state.log[-1].startswith(f"[{sn}]"):
                    st.session_state.log[-1] = entry
                else:
                    st.session_state.log.append(entry)

    icons    = {"pending": "⬜", "running": "🔵", "done": "✅", "failed": "❌"}
    statuses = st.session_state.step_status
    n_done   = sum(1 for s in statuses.values() if s == "done")
    n_total  = len(STEPS)
    cur_step = st.session_state.current_step
    cur_frac = st.session_state.current_step_frac
    cur_msg  = st.session_state.current_step_msg

    # Overall progress bar
    pipeline_done = not st.session_state.running and n_done == n_total and n_total > 0
    overall = min(1.0, (n_done + cur_frac) / n_total if st.session_state.running else n_done / n_total)
    st.progress(overall, text="Finished ✅" if pipeline_done else f"{n_done} / {n_total} steps complete")

    # Current step + its own progress bar (only while running)
    if st.session_state.running and cur_step:
        st.markdown(f"**{cur_step.replace('_', ' ')}**")
        st.progress(cur_frac, text=cur_msg or "Working…")

    # Elapsed timer
    t_start = st.session_state.pipeline_start_time
    t_end   = st.session_state.pipeline_end_time
    if t_start is not None:
        elapsed = (time.time() if st.session_state.running else t_end) - t_start
        m, s = divmod(int(elapsed), 60)
        h, m = divmod(m, 60)
        timer_str = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
        label = "Elapsed" if st.session_state.running else "Total time"
        st.metric(label, timer_str)

    st.divider()

    # Step grid
    cols = st.columns(n_total)
    for col, step_name in zip(cols, STEPS):
        status = statuses.get(step_name, "pending")
        col.markdown(
            f"<div style='text-align:center;font-size:1.4rem'>{icons[status]}</div>"
            f"<div style='text-align:center;font-size:0.7rem;color:gray'>{step_name.replace('_', ' ')}</div>",
            unsafe_allow_html=True,
        )

    st.divider()

    # Log
    st.code(
        "\n".join(st.session_state.log[-100:]) if st.session_state.log else "Waiting for pipeline to start...",
        language=None,
    )

    if st.session_state.running:
        time.sleep(1)
        st.rerun()

# ---- Tab 2: Results ----
with tab_results:
    metrics = st.session_state.results.get("metrics")

    if metrics is None:
        st.info("Run the pipeline to see results.")
    else:
        st.subheader("Model metrics")
        col_u, col_v = st.columns(2)
        col_u.metric("RMSE  u", f"{metrics['rmse_u']:.4f} m/s")
        col_u.metric("R²  u",   f"{metrics['r2_u']:.4f}")
        col_v.metric("RMSE  v", f"{metrics['rmse_v']:.4f} m/s")
        col_v.metric("R²  v",   f"{metrics['r2_v']:.4f}")

        st.subheader("Feature importances")
        fi_u = metrics["feature_importance_u"]
        fi_v = metrics["feature_importance_v"]
        features = list(fi_u.keys())

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, fi, label in [(axes[0], fi_u, "u-velocity"), (axes[1], fi_v, "v-velocity")]:
            vals = [fi[f] for f in features]
            idx  = np.argsort(vals)[::-1]
            ax.barh([features[i] for i in idx], [vals[i] for i in idx], color="steelblue")
            ax.set_xlabel("Importance")
            ax.set_title(f"Feature importances — {label}")
            ax.invert_yaxis()
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ---- Tab 3: Animation ----
with tab_animation:
    anim_paths = st.session_state.results.get("animation_paths") or []
    frame_dir_path = Path(current_config.frame_dir) if config_valid else Path("SWOTxAI/frames")

    if anim_paths:
        st.subheader("Animation")
        for anim_path in anim_paths:
            if Path(anim_path).exists():
                with open(anim_path, "rb") as f:
                    st.download_button(
                        f"⬇ Download {Path(anim_path).name}",
                        data=f,
                        file_name=Path(anim_path).name,
                        key=anim_path,
                    )

    # Frame scrubber
    frame_files = sorted(frame_dir_path.glob("*.png")) if frame_dir_path.exists() else []
    if frame_files:
        st.subheader(f"Frame preview ({len(frame_files)} frames)")
        idx = st.slider("Frame", 0, len(frame_files) - 1, 0)
        st.image(str(frame_files[idx]), width='stretch',
                 caption=frame_files[idx].stem.replace("_", " "))
    elif not anim_paths:
        st.info("Run the pipeline to generate frames and animation.")

# ---- Tab 4: Batch ----
with tab_batch:

    # ---------------------------------------------------------------------------
    # Batch background thread
    # ---------------------------------------------------------------------------
    def _run_batch_thread(batch_cfg: BatchConfig, use_cache_: bool, q_: queue.Queue) -> None:
        def _shared_cb(step_name: str, frac: float, msg: str) -> None:
            q_.put({"type": "shared_step", "step": step_name, "frac": frac, "msg": msg})

        def _job_status_cb(run_id: str, status: str, results) -> None:
            q_.put({"type": "job_status", "run_id": run_id, "status": status, "results": results})

        try:
            all_results = run_batch(
                batch_cfg,
                progress_cb=_shared_cb,
                job_status_cb=_job_status_cb,
                use_cache=use_cache_,
            )
            q_.put({"type": "done", "results": all_results})
        except Exception as exc:
            q_.put({"type": "error", "msg": str(exc)})
        finally:
            q_.put({"type": "__done__"})

    # ---------------------------------------------------------------------------
    # Drain the batch queue
    # ---------------------------------------------------------------------------
    bq = st.session_state.get("batch_msg_queue")
    if bq:
        while not bq.empty():
            item = bq.get_nowait()
            t = item["type"]
            if t == "__done__":
                st.session_state.batch_running  = False
                st.session_state.batch_end_time = time.time()
            elif t == "shared_step":
                st.session_state.batch_shared_step_status[item["step"]] = (
                    "done" if item["frac"] >= 1.0 else "running"
                )
            elif t == "job_status":
                rid, status, res = item["run_id"], item["status"], item["results"]
                st.session_state.batch_job_statuses[rid] = status
                if res:
                    if status == "done":
                        st.session_state.batch_job_results[rid] = res.get("metrics", {})
                        st.session_state.batch_job_logs[rid]    = res.get("log", [])
                    elif status == "failed":
                        st.session_state.batch_job_results[rid] = {"error": res.get("error", "")}
            elif t == "error":
                st.session_state.batch_running = False
                st.error(f"Batch error: {item['msg']}")

    # ---------------------------------------------------------------------------
    # Section 1 — Settings
    # ---------------------------------------------------------------------------
    st.subheader("Batch Settings")
    st.caption(
        "Shared data steps (load, regrid, interpolation) run **once** for all jobs. "
        "Per-job steps (flatten, train, evaluate, animate) run in parallel."
    )
    batch_max_workers = st.slider(
        "Max parallel workers", 1, max(1, os.cpu_count() or 4), 2,
        help="Each worker runs one job's ML steps simultaneously. CPU cores are divided evenly across workers.",
    )

    st.divider()

    # ---------------------------------------------------------------------------
    # Section 2 — Job table
    # ---------------------------------------------------------------------------
    st.subheader("Jobs")
    st.caption(
        "Each row is one experiment. All jobs share the data source and domain configured in the sidebar."
    )

    # Add / Delete Selected buttons
    col_add, col_del, _ = st.columns([1, 1, 6])
    if col_add.button("＋ Add Job", disabled=st.session_state.batch_running):
        idx_new = len(st.session_state.batch_jobs) + 1
        st.session_state.batch_jobs.append({
            "_delete":      False,
            "run_id":       f"job_{idx_new:02d}",
            "features":     ",".join(AVAILABLE_FEATURES),
            "stencil_k":    3,
            "n_estimators": 50,
            "max_depth":    15,
            "random_state": 42,
        })
        st.rerun()
    if col_del.button("🗑 Delete Selected", disabled=st.session_state.batch_running):
        st.session_state.batch_jobs = [
            j for j in st.session_state.batch_jobs if not j.get("_delete", False)
        ]
        st.rerun()

    edited = st.data_editor(
        st.session_state.batch_jobs,
        num_rows="fixed",
        width='stretch',
        disabled=st.session_state.batch_running,
        column_config={
            "_delete":      st.column_config.CheckboxColumn("Delete", default=False),
            "run_id":       st.column_config.TextColumn("Run ID", help="Unique name for this job"),
            "features":     st.column_config.TextColumn("Features", help="Comma-separated list"),
            "stencil_k":    st.column_config.SelectboxColumn("Stencil K", options=[1, 3, 5, 7]),
            "n_estimators": st.column_config.NumberColumn("N Estimators", min_value=10, max_value=500, step=10),
            "max_depth":    st.column_config.NumberColumn("Max Depth", min_value=3, max_value=50),
            "random_state": st.column_config.NumberColumn("Random State", min_value=0),
        },
        key="batch_job_editor",
    )
    st.session_state.batch_jobs = edited if edited is not None else st.session_state.batch_jobs

    # Validate only non-deleted rows
    import re as _re
    active_jobs = [j for j in st.session_state.batch_jobs if not j.get("_delete", False)]
    batch_errors = []
    run_ids = [(j.get("run_id") or "").strip() for j in active_jobs]
    for j in active_jobs:
        rid = (j.get("run_id") or "").strip()
        if not rid:
            batch_errors.append("All jobs must have a Run ID.")
            break
        if not _re.match(r'^[A-Za-z0-9_\-]+$', rid):
            batch_errors.append(f"Run ID '{rid}' contains invalid characters.")
    if len(run_ids) != len(set(run_ids)):
        batch_errors.append("Run IDs must be unique.")
    if not active_jobs:
        batch_errors.append("Add at least one job.")

    for err in batch_errors:
        st.warning(err)

    st.divider()

    # ---------------------------------------------------------------------------
    # Section 3 — Run button
    # ---------------------------------------------------------------------------
    batch_required_ok = bool((using_preset or (swot_path and hfr_path)) and not batch_errors)
    batch_run_btn = st.button(
        "▶ Run Batch",
        disabled=not batch_required_ok or st.session_state.batch_running or st.session_state.running,
        type="primary",
    )

    if batch_run_btn and batch_required_ok and not st.session_state.batch_running:
        job_specs = []
        for j in st.session_state.batch_jobs:
            if j.get("_delete", False):
                continue
            raw_features = [f.strip() for f in (j.get("features") or "").split(",") if f.strip()]
            valid_features = [f for f in raw_features if f in AVAILABLE_FEATURES]
            job_specs.append(JobSpec(
                run_id       = (j.get("run_id") or "").strip(),
                features     = valid_features or list(AVAILABLE_FEATURES),
                stencil_k    = int(j.get("stencil_k") or 3),
                n_estimators = int(j.get("n_estimators") or 50),
                max_depth    = int(j.get("max_depth") or 15),
                random_state = int(j.get("random_state") or 42),
            ))

        batch_cfg_obj = BatchConfig(
            jobs             = job_specs,
            max_workers      = batch_max_workers,
            swot_path        = swot_path,
            hfr_path         = hfr_path,
            era5_path        = era5_path,
            goes_dir         = goes_dir or None,
            sw_corner        = [sw_lon, sw_lat],
            ne_corner        = [ne_lon, ne_lat],
            mission          = mission,
            sph_calval_path  = sph_calval_path,
            sph_science_path = sph_science_path,
            cycles_start     = int(cycles_start),
            cycles_end       = int(cycles_end),
            cache_dir        = cache_dir,
            fps              = fps,
            dpi              = dpi,
            swot_pkl_path    = swot_pkl_path,
            hfr_pkl_path     = hfr_pkl_path,
            region           = pcfg.get("region") if using_preset else None,
        )

        st.session_state.batch_running           = True
        st.session_state.batch_start_time        = time.time()
        st.session_state.batch_end_time          = None
        st.session_state.batch_job_statuses      = {j.run_id: "pending" for j in job_specs}
        st.session_state.batch_job_results       = {}
        st.session_state.batch_job_logs          = {}
        st.session_state.batch_shared_step_status = {s: "pending" for s in SHARED_STEPS}

        bq_new = queue.Queue()
        st.session_state.batch_msg_queue = bq_new
        threading.Thread(
            target=_run_batch_thread,
            args=(batch_cfg_obj, use_cache, bq_new),
            daemon=True,
        ).start()

    st.divider()

    # ---------------------------------------------------------------------------
    # Section 4 — Shared step progress
    # ---------------------------------------------------------------------------
    if st.session_state.batch_start_time is not None:
        st.subheader("Shared Data Steps")
        icons = {"pending": "⬜", "running": "🔵", "done": "✅", "failed": "❌"}
        sh_cols = st.columns(len(SHARED_STEPS))
        for col, sname in zip(sh_cols, SHARED_STEPS):
            status = st.session_state.batch_shared_step_status.get(sname, "pending")
            col.markdown(
                f"<div style='text-align:center;font-size:1.4rem'>{icons[status]}</div>"
                f"<div style='text-align:center;font-size:0.7rem;color:gray'>{sname.replace('_', ' ')}</div>",
                unsafe_allow_html=True,
            )

        # Elapsed timer
        bt_start = st.session_state.batch_start_time
        bt_end   = st.session_state.batch_end_time
        if bt_start:
            b_elapsed = (time.time() if st.session_state.batch_running else bt_end) - bt_start
            bm, bs = divmod(int(b_elapsed), 60)
            bh, bm = divmod(bm, 60)
            btimer = f"{bh:02d}:{bm:02d}:{bs:02d}" if bh else f"{bm:02d}:{bs:02d}"
            st.metric("Elapsed" if st.session_state.batch_running else "Total Time", btimer)

        st.divider()

        # ---------------------------------------------------------------------------
        # Section 5 — Per-job status
        # ---------------------------------------------------------------------------
        st.subheader("Job Progress")
        job_statuses  = st.session_state.batch_job_statuses
        job_results   = st.session_state.batch_job_results
        job_logs_map  = st.session_state.batch_job_logs
        status_icons  = {"pending": "⬜ Pending", "running": "🔵 Running",
                         "done": "✅ Done", "failed": "❌ Failed"}
        n_done_jobs   = sum(1 for s in job_statuses.values() if s == "done")
        n_total_jobs  = len(job_statuses)
        if n_total_jobs:
            st.progress(n_done_jobs / n_total_jobs,
                        text=f"{n_done_jobs} / {n_total_jobs} jobs complete")

        for j in st.session_state.batch_jobs:
            rid    = j.get("run_id", "")
            status = job_statuses.get(rid, "pending")
            res    = job_results.get(rid, {})
            label  = f"{status_icons.get(status, status)}  —  {rid}"
            with st.expander(label, expanded=(status == "failed")):
                st.caption(
                    f"Features: `{j.get('features', '')}` | "
                    f"Stencil K: {j.get('stencil_k')} | "
                    f"N Estimators: {j.get('n_estimators')} | "
                    f"Max Depth: {j.get('max_depth')}"
                )
                if status == "done" and res and "error" not in res:
                    mc1, mc2 = st.columns(2)
                    mc1.metric("RMSE u", f"{res.get('rmse_u', 0):.4f} m/s")
                    mc1.metric("R² u",   f"{res.get('r2_u', 0):.4f}")
                    mc2.metric("RMSE v", f"{res.get('rmse_v', 0):.4f} m/s")
                    mc2.metric("R² v",   f"{res.get('r2_v', 0):.4f}")
                elif "error" in res:
                    st.error(res["error"])
                logs = job_logs_map.get(rid, [])
                if logs:
                    st.code("\n".join(logs[-50:]), language=None)

        st.divider()

        # ---------------------------------------------------------------------------
        # Section 6 — Comparison table
        # ---------------------------------------------------------------------------
        done_jobs = [(rid, r) for rid, r in job_results.items() if "error" not in r and "rmse_u" in r]
        if done_jobs:
            st.subheader("Results Comparison")
            rows = []
            job_map = {j.get("run_id"): j for j in st.session_state.batch_jobs}
            for rid, r in done_jobs:
                j = job_map.get(rid, {})
                rows.append({
                    "Run ID":       rid,
                    "Features":     j.get("features", ""),
                    "Stencil K":    j.get("stencil_k", ""),
                    "N Estimators": j.get("n_estimators", ""),
                    "Max Depth":    j.get("max_depth", ""),
                    "RMSE u":       round(r.get("rmse_u", 0), 4),
                    "RMSE v":       round(r.get("rmse_v", 0), 4),
                    "R² u":         round(r.get("r2_u", 0), 4),
                    "R² v":         round(r.get("r2_v", 0), 4),
                })
            df = pd.DataFrame(rows).sort_values("R² u", ascending=False)
            st.dataframe(df, width='stretch', hide_index=True)
            csv = df.to_csv(index=False).encode()
            st.download_button("⬇ Download CSV", data=csv,
                               file_name="batch_results.csv", mime="text/csv")

    if st.session_state.batch_running:
        time.sleep(1)
        st.rerun()
