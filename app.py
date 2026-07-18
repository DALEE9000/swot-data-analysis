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

# Preset definitions live in swotxai.presets (shared with the pooled
# multi-region runner). Pooled entries carry "multi": True and dispatch to
# run_multiregion instead of run_pipeline; per-region entries point at the
# pre-built S3 artifacts (SWOT/HFR pkls, ERA5 winds, GOES SST) with raw
# sources as fallback.
from swotxai.presets import build_presets

PRESETS = build_presets()

# ---------------------------------------------------------------------------
# Page setup + deep-ocean theme
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="SWOTxAI",
    page_icon="🌊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Chart colors validated for CVD separation + contrast on the navy surface
ACCENT     = "#22d3ee"   # UI accent — bright cyan
CHART_U    = "#0891b2"   # u-velocity marks (deep cyan)
CHART_V    = "#9085e9"   # v-velocity marks (violet)
INK        = "#e2e8f0"
INK_MUTED  = "#8ba3bf"
GRID_LINE  = "#1e3050"
SURFACE    = "#0b1626"
SURFACE_2  = "#101f33"

# Theme handed to the plotly figure builders in swotxai.experiments / viz3d
PLOTLY_THEME = {
    "color_u":     CHART_U,
    "color_v":     CHART_V,
    "ink":         INK,
    "ink_muted":   INK_MUTED,
    "grid":        GRID_LINE,
    "surface":     SURFACE,
    "surface_alt": SURFACE_2,
    "accent":      ACCENT,
}
PLOTLY_CONFIG = {"displayModeBar": False}

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&display=swap');

html, body { font-family: 'Space Grotesk', 'Segoe UI', system-ui, sans-serif; }
[data-testid="stAppViewContainer"] :is(p, h1, h2, h3, h4, h5, h6, label, li, td, th, input, textarea, button),
[data-testid="stSidebar"] :is(p, h1, h2, h3, h4, h5, h6, label, li, input, textarea, button) {
    font-family: 'Space Grotesk', 'Segoe UI', system-ui, sans-serif;
}
/* never touch Streamlit's icon font — glyphs turn into overlapping text */
[data-testid="stIconMaterial"], [class*="material-symbols"] {
    font-family: 'Material Symbols Rounded' !important;
}
code, pre { font-family: 'Cascadia Code', 'Consolas', monospace; }

/* Layered abyssal background */
[data-testid="stAppViewContainer"] {
    background:
        radial-gradient(1100px 520px at 85% -10%, rgba(34,211,238,.09), transparent 60%),
        radial-gradient(900px 520px at -10% 110%, rgba(59,130,246,.10), transparent 55%),
        linear-gradient(180deg, #0b1626 0%, #091120 100%);
}
[data-testid="stHeader"] { background: transparent; }
[data-testid="stAppDeployButton"] { display: none; }
[data-testid="stMainBlockContainer"] { padding-top: 2.4rem; }

/* Inputs & selects — visible fields on the dark surface */
[data-testid="stTextInput"] div[data-baseweb="input"],
[data-testid="stNumberInput"] div[data-baseweb="input"],
div[data-baseweb="select"] > div {
    background: #0d1b2e;
    border: 1px solid rgba(139,163,191,.28);
    border-radius: 8px;
}
[data-testid="stTextInput"] div[data-baseweb="input"]:focus-within,
[data-testid="stNumberInput"] div[data-baseweb="input"]:focus-within,
div[data-baseweb="select"] > div:focus-within {
    border-color: rgba(34,211,238,.65);
    box-shadow: 0 0 0 1px rgba(34,211,238,.35);
}
[data-baseweb="input"] input { background: transparent; }

/* Inline code */
[data-testid="stMarkdownContainer"] code, [data-testid="stCaptionContainer"] code {
    color: #67e8f9 !important; background: rgba(34,211,238,.09); border-radius: 5px;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0d1b2e 0%, #0a1424 100%);
    border-right: 1px solid rgba(34,211,238,.14);
}
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 {
    color: #a5f3fc; letter-spacing: .04em; text-transform: uppercase; font-size: .82rem;
}

/* Hero */
.swx-hero { padding: .2rem 0 .6rem 0; }
.swx-hero h1 {
    margin: 0; font-size: 2.6rem; font-weight: 700; letter-spacing: .02em;
    background: linear-gradient(90deg, #67e8f9 0%, #22d3ee 35%, #3b82f6 90%);
    -webkit-background-clip: text; background-clip: text; color: transparent;
}
.swx-hero h1 .x { color: #3b82f6; -webkit-text-fill-color: #3b82f6; font-weight: 400; }
.swx-hero p {
    margin: .15rem 0 0 2px; color: #5c7290; font-size: .70rem;
    letter-spacing: .38em; text-transform: uppercase;
}

/* Tabs */
button[data-baseweb="tab"] {
    color: #8ba3bf; letter-spacing: .05em; text-transform: uppercase; font-size: .8rem;
}
button[data-baseweb="tab"][aria-selected="true"] { color: #22d3ee; }
[data-baseweb="tab-highlight"] {
    background: linear-gradient(90deg, #22d3ee, #3b82f6);
    box-shadow: 0 0 14px rgba(34,211,238,.55); height: 2px;
}
[data-baseweb="tab-border"] { background: rgba(34,211,238,.12); }

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(160deg, rgba(34,211,238,.07), rgba(16,31,51,.72));
    border: 1px solid rgba(34,211,238,.18);
    border-radius: 14px; padding: 14px 18px;
}
[data-testid="stMetricValue"] { color: #22d3ee; font-weight: 600; }
[data-testid="stMetricLabel"] { color: #8ba3bf; letter-spacing: .08em; text-transform: uppercase; }

/* Primary button */
[data-testid="stBaseButton-primary"] {
    background: linear-gradient(90deg, #0891b2 0%, #2563eb 100%);
    border: 0; border-radius: 10px; letter-spacing: .06em;
    box-shadow: 0 4px 22px rgba(34,211,238,.30);
    transition: box-shadow .2s, filter .2s;
}
[data-testid="stBaseButton-primary"]:hover {
    filter: brightness(1.15); box-shadow: 0 4px 30px rgba(34,211,238,.55);
}
[data-testid="stBaseButton-primary"]:disabled {
    background: #14243c; color: #46587a; box-shadow: none;
}
[data-testid="stBaseButton-secondary"] {
    border: 1px solid rgba(34,211,238,.30); border-radius: 10px;
    background: rgba(16,31,51,.6); color: #a5f3fc;
}
[data-testid="stBaseButton-primary"], [data-testid="stBaseButton-secondary"] { white-space: nowrap; }

/* Progress bars — dark track, glowing gradient fill */
[data-testid="stProgress"] div[role="progressbar"] > div > div {
    background: #14243c; border-radius: 99px;
}
[data-testid="stProgress"] div[role="progressbar"] > div > div > div {
    background: linear-gradient(90deg, #22d3ee, #3b82f6);
    box-shadow: 0 0 10px rgba(34,211,238,.5); border-radius: 99px;
}

/* Expanders / code / dataframes */
[data-testid="stExpander"] details {
    border: 1px solid rgba(34,211,238,.14); border-radius: 12px;
    background: rgba(16,31,51,.55);
}
[data-testid="stCode"] pre, .stCode {
    background: #0a1424 !important;
    border: 1px solid rgba(34,211,238,.10); border-radius: 10px;
}

/* Pipeline step chips */
.swx-step {
    display: flex; flex-direction: column; align-items: center;
    justify-content: center; min-height: 58px;
    text-align: center; padding: 6px 2px; border-radius: 10px;
    border: 1px solid; font-size: .62rem; letter-spacing: .07em;
    text-transform: uppercase; line-height: 1.25;
}
.swx-step .dot {
    display: block; width: 9px; height: 9px; border-radius: 50%;
    margin: 0 auto 6px;
}
.swx-step.pending { border-color: #1a2a44; color: #5c7290; background: rgba(16,31,51,.35); }
.swx-step.pending .dot { background: #24344d; }
.swx-step.running { border-color: rgba(34,211,238,.55); color: #22d3ee; background: rgba(34,211,238,.06); }
.swx-step.running .dot { background: #22d3ee; animation: swx-pulse 1.4s ease-out infinite; }
.swx-step.done { border-color: rgba(52,211,153,.35); color: #34d399; background: rgba(52,211,153,.05); }
.swx-step.done .dot { background: #34d399; }
.swx-step.failed { border-color: rgba(248,113,113,.55); color: #f87171; background: rgba(248,113,113,.07); }
.swx-step.failed .dot { background: #f87171; }
@keyframes swx-pulse {
    0%   { box-shadow: 0 0 0 0 rgba(34,211,238,.55); }
    70%  { box-shadow: 0 0 0 9px rgba(34,211,238,0); }
    100% { box-shadow: 0 0 0 0 rgba(34,211,238,0); }
}
</style>
""", unsafe_allow_html=True)


def _parse_hidden_layers(text: str) -> list[int]:
    """Parse hidden-layer widths from user text, accepting '256,256,128',
    '[256, 256, 128]', or '(256 256 128)'. Returns [] if anything is invalid
    (which invalidates the config and surfaces a warning instead of crashing)."""
    cleaned = str(text)
    for ch in "[]()":
        cleaned = cleaned.replace(ch, "")
    tokens = cleaned.replace(";", ",").replace(" ", ",").split(",")
    vals: list[int] = []
    for tok in tokens:
        if not tok.strip():
            continue
        try:
            vals.append(int(tok.strip()))
        except ValueError:
            return []
    return vals


def _step_chip(name: str, status: str) -> str:
    return (f"<div class='swx-step {status}'><span class='dot'></span>"
            f"{name.replace('_', ' ')}</div>")


st.markdown("""
<div class="swx-hero">
  <h1>SWOT<span class="x">×</span>AI</h1>
  <p>Surface Water &amp; Ocean Topography × Machine Learning</p>
</div>
""", unsafe_allow_html=True)

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
    ("ann_epochs", []),
    ("msg_queue", queue.Queue()),
    ("pipeline_start_time", None),
    ("pipeline_end_time", None),
    # Batch mode
    ("batch_running", False),
    ("batch_jobs", []),
    ("batch_jobs_edited", []),
    ("batch_error", None),
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

    _default_cfg_path = Path("config.yaml")
    if "loaded_config" not in st.session_state and _default_cfg_path.exists():
        try:
            st.session_state["loaded_config"] = load_config(_default_cfg_path)
        except Exception:
            pass
    base = st.session_state.get("loaded_config", SWOTConfig())

    # Initialise widget session-state keys exactly once from base config.
    # After first render, keys persist and value= is NOT used.
    _widget_defaults = {
        "swot_path":       base.swot_path,
        "hfr_path":        base.hfr_path,
        "era5_pkl_path":   base.era5_pkl_path,
        "goes_nc_path":    base.goes_nc_path or "",
        "sw_lon":          float(base.sw_corner[0]),
        "sw_lat":          float(base.sw_corner[1]),
        "ne_lon":          float(base.ne_corner[0]),
        "ne_lat":          float(base.ne_corner[1]),
        "mission":         base.mission,
        "sph_calval_path": base.sph_calval_path,
        "sph_science_path":base.sph_science_path,
        "model_kind":      base.model,
        "rf_n_estimators": base.rf_n_estimators,
        "rf_max_depth":    base.rf_max_depth,
        "stencil_k":       base.stencil_k,
        "ann_hidden":      ",".join(str(h) for h in base.ann_hidden_layers),
        "ann_activation":  base.ann_activation,
        "ann_dropout":     float(base.ann_dropout),
        # snap to the widget option grids so select_sliders accept the value
        "ann_lr":          min([1e-4, 3e-4, 1e-3, 3e-3, 1e-2], key=lambda o: abs(o - float(base.ann_lr))),
        "ann_batch_size":  min([1024, 2048, 4096, 8192, 16384], key=lambda o: abs(o - int(base.ann_batch_size))),
        "ann_max_epochs":  int(base.ann_max_epochs),
        "ann_patience":    int(base.ann_patience),
        "anim_name":       base.run_id,
        "fps":             base.fps,
        "dpi":             base.dpi,
        "run_id":          base.run_id,
        "cache_dir":       base.cache_dir,
    }
    for _k, _v in _widget_defaults.items():
        if _k not in st.session_state:
            st.session_state[_k] = _v

    # --- Data sources ---
    st.subheader("Data sources")
    multi_preset = bool(pcfg.get("multi"))
    if multi_preset:
        st.info(
            f"**{preset}** trains ONE model on the pooled data of "
            f"**{', '.join(pcfg['regions'])}** — each region's SWOT/HFR/ERA5/GOES "
            "artifacts stream from S3, flatten per region, then train together. "
            "Metrics report pooled + per-region scores; inference/animation are "
            "skipped (single-grid concepts)."
        )
        swot_path = hfr_path = ""
        swot_pkl_path = hfr_pkl_path = None
        era5_pkl_path = ""
        goes_nc_path = ""
    elif using_preset:
        if pcfg.get("hfr_pkl") and pcfg.get("swot_pkl"):
            st.info(f"Pre-processed SWOT and HFR data loaded from S3 for **{preset}**.")
        else:
            st.info(
                f"**{preset}** builds its dataset from source on the first run "
                "(SWOT regridded from raw granules, HFR cut from the "
                "full-network archive), then caches it — expect the first run "
                "to be much slower than later ones."
            )
        swot_path    = pcfg.get("swot_path", "")
        hfr_path     = pcfg.get("hfr_path", "")
        swot_pkl_path = pcfg["swot_pkl"]
        hfr_pkl_path  = pcfg["hfr_pkl"]
        era5_pkl_path = pcfg.get("era5_pkl", "")
        goes_nc_path  = pcfg.get("goes_nc", "")
        extras = [x for x, p in (("ERA5 winds", era5_pkl_path), ("GOES SST", goes_nc_path)) if p]
        if extras:
            st.caption(f"{' + '.join(extras)} available from S3 — loaded only when the "
                       "matching features (era5_u/era5_v/SST) are selected below.")
    else:
        st.caption("SWOT and HFR are required. ERA5 and GOES are optional — leave blank to skip.")
        swot_path    = st.text_input("SWOT path (S3 or local) *", key="swot_path")
        hfr_path     = st.text_input(
            "HFR path (.nc) *", key="hfr_path",
            placeholder="e.g. data/HFR/hfr_uswc.nc — no default; region-specific ground truth",
            help="High-frequency-radar surface velocity NetCDF with u/v variables — the "
                 "training target. There is no universal default: it depends on your domain. "
                 "The region presets skip this field by loading a pre-processed HFR pkl from S3.",
        )
        swot_pkl_path = None
        hfr_pkl_path  = None
    if not using_preset:
        era5_pkl_path = st.text_input("ERA5 pkl path (S3 or local) — optional", key="era5_pkl_path")
        goes_nc_path = st.text_input("GOES SST NC path (S3 or local) — optional", key="goes_nc_path")

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
        sw_lon = col1.number_input("SW lon", step=0.5, key="sw_lon")
        sw_lat = col2.number_input("SW lat", step=0.5, key="sw_lat")
        ne_lon = col1.number_input("NE lon", step=0.5, key="ne_lon")
        ne_lat = col2.number_input("NE lat", step=0.5, key="ne_lat")
        mission = st.radio("Mission phase", ["calval", "science"], horizontal=True, key="mission")
        sph_calval_path  = st.text_input("Calval orbit file (.zip)", key="sph_calval_path")
        sph_science_path = st.text_input("Science orbit file (.zip)", key="sph_science_path")

    # --- Features ---
    st.subheader("Features")
    selected_features = st.multiselect(
        "Model input features",
        options=AVAILABLE_FEATURES,
        default=base.features,
    )

    # --- Model selection + hyperparameters ---
    st.subheader("Model")
    _MODEL_LABELS = {"rf": "Random Forest", "ann": "Neural Network (MLP)"}
    model_kind = st.radio(
        "Model type",
        options=list(_MODEL_LABELS),
        format_func=_MODEL_LABELS.get,
        horizontal=True,
        key="model_kind",
    )
    stencil_k = st.select_slider("stencil k (must be odd)", options=[1, 3, 5, 7], key="stencil_k")

    # Defaults so both branches always define every config field
    rf_n_estimators = st.session_state["rf_n_estimators"]
    rf_max_depth    = st.session_state["rf_max_depth"]
    rf_use_gpu, rf_use_lgbm = False, False
    ann_hidden_layers = _parse_hidden_layers(st.session_state["ann_hidden"])
    ann_activation  = st.session_state["ann_activation"]
    ann_dropout     = st.session_state["ann_dropout"]
    ann_lr          = st.session_state["ann_lr"]
    ann_batch_size  = st.session_state["ann_batch_size"]
    ann_max_epochs  = st.session_state["ann_max_epochs"]
    ann_patience    = st.session_state["ann_patience"]

    if model_kind == "rf":
        rf_n_estimators = st.slider("n_estimators", 10, 500, step=10, key="rf_n_estimators")
        rf_max_depth    = st.slider("max_depth", 3, 50, key="rf_max_depth")
        try:
            import cuml  # noqa: F401
            _cuml_available = True
        except ImportError:
            _cuml_available = False
        rf_use_gpu = st.checkbox(
            "Use GPU (cuML / RAPIDS)",
            value=_cuml_available,
            help="Auto-detected based on whether cuML is installed.",
        )
        try:
            import lightgbm as lgb  # noqa: F401
            _lgbm_available = True
        except ImportError:
            _lgbm_available = False
        rf_use_lgbm = st.checkbox(
            "Use LightGBM GPU (fallback if cuML unavailable)",
            value=(not _cuml_available and _lgbm_available),
            help="Uses LightGBM with device='gpu'. Ignored if Use GPU (cuML) is also checked — LightGBM takes priority when enabled.",
            disabled=not _lgbm_available,
        )
    else:
        try:
            import torch
            _torch_dev = "CUDA GPU" if torch.cuda.is_available() else "CPU"
            st.caption(f"PyTorch {torch.__version__} — training on **{_torch_dev}**")
        except ImportError:
            st.error("PyTorch is not installed. Run: `pip install -e .[ann]`")
        _hidden_str = st.text_input(
            "Hidden layers (comma-separated)",
            key="ann_hidden",
            help="Widths of the MLP hidden layers, e.g. 256,256,128 (brackets/spaces are fine too)",
        )
        ann_hidden_layers = _parse_hidden_layers(_hidden_str)
        if not ann_hidden_layers:
            st.warning("Hidden layers must be positive integers, e.g. 256,256,128")
        ann_activation = st.selectbox(
            "Activation function",
            options=["silu", "relu", "gelu", "tanh"],
            key="ann_activation",
            help="Nonlinearity between hidden layers. silu (default) and gelu are smooth "
                 "modern choices; relu is the classic; tanh saturates and is rarely best.",
        )
        ann_dropout    = st.slider("Dropout", 0.0, 0.5, step=0.05, key="ann_dropout")
        ann_lr         = st.select_slider(
            "Learning rate",
            options=[1e-4, 3e-4, 1e-3, 3e-3, 1e-2],
            key="ann_lr",
            format_func=lambda v: f"{v:g}",
        )
        ann_batch_size = st.select_slider(
            "Batch size", options=[1024, 2048, 4096, 8192, 16384], key="ann_batch_size",
        )
        ann_max_epochs = st.slider("Max epochs", 10, 500, step=10, key="ann_max_epochs")
        ann_patience   = st.slider("Early-stopping patience", 5, 50, step=5, key="ann_patience")

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
    _anim_base   = (f"animations/{_anim_region}/{model_kind}" if _anim_region
                    else f"animations/{model_kind}")
    anim_name    = st.text_input(
        "Animation name — optional",
        key="anim_name",
        help="Stem for output files, saved as animations/[region/]{model}/{name}_pass_N.mp4. "
             "Leave blank to name outputs after the run's unique experiment ID.",
    )
    anim_output  = f"{_anim_base}/{anim_name}" if anim_name else ""
    if anim_name:
        st.caption(f"Animations → `{_anim_base}/{anim_name}_pass_N.mp4`")
    else:
        st.caption(f"Animations → `{_anim_base}/<experiment_id>_pass_N.mp4`")
    fps = st.slider("FPS", 1, 30, key="fps")
    dpi = st.slider("DPI", 72, 300, step=10, key="dpi")

    # --- Caching ---
    st.subheader("Caching")
    run_id = st.text_input("Run ID — optional", key="run_id",
                           help="Short name (letters, digits, _ or - only) that keys cached weights, "
                                "so later runs with the same ID reuse them. Leave blank to key "
                                "everything by the run's unique experiment ID (each run trains fresh).")
    if using_preset:
        cache_dir = base.cache_dir
        _region = pcfg.get("region", "")
        st.caption(
            f"Model outputs cached to `experiments/{_region}/{model_kind}/` "
            f"(weights, inference — reused when Run ID matches). Shared data "
            f"(flattened, cycles, HFR) lives at `…/{_region}/` and is reused by both models."
        )
    else:
        cache_dir = st.text_input("Cache dir", key="cache_dir",
                                  help="Relative to repo root. Intermediate results are pickled here so slow steps don't re-run.")
        st.caption(f"Cache will be saved to: `{Path(cache_dir).resolve()}`")
    use_cache = st.checkbox("Use cached steps", value=True,
                            help="Uncheck to force all steps to re-run from scratch.")

    try:
        current_config = SWOTConfig(
            swot_path=swot_path, hfr_path=hfr_path, era5_pkl_path=era5_pkl_path,
            goes_nc_path=goes_nc_path or None,
            sw_corner=[sw_lon, sw_lat], ne_corner=[ne_lon, ne_lat],
            mission=mission,
            sph_calval_path=sph_calval_path, sph_science_path=sph_science_path,
            model=model_kind,
            features=selected_features,
            stencil_k=stencil_k, random_state=42,
            rf_n_estimators=rf_n_estimators, rf_max_depth=rf_max_depth,
            rf_use_gpu=rf_use_gpu, rf_use_lgbm=rf_use_lgbm,
            ann_hidden_layers=ann_hidden_layers, ann_activation=ann_activation,
            ann_dropout=ann_dropout,
            ann_lr=ann_lr, ann_batch_size=int(ann_batch_size),
            ann_max_epochs=int(ann_max_epochs), ann_patience=int(ann_patience),
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

    required_ok = bool(using_preset or (swot_path and hfr_path))
    if not required_ok:
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
    st.session_state.ann_epochs          = []
    st.session_state.step_status         = {s: "pending" for s in STEPS}
    st.session_state.pipeline_start_time = time.time()
    st.session_state.pipeline_end_time   = None

    q = queue.Queue()
    st.session_state.msg_queue = q

    _multi_spec = ({"mission": pcfg["mission"], "regions": pcfg["regions"]}
                   if pcfg.get("multi") else None)

    def _run(cfg, use_cache_, q_, stop_, multi_=_multi_spec):
        def _progress(step_name, frac, msg):
            if stop_.is_set():
                raise RuntimeError("Pipeline stopped by user.")
            q_.put({"step": step_name, "frac": frac, "msg": msg})

        try:
            if multi_:
                from swotxai.multiregion import run_multiregion
                results = run_multiregion(
                    cfg, mission=multi_["mission"], regions=multi_["regions"],
                    progress_cb=_progress, use_cache=use_cache_,
                )
            else:
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
tab_pipeline, tab_results, tab_animation, tab_batch, tab_experiments = st.tabs(
    ["Pipeline", "Results", "Animation", "Batch", "Experiments"]
)

# ---- Tab 1: Pipeline progress ----
with tab_pipeline:
    _pipe_run_every = 1 if st.session_state.running else None
    @st.fragment(run_every=_pipe_run_every)
    def _pipeline_progress():
        q = st.session_state.get("msg_queue")
        if q:
            while not q.empty():
                item = q.get_nowait()
                sn, frac, msg = item["step"], item["frac"], item["msg"]
                if sn == "train_epoch":
                    import json as _json
                    try:
                        st.session_state.ann_epochs.append(_json.loads(msg))
                    except Exception:
                        pass
                elif sn == "__done__":
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

        statuses = st.session_state.step_status
        n_done   = sum(1 for s in statuses.values() if s == "done")
        n_total  = len(STEPS)
        cur_step = st.session_state.current_step
        cur_frac = st.session_state.current_step_frac
        cur_msg  = st.session_state.current_step_msg

        pipeline_done = not st.session_state.running and n_done == n_total and n_total > 0
        overall = min(1.0, (n_done + cur_frac) / n_total if st.session_state.running else n_done / n_total)
        st.progress(overall, text="Finished ✅" if pipeline_done else f"{n_done} / {n_total} steps complete")

        if st.session_state.running and cur_step:
            st.markdown(f"**{cur_step.replace('_', ' ')}**")
            st.progress(cur_frac, text=cur_msg or "Working…")

        # Live ANN training monitor — one row per epoch, streamed in real time
        epochs = st.session_state.get("ann_epochs") or []
        if epochs:
            last = epochs[-1]
            st.markdown("**Training monitor**")
            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("Epoch", f"{last['epoch']} / {last['max_epochs']}")
            c2.metric("Train loss", f"{last['train_loss']:.5f}")
            c3.metric("Val loss", f"{last['val_loss']:.5f}")
            c4.metric("Best val", f"{last['best_val_loss']:.5f}",
                      delta=f"@ epoch {last['best_epoch']}", delta_color="off")
            _tm, _ts = divmod(int(last["elapsed_s"]), 60)
            c5.metric("Train time", f"{_tm:02d}:{_ts:02d}",
                      delta=f"{last['epoch_s']:.1f} s/epoch", delta_color="off")

            loss_df = pd.DataFrame(epochs).set_index("epoch")[["train_loss", "val_loss"]]
            loss_df.columns = ["train", "val"]
            st.line_chart(loss_df, color=[CHART_U, CHART_V], height=240)

            with st.expander(f"Epoch log ({len(epochs)} epochs)", expanded=False):
                st.dataframe(
                    pd.DataFrame(epochs)[
                        ["epoch", "train_loss", "val_loss", "lr", "epoch_s", "elapsed_s"]
                    ].rename(columns={
                        "train_loss": "train loss", "val_loss": "val loss",
                        "epoch_s": "epoch (s)", "elapsed_s": "elapsed (s)",
                    }),
                    width="stretch", hide_index=True, height=240,
                )

        t_start = st.session_state.pipeline_start_time
        t_end   = st.session_state.pipeline_end_time
        if t_start is not None and (st.session_state.running or t_end is not None):
            elapsed = (time.time() if st.session_state.running else t_end) - t_start
            m, s = divmod(int(elapsed), 60)
            h, m = divmod(m, 60)
            timer_str = f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"
            st.metric("Elapsed" if st.session_state.running else "Total time", timer_str)

        st.divider()

        cols = st.columns(n_total)
        for col, step_name in zip(cols, STEPS):
            status = statuses.get(step_name, "pending")
            col.markdown(_step_chip(step_name, status), unsafe_allow_html=True)

        st.divider()
        st.code(
            "\n".join(st.session_state.log[-100:]) if st.session_state.log else "Waiting for pipeline to start...",
            language=None,
        )

    _pipeline_progress()

# ---- Tab 2: Results ----
with tab_results:
    metrics = st.session_state.results.get("metrics")

    if metrics is None:
        st.info("Run the pipeline to see results.")
    else:
        experiment = st.session_state.results.get("experiment")
        if experiment:
            st.caption(f"Experiment recorded as `{experiment['experiment_id']}` — see the Experiments tab.")
        st.subheader("Model metrics")
        col_u, col_v = st.columns(2)
        col_u.metric("RMSE  u", f"{metrics['rmse_u']:.4f} m/s")
        col_u.metric("R²  u",   f"{metrics['r2_u']:.4f}")
        col_v.metric("RMSE  v", f"{metrics['rmse_v']:.4f} m/s")
        col_v.metric("R²  v",   f"{metrics['r2_v']:.4f}")

        per_region = metrics.get("per_region")
        if per_region:
            st.subheader("Per-region breakdown")
            st.caption("One pooled model, scored on each region's own passes "
                       "(same evaluation convention as single-region runs).")
            st.dataframe(
                [
                    {"region": rid,
                     "RMSE u": round(m.get("rmse_u", float("nan")), 4),
                     "R² u":   round(m.get("r2_u",  float("nan")), 4),
                     "RMSE v": round(m.get("rmse_v", float("nan")), 4),
                     "R² v":   round(m.get("r2_v",  float("nan")), 4)}
                    for rid, m in per_region.items()
                ],
                hide_index=True, width='stretch',
            )

        st.subheader("Feature importances")
        st.caption(
            "Permutation importance (increase in MSE when a feature's stencil block is shuffled)"
            if model_kind == "ann" else
            "Impurity-based importance from the random forest"
        )
        fi_u = metrics["feature_importance_u"]
        fi_v = metrics["feature_importance_v"]
        features = list(fi_u.keys())

        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        fig.patch.set_alpha(0)
        for ax, fi, label, color in [(axes[0], fi_u, "u-velocity", CHART_U),
                                     (axes[1], fi_v, "v-velocity", CHART_V)]:
            vals = [fi[f] for f in features]
            idx  = np.argsort(vals)[::-1]
            ax.set_facecolor("none")
            ax.barh([features[i] for i in idx], [vals[i] for i in idx],
                    color=color, height=0.62, zorder=3)
            ax.set_xlabel("Importance", color=INK_MUTED, fontsize=9)
            ax.set_title(f"Feature importances — {label}", color=INK,
                         fontsize=11, loc="left", pad=10)
            ax.invert_yaxis()
            ax.tick_params(colors=INK_MUTED, labelsize=9)
            for side in ("top", "right", "left"):
                ax.spines[side].set_visible(False)
            ax.spines["bottom"].set_color(GRID_LINE)
            ax.grid(axis="x", color=GRID_LINE, linewidth=0.6, zorder=0)
            ax.set_axisbelow(True)
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

# ---- Tab 3: Animation ----
with tab_animation:
    anim_paths = st.session_state.results.get("animation_paths") or []
    # Prefer the completed run's frame dir (named after its experiment ID when
    # no animation name was given); fall back to the configured dir.
    _frame_dir = (st.session_state.results.get("frame_dir")
                  or (current_config.frame_dir if config_valid else ""))
    frame_dir_path = Path(_frame_dir) if _frame_dir else Path("frames")

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

    # ---- 3D SSH surface ----
    st.divider()
    st.subheader("3D sea surface")

    @st.cache_resource(show_spinner="Loading cached inference data...")
    def _load_inference_dicts(path_str: str, _mtime: float):
        import pickle
        with open(path_str, "rb") as f:
            return pickle.load(f)

    _swot_dict = st.session_state.results.get("swot_dict")
    _hfr_dict  = st.session_state.results.get("hfr_dict")
    if _swot_dict is None and config_valid:
        _inf_path = current_config.cache_path("inference")
        if _inf_path.exists():
            if st.toggle(
                "Load cached inference data", key="s3d_load",
                help="No completed run in this session — load the configured "
                     "run's cached predictions from disk instead.",
            ):
                _swot_dict, _hfr_dict = _load_inference_dicts(
                    str(_inf_path), _inf_path.stat().st_mtime
                )

    if _swot_dict is None:
        st.info("Run the pipeline (or load a cached run above) to view the sea surface in 3D.")
    else:
        from swotxai.viz3d import ssh_surface_figure, surface_elevation_var

        _s3d_pairs = sorted(
            (int(cyc), j)
            for cyc, ds_list in _swot_dict.items()
            for j, ds in enumerate(ds_list or [])
            if ds is not None and surface_elevation_var(ds) is not None
        )
        if not _s3d_pairs:
            st.info("No cycle in this run carries an SSH field to draw.")
        else:
            st.caption(
                "Sea-surface height as elevation (relief exaggerated), colored by "
                "surface velocity — rotate to see geostrophic flow wrapping the "
                "highs and lows. Compare the model drape against HFR ground truth."
            )
            c_cyc, c_pass, c_drape = st.columns([1.5, 1, 3.5])
            _s3d_cycle = c_cyc.selectbox(
                "Cycle", sorted({c for c, _ in _s3d_pairs}), key="s3d_cycle",
            )
            _s3d_passes = [j for c, j in _s3d_pairs if c == _s3d_cycle]
            _s3d_pass = c_pass.radio("Pass", _s3d_passes, horizontal=True, key="s3d_pass")

            _swot_ds = _swot_dict[str(_s3d_cycle)][_s3d_pass]
            _hfr_list = (_hfr_dict or {}).get(str(_s3d_cycle), [])
            _hfr_ds = _hfr_list[_s3d_pass] if _s3d_pass < len(_hfr_list) else None

            _drapes = {}
            if "ssv_pred" in _swot_ds:
                _drapes["Model prediction"] = (_swot_ds["ssv_pred"], 0.3)
            if _hfr_ds is not None and "ssv" in _hfr_ds:
                _drapes["HFR ground truth"] = (_hfr_ds["ssv"], 0.3)
            if "gos_filtered" in _swot_ds:
                _drapes["SWOT geostrophic"] = (_swot_ds["gos_filtered"], 2.0)

            if not _drapes:
                st.info("This cycle has no velocity field to drape on the surface.")
            else:
                _drape_name = c_drape.radio(
                    "Color by", list(_drapes), horizontal=True, key="s3d_drape",
                )
                _drape_da, _drape_cmax = _drapes[_drape_name]
                _s3d_fig = ssh_surface_figure(
                    _swot_ds, _drape_da, _drape_name,
                    cmax=_drape_cmax, theme=PLOTLY_THEME,
                )
                st.plotly_chart(_s3d_fig, width="stretch", config=PLOTLY_CONFIG)

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

        def _job_progress_cb(run_id: str, step: str, frac: float, msg: str) -> None:
            if step == "train_epoch":
                q_.put({"type": "job_epoch", "run_id": run_id, "msg": msg})
            else:
                q_.put({"type": "job_step", "run_id": run_id,
                        "step": step, "frac": frac, "msg": msg})

        try:
            all_results = run_batch(
                batch_cfg,
                progress_cb=_shared_cb,
                job_status_cb=_job_status_cb,
                job_progress_cb=_job_progress_cb,
                use_cache=use_cache_,
            )
            q_.put({"type": "done", "results": all_results})
        except Exception as exc:
            import traceback
            q_.put({"type": "error", "msg": traceback.format_exc()})
        finally:
            q_.put({"type": "__done__"})

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
        "Each row is one experiment. **＋ Add Job snapshots the current sidebar configuration** "
        "(model, features, stencil, hyperparameters) into a new row — tweak cells afterwards to "
        "vary jobs. All jobs share the data source and domain configured in the sidebar."
    )

    # Add / Delete Selected buttons (outside fragment so they can trigger full reruns)
    _EDITOR_KEY = "batch_job_editor"
    col_add, col_del, _ = st.columns([1, 1.5, 5.5])
    if col_add.button("＋ Add Job", disabled=st.session_state.batch_running,
                      help="Snapshot the sidebar's current model + hyperparameters as a new job row."):
        # Use batch_jobs_edited (latest with user edits) as the base, then append new row
        current = st.session_state.batch_jobs_edited or st.session_state.batch_jobs
        if not isinstance(current, list):
            try:
                current = list(current.to_dict("records"))
            except Exception:
                current = []
        existing_ids = {(j.get("run_id") or "").strip() for j in current}
        idx_new = len(current) + 1
        while f"job_{idx_new:02d}" in existing_ids:
            idx_new += 1
        new_base = list(current) + [{
            "_delete":        False,
            "run_id":         f"job_{idx_new:02d}",
            # snapshot of the sidebar configuration, exactly as currently set
            "model":          model_kind,
            "features":       ",".join(selected_features or AVAILABLE_FEATURES),
            "stencil_k":      int(stencil_k),
            "random_state":   42,
            "n_estimators":   int(rf_n_estimators),
            "max_depth":      int(rf_max_depth),
            "ann_hidden":     ",".join(str(h) for h in (ann_hidden_layers or [256, 256, 128])),
            "ann_activation": ann_activation,
            "ann_dropout":    float(ann_dropout),
            "ann_lr":         float(ann_lr),
            "ann_max_epochs": int(ann_max_epochs),
        }]
        st.session_state.batch_jobs = new_base
        st.session_state.batch_jobs_edited = new_base
        st.session_state.pop(_EDITOR_KEY, None)
        st.rerun()
    if col_del.button("🗑 Delete Selected", disabled=st.session_state.batch_running):
        current = st.session_state.batch_jobs_edited or st.session_state.batch_jobs
        if not isinstance(current, list):
            try:
                current = list(current.to_dict("records"))
            except Exception:
                current = []
        new_base = [j for j in current if not j.get("_delete", False)]
        st.session_state.batch_jobs = new_base
        st.session_state.batch_jobs_edited = new_base
        st.session_state.pop(_EDITOR_KEY, None)
        st.rerun()

    # Isolated fragment: the data_editor lives here so no external fragment rerun can disturb it
    @st.fragment
    def _jobs_editor():
        # batch_jobs is the STABLE base — never overwritten from inside this fragment.
        # Edits are accumulated in batch_jobs_edited so the base data never changes
        # between fragment reruns, which prevents the data_editor from resetting its
        # diff state and reverting user inputs.
        base = st.session_state.batch_jobs
        if not isinstance(base, list):
            try:
                base = list(base.to_dict("records"))
            except Exception:
                base = []
            st.session_state.batch_jobs = base

        if not base:
            st.info("No jobs yet — configure the sidebar (model, features, hyperparameters), "
                    "then click ＋ Add Job to snapshot it as a job row.")
            return

        edited = st.data_editor(
            base,
            num_rows="fixed",
            width="stretch",
            disabled=st.session_state.batch_running,
            key=_EDITOR_KEY,
            column_config={
                "_delete":        st.column_config.CheckboxColumn("Delete", default=False),
                "run_id":         st.column_config.TextColumn("Run ID", help="Unique name for this job"),
                "model":          st.column_config.SelectboxColumn("Model", options=["rf", "ann"],
                                                                   help="rf = random forest, ann = neural network (MLP)"),
                "features":       st.column_config.TextColumn("Features", help="Comma-separated list"),
                "stencil_k":      st.column_config.SelectboxColumn("Stencil K", options=[1, 3, 5, 7]),
                "random_state":   st.column_config.NumberColumn("Random State", min_value=0),
                "n_estimators":   st.column_config.NumberColumn("N Estimators (RF)", min_value=10, max_value=500, step=10),
                "max_depth":      st.column_config.NumberColumn("Max Depth (RF)", min_value=3, max_value=50),
                "ann_hidden":     st.column_config.TextColumn("Hidden Layers (ANN)", help="e.g. 256,256,128"),
                "ann_activation": st.column_config.SelectboxColumn("Activation (ANN)",
                                                                   options=["silu", "relu", "gelu", "tanh"]),
                "ann_dropout":    st.column_config.NumberColumn("Dropout (ANN)", min_value=0.0, max_value=0.5, step=0.05),
                "ann_lr":         st.column_config.NumberColumn("LR (ANN)", min_value=0.00001, max_value=0.1,
                                                                step=0.0001, format="%.4f"),
                "ann_max_epochs": st.column_config.NumberColumn("Max Epochs (ANN)", min_value=10, max_value=500, step=10),
            },
        )
        if edited is not None:
            st.session_state.batch_jobs_edited = (
                edited.to_dict("records") if hasattr(edited, "to_dict") else list(edited)
            )

    _jobs_editor()

    # Validate only non-deleted rows (use batch_jobs_edited which reflects live editor state)
    import re as _re
    _live_jobs = st.session_state.batch_jobs_edited or st.session_state.batch_jobs
    active_jobs = [j for j in _live_jobs if not j.get("_delete", False)]
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
        _jobs_to_run = st.session_state.batch_jobs_edited or st.session_state.batch_jobs
        for j in _jobs_to_run:
            if j.get("_delete", False):
                continue
            raw_features = [f.strip() for f in (j.get("features") or "").split(",") if f.strip()]
            valid_features = [f for f in raw_features if f in AVAILABLE_FEATURES]
            row_hidden = _parse_hidden_layers(j.get("ann_hidden") or "")
            job_specs.append(JobSpec(
                run_id            = (j.get("run_id") or "").strip(),
                model             = (j.get("model") or "rf").strip(),
                features          = valid_features or list(AVAILABLE_FEATURES),
                stencil_k         = int(j.get("stencil_k") or 3),
                rf_n_estimators   = int(j.get("n_estimators") or 50),
                rf_max_depth      = int(j.get("max_depth") or 15),
                random_state      = int(j.get("random_state") or 42),
                # per-row ANN hyperparams; sidebar values as fallback for old rows
                ann_hidden_layers = row_hidden or ann_hidden_layers or [256, 256, 128],
                ann_activation    = (j.get("ann_activation") or ann_activation),
                ann_dropout       = float(j["ann_dropout"] if j.get("ann_dropout") is not None else ann_dropout),
                ann_lr            = float(j.get("ann_lr") or ann_lr),
                ann_max_epochs    = int(j.get("ann_max_epochs") or ann_max_epochs),
            ))

        batch_cfg_obj = BatchConfig(
            jobs             = job_specs,
            max_workers      = batch_max_workers,
            swot_path        = swot_path,
            hfr_path         = hfr_path,
            era5_pkl_path    = era5_pkl_path,
            goes_nc_path     = goes_nc_path or None,
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
        st.session_state.batch_error             = None
        st.session_state.batch_job_statuses      = {j.run_id: "pending" for j in job_specs}
        st.session_state.batch_job_results       = {}
        st.session_state.batch_job_logs          = {}
        st.session_state.batch_job_epochs        = {}
        st.session_state.batch_job_steps         = {}
        st.session_state.batch_shared_step_status = {s: "pending" for s in SHARED_STEPS}

        bq_new = queue.Queue()
        st.session_state.batch_msg_queue = bq_new
        threading.Thread(
            target=_run_batch_thread,
            args=(batch_cfg_obj, use_cache, bq_new),
            daemon=True,
        ).start()
        st.rerun()  # re-render immediately so the button grays out

    st.divider()

    # ---------------------------------------------------------------------------
    # Section 4-6 — Progress (only auto-reruns when the batch is actually running)
    # ---------------------------------------------------------------------------
    if st.session_state.batch_error:
        st.error(f"Batch error: {st.session_state.batch_error}")

    _batch_run_every = 1 if st.session_state.batch_running else None
    @st.fragment(run_every=_batch_run_every)
    def _batch_progress():
        bq = st.session_state.get("batch_msg_queue")
        if bq:
            while not bq.empty():
                item = bq.get_nowait()
                t = item["type"]
                if t == "__done__":
                    st.session_state.batch_running  = False
                    st.session_state.batch_end_time = time.time()
                    st.rerun()  # full rerun so job editor re-enables and run button un-disables
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
                elif t == "job_epoch":
                    import json as _json
                    try:
                        st.session_state.setdefault("batch_job_epochs", {}).setdefault(
                            item["run_id"], []).append(_json.loads(item["msg"]))
                    except Exception:
                        pass
                elif t == "job_step":
                    st.session_state.setdefault("batch_job_steps", {})[item["run_id"]] = (
                        item["step"], item["frac"], item["msg"]
                    )
                elif t == "error":
                    st.session_state.batch_running = False
                    st.session_state.batch_error   = item["msg"]

        if st.session_state.batch_start_time is None:
            return

        st.subheader("Shared Data Steps")
        sh_cols = st.columns(len(SHARED_STEPS))
        for col, sname in zip(sh_cols, SHARED_STEPS):
            status = st.session_state.batch_shared_step_status.get(sname, "pending")
            col.markdown(_step_chip(sname, status), unsafe_allow_html=True)

        bt_start = st.session_state.batch_start_time
        bt_end   = st.session_state.batch_end_time
        if bt_start:
            b_elapsed = (time.time() if st.session_state.batch_running else bt_end) - bt_start
            bm, bs = divmod(int(b_elapsed), 60)
            bh, bm = divmod(bm, 60)
            btimer = f"{bh:02d}:{bm:02d}:{bs:02d}" if bh else f"{bm:02d}:{bs:02d}"
            st.metric("Elapsed" if st.session_state.batch_running else "Total Time", btimer)

        st.divider()
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

        _display_jobs = st.session_state.batch_jobs_edited or st.session_state.batch_jobs
        for j in _display_jobs:
            rid    = j.get("run_id", "")
            status = job_statuses.get(rid, "pending")
            res    = job_results.get(rid, {})
            label  = f"{status_icons.get(status, status)}  —  {rid}"
            with st.expander(label, expanded=(status == "failed")):
                if j.get("model") == "ann":
                    _hp = (f"Hidden: {j.get('ann_hidden')} | Activation: {j.get('ann_activation')} | "
                           f"Dropout: {j.get('ann_dropout')} | LR: {j.get('ann_lr')} | "
                           f"Max Epochs: {j.get('ann_max_epochs')}")
                else:
                    _hp = f"N Estimators: {j.get('n_estimators')} | Max Depth: {j.get('max_depth')}"
                st.caption(
                    f"Model: {j.get('model', 'rf')} | "
                    f"Features: `{j.get('features', '')}` | "
                    f"Stencil K: {j.get('stencil_k')} | {_hp}"
                )

                # Live per-job progress: current step, then the training stream
                if status == "running":
                    _jstep, _jfrac, _jmsg = st.session_state.get(
                        "batch_job_steps", {}).get(rid, (None, 0.0, ""))
                    if _jstep:
                        st.progress(min(1.0, max(0.0, _jfrac)),
                                    text=f"{_jstep}: {_jmsg[:90]}")
                _jep = st.session_state.get("batch_job_epochs", {}).get(rid) or []
                if _jep:
                    _jlast = _jep[-1]
                    je1, je2, je3, je4 = st.columns(4)
                    je1.metric("Epoch", f"{_jlast['epoch']} / {_jlast['max_epochs']}")
                    je2.metric("Train loss", f"{_jlast['train_loss']:.5f}")
                    je3.metric("Val loss", f"{_jlast['val_loss']:.5f}")
                    je4.metric("Best val", f"{_jlast['best_val_loss']:.5f}",
                               delta=f"@ epoch {_jlast['best_epoch']}", delta_color="off")
                    _jloss = pd.DataFrame(_jep).set_index("epoch")[["train_loss", "val_loss"]]
                    _jloss.columns = ["train", "val"]
                    st.line_chart(_jloss, color=[CHART_U, CHART_V], height=180)

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
        done_jobs = [(rid, r) for rid, r in job_results.items() if "error" not in r and "rmse_u" in r]
        if done_jobs:
            st.subheader("Results Comparison")
            rows = []
            _ref_jobs = st.session_state.batch_jobs_edited or st.session_state.batch_jobs
            job_map = {j.get("run_id"): j for j in _ref_jobs}
            for rid, r in done_jobs:
                j = job_map.get(rid, {})
                if j.get("model") == "ann":
                    hp_str = (f"hidden={j.get('ann_hidden')} act={j.get('ann_activation')} "
                              f"drop={j.get('ann_dropout')} lr={j.get('ann_lr')}")
                else:
                    hp_str = f"trees={j.get('n_estimators')} depth={j.get('max_depth')}"
                rows.append({
                    "Run ID":       rid,
                    "Model":        j.get("model", "rf"),
                    "Features":     j.get("features", ""),
                    "Stencil K":    j.get("stencil_k", ""),
                    "Hyperparams":  hp_str,
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

    _batch_progress()

# ---- Tab 5: Experiments ----
with tab_experiments:
    from swotxai.experiments import load_experiments

    st.subheader("Experiment Registry")
    st.caption(
        "Every completed run is recorded automatically — full config, data inputs, "
        "features, stenciling, hyperparameters, metrics, and timings — under a unique ID. "
        "Registries are partitioned per model in "
        "`experiments/registry/{rf,ann}/`."
    )

    exp_model = st.radio("Show", ["all", "rf", "ann"], horizontal=True, key="exp_model_filter")
    records = load_experiments(None if exp_model == "all" else exp_model)

    if not records:
        st.info("No experiments recorded yet — run the pipeline to completion to create the first entry.")
    else:
        rows = []
        for r in records:
            m = r.get("metrics", {})
            hp = r.get("hyperparameters", {})
            if r.get("model") == "rf":
                hp_str = f"trees={hp.get('rf_n_estimators')} depth={hp.get('rf_max_depth')}"
            else:
                hidden = hp.get("ann_hidden_layers", [])
                hp_str = f"hidden={'-'.join(str(h) for h in hidden)} lr={hp.get('ann_lr')}"
            rows.append({
                "Experiment ID": r.get("experiment_id", ""),
                "Recorded":      r.get("recorded_at", ""),
                "Model":         r.get("model", ""),
                "Run ID":        r.get("run_id", ""),
                "Region":        (r.get("inputs") or {}).get("region") or "",
                "Mission":       (r.get("inputs") or {}).get("mission") or "",
                "Stencil K":     r.get("stencil_k", ""),
                "Features":      ",".join(r.get("features", [])),
                "Hyperparams":   hp_str,
                "RMSE u":        round(m["rmse_u"], 4) if "rmse_u" in m else None,
                "RMSE v":        round(m["rmse_v"], 4) if "rmse_v" in m else None,
                "R² u":          round(m["r2_u"], 4) if "r2_u" in m else None,
                "R² v":          round(m["r2_v"], 4) if "r2_v" in m else None,
            })
        exp_df = pd.DataFrame(rows)
        st.dataframe(exp_df, width="stretch", hide_index=True)
        st.download_button(
            "⬇ Download registry CSV",
            data=exp_df.to_csv(index=False).encode(),
            file_name="experiments.csv",
            mime="text/csv",
        )

        st.divider()
        st.subheader("What drives R²")
        from swotxai.experiments import (
            importance_bar_figure,
            importance_heatmap_figure,
            leaderboard_figure,
            scored_experiments,
        )

        n_scored = len(scored_experiments(records))
        if n_scored < 2:
            st.info(
                "The leaderboard needs at least two experiments with recorded R² "
                f"({n_scored} so far). Run more configurations — the Batch tab can "
                "sweep features and stencil sizes over the same data."
            )
        else:
            MAX_LEADERBOARD = 25
            st.caption(
                "Experiments ranked by mean R² — best on the left; hover a point "
                "for that run's full metadata. The matrix underneath shows which "
                "features each run used, plus its stencil size and backend: "
                "ingredients shared by the leftmost columns are what's driving "
                "R² up."
                + (f" Showing the top {MAX_LEADERBOARD} of {n_scored} scored runs."
                   if n_scored > MAX_LEADERBOARD else "")
            )
            lb_fig = leaderboard_figure(
                records, max_experiments=MAX_LEADERBOARD, theme=PLOTLY_THEME,
            )
            st.plotly_chart(lb_fig, width="stretch", config=PLOTLY_CONFIG)

            st.subheader("Feature importances across experiments")
            fi_component = st.radio(
                "Component", ["mean", "u", "v"], horizontal=True,
                key="exp_fi_component",
                help="Importances from the u model, the v model, or their mean.",
            )
            fi_fig = importance_heatmap_figure(
                records, component=fi_component,
                max_experiments=MAX_LEADERBOARD, theme=PLOTLY_THEME,
            )
            if fi_fig is None:
                st.info("None of the shown runs recorded feature importances.")
            else:
                st.caption(
                    "Columns follow the leaderboard ranking. Cells are each "
                    "feature's share of that run's total importance, so RF "
                    "(impurity) and ANN (permutation) runs are comparable in "
                    "pattern even though their raw units differ. Hover a cell "
                    "for exact u/v shares."
                )
                st.plotly_chart(fi_fig, width="stretch", config=PLOTLY_CONFIG)

            st.subheader("R² response surface")
            from swotxai.experiments import (
                available_response_params,
                response_heatmap_figure,
                response_surface_figure,
            )

            rs_params = available_response_params(records)
            if len(rs_params) < 2:
                st.info(
                    "Needs runs that vary at least two numeric knobs (stencil "
                    "size, feature count, trees, learning rate, ...). Sweep them "
                    "in the Batch tab to fill this in."
                )
            else:
                from swotxai.experiments import describe_param
                c_x, c_y = st.columns(2)
                rs_x = c_x.selectbox("X parameter", rs_params, index=0, key="rs_x",
                                     help=describe_param(st.session_state.get("rs_x", rs_params[0])))
                rs_y = c_y.selectbox(
                    "Y parameter", rs_params,
                    index=1 if len(rs_params) > 1 else 0, key="rs_y",
                    help=describe_param(st.session_state.get("rs_y", rs_params[min(1, len(rs_params) - 1)])),
                )
                with st.expander("What do these parameters mean?"):
                    for _p in rs_params:
                        st.markdown(f"- `{_p}` — {describe_param(_p)}")
                if rs_x == rs_y:
                    st.warning("Pick two different parameters.")
                else:
                    rs_hm = response_heatmap_figure(records, rs_x, rs_y, theme=PLOTLY_THEME)
                    if rs_hm is None:
                        st.info(
                            f"No run varies both `{rs_x}` and `{rs_y}` — sweep "
                            "them together in the Batch tab."
                        )
                    else:
                        st.caption(
                            "Mean R² (u/v averaged) over every run at each "
                            "parameter combination — empty cells are untested "
                            "combos. Averages hide whatever else differed "
                            "between runs, so this reads cleanest after a "
                            "controlled sweep."
                        )
                        st.plotly_chart(rs_hm, width="stretch", config=PLOTLY_CONFIG)
                        if st.toggle("3D scatter view", key="rs_3d", value=False):
                            rs_sf = response_surface_figure(
                                records, rs_x, rs_y, theme=PLOTLY_THEME,
                            )
                            st.plotly_chart(rs_sf, width="stretch", config=PLOTLY_CONFIG)

        st.divider()
        st.subheader("Experiment detail")
        sel = st.selectbox(
            "Experiment",
            options=[r["experiment_id"] for r in records],
            key="exp_detail_select",
        )
        detail = next((r for r in records if r["experiment_id"] == sel), None)
        if detail:
            bar_fig = importance_bar_figure(detail, theme=PLOTLY_THEME)
            if bar_fig is not None:
                st.caption(
                    "Raw feature importances for this run "
                    "(impurity for RF, permutation ΔMSE for ANN)."
                )
                st.plotly_chart(bar_fig, width="stretch", config=PLOTLY_CONFIG)
            st.json(detail, expanded=False)
