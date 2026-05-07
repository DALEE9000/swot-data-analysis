# SWOTxAI

Machine learning pipeline that combines **SWOT satellite altimetry** (sea surface height, geostrophic velocity) with optional **ERA5 winds** and **GOES SST** to infer subsurface velocity (SSV), validated against **HFR (high-frequency radar)** ground truth. Results are visualized through a multi-panel animation and an interactive Streamlit GUI.

---

## Installation

Requires Python ≥ 3.10.

```bash
pip install -e ".[dev]"
```

## Installation From the Columbia LEAP Jupyter Hub

```bash
cd swot-data-analysis
pip install -e .
pip install -r requirements-gpu.txt
```

## Installation From Vast.ai

Use the **NVIDIA RAPIDS** template when renting the instance. Connect with port forwarding so the Streamlit UI is accessible locally:

```bash
ssh -p <PORT> -i ~/.ssh/id_ed25519 -L 8501:localhost:8501 root@<IP>
```

Then on the instance:

```bash
git clone https://github.com/DALEE9000/swot-data-analysis.git
cd swot-data-analysis
pip install -e .
```

Install GPU dependencies:

```bash
pip install -r requirements-gpu.txt
```

Verify cuML is working:

```bash
python -c "from cuml.ensemble import RandomForestRegressor; print('OK')"
```

## Running the app

```bash
streamlit run app.py
```

## Running the app from a Vast.ai instance

```bash
streamlit run app.py --server.port 8501
```

Then open `localhost:8501` in your browser.

---

## File hierarchy

```
SWOT-data-analysis/
│
├── app.py                          # Streamlit GUI entry point (3 tabs: Pipeline, Results, Animation + Batch)
├── config.yaml                     # Template config — copy and edit for your run
├── pyproject.toml                  # Package metadata and dependencies
│
├── src/
│   ├── swotxai/                    # Main ML pipeline package
│   │   ├── pipeline.py             # 12-step pipeline orchestrator + step-level pickle cache
│   │   ├── config.py               # SWOTConfig dataclass; load_config / save_config (YAML)
│   │   ├── swotxai_utils.py        # Core ML logic: regrid, interp_to_swot, RF flatten/train/predict
│   │   ├── animation_utils.py      # Frame generation (generate_frames) and MP4 assembly
│   │   └── batch.py                # Batch mode — parallel hyperparameter sweep over shared data steps
│   │
│   └── swot/                       # Legacy SWOT utilities (used by examples/ notebooks)
│       ├── data_loaders.py         # SWOT L3 NetCDF loading and subsetting
│       ├── download_swaths.py      # Pass / swath selection from orbit shapefiles
│       ├── swot_utils.py           # General SWOT utility functions
│       ├── interp_utils.py         # Interpolation helpers
│       ├── plotting_scripts.py     # Cartopy-based plotting utilities
│       ├── download_VIIRS.py       # VIIRS SST download from THREDDS
│       ├── thredds.py              # THREDDS server access
│       └── download_swot_orbit.sh  # Shell script to fetch orbit shapefiles
│
├── orbit_data/                     # SWOT orbit shapefiles (not tracked)
│   ├── sph_calval_swath.zip        #   1-day repeat calibration/validation phase
│   └── sph_science_swath.zip       #   21-day science phase
│
├── cache/                          # Step-level pickle cache (not tracked)
│   └── <run_id>/                   #   one subdirectory per run_id
│       ├── cycle_data.pkl
│       ├── swot_regridded.pkl
│       ├── era5.pkl
│       ├── goes.pkl
│       ├── swot_features.pkl
│       ├── hfr_interp.pkl
│       ├── flattened.pkl
│       ├── rf_u.pkl / rf_v.pkl
│       ├── rf_meta.pkl
│       └── inference.pkl
│
├── frames/                         # PNG animation frames output
├── SWOTxAI/                        # Experiment outputs and scratch notebooks
├── scratch_notebooks/              # Development scratch scripts
├── examples/                       # Tutorial notebooks (not tracked)
└── swot_documentation/             # Reference documentation and notes
```

---

## Pipeline steps

The pipeline is a 12-step sequential chain. Each step pickles its output; re-running skips cached steps automatically.

| # | Step | Description |
|---|------|-------------|
| 1 | `load_swot` | Find passes over domain and load SWOT L3 NetCDF cycles |
| 2 | `regrid` | Interpolate swath data onto a regular lat/lon grid |
| 3 | `load_era5` | Load ERA5 surface wind (u, v) from local or S3 NetCDF |
| 4 | `load_goes` | Load GOES SST from a single `.nc` file or S3 path |
| 5 | `interp_sources` | Interpolate ERA5 winds and GOES SST onto the SWOT grid |
| 6 | `load_hfr` | Load HFR ground-truth velocity |
| 7 | `interp_hfr` | Interpolate HFR onto SWOT grid with 24-hour rolling mean |
| 8 | `flatten` | Build feature matrix with spatial stencil (k × k neighbourhood) |
| 9 | `train` | Fit `RandomForestRegressor` for u and v SSV components |
| 10 | `evaluate` | Compute RMSE and R² on held-out test set; feature importances |
| 11 | `inference` | Run RF predictions for all cycles; build `swot_dict` / `hfr_dict` |
| 12 | `animate` | Generate per-cycle PNG frames and assemble per-pass MP4s |

---

## Configuration

Key fields in `config.yaml`:

| Field | Description |
|-------|-------------|
| `swot_path` | S3 or local path to SWOT L3 NetCDF files |
| `hfr_path` | Path to HFR NetCDF with `u`, `v` velocity components |
| `era5_path` | *(optional)* Path to ERA5 NetCDF with surface winds |
| `goes_dir` | *(optional)* Path to a GOES SST `.nc` file or S3 URI |
| `sw_corner` / `ne_corner` | Bounding box `[lon, lat]` |
| `mission` | `"calval"` (1-day repeat, cycles 474–578) or `"science"` (21-day, cycles 1–16) |
| `features` | RF input features — any of `mdt, ssha_filtered, ugos_filtered, vgos_filtered, ugosa_filtered, vgosa_filtered, era5_u, era5_v, SST` |
| `stencil_k` | Spatial context window size (odd integer: 1, 3, 5, 7) |
| `run_id` | Unique name for this experiment's cache (letters/digits/`_`/`-`) |

---

## Region presets

Two pre-processed presets are available directly from the GUI (no local SWOT/HFR data needed):

- **US West Coast (calval)** — cycles 474–578, domain `[-127, 37.5]` → `[-123, 42.5]`
- **US East-Gulf Coast (calval)** — cycles 474–578, East Coast domain
