# SWOTxAI

Machine learning pipeline that combines **SWOT satellite altimetry** (sea surface height, geostrophic velocity) with optional **ERA5 winds** and **GOES SST** to infer subsurface velocity (SSV), validated against **HFR (high-frequency radar)** ground truth. Results are visualized through a multi-panel animation and an interactive Streamlit GUI.

---

## Local

Requires Python ≥ 3.10.

```bash
pip install -e ".[dev]"
streamlit run app.py
```

---

## Columbia LEAP JupyterHub

### Install

```bash
cd ~/swot-data-analysis
pip install -e .
conda install -c nvidia cuda-toolkit=12.9
pip install cuml-cu12 --extra-index-url=https://pypi.nvidia.com
```

> If cuML fails due to a CUDA driver mismatch, set `use_gpu: false` in `config.yaml` to fall back to sklearn on CPU.

### Run

```bash
streamlit run app.py --server.port 8501
```

Open in your browser:

```
https://leap.2i2c.cloud/user/<USERNAME>/proxy/8501/
```

---

## Vast.ai

### Setup

Use the **NVIDIA RAPIDS** template — cuML is pre-installed. Connect with port forwarding:

```bash
ssh -p <PORT> -i ~/.ssh/id_ed25519 -L 8501:localhost:8501 root@<IP>
```

Clone and install:

```bash
git clone https://github.com/DALEE9000/swot-data-analysis.git
cd swot-data-analysis
pip install -e .
```

Verify cuML:

```bash
python -c "from cuml.ensemble import RandomForestRegressor; print('OK')"
```

Check version and CUDA compatibility:

```bash
python -c "import cuml; print(cuml.__version__)"
python -c "import cuml; print(cuml.__file__)"   # should be cu12, not cu11
nvidia-smi --query-gpu=name,compute_cap --format=csv
```

If cuML is the CUDA 11 build but your system has CUDA 12 (check `nvidia-smi`), FIL inference will fail at prediction time with `All cuML FIL configurations failed`. Fix with:

```bash
conda install -c rapidsai -c conda-forge -c nvidia cuml=24.8 cuda-version=12.6 -y
```

Then delete the training cache and retrain — models pickled with the old cuML build are incompatible:

```bash
rm cache/<run_id>/rf_u.pkl cache/<run_id>/rf_v.pkl cache/<run_id>/rf_meta.pkl cache/<run_id>/evaluate.pkl cache/<run_id>/inference.pkl
```

If fixing cuML is not practical, use LightGBM GPU instead (works with CUDA 12 out of the box):

```yaml
# config.yaml
use_gpu: false
use_lgbm: true
```

### Run

Use `nohup` so training survives SSH disconnects:

```bash
nohup streamlit run app.py --server.port 8501 > nohup.out 2>&1 &
```

Open `localhost:8501` in your browser. Useful commands:

```bash
tail -f nohup.out                        # monitor logs
ps aux | grep streamlit | grep -v grep   # check if running
kill <PID>                               # stop by PID
pkill -f "streamlit run"                 # stop by name
```

> **Important:** Do not destroy your instance between sessions — just stop it. The SWOT/HFR cache in `cache/` is local and will be lost if the instance is destroyed.

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
├── experiments/                    # Per-region data mirrors, model weights, registry
│   ├── {region}/                   #   swot_cycles/ hfr_target/ era5/ goes/ (S3 mirrors)
│   │   └── {rf,ann}/               #   trained model weights + inference per backend
│   └── registry/{rf,ann}/          #   experiments.jsonl + experiments_summary.csv (tracked)
├── scripts/                        # Data campaign tooling (colocation, aggregation, ERA5/GOES)
├── notebooks/                      # Analysis notebooks
├── frames/                         # PNG animation frames output
├── animations/                     # Assembled MP4s
├── figures/                        # Region overview figures
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
| 3 | `load_era5` | Load ERA5 surface wind (u, v) from S3 pkl (fast) or NetCDF fallback |
| 4 | `load_goes` | Load GOES SST from S3 pkl |
| 5 | `interp_sources` | Interpolate ERA5 winds and GOES SST onto the SWOT grid |
| 6 | `load_hfr` | Load HFR ground-truth velocity |
| 7 | `interp_hfr` | Interpolate HFR onto SWOT grid with 25-hour rolling mean (detiding) |
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
| `era5_pkl_path` | *(optional)* S3 or local path to processed ERA5 pkl; auto-saved on first run from `era5_path` |
| `goes_nc_path` | *(optional)* S3 or local path to GOES SST NetCDF file |
| `sw_corner` / `ne_corner` | Bounding box `[lon, lat]` |
| `mission` | `"calval"` (1-day repeat, cycles 474–578) or `"science"` (21-day, cycles 1–16) |
| `features` | RF input features — any of `mdt, ssha_filtered, ugos_filtered, vgos_filtered, ugosa_filtered, vgosa_filtered, era5_u, era5_v, SST` |
| `stencil_k` | Spatial context window size (odd integer: 1, 3, 5, 7) |
| `run_id` | Unique name for this experiment's cache (letters/digits/`_`/`-`) |
| `sklearn_n_jobs` | CPU parallelism for sklearn RF (`-1` = all cores). Note: `n_jobs=-1` causes slight non-determinism run-to-run even with a fixed `random_state` due to thread scheduling. Set to `1` for fully reproducible results at the cost of speed. |

---

## Data requirements for model runs

See **[docs/model-data.md](docs/model-data.md)** for exactly which files each
model run needs (per region × mission), what's optional, local-vs-S3
resolution, sizes, and re-download commands.

## Region presets

Thirteen pre-processed presets (defined in `src/swotxai/presets.py`) are available directly from the GUI — no local SWOT/HFR data needed:

- **ALL regions — pooled (science / calval)** — trains ONE model on every colocated region for the mission (science: uswc, usegc, gak, glna, prvi, ushi; calval: uswc, usegc, gak, ushi). Per-region data is flattened separately then merged round-robin so the temporal 80/20 split covers every region; the Results tab shows pooled metrics plus a per-region breakdown. Inference/animation are skipped (single-grid concepts).
- **Seven "(science)" presets** — one per HFR network (see the region table below); cycles 1–16, full network footprints, colocation pkls streamed from S3 (or read from a local mirror when present)
- **Four "(calval)" presets** — uswc, usegc, gak, ushi; cycles 474–578, full network footprints (glna/prvi have no cal/val passes; akns has no ground truth in the window)

Every preset also carries the region's ERA5 10 m winds and (except akns) GOES SST paths — these load **only when** the matching features (`era5_u`/`era5_v`/`SST`) are selected, so leaving them out of the feature list costs nothing.

---

## Data on S3 (`s3://swot-ai-ssv`)

All project data lives in one bucket. **Reads are anonymous** (`s3fs.S3FileSystem(anon=True)`); writes require an authenticated `aws login`. Storage ≈ 120 GB; the cost driver is egress on repeated large reads, so keep local mirrors of anything you read often.

### Cost guards

Egress is the only real cost risk (~$0.09/GB out; storage is ~$5/month; uploads
and requests are ≈ free). A July 2026 colocation campaign learned this the hard
way: per-pass slab streaming from S3 totalled ~4 TB (~$290) — always mirror data
locally before compute jobs read it repeatedly (training presets already read
local mirrors under `experiments/…` first). Two guards are installed
on the account (539247449365):

1. **Budget alerts** — AWS Budget `monthly-cost-guard` ($20/month) emails at
   $10 (50%) and $20 (100%) actual spend.
2. **Circuit breaker** — a fourth alert on the same budget fires at **$10
   absolute** and publishes to SNS topic `budget-circuit-breaker` (us-east-1),
   which invokes Lambda `budget-circuit-breaker` (role `budget-breaker-lambda`,
   sole permission `s3:PutBucketPublicAccessBlock`). The Lambda enables all four
   **Block Public Access** switches on `swot-ai-ssv`, immediately cutting
   anonymous egress — the dominant cost vector — while authenticated access and
   local mirrors keep working.

   Caveats: billing data lags ~8–24 h and budgets evaluate a few times a day, so
   the breaker bounds damage to hours, not seconds — it is a damage limiter, not
   a hard cap. It does **not** auto-reset; after a trip, restore public reads
   with:

   ```bash
   aws s3api delete-public-access-block --bucket swot-ai-ssv
   ```

   The alert re-arms each calendar month. The Lambda's action was test-fired and
   verified 2026-07-17.

Related: `CLAUDE.md` carries a mandatory rule that the AI assistant must give a
dollar estimate and get explicit approval before any S3 operation.

### Layout

```
s3://swot-ai-ssv/
├── HFR/{region}/                          # full-network hourly HFR archives (NetCDF4)
├── SWOT_L3/
│   ├── calval/Expert_reproc_v3_{region}_calval/       # raw calval granules
│   └── science/Expert_reproc_v3_{region}_science/     # science granule segments,
│       └── cycle_001 … cycle_016/                     #   lat-trimmed per region
└── experiments/{region}/
    ├── swot_cycles/swot_expert_reproc_v3_{region}_science.pkl   # regridded SWOT pkls
    │                swot_expert_reproc_v3_calval_{region}_474_578.pkl
    ├── hfr_target/hfr_science_{region}[_{res}].pkl              # colocation targets
    │              hfr_calval_{region}.pkl
    ├── era5/era5wind_{mission}_{region}_10m.pkl    # ERA5 10 m winds, hourly,
    │                                               #   era5_u/era5_v (NCAR AWS mirror)
    └── goes/goes_sst_{mission}_{region}.nc         # GOES ABI-L2-SSTF at SWOT pass
                                                    #   hours, DQF<=1 (no akns: out of
                                                    #   geo view; glna winter SST noisy)
```

### Regions — the seven HFR networks (science phase, cycles 1–16 = Jul 2023 – Jun 2024)

| region | network | data footprint | SWOT passes | colocated | notes |
|--------|---------|----------------|-------------|-----------|-------|
| `uswc` | US West Coast | 31.7–49.4°N, 126.3–117.1°W | 430 | 430 | |
| `usegc` | US East + Gulf Coast | 23.3–43.9°N, 97.2–68.5°W | 890 | 890 | largest network |
| `gak` | Gulf of Alaska | 54.1–54.3°N (Prince Rupert site) | 104 | 104 | default target uses 2 km |
| `akns` | Alaska North Slope | 70.5–72.6°N | 551 | **0** | network dark since 2022 — no HFR in science window; SWOT pkl useful for inference only |
| `glna` | Great Lakes (Straits of Mackinac) | 45.8–45.9°N | 92 | 78 | SWOT-over-lakes unvalidated |
| `prvi` | Puerto Rico / USVI (CARICOOS) | 16.3–19.1°N | 151 | 140 | only tropical ground truth |
| `ushi` | Hawaii (Oahu) | 19.8–21.6°N | 162 | 146 | |

### HFR aggregates (`HFR/{region}/`)

Full-period hourly surface currents (`u`, `v`, float32) merged from raw HFRNet files.
Naming: `{region}_{res}_Resolution_hourly_{startYear}_{endYear}.nc4`.

- Resolutions: `uswc` 500m/1km/2km/6km · `usegc` 1km/2km/6km · `gak` 1km/2km/6km · `akns` 6km · `glna` 500m/1km/2km/6km · `prvi` 2km/6km · `ushi` 1km/2km/6km
- Domains that were **regridded historically ship as one file per grid era** (e.g. `gak_1km_..._2017_2019.nc4` + `..._2018_2025_grid1.nc4`); `usegc_1km` is split at the 2020-01-01 boundary (`2012_2019` + `2020_2025`)
- Time axes are strictly monotonic but **not gap-free** (network outages remain as gaps; `akns` is seasonal and effectively ends in 2022)

### SWOT science pkls (`experiments/{region}/swot_cycles/`)

Pickled `dict` mapping cycle key (`"001"`…`"016"`) → list of per-pass `xarray.Dataset`s regridded to a regular lat/lon grid. Float32; variables `mdt, ssha_filtered, ugos_filtered, vgos_filtered, ugosa_filtered, vgosa_filtered, quality_flag`; **every pass carries a `time` coordinate** (required for temporal colocation).

### HFR colocation targets (`experiments/{region}/hfr_target/`)

The supervised-learning targets: HFR `u`/`v` interpolated onto each pass's SWOT grid, temporally matched to the overflight.

- `hfr_science_{region}.pkl` — the default the GUI presets load (6 km source; `gak` uses 2 km)
- `hfr_science_{region}_{res}.pkl` — same colocation built from that resolution's archive
- `hfr_calval_{region}.pkl` — legacy cal/val-phase targets (`uswc`/`usegc` only)

**Temporal convention (all science targets):** 25-hour centered rolling mean — the standard HFR detiding filter (removes diurnal + semidiurnal tides) — evaluated at the hourly sample nearest each SWOT pass time. Passes with no HFR sample within ±36 h are dropped rather than matched to stale data. The rolling mean is computed *only at pass times* (windowed), never as a full smoothed time series — the raw hourly aggregates remain the archival source, and continuous rolled products can be derived on demand.

### Rebuild tooling (`scripts/`)

| script | purpose |
|--------|---------|
| `build_science_colocations.py` | end-to-end per region: AVISO granule download → SWOT pkl → HFR target → upload → granule sync (5 phases, resume-safe via manifest + per-cycle checkpoints) |
| `aggregate_hfr.py` | raw hourly HFRNet files → single full-period archives (multi-grid-era aware, memory-bounded streaming, crash-resumable) |
| `build_hfr_multires.py` | per-resolution colocation targets built from the local raw hourly archives |

AVISO source: `swot_products/l3_karin_nadir/l3_lr_ssh/v3_0/Expert/reproc` (SFTP; credentials in `src/swot/download_swaths.py`). Raw hourly HFR source files are kept locally under `HFR/Code/Data/*_Resolution_hourly/` (not on S3).
