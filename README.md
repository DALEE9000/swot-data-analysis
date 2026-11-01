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

## Replication

`docs/replication.md` walks three levels: regenerate the figures from tracked results (minutes), rebuild the data and retrain the champion from public sources (days), and rerun the LLM-guided search (not replayable; audit trail in `database.jsonl`). Exact package versions: `requirements-lock.txt`. Data archive manifest with SHA-256: section 2.7 there. A visual map of the whole chain, sources to figures, is `docs/pipeline-map.md`.

---

## File hierarchy

```
SWOT-data-analysis/
├── app.py                          # Streamlit GUI entry point
├── config.yaml                     # Template config — copy and edit for your run
├── pyproject.toml                  # Package metadata and dependencies
├── CLAUDE.md                       # Working rules for the coding agent (S3 cost rule, research log)
│
├── src/
│   ├── swotxai/                    # Main ML package
│   │   ├── config.py               # SWOTConfig dataclass; load_config / save_config (YAML)
│   │   ├── pipeline/               # 12-step cached pipeline: orchestrator, steps_data, steps_ml, steps_viz
│   │   ├── models/{rf,ann}/        # Backends: random forest, PyTorch MLP (weights + training code)
│   │   ├── evolve/                 # AlphaEvolve harness: loop, mutator, sandbox, candidate database
│   │   ├── data_utils.py           # Regrid, interpolation, stencil flattening
│   │   ├── animation_utils.py      # Frame generation and MP4 assembly
│   │   ├── presets.py / multiregion.py / batch.py / experiments.py
│   │   └── swotxai_utils.py        # Older RF helpers still imported by the pipeline
│   └── swot/                       # Legacy SWOT utilities; find_swaths() is still used by the pipeline
│
├── scripts/                        # Command-line entry points (index below)
├── tests/
│
├── docs/
│   ├── research-log.md             # Dated log of every change and finding (newest first)
│   ├── research-plan.md            # Phased plan with tick-boxes
│   ├── glossary.md  model-data.md  preprint-style-checklist.md
│   ├── cleanup-plan.md             # This layout and what moved where
│   ├── pipeline-map.md             # Mermaid map: sources -> mirrors -> rows -> model -> tables -> figures
│   ├── replication.md              # Three replication levels, commands, data-archive manifest
│   ├── updates/                    # Dated research-update memos
│   └── figures/
│       ├── README.md               # figure library index: every figure, what it shows, which script
│       ├── baselines/              # Phase-1 figure set + README describing each (scripts/baselines_figures.py)
│       ├── poster/                 # Fitness scatter, architecture, importances (scripts/poster_figures.py)
│       └── regions/                # Region-overview maps
│
├── papers/                         # PDF library + NOTES.md; handbooks/ holds the SWOT L3 product docs
├── orbit_data/                     # SWOT orbit shapefiles (sph_calval_swath.zip, sph_science_swath.zip)
├── ne_10m_bathymetry_all/          # Natural Earth bathymetry shapefiles for cartopy plots
│
├── experiments/                    # Data mirrors and runs (not tracked except evolve/ provenance + registry/)
│   ├── {region}/                   #   swot_cycles/ hfr_target/ era5/ goes/ duacs/ flattened/ {rf,ann}/
│   ├── evolve/{run}/               #   candidates/ solo/ baselines/ database.jsonl state.json logs
│   └── registry/{rf,ann}/          #   experiments.jsonl + experiments_summary.csv (tracked)
├── cache/                          # Step-level pickle cache (not tracked); delete a file to recompute a step
├── frames/  animations/            # PNG frames and MP4s per region/backend/run (not tracked)
└── archive/                        # Parked one-off output, scratch notebooks, old movies (not tracked)
```

### Scripts index

| Script | Does | Writes to |
|---|---|---|
| `evolve.py` | AlphaEvolve search over the ANN training code; resumable by `--name` | `experiments/evolve/{run}/` |
| `solo_vs_pooled.py` | Retrain a run's champion per region (solo) for comparison with the pooled model | `experiments/evolve/{run}/solo/` |
| `baselines_eval.py` | Phase-1 table: baseline ladder, holdout protocols, block bootstrap | `experiments/evolve/{run}/baselines/results.{json,md}` |
| `baselines_figures.py` | Phase-1 figure set (skill ladder, RMSE change, truth tiers, overshoot, coverage, benchmark grids, two orbits) | `docs/figures/science/` |
| `hfr_tiers.py` | Alternative HFR truth tiers (raw, 25-h, Butterworth) for held-out rows | `experiments/evolve/{run}/baselines/tiers_{region}.npz` |
| `hfr_rotary_spectrum.py` | Rotary spectrum of hourly radar currents with the three target-filter responses; `--selftest` on synthetic data | `docs/figures/science/hfr_rotary_spectrum.*` |
| `duacs_pull.py` | DUACS L4 daily geostrophy per region from Copernicus Marine | `experiments/{region}/duacs/` |
| `animate_evolve_champion.py` | Retrain champion, run inference + animate every region | `frames/`, `animations/{region}/ann/evolve_{cid}/` |
| `champion_importance.py` | Grouped permutation importance for a champion | `docs/figures/poster/champion_importance.{json,png,svg}` |
| `poster_figures.py` | Fitness-vs-candidate scatter and architecture diagrams | `docs/figures/poster/` |
| `build_science_colocations.py` | Science-phase SWOT × HFR colocation campaign (S3; cost rule applies) | `experiments/{region}/swot_cycles/`, `hfr_target/` |
| `build_hfr_multires.py` | 25-h HFR targets at every available radar resolution | `experiments/{region}/hfr_target/` |
| `build_era5_winds.py` | ERA5 10-m winds per region from the public NCAR mirror | `experiments/{region}/era5/` |
| `build_goes_sst.py` | GOES SST at SWOT pass hours from NOAA public S3 | `experiments/{region}/goes/` |
| `aggregate_hfr.py` | Aggregate raw hourly HFR files into per-(region, resolution) NetCDFs | S3 `HFR/` (cost rule applies) |
| `repartition_usegc1km.py` | One-off: resplit the two usegc 1-km aggregate segments at 2020-01-01 | S3 `HFR/` |

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

## AlphaEvolve (LLM-guided code evolution)

Evolves the ANN training code itself: Claude proposes full-module mutations of
a candidate `train_and_predict()` (seeded from the production MLP), each
candidate trains locally in a sandboxed subprocess (no file/network/OS access,
cloud credentials stripped, hard timeout), and is scored on a **temporally
held-out** split — fitness = mean of R²(u) and R²(v). Results append to
`experiments/evolve/{name}/database.jsonl`; watch live in the app's **Evolve** tab.

```bash
pip install -e ".[ann,evolve]"          # adds the anthropic SDK

# Free smoke test — no API calls (mock mutator jitters hyperparameters):
python scripts/evolve.py --config config.yaml --name test_mock --mock-llm \
    --generations 2 --children 2

# Zero-cost LLM evolution on your existing subscriptions (no API credits):
python scripts/evolve.py --config config.yaml --name evolve_uswc_v1 \
    --mutator claude-cli --generations 10 --children 4      # `claude -p` (Claude sub)
python scripts/evolve.py --config config.yaml --name evolve_uswc_v1 \
    --mutator codex-cli --generations 10 --children 4       # `codex exec` (ChatGPT sub)

# Direct Anthropic API — prints a dollar estimate and asks for confirmation
# first; --budget-usd is a hard cap checked before every API call:
python scripts/evolve.py --config config.yaml --name evolve_uswc_v1 \
    --mutator claude-api --generations 10 --children 4 --budget-usd 10

# Paper-protocol campaign (decided 2026-10-29): pooled science WITHOUT gak/glna
# (no SWOT geostrophy at their radar cells); fitness = mean over regions of
# per-region mean(R2_u, R2_v) on SWOT-valid held-out rows >= 30 days after
# the last training pass (25-h detided HFR target). Candidates also record
# the legacy pooled score as fitness_pooled and the per-region breakdown:
python scripts/evolve.py --name evolve_science_protocol_v1 --pooled science \
    --regions uswc,usegc,prvi,ushi --fitness per_region_buffered --fitness-buffer-days 30 \
    --features mdt,ssha_filtered,ugos_filtered,vgos_filtered,ugosa_filtered,vgosa_filtered,era5_u,era5_v \
    --mutator claude-cli --generations 35 --children 4 --timeout-min 60
```

Notes:
- **Launch from the app instead:** the Evolve tab has a "Launch a new run"
  panel (features, stencil k, generations, children, mutator, pooled domain) —
  it spawns the same detached, resumable process the CLI does.
- **`--features` / `--stencil-k`** override `config.yaml` per run; a run's
  dataset is frozen at first launch, so changing them requires a new `--name`.
- **`--fitness per_region_buffered`** needs a dataset frozen by the
  2026-10-29 harness (it stores per-row pass times as `test_time` in
  `data.npz` and `region_train_end` in `data_meta.json`); older frozen runs
  can only use the legacy `pooled` fitness. `--regions` subsets the pooled
  mission for both the data and the fitness.
- **Mutators are agent-agnostic** (`--mutator`): `claude-cli` / `codex-cli`
  shell out to a locally installed CLI agent and run on its subscription
  (counts toward its usage windows; $0 per token). `claude-api` needs
  `ANTHROPIC_API_KEY` (parent process only — it is stripped from candidate
  subprocesses); default model `claude-opus-4-8` (~$0.17/candidate ≈ $7 for a
  10×4 run), `--llm-model claude-sonnet-5` is ~2.5× cheaper.
- Re-running with the same `--name` **resumes** (completed candidates are
  skipped) — safe across hibernation.
- `--train-max-epochs 60` (default) screens candidates in ~4 min each on the
  GTX 1650 Ti; raise it (or retrain the winner at 200 epochs) for final numbers.
- All data reads are local (pipeline cache); the evolve loop never touches S3.

### Champion post-processing

```bash
# Animate every region in the run's pool with the champion (retrains it once
# on the run's frozen data with the exact run params, then drives the normal
# inference+animate steps; outputs under animations/{region}/ann/evolve_{cid}):
python scripts/animate_evolve_champion.py --run evolve_all_science_v1

# Pooled champion vs region-solo models of the same form (same code, params,
# budget; per-region temporal holdouts identical to the pooled breakdown).
# Writes experiments/evolve/{run}/solo/comparison.{json,md}:
python scripts/solo_vs_pooled.py --run evolve_all_science_v1 --seeds 42,43,44
# Retrain a single-region run's champion on sister regions of the same mission
# (runs the data steps from local mirrors if no flatten cache exists):
python scripts/solo_vs_pooled.py --run evolve_uswc_v1 --regions usegc,ushi --seeds 42,43,44
```

Both are resumable (champion predictions / finished region seeds are
checkpointed and skipped on rerun) and read only local data.

```bash
# Phase-1 evaluation (paper baselines): scores the champion and its solo twin
# against SWOT L3 geostrophy, geostrophy+Ekman (classical 3%/45 deg and a
# least-squares wind fit), a ridge on the 72 stencil columns, wind-only and
# training-mean baselines -- on identical held-out rows, under all-test and
# 30-day-buffered protocols, on all rows and on the SWOT-valid subset, with
# block-bootstrap CIs over (cycle, pass) and an effective-N estimate.
# Needs solo/{rid}/data.npz from solo_vs_pooled.py. Replays per-row
# cycle/pass/time/lat/lon from the cached inputs (checkpointed as
# baselines/meta_{rid}.npz) and asserts it reproduces Y_test exactly.
# Writes experiments/evolve/{run}/baselines/results.{json,md}:
python scripts/baselines_eval.py --run evolve_all_science_v1
python scripts/baselines_eval.py --selftest   # metric + bootstrap self-check
# Calval (1-day repeat) table: single-region run; the record ends 3 weeks
# after training so the buffer is 7 days, not 30:
python scripts/baselines_eval.py --run evolve_uswc_v1 --buffer-days 7
# Sister regions of a single-region run (usegc, ushi calval) are added by
# retraining its champion there first (below), then --regions uswc,usegc,ushi.

# HFR truth tiers for the held-out rows (raw hourly, 18-h and 48-h zero-phase
# Butterworth, plus a rebuilt 25-h mean that must reproduce the stored
# target exactly) from the hourly HFRNet files on the external drive
# (Z:\data_repository\columbia_oceanography\HFR\Code\Data). Writes
# baselines/tiers_{rid}.npz; baselines_eval.py picks them up automatically:
python scripts/hfr_tiers.py --regions uswc,usegc,prvi,ushi

# DUACS L4 daily geostrophy (Copernicus Marine, free account; run
# `Scripts/copernicusmarine.exe login` once in a real terminal). Writes
# experiments/{rid}/duacs/*.nc; baselines_eval.py adds duacs_* baselines
# when the file exists:
python scripts/duacs_pull.py
# Calval window (the eval concatenates every duacs_l4_{rid}_*.nc it finds):
python scripts/duacs_pull.py --regions uswc,usegc,ushi --start 2023-03-01T00:00:00 --end 2023-08-31T23:59:59

# Paper figures for the baselines (docs/figures/science/*.png|svg; the
# README there describes each one):
python scripts/baselines_figures.py --run evolve_all_science_v1
# Rotary spectrum of the hourly radar + filter responses (needs the hourly
# archive: HFR_DATA_ROOT or Z:; --selftest runs on a synthetic series):
python scripts/hfr_rotary_spectrum.py --regions usegc,uswc --start 2024-03-01 --end 2024-06-30
```

```bash
# Poster/paper figures (white background, Computer Modern):
# fitness-vs-candidate scatter + champion architecture diagram
# -> docs/figures/poster/*.png (300 dpi) and *.svg
python scripts/poster_figures.py --run evolve_all_science_v1

# Champion feature importances (grouped permutation on the temporal holdout;
# retrains the champion once ~20 min GPU, then checkpointed in
# docs/figures/poster/champion_importance.json — replotting is instant):
python scripts/champion_importance.py --run evolve_all_science_v1
```

---

## Data requirements for model runs

See **[docs/model-data.md](docs/model-data.md)** for exactly which files each
model run needs (per region × mission), what's optional, local-vs-S3
resolution, sizes, and re-download commands.

Dated research notes (findings, experiment narratives, project history) live in
**[docs/research-log.md](docs/research-log.md)**.

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

AVISO source: `swot_products/l3_karin_nadir/l3_lr_ssh/v3_0/Expert/reproc` (SFTP; set `AVISO_USER` and `AVISO_PW` in the environment, nothing is stored in the repo). Raw hourly HFR source files are kept locally under `HFR/Code/Data/*_Resolution_hourly/` (not on S3); set `HFR_DATA_ROOT` to point the two HFR scripts elsewhere. Public route from HFRNet THREDDS: `docs/replication.md` section 2.3.
