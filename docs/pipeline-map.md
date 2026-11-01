# Pipeline map

One picture of how data becomes the numbers in the paper. Three stages: get the
data onto disk, turn it into training rows and a model, score the model against
the baselines. Boxes are files or folders; arrows carry the script or module
that makes the right-hand side from the left. Renders on GitHub and in VS Code
(Mermaid). Last checked against the code 2026-10-31.

## Stage 1: sources to local mirrors

```mermaid
flowchart LR
  subgraph SRC["Public and account sources"]
    AVISO["AVISO+ SFTP<br/>SWOT L3 Expert v3 granules"]
    ORB["orbit_data/<br/>science + calval swath shapefiles"]
    HFRNET["HFRNet hourly radar<br/>(our copy: HFR/Code/Data, HDOP 1.0)"]
    ERA5["NCAR ERA5 mirror on S3<br/>10-m u, v hourly"]
    GOES["NOAA GOES-16/18 on S3<br/>ABI SST hourly"]
    CMEMS["Copernicus Marine<br/>DUACS L4 daily geostrophy"]
  end

  ORB -->|find_swaths: which passes cross each radar| COLO
  AVISO -->|build_science_colocations.py<br/>download, regrid to lat/lon, keep 6 vars| COLO["experiments/{region}/swot_cycles/<br/>regridded SWOT per cycle (pkl)"]
  HFRNET -->|aggregate_hfr.py| AGG["HFR/{region}_{res}_Resolution_hourly_*.nc4<br/>full-period hourly archive"]
  AGG -->|build_science_colocations.py<br/>25-h running mean centred on pass time| TGT["experiments/{region}/hfr_target/<br/>u, v target per pass (pkl)"]
  AGG -->|build_hfr_multires.py| TGTR["hfr_target/ at 500 m, 1, 2 km"]
  ERA5 -->|build_era5_winds.py| WIND["experiments/{region}/era5/"]
  GOES -->|build_goes_sst.py<br/>calval only| SST["experiments/{region}/goes/"]
  CMEMS -->|duacs_pull.py| DUACS["experiments/{region}/duacs/<br/>duacs_l4_{region}_*.nc"]
```

## Stage 2: mirrors to training rows to a model

```mermaid
flowchart LR
  COLO["swot_cycles/"] --> S1
  TGT["hfr_target/"] --> S6
  WIND["era5/"] --> S3
  SST["goes/"] --> S4

  subgraph PIPE["src/swotxai/pipeline (12 cached steps, config.yaml + presets.py)"]
    direction LR
    S1["1 load_swot"] --> S2["2 regrid"] --> S3["3 load_era5"] --> S4["4 load_goes"] --> S5["5 interp_sources<br/>wind, SST onto SWOT grid"]
    S5 --> S6["6 load_hfr"] --> S7["7 interp_hfr<br/>target onto SWOT grid"] --> S8["8 flatten<br/>3x3 stencil, 9 cells x features = row"]
    S8 --> S9["9 train (rf or ann backend)"] --> S10["10 evaluate"] --> S11["11 inference"] --> S12["12 animate"]
  end

  S8 --> FLAT["experiments/{region}/flattened/<br/>flattened_{mission}_{region}_k3_*.pkl"]
  FLAT -->|"evolve.py: pool regions, temporal 80/20 split per region"| NPZ["experiments/evolve/{run}/data.npz + data_meta.json<br/>X_train Y_train X_test Y_test test_time"]

  subgraph EVO["AlphaEvolve harness (src/swotxai/evolve)"]
    direction LR
    SEED["seed candidate.py<br/>(MLP training code)"] --> MUT["mutator<br/>claude-cli proposes edits"] --> CAND["candidates/genNNN_MM/candidate.py"]
    CAND -->|"sandbox run on data.npz"| FIT["fitness<br/>pooled R2 or per-region buffered"]
    FIT --> DB["database.jsonl<br/>idea, metrics, parent"] --> MUT
  end

  NPZ --> CAND
  CAND -->|"predictions.npz on held-out rows"| PRED["champion predictions"]
  NPZ -->|"solo_vs_pooled.py: retrain champion per region, seeds 42-44"| SOLO["solo/{region}/seed4X/pred.npz<br/>comparison.json"]
  CAND -->|"animate_evolve_champion.py"| ANIM["frames/, animations/{region}/ann/evolve_{cid}/"]
```

## Stage 3: predictions to the paper's tables and figures

```mermaid
flowchart LR
  NPZ["data.npz + data_meta.json"] --> EVAL
  PRED["champion predictions.npz"] --> EVAL
  SOLO["solo/ seed predictions"] --> EVAL
  FLAT["flattened/ cache"] -->|"replay_meta: cycle, pass, time, lat, lon per row"| META["baselines/meta_{region}.npz<br/>meta_train_{region}.npz"]
  META --> EVAL
  DUACS["duacs/"] -->|"sample at row time, lat, lon"| EVAL
  subgraph TRUTH["Alternative truth tiers (re-score only, no retraining)"]
    direction LR
    AGG["HFR hourly archive"] -->|"hfr_tiers.py, held-out rows only"| TIERS["baselines/tiers_{region}.npz<br/>raw hourly, 25-h mean (check),<br/>Butterworth 18-h, Butterworth 48-h"]
  end
  TIERS -->|"same predictions, different yardstick"| EVAL

  subgraph EVAL["baselines_eval.py"]
    direction TB
    LAD["baseline ladder, fitted on training rows:<br/>train mean, per-cell climatology, wind-only,<br/>SWOT geostrophy raw / calibrated / + wind,<br/>per-cell linear, DUACS, ridge-72, champion pooled / solo"]
    PROT["4 protocols: all rows, SWOT-valid,<br/>30-day buffer (7 for calval), buffer + SWOT-valid"]
    BOOT["per-region R2, RMSE;<br/>block bootstrap over (cycle, pass); N_eff"]
    LAD --> PROT --> BOOT
  end

  EVAL --> RES["baselines/results.json + results.md<br/>regions x tiers x protocols x models"]
  RES -->|"baselines_figures.py"| FIGS["docs/figures/science/skill_ladder-uswc_two_orbits"]
```

(The poster figures, fitness scatter and architecture from `database.jsonl` and
feature importances from `data.npz`, are a side branch off stage 2 and are left
off the chart; commands in the README.)

## Reading notes

- Stage 1 is the only place that costs money or needs accounts. Everything
  after it reads local mirrors. The S3 bucket `swot-ai-ssv` is our own backup
  of stage 1 outputs and is never read by compute jobs.
- The target is fixed at stage 1: a 25-hour running mean of the radar current,
  centred on the SWOT pass time. The Butterworth tiers in stage 3 do not
  retrain anything; they re-score the same predictions against alternative
  definitions of truth (`docs/research-log.md`, 2026-10-29 and 10-31). A
  Butterworth low-pass is a frequency-domain smoother: it passes oscillations
  slower than its cutoff period (18 h or 48 h here) with almost no change and
  attenuates faster ones progressively, with a flat response in the pass band
  and no ripple. Order 2 sets how steep the roll-off is; running it forward and
  backward (`filtfilt`) cancels the phase shift so the filtered value sits at
  the same hour as the raw one. Against the 25-hour boxcar, which has side
  lobes that let some tidal energy leak through, it is the cleaner way to ask
  "what is the sub-daily-free current at this hour".
- The evolve harness does not use the pipeline's `train` step. It freezes the
  flattened rows once into `data.npz` and lets each candidate's `candidate.py`
  own the whole training loop. The champion's weights are not saved; anything
  that needs predictions on new rows retrains it from `candidate.py` and
  `params.json` (`solo_vs_pooled.py`, `animate_evolve_champion.py`).
- Two independent runs exist: `evolve_all_science_v1` (21-day orbit, pooled
  over uswc, usegc, prvi, ushi, 72 input columns) and `evolve_uswc_v1` (1-day
  calval orbit, uswc only, 81 columns including SST). Stage 3 runs once per
  run; uswc_two_orbits puts the two side by side.
- Where to look when something is wrong: a missing region in stage 3 means
  its `flattened/` cache or `data.npz` span is missing (stage 2); a NaN tier
  means too few radar hours around the pass (stage 3, `hfr_tiers.py` coverage
  rule); a DUACS row count under 50 means the pull window does not cover the
  held-out dates (stage 1).
