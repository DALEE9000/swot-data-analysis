# Data requirements for running the ML models

Every model run — Random Forest (`rf`) or PyTorch MLP (`ann`), single-region
or pooled — consumes the same four kinds of input files. This page lists
exactly which files each kind of run needs, where they live, and what is
optional.

## The four input files (per region × mission)

| File | Local path (mirrors S3 1:1) | Role | Required? |
|------|------------------------------|------|-----------|
| SWOT pkl | `experiments/{region}/swot_cycles/swot_expert_reproc_v3_{region}_science.pkl`<br>`…/swot_expert_reproc_v3_calval_{region}_474_578.pkl` | regridded SWOT passes with per-pass time coords — the feature source (`mdt`, `ssha_filtered`, `ugos/vgos…`) | **yes** |
| HFR target | `experiments/{region}/hfr_target/hfr_{mission}_{region}.pkl` | 25-hour-mean HFR velocity colocated at each pass — the training target (`u`, `v`) | **yes** |
| ERA5 winds | `experiments/{region}/era5/era5wind_{mission}_{region}_10m.pkl` | hourly 10 m winds (`era5_u`, `era5_v`) | only when `era5_u`/`era5_v` are in `features` |
| GOES SST | `experiments/{region}/goes/goes_sst_{mission}_{region}.nc` | SST snapshots at SWOT pass hours (`SST`) | only when `SST` is in `features` |

The pipeline is **local-first**: it checks these paths, and only if a file is
absent does it stream from `s3://swot-ai-ssv/` (same key). ERA5/GOES loading
is **feature-gated** — if the matching feature names aren't selected, the
files are not opened at all, so presets can carry the paths at zero cost.

Both backends consume identical inputs; `rf` vs `ann` changes only
hyperparameters and where weights are written
(`experiments/{region}/{rf,ann}/`). No extra data is needed to switch models.

## What each kind of run needs

**Single-region preset run** — the region's 4 files for the chosen mission.
All are mirrored locally for every trainable region (see coverage below).

**Pooled run ("ALL regions — pooled")** — the 4 files for *every* region in
the mission's trainable set. Regions flatten separately, then one model
trains on the round-robin-merged matrix.

**Batch tab / model-space sweeps** — same files as the underlying preset;
flattened matrices are cached per feature-set under
`experiments/{region}/flattened/` and rebuilt automatically when absent.

**Custom (non-preset) run** — built from raw sources instead of pkls:
- `swot_path`: SWOT L3 granules (local dir or `s3://swot-ai-ssv/SWOT_L3/{mission}/…`)
- `hfr_path`: an HFR archive NetCDF (`s3://swot-ai-ssv/HFR/{region}/…` or local)
- `orbit_data/sph_{calval,science}_swath.zip` (in the repo) — required to
  find passes over the domain
First runs are slow (regrid + colocation from scratch) and, if sources are on
S3, incur egress — see the cost rule in `CLAUDE.md`.

## Coverage — which region × mission combos are trainable

| Mission | Trainable regions | Not trainable |
|---------|-------------------|----------------|
| science (cycles 1–16) | uswc, usegc, gak, glna, prvi, ushi | akns (HFR network dark since 2022 — empty target) |
| calval (cycles 474–578) | uswc, usegc, gak, ushi | glna/prvi (no passes), akns (no ground truth) |

GOES SST exists for every trainable region (akns alone sits outside
geostationary view). glna SST includes winter ice/cloud contamination —
treat that feature skeptically there.

## Sizes (local mirror, all 40 files ≈ 17 GB)

Largest: usegc science SWOT pkl (2.4 GB), usegc science HFR target (2.4 GB),
usegc calval SWOT pkl (1.5 GB), usegc calval target (1.1 GB), usegc ERA5
science (844 MB), usegc GOES science (547 MB). Small regions (gak, glna,
ushi, prvi) are tens-of-MB per file.

Optional extras kept on S3 only (download deliberately, per the cost rule):
multi-resolution HFR targets `hfr_science_{region}_{res}.pkl`
(500m/1km/2km variants, ~10 GB, for the resolution-comparison study).

## Re-downloading mirrors

If a mirror is deleted, either let the pipeline stream it on demand or pull
it explicitly (egress ≈ $0.09/GB):

```bash
aws s3 cp s3://swot-ai-ssv/experiments/{region}/{kind}/{file} experiments/{region}/{kind}/{file}
```

## Outputs a run produces

- weights: `experiments/{region}/{model}/weights/…` (`.joblib` for rf, `.pt` for ann)
- inference + flattened caches: `experiments/{region}/{model}/inference/`, `experiments/{region}/flattened/`
- registry record: `experiments/registry/{model}/experiments.jsonl` (+ summary CSV) — tracked in git
- animations: `frames/` + `animations/` (single-region runs only; pooled runs skip them)
