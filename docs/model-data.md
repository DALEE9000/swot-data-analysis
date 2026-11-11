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

## Radar archive day coverage (all 19 regions)

Days with at least one finite radar cell, from the hourly archives on Z:
(`HFR_DATA_ROOT`), computed 2026-11-08 by
`python scripts/animate_hfr_regions.py --coverage` (it reads the daily-mean
cache that `animate_hfr_swot.py --build-daily` keeps under
`HFR_ANIM_ROOT/hfr_daily/`; rebuild the cache to refresh). All archives are
now the full public records: US HFRNet from the NCEI archive through Aug 2026
(the Oct 2023 and 2015-2018 holes refetched; the remaining gaps are outages
also absent on the server), European node 2019 (TirLig 2016) through Oct 2026
except COSYNA (to Jun 2023, later months hang the node) and Vestlandet (from
Oct 2023), IMOS ACORN 2009/2010 through Jun 2026 (the QC product lags about
four months; the multi-year holes are station outages or data not yet QC'd on
the server, see the research log of 2026-11-07).

| Region | First | Last | Days in span | Days with data | Missing | Gaps of 7 days or more |
|---|---|---|---|---|---|---|
| uswc | 2012-01-01 | 2026-08-31 | 5,357 | 5,249 | 108 | 2012-06-15 to 2012-06-30 (16); 2013-04-07 to 2013-05-10 (34); 2013-06-01 to 2013-06-14 (14); 2014-04-15 to 2014-04-30 (16); 2017-04-01 to 2017-04-14 (14); 2018-06-01 to 2018-06-14 (14) |
| usegc | 2012-01-01 | 2026-08-31 | 5,357 | 5,306 | 51 | 2013-04-07 to 2013-05-18 (42); 2025-07-23 to 2025-07-31 (9) |
| gak | 2017-11-08 | 2026-08-31 | 3,219 | 2,913 | 306 | 2018-12-12 to 2018-12-18 (7); 2019-08-27 to 2019-09-09 (14); 2019-11-20 to 2019-12-02 (13); 2021-11-19 to 2021-11-29 (11); 2025-07-23 to 2025-11-23 (124); 2025-12-15 to 2026-03-16 (92); 2026-03-23 to 2026-04-07 (16); 2026-05-15 to 2026-05-28 (14) |
| glna | 2022-08-01 | 2026-08-31 | 1,492 | 942 | 550 | 2023-01-30 to 2023-02-06 (8); 2023-02-08 to 2023-02-14 (7); 2023-02-24 to 2023-03-05 (10); 2023-05-04 to 2023-05-23 (20); 2024-01-18 to 2024-01-31 (14); 2024-08-11 to 2024-11-22 (104); 2025-01-03 to 2025-03-19 (76); 2025-04-30 to 2025-09-07 (131); 2025-09-10 to 2025-09-18 (9); 2026-01-02 to 2026-01-08 (7); 2026-01-16 to 2026-02-17 (33); 2026-02-21 to 2026-03-11 (19); 2026-04-21 to 2026-05-18 (28) |
| prvi | 2010-01-27 | 2026-08-31 | 6,061 | 4,872 | 1,189 | 2010-02-06 to 2010-02-24 (19); 2010-06-05 to 2010-06-14 (10); 2010-07-24 to 2011-05-04 (285); 2011-05-12 to 2011-07-06 (56); 2011-07-13 to 2011-08-01 (20); 2011-09-06 to 2011-10-02 (27); 2011-10-08 to 2011-10-19 (12); 2011-11-17 to 2011-12-03 (17); 2011-12-09 to 2011-12-18 (10); 2012-02-07 to 2012-02-15 (9); 2012-03-05 to 2012-03-16 (12); 2012-03-23 to 2012-04-01 (10); 2012-04-05 to 2012-04-16 (12); 2012-05-01 to 2012-05-17 (17); 2012-05-24 to 2012-07-19 (57); 2012-08-23 to 2012-10-08 (47); 2012-11-14 to 2013-02-06 (85); 2013-04-18 to 2013-05-06 (19); 2013-05-25 to 2013-06-15 (22); 2013-07-05 to 2013-07-18 (14); 2013-09-03 to 2013-09-15 (13); 2013-10-04 to 2013-10-29 (26); 2013-12-18 to 2014-01-22 (36); 2017-09-19 to 2018-02-19 (154); 2022-09-19 to 2022-10-04 (16); 2025-01-08 to 2025-01-21 (14); 2025-07-28 to 2025-08-05 (9) |
| ushi | 2010-07-29 | 2026-08-31 | 5,878 | 5,457 | 421 | 2012-12-06 to 2013-01-26 (52); 2013-04-04 to 2013-05-06 (33); 2016-10-10 to 2016-10-30 (21); 2019-10-16 to 2019-10-22 (7); 2020-04-10 to 2020-04-19 (10); 2021-08-09 to 2021-08-15 (7); 2022-01-03 to 2022-01-09 (7); 2022-01-11 to 2022-01-20 (10); 2022-02-12 to 2022-06-23 (132); 2022-08-01 to 2022-10-21 (82) |
| eu_biscay | 2019-01-01 | 2026-10-31 | 2,861 | 2,264 | 597 | 2019-05-05 to 2019-06-26 (53); 2019-08-16 to 2019-09-17 (33); 2019-10-01 to 2019-10-08 (8); 2019-11-04 to 2019-11-26 (23); 2019-12-03 to 2019-12-10 (8); 2019-12-18 to 2019-12-24 (7); 2019-12-28 to 2020-01-08 (12); 2021-06-04 to 2021-07-01 (28); 2021-12-05 to 2022-01-04 (31); 2023-01-21 to 2023-01-29 (9); 2024-03-22 to 2024-03-28 (7); 2024-06-13 to 2024-10-13 (123); 2024-12-11 to 2025-01-18 (39); 2025-04-20 to 2025-05-09 (20); 2025-07-24 to 2025-07-31 (8); 2026-04-11 to 2026-05-30 (50); 2026-07-17 to 2026-09-11 (57) |
| eu_galicia | 2019-04-09 | 2026-10-20 | 2,752 | 2,454 | 298 | 2019-07-30 to 2019-08-29 (31); 2019-12-23 to 2020-01-19 (28); 2025-09-16 to 2025-09-22 (7); 2025-10-04 to 2025-10-19 (16); 2025-11-09 to 2025-11-16 (8); 2025-11-19 to 2026-02-13 (87); 2026-05-29 to 2026-06-10 (13); 2026-06-21 to 2026-06-27 (7) |
| eu_wmed | 2019-01-01 | 2026-10-31 | 2,861 | 2,861 | 0 |  |
| eu_tirlig | 2016-08-04 | 2026-10-31 | 3,741 | 3,699 | 42 | 2018-04-01 to 2018-04-17 (17); 2020-07-04 to 2020-07-13 (10) |
| eu_gbight | 2019-01-08 | 2023-09-04 | 1,701 | 1,524 | 177 | 2019-01-11 to 2019-01-29 (19); 2019-04-20 to 2019-05-01 (12); 2020-03-10 to 2020-04-01 (23); 2020-10-28 to 2020-11-04 (8); 2021-10-02 to 2021-10-13 (12); 2022-02-20 to 2022-03-07 (16); 2022-11-11 to 2022-11-18 (8); 2023-01-01 to 2023-01-17 (17); 2023-01-25 to 2023-02-01 (8); 2023-08-15 to 2023-08-21 (7) |
| eu_norway | 2023-10-09 | 2026-10-31 | 1,119 | 1,097 | 22 | 2025-07-17 to 2025-07-30 (14) |
| au_rot | 2010-02-23 | 2026-06-30 | 5,972 | 3,943 | 2,029 | 2010-02-24 to 2010-03-09 (14); 2010-05-12 to 2010-05-18 (7); 2010-06-22 to 2010-07-13 (22); 2010-07-15 to 2010-07-28 (14); 2010-10-27 to 2010-11-10 (15); 2010-11-13 to 2010-11-20 (8); 2011-03-01 to 2011-03-24 (24); 2012-01-07 to 2012-02-02 (27); 2012-06-01 to 2012-06-30 (30); 2013-08-09 to 2013-08-15 (7); 2013-09-30 to 2013-10-07 (8); 2013-11-21 to 2013-12-03 (13); 2014-11-11 to 2014-11-17 (7); 2019-04-19 to 2019-04-26 (8); 2019-06-08 to 2022-06-05 (1094); 2022-08-26 to 2022-09-04 (10); 2022-11-15 to 2022-12-13 (29); 2023-01-05 to 2023-01-15 (11); 2023-02-02 to 2023-02-28 (27); 2023-03-03 to 2023-04-18 (47); 2024-02-27 to 2024-03-13 (16); 2024-05-04 to 2025-07-22 (445) |
| au_turq | 2009-03-27 | 2026-06-30 | 6,305 | 4,672 | 1,633 | 2009-04-01 to 2009-04-20 (20); 2009-05-29 to 2009-07-12 (45); 2009-07-25 to 2009-08-02 (9); 2009-12-14 to 2009-12-20 (7); 2010-05-21 to 2010-06-07 (18); 2010-06-21 to 2010-06-29 (9); 2010-12-25 to 2011-01-03 (10); 2012-02-13 to 2012-02-23 (11); 2012-06-09 to 2012-06-15 (7); 2012-07-01 to 2014-12-31 (914); 2015-07-01 to 2015-07-31 (31); 2016-07-09 to 2016-08-07 (30); 2019-07-24 to 2019-08-13 (21); 2019-08-24 to 2019-09-01 (9); 2020-05-25 to 2020-06-14 (21); 2020-12-18 to 2020-12-24 (7); 2021-02-27 to 2021-11-22 (269); 2022-06-13 to 2022-06-22 (10); 2023-03-17 to 2023-03-26 (10); 2023-09-09 to 2023-09-18 (10); 2024-02-09 to 2024-02-19 (11); 2025-04-17 to 2025-04-26 (10) |
| au_cof | 2012-03-08 | 2026-06-30 | 5,228 | 3,118 | 2,110 | 2012-04-01 to 2012-04-27 (27); 2013-02-22 to 2013-03-03 (10); 2013-05-18 to 2013-05-31 (14); 2013-06-03 to 2013-06-30 (28); 2014-05-17 to 2014-11-27 (195); 2015-02-05 to 2015-02-15 (11); 2015-03-12 to 2015-03-22 (11); 2015-05-08 to 2015-05-24 (17); 2015-09-09 to 2015-10-10 (32); 2016-11-20 to 2017-01-23 (65); 2017-01-27 to 2017-02-08 (13); 2017-12-20 to 2018-01-29 (41); 2018-02-21 to 2018-03-13 (21); 2018-11-07 to 2018-12-11 (35); 2019-01-29 to 2019-02-26 (29); 2019-08-01 to 2019-08-07 (7); 2020-12-12 to 2021-01-05 (25); 2021-01-07 to 2021-02-11 (36); 2021-02-19 to 2023-12-13 (1028); 2024-01-21 to 2024-06-24 (156); 2024-07-05 to 2024-08-07 (34); 2024-10-14 to 2025-03-26 (164); 2025-11-06 to 2025-11-12 (7) |
| au_newc | 2017-11-30 | 2026-06-30 | 3,135 | 1,981 | 1,154 | 2018-03-22 to 2018-03-29 (8); 2018-06-14 to 2018-07-04 (21); 2018-07-12 to 2019-07-14 (368); 2019-11-25 to 2019-12-09 (15); 2020-01-08 to 2020-01-16 (9); 2020-01-24 to 2020-01-30 (7); 2020-02-19 to 2020-03-03 (14); 2020-03-23 to 2020-03-31 (9); 2020-07-18 to 2020-08-02 (16); 2020-11-16 to 2020-11-22 (7); 2020-12-16 to 2021-01-31 (47); 2021-12-24 to 2022-01-12 (20); 2022-10-06 to 2023-03-21 (167); 2023-07-20 to 2023-07-26 (7); 2023-08-13 to 2023-09-12 (31); 2023-09-20 to 2023-09-26 (7); 2024-02-16 to 2024-03-03 (17); 2024-04-25 to 2024-05-05 (11); 2024-05-23 to 2024-05-31 (9); 2024-07-30 to 2024-08-28 (30); 2024-11-05 to 2025-06-10 (218); 2025-12-19 to 2025-12-27 (9) |
| au_sag | 2009-11-23 | 2026-03-23 | 5,965 | 3,842 | 2,123 | 2009-11-25 to 2009-12-01 (7); 2009-12-21 to 2009-12-31 (11); 2010-01-28 to 2010-02-27 (31); 2010-03-02 to 2010-04-13 (43); 2010-05-03 to 2010-05-13 (11); 2010-05-21 to 2010-06-11 (22); 2010-06-15 to 2010-06-21 (7); 2010-09-30 to 2010-10-31 (32); 2010-11-08 to 2010-11-15 (8); 2011-03-20 to 2011-03-29 (10); 2011-05-19 to 2011-05-25 (7); 2011-08-11 to 2011-08-24 (14); 2012-05-23 to 2012-06-07 (16); 2012-06-13 to 2012-07-11 (29); 2012-07-13 to 2012-07-31 (19); 2012-10-20 to 2012-11-05 (17); 2013-01-13 to 2013-01-19 (7); 2013-02-01 to 2013-02-20 (20); 2013-04-21 to 2013-05-08 (18); 2014-04-26 to 2014-05-12 (17); 2014-11-27 to 2014-12-04 (8); 2015-02-28 to 2015-03-09 (10); 2016-02-04 to 2016-02-10 (7); 2017-03-16 to 2017-03-22 (7); 2018-04-01 to 2018-04-30 (30); 2018-07-17 to 2018-09-05 (51); 2018-12-21 to 2019-01-05 (16); 2019-02-03 to 2019-03-06 (32); 2019-05-19 to 2019-06-26 (39); 2019-07-28 to 2019-08-04 (8); 2019-08-07 to 2019-08-13 (7); 2020-05-30 to 2020-06-07 (9); 2020-06-15 to 2020-07-19 (35); 2020-10-23 to 2020-10-31 (9); 2020-12-05 to 2021-09-07 (277); 2021-10-22 to 2021-11-07 (17); 2021-11-25 to 2023-08-10 (624); 2024-02-21 to 2025-03-11 (385); 2025-03-13 to 2025-03-19 (7) |
| au_nwa | 2021-04-01 | 2026-06-30 | 1,917 | 1,523 | 394 | 2021-06-10 to 2021-06-24 (15); 2021-07-01 to 2021-07-23 (23); 2021-07-25 to 2021-08-01 (8); 2021-08-03 to 2021-08-20 (18); 2021-09-02 to 2021-09-12 (11); 2021-09-14 to 2021-09-23 (10); 2021-12-22 to 2021-12-30 (9); 2022-02-15 to 2022-03-01 (15); 2022-05-27 to 2022-10-24 (151); 2024-07-03 to 2024-07-10 (8); 2025-08-26 to 2025-09-09 (15); 2026-03-27 to 2026-05-13 (48) |
| au_corl | 2019-04-18 | 2026-06-30 | 2,631 | 2,182 | 449 | 2019-05-06 to 2019-05-16 (11); 2019-08-24 to 2019-09-01 (9); 2019-12-13 to 2019-12-19 (7); 2020-05-25 to 2020-06-14 (21); 2020-11-17 to 2020-12-04 (18); 2020-12-18 to 2020-12-24 (7); 2021-07-27 to 2021-08-04 (9); 2021-12-27 to 2022-01-05 (10); 2022-01-23 to 2022-01-31 (9); 2022-05-11 to 2022-05-17 (7); 2022-10-21 to 2022-11-01 (12); 2022-11-29 to 2023-03-08 (100); 2023-03-17 to 2023-03-27 (11); 2023-04-17 to 2023-04-30 (14); 2023-09-09 to 2023-09-18 (10); 2023-10-18 to 2023-10-26 (9); 2024-05-15 to 2024-05-28 (14); 2025-12-07 to 2026-02-23 (79) |

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
