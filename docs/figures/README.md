# Figure library

Every figure the project produces, by topic folder, with what it shows and the
script that regenerates it. Files are not numbered; numbering happens in the
paper. `science/README.md` carries the full per-figure description, protocol
and caveats for the baselines set; the calval set uses the same figure
definitions on the 1-day-repeat campaign. PNG at 300 dpi and SVG for every
entry.

## science/ — Phase 1 on the 21-day science orbit (cycles 1-16, Jul 2023 - Jun 2024)

Script: `scripts/baselines_figures.py --run evolve_all_science_v1`. Headline
protocol: SWOT-valid rows, 30-day buffer after the last training pass (cycles
15-16, Apr-Jun 2024). Regions usegc, uswc, prvi, ushi. Champion `gen031_00`,
pooled, 72 input columns.

| File | Shows | Use |
|---|---|---|
| `skill_ladder` | Per-region R² of u and v for every model, train mean to champion, with block-bootstrap intervals; predictors named on each label | main result figure |
| `rmse_change` | The same comparison as percent RMSE change relative to calibrated SWOT geostrophy | field's currency, companion to the ladder |
| `truth_tiers` | Champion and the 5-parameter linear model re-scored against raw hourly, 18-h low-pass, 25-h mean and 48-h low-pass truth | the result does not depend on the target filter |
| `geostrophy_overshoot` | Joint density of raw SWOT geostrophic velocity against radar; slope well below one | why raw geostrophy is a poor predictor and calibration is needed |
| `coverage_map` | Held-out radar cells coloured by the fraction of passes with SWOT geostrophy | where the comparison happens; sample sizes |
| `benchmark_grid_u`, `benchmark_grid_v` | Heatmap of the whole evaluation grid: regions × truth tiers × protocols × models, one component each | audit view of the ~1000-row results table |
| `uswc_two_orbits` | Same radar (US West Coast), science 21-day orbit beside calval 1-day orbit, full ladder on each | sampling density changes what a network can learn |

## calval/ — Phase 1 on the 1-day calval orbit (cycles 474-578, Apr - Jul 2023)

Script: `scripts/baselines_figures.py --run evolve_uswc_v1`. Headline protocol:
SWOT-valid rows, 7-day buffer (the record ends three weeks after training; test
26 Jun - 10 Jul 2023). Regions usegc, uswc, ushi (prvi and glna have no calval
passes; gak dropped). Champion `gen020_02`, evolved on uswc with GOES SST as an
input, 81 columns; the uswc row uses its campaign predictions, usegc and ushi
rows are the same code retrained on each region (mean of seeds 42-44).

| File | Shows | Use |
|---|---|---|
| `skill_ladder` | As the science ladder, three regions | calval main result |
| `rmse_change` | As the science RMSE panel | companion |
| `truth_tiers` | usegc and uswc (Hawaii omitted: only 127 rows are finite in every tier) | filter sensitivity on the calval orbit |
| `benchmark_grid_u`, `benchmark_grid_v` | Full grid; tier columns with no data are hidden | audit view |

## hfr_target/ — how the radar target is defined (mission-agnostic)

Script: `scripts/hfr_rotary_spectrum.py` (needs the hourly archive: `HFR_DATA_ROOT` or Z:; `--selftest` for a synthetic check).

| File | Shows | Use |
|---|---|---|
| `hfr_rotary_spectrum` | Top: rotary power spectrum of hourly radar currents, clockwise and counterclockwise, with tidal and inertial lines. Middle: fraction of variance passed by the 25-h running mean and the 18-h and 48-h Butterworth filters on the same frequency axis. Bottom: ten days of one cell raw and under the three filters | justifies the 25-h target; shows where tidal and inertial energy sits |

## poster/ — the evolutionary search itself

Scripts: `scripts/poster_figures.py --run evolve_all_science_v1`,
`scripts/champion_importance.py --run evolve_all_science_v1`.

| File | Shows | Use |
|---|---|---|
| `fitness_evolve_all_science_v1` | Every candidate of the run in order, coloured by generation, with the best-so-far envelope; seed and champion marked | how the search progressed |
| `champion_architecture_gen031_00` | Block diagram of the champion's network as written in its candidate.py | what the search found |
| `champion_architecture_simple` | Same, reduced to layers and widths for a slide | talk version |
| `champion_importance` (+ `.json`) | Grouped permutation importance of the champion's input groups on the temporal holdout | which inputs the network leans on |

## holdout/ — out-of-time and out-of-region evaluation (cycles 017-058)

Built by `scripts/holdout_eval.py --figures` from `experiments/holdout/results.json`. Every model is trained or fitted on cycles 001-016 (Jul 2023 to Jun 2024) and scored on cycles 017-058 (Jul 2024 to Aug 2026, the end of the ERA5 record), rows with clean pairing, finite SWOT geostrophy and ERA5 inside its record. glna, gak and eu_gbight have no clean test rows and are absent. In the grid, `< -1` marks a blown-up fit and `n/a` a baseline skipped for too few rows (au_cof's year-one geostrophy calibration is broken, read its R² columns only).

| figure | what it shows | role |
|---|---|---|
| `skill_through_time` | One panel per region: percent RMSE change against calibrated SWOT geostrophy, per cycle, for the champion trained on the region's own continent, on all regions and on the other continents, beside DUACS and the cell climatology; training year shaded | drift of the calibration over two more years |
| `skill_through_time_quarterly` | The same quantity pooled by quarter with a 95% bootstrap interval over passes (single cycles as faint dots), plus a row-weighted linear trend per model with its slope in percent per year and interval printed in the panel | the drift statistic: a slope indistinguishable from zero is "no erosion" as a number |
| `loro_us` | Bars of R² per component (u, v) for the four scorable US regions under five training sets: own continent, all 19, US without this region, other continents, DUACS + wind; bars: average the seeds' predictions row by row first, then compute one R² of that averaged prediction; dots: R² of each seed's own prediction, one per seed. The bar is not the mean of the dots and sits above them where seeds disagree, because averaging predictions cancels their independent errors. Bars with a single seed (US without this region, DUACS) have no dots | the leave-one-region-out result and the seed spread in one picture |
| `transfer_grid` | Heat map of R² (mean of u, v): test regions by continent against the seven training sets, the leave-one-region-out model ("US w/o own region": each US row predicted by the recipe trained on the other five US regions) and three baselines; outlined cells are in-continent; us, all and not us are three-seed means | the generalization table |
| `transfer_map` | World map, one marker per radar region, coloured by R² of the champion trained on the other continents (top) and on the region's own continent (bottom) | the one-glance version |

## hfr_animations/ — daily-mean surface currents, every radar region

Script: `scripts/animate_hfr_regions.py` (needs the hourly archives under
`HFR_DATA_ROOT` or Z:). MP4s are gitignored; regenerate with the script.
One frame per calendar day with data: daily-mean speed (cmocean `speed`,
0 to 1 m/s on every region so they compare), arrows subsampled to about
30 across the grid, 50 m coastline, cell count in the title. 1600 px wide
(200 dpi), Computer Modern text, libx264 CRF 18 High profile, 24 fps.

| File | Shows |
|---|---|
| `<region>_daily.mp4` | US HFRNet 6 km archives over their full record (uswc, usegc 2012-2024; prvi, ushi 2010-2025; gak 2017-2025; glna 2022-2025) and the European and IMOS regions over Jul 2023 - Jul 2024 |
| `<region>_hfr_swot.mp4` | `scripts/animate_hfr_swot.py`: HFR daily mean on the left, SWOT L3 geostrophic current (ugos/vgos filtered) on the swaths that crossed the box that day on the right, same colour and arrow scales; every calendar day of the union of the two records, and when no pass fell on a day the most recent one up to 10 days old is shown with its age in the title. 2400 px wide, panels side by side for tall boxes and stacked for wide ones |
| `<continent>_hfr_swot.mp4` | Same two panels over one map per continent (`north_america`, `europe`, `australia`), every region drawn on it |
| `<name>_hfr_swot_since2023.mp4` | The same movies cut to 2023-04-01 onward (calval and science SWOT era), since the full-record films spend their first 3 minutes before SWOT existed |

## regions/ — study-area maps

Made in an earlier session (notebook); no regenerating script tracked.

| File | Shows |
|---|---|
| `regions_global` | The seven HFRNet networks on a world map |
| `regions_overview` | Regional bounding boxes with SWOT swath coverage |
| `regions_panels` | One panel per region, radar footprint and SWOT passes |
| `global_hfr_vs_swot` | Global radar coverage against SWOT science-orbit swaths |

## Conventions

u is teal `#0e7490`, v is orange `#ea580c`, ink `#1a1a2e`, muted `#5a6472`,
grid `#d9dee5`; in the spectrum figure the target filter is teal and the two
Butterworth tiers amber and violet. Computer Modern text (mathtext for symbols
the cmr10 font lacks, and "at least" rather than a >= glyph); white
background. Captions state the protocol: target, row filter, buffer,
bootstrap. New figures go in a topic folder here and get a row in this file.
