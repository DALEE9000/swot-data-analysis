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
| `truth_tiers` | uswc only (tiers need the hourly archive; usegc and ushi tiers pending Z:) | filter sensitivity on the calval orbit |
| `benchmark_grid_u`, `benchmark_grid_v` | Full grid; tier columns with no data are hidden | audit view |

## hfr_target/ — how the radar target is defined (mission-agnostic)

Script: `scripts/hfr_rotary_spectrum.py` (needs the hourly archive: `HFR_DATA_ROOT` or Z:; `--selftest` for a synthetic check).

| File | Shows | Use |
|---|---|---|
| `hfr_rotary_spectrum` (render pending Z:) | Top: rotary power spectrum of hourly radar currents, clockwise and counterclockwise, with tidal and inertial lines. Middle: fraction of variance passed by the 25-h running mean and the 18-h and 48-h Butterworth filters on the same frequency axis. Bottom: ten days of one cell raw and under the three filters | justifies the 25-h target; shows where tidal and inertial energy sits |

## poster/ — the evolutionary search itself

Scripts: `scripts/poster_figures.py --run evolve_all_science_v1`,
`scripts/champion_importance.py --run evolve_all_science_v1`.

| File | Shows | Use |
|---|---|---|
| `fitness_evolve_all_science_v1` | Every candidate of the run in order, coloured by generation, with the best-so-far envelope; seed and champion marked | how the search progressed |
| `champion_architecture_gen031_00` | Block diagram of the champion's network as written in its candidate.py | what the search found |
| `champion_architecture_simple` | Same, reduced to layers and widths for a slide | talk version |
| `champion_importance` (+ `.json`) | Grouped permutation importance of the champion's input groups on the temporal holdout | which inputs the network leans on |

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
