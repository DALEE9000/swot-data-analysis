# Baselines figures (Phase 1)

Generated 2026-10-28 by `scripts/baselines_figures.py` from
`experiments/evolve/evolve_all_science_v1/baselines/results.json` (the
output of `scripts/baselines_eval.py`). PNG at 300 dpi and SVG for each.
White background, Computer Modern, the same tokens as the poster figures.
Colour: teal is the east-west component u, orange is the north-south
component v, throughout.

Regenerate:

```bash
python scripts/baselines_eval.py --run evolve_all_science_v1   # the numbers
python scripts/baselines_figures.py --run evolve_all_science_v1 # the figures
```

What every figure shares: the data are the held-out rows of the pooled
science run (SWOT science cycles 13 to 16, late March to mid June 2024),
scored against HF radar surface currents. The radar truth is the project's
standard target, a 25-hour centred running mean of the hourly field, unless
a figure says otherwise. The four regions shown are the ones with usable
SWOT coverage; the Gulf of Alaska and Great Lakes radars have no SWOT L3
geostrophic values at their cells and are omitted.

## What is regressed against what

Every row in the tables and figures predicts the same thing: the radar's
25-hour detided surface current at one grid point and one SWOT pass time,
as two separate targets, u (eastward) and v (northward). Every fitted
model is fitted on the region's training rows (SWOT cycles 1 to 12) and
applied unchanged to the held-out rows (cycles 13 to 16). Only the
right-hand side changes. g = SWOT L3 geostrophic velocity at the centre
cell (ug, vg); d = DUACS L4 geostrophic velocity sampled to the same point
and time; W = ERA5 10-m wind (Wu, Wv); X72 = the eight input variables at
each of the nine cells of the 3x3 stencil. "Parameters" are per component.

| results key | predicts u from | fitted parameters | note |
|---|---|---|---|
| `train_mean` | a constant | 1 (the training-row mean) | floor of the ladder |
| `cell_mean` | training mean at this radar grid point | 1 per cell | climatology; location memorised, nothing else |
| `cell_mean+wind_lsq` | cell mean + (1, Wu, Wv) fitted to the residual | 1 per cell + 3 | |
| `cell_linear_geos_wind` | 1, ug, vg, Wu, Wv fitted separately in each cell (>= 20 training rows; else the global fit) | 5 per cell | cell-specific calibration and wind response: what repeat tracks let a network learn |
| `wind_only_lsq` | 1, Wu, Wv | 3 | wind alone, rotated response allowed |
| `swot_geos` | ug | 0 | raw SWOT velocity used as is |
| `geos+ekman_3pct45deg` | ug + 0.03 \|W\| turned 45 deg right | 0 | textbook Ekman, nothing fitted |
| `geos+wind_lsq` | ug + (1, Wu, Wv) fitted to the residual (HFR - g) | 3 | geostrophy enters with slope 1 |
| `geos_calibrated` | 1, ug | 2 | slope came out 0.25 to 0.53: the overshoot |
| `geos_cal+wind_lsq` | 1, ug, vg, Wu, Wv | 5 | the linear model that matches the MLP on the West Coast |
| `duacs_geos` | du | 0 | raw DUACS velocity |
| `duacs_calibrated` | 1, du | 2 | |
| `duacs_cal+wind_lsq` | 1, du, dv, Wu, Wv | 5 | |
| `ridge72_solo` | X72 (standardised, NaN to column mean), L2 penalty | 72 + 1 | linear model with exactly the MLP's inputs |
| `champion_pooled` | X72 through the evolved MLP, trained on all six regions | thousands (joint u, v) | campaign predictions of `gen031_00` |
| `champion_solo` | X72 through the same MLP, trained on this region only | thousands | mean of seeds 42, 43, 44 |

v is predicted the same way with vg, dv in place of ug, du (and both wind
components again). Reading down the table is reading how much each
addition to the right-hand side buys with the target held fixed; the gap
between `ridge72_solo` and the champion rows is what the nonlinearity buys.

## Where the "AlphaEvolve champion MLP" comes from

The neural network in these figures is **not** the production `ann`
backend of the app. It is candidate `gen031_00` of the AlphaEvolve run
`evolve_all_science_v1` (July 2026): an LLM-guided evolutionary search
over the ANN training code, 35 generations of 4 children, pooled science
data from all six radar regions, 8 features (SST excluded), 3-by-3 stencil,
fitness = mean held-out R² on a temporal split. The champion's code is
`experiments/evolve/evolve_all_science_v1/candidates/gen031_00/candidate.py`
and its idea trail is in that run's `database.jsonl`.

Its form, in words: fit an exact ridge regression on the 72 stencil
columns first; train a plain feedforward multilayer perceptron (MLP) on
the ridge residuals, with wind pseudo-stress, current-speed magnitudes and
per-feature stencil-validity fractions added as inputs; component-balanced
Huber loss; early stopping validated on the most recent training rows; a
small seed ensemble within the time budget. No convolutions, no skip
connections.

Two rows use it:

- **pooled training** — the predictions the candidate wrote during the
  campaign (`candidates/gen031_00/predictions.npz`), trained once on all
  six regions. These are the predictions behind the 0.4458 pooled fitness.
- **region-only retrain** — the same `candidate.py` and `params.json`
  retrained from scratch on one region's training rows by
  `scripts/solo_vs_pooled.py` (mean of seeds 42, 43, 44; per-seed scores in
  `experiments/evolve/evolve_all_science_v1/solo/comparison.md`), scored on the
  identical held-out rows.

The "ridge on 3x3 stencil" row is the linear model with exactly the same
72 inputs, so the gap between it and the champion is what the hidden
layers buy. The production random forest and plain MLP are absent because
no saved weights exist for them under the temporal protocol.

---

## skill_ladder — how much variance each level of model explains

Since 2026-10-30 skill_ladder, rmse_change and truth_tiers use the headline protocol (rows at
least 30 days after the last training pass, i.e. cycles 15-16) rather than
all four test cycles; the unbuffered numbers remain in `results.md`.

Row labels carry two lines: the model name, then its predictors (the
right-hand side of the regression) with the fitted parameter count; the
caption restates that the target is the same in every row. The same
convention is used in rmse_change (two-line labels), truth_tiers (legend) and benchmark_grid
(label: predictors). Symbols: g SWOT L3 geostrophy, d DUACS L4, W ERA5
wind; see "What is regressed against what" above.

**What it shows.** For each region, the coefficient of determination R²
of eight predictors, ordered from the simplest to the most complex. Each
row has two markers, u and v. The horizontal bar is a 95 percent confidence
interval from a block bootstrap in which whole SWOT passes are resampled,
not individual grid cells, because neighbouring cells of one pass are not
independent samples.

**The rungs, top to bottom.** The training-period mean (zero skill by
construction). DUACS L4 geostrophic velocity, the conventional multi-satellite
map, after a per-region linear calibration fitted on training rows. DUACS
plus a least-squares fit to the instantaneous ERA5 wind. SWOT L3 geostrophic
velocity at the centre stencil cell, calibrated the same way. SWOT plus the
same wind fit, a five-parameter linear model. A ridge regression on all 72
stencil columns, the linear model with the same inputs the neural network
sees. The evolved MLP trained on all regions pooled. The same MLP retrained
on the region alone.

**How to read it.** The distance between the SWOT-plus-wind rung and the
evolved-MLP rung is what nonlinearity and spatial context buy. On the East
and Gulf Coast that distance is real and the intervals separate. On the
West Coast it is zero: a five-parameter linear model matches the evolved
network. The DUACS rungs sit at or above the SWOT rungs in every region,
which is the central finding of Phase 1. Rows are restricted to cells where
the SWOT centre cell is finite so that every rung is scored on identical
data.

---

## rmse_change — the same comparison in the field's currency

**What it shows.** Percent change in root-mean-square error relative to
the calibrated SWOT L3 geostrophic velocity on the same rows. Negative bars
are better than SWOT geostrophy. This is the form in which Martin et al.
(2023, 2024) and others state results ("x percent better than DUACS"), so
the numbers can be placed next to theirs; here the reference is SWOT
geostrophy because that is the project's input.

**How to read it.** The bar for the training mean is positive and large
where SWOT geostrophy carries real information (East and Gulf Coast). The
West Coast bars show that almost all of the improvement over geostrophy
comes from adding wind, with the same roughly 8 percent (u) and 19 percent
(v) reached by the linear model, the ridge and the evolved network alike.

---

## truth_tiers — skill as a function of the truth's timescale

**What it shows.** The same predictions scored against four versions of
the radar truth: the raw hourly field at the hour nearest the pass; an
18-hour zero-phase Butterworth low-pass (order 2); the project's standard
25-hour running mean; and a 48-hour Butterworth low-pass. The low-pass
tiers were built from the continuous hourly HFRNet record on the external
drive and sampled at the pass hour. Two models are drawn, the evolved MLP
(solid, circles) and the SWOT-plus-wind linear model (dashed, squares).
Rows are the subset where all four tiers are defined, so differences
between tiers are not coverage differences; that subset is smaller than
the main table, and on the East and Gulf Coast it is the well-covered 27
percent of rows, where skill is higher than in skill_ladder.

**How to read it.** On the East and Gulf Coast, skill rises from about 0.3
against raw hourly currents to about 0.63 against the 48-hour low-pass.
The difference is the sub-daily variance (tides and near-inertial motion)
that no single-snapshot predictor can reach; it is the "noise floor"
Kachelein et al. (2024) describe. The 18-hour tier, which keeps
near-inertial motion, sits between. On the West Coast the curves are flat
and low: the floor is a small part of the story there. Where the curves
are flat or fall with more smoothing (Puerto Rico, Hawaii), the model is
not capturing the slow signal either.

---

## geostrophy_overshoot — why raw SWOT velocity is a worse predictor than the mean

**What it shows.** Joint density (hexagonal bins, log colour scale) of the
SWOT L3 geostrophic velocity at the radar cell against the radar velocity,
for u and v, on the East and Gulf Coast and the West Coast. The dotted line
is one-to-one. The coloured line is the least-squares fit of radar on
SWOT, which is the calibration applied to the "calibrated" baselines. The
annotation gives the correlation, the ratio of standard deviations
(SWOT over radar), and the row count.

**How to read it.** The cloud is tilted well below the one-to-one line in
every panel: SWOT velocity has 1.2 to 1.4 times the radar's amplitude and
the regression slope is 0.25 to 0.5. Used raw, SWOT geostrophy therefore
has negative R² everywhere (skill_ladder would show it far off the left edge).
The likely cause is small-scale noise in the sea-surface slope that the
20 to 30 km L3 filter does not remove; Tchonang et al. (2026) found the
optimum smoothing for in situ comparison in the California Current to be
about 70 km. The correlation, 0.63 to 0.72 in the east and 0.32 to 0.34 in
the west, is the information content; the slope is the price of not
smoothing enough.

---

## coverage_map — where the comparison happens

**What it shows.** Every held-out radar grid point, coloured by the
fraction of its passes in which the SWOT L3 geostrophic velocity is
defined there. Dark is always covered; white is never. The grids are the
swath-aligned SWOT grids, so the same radar cell appears at slightly
different coordinates on different passes and the diagonal banding traces
the ground tracks.

**How to read it.** The light stripes are the nadir gap between the two
50-km swaths and the swath edges, where the L3 filtered product is
masked; the lighter coastal fringe is the land mask. About 30 percent of
the radar rows have no SWOT geostrophy at all, and in those rows the
model predicts from mean dynamic topography and wind alone. This is also
why the Gulf of Alaska and Great Lakes radars do not appear in the other
figures: their footprints fall almost entirely in masked SWOT cells.

## benchmark_grid_u / _v — the whole evaluation grid on one page

**What it shows.** Every scored cell in `baselines/results.json` as one
heatmap tile: six region panels, the full model list down the rows (the count
in the title is the number of scored cells), and
twenty columns made of five truth tiers (header) times four row
protocols (test cycles all or 30-day buffered; rows all or SWOT-valid).
Colour and the printed number are held-out R² for the component named
in the filename, clipped to [-0.5, 0.7]; values below -1 are printed as
"<-1". Grey tiles are combinations that were not scored (no SWOT rows
for the SWOT-based models, DUACS not finite, or no hourly HFR record for
the tiers in gak and glna). The black box marks the paper's headline
column: 25-h target, SWOT-valid rows, 30-day buffer.

**How to read it.** This is the audit view of the thousand-row
`results.md`, not a paper figure. Read down a column to see the model
ladder under one protocol; read across a row to see how one model's
score moves as the truth tier or row filter changes. Two things stand
out: the ladder's ordering is stable across columns on the East and
West Coast (the conclusions do not depend on the protocol choice), and
the SWOT-raw rows are dark red in every tier, which is Finding 4 of the
research log in one glance. Hawaii and Alaska show sign changes between
adjacent columns, which is what "too few passes" looks like.

**Built by** `fig_benchmark_grid()` in `scripts/baselines_figures.py`;
rerun with `python scripts/baselines_figures.py`.

## uswc_two_orbits — same radar, two SWOT orbits

**What it shows.** The full ladder for the US West Coast radar under the
two orbits SWOT has flown: left, the science phase (21-day repeat; held-out
rows at least 30 days after training, April to June 2024, scored with the
pooled science champion); right, the calval phase (1-day repeat; held-out
rows at least 7 days after training, 26 June to 10 July 2023, scored with
the calval champion `gen020_02` of run `evolve_uswc_v1`, which also had
GOES SST as an input). Same marker and bar conventions as skill_ladder.

**How to read it.** Under 21-day sampling every model from the
5-parameter linear fit upward sits at the same 0.2 on u and 0.1 on v: the
network adds nothing. Under daily sampling the per-grid-point linear model
climbs to 0.38 / 0.36 and the network to 0.43 / 0.56, while the global
linear models stay where they were. Daily revisits give about 84 training
samples per cell instead of 12, enough to learn a cell-specific response,
and something nonlinear on top of it, mostly in v. The two panels share a
radar and a season-sized test window, so the difference is sampling
density (and possibly the SST input), not geography. Climatology, by
contrast, is positive on calval (training and test in the same season)
and negative on science (seasons differ).

**Built by** `fig_two_orbits()` in `scripts/baselines_figures.py`; needs
both `baselines/results.json` files (`--run`, `--calval-run`).

## hfr_rotary_spectrum — where the radar's energy sits, and what each target filter keeps

Script: `scripts/hfr_rotary_spectrum.py` (not `baselines_figures.py`; it needs
the hourly radar archive on Z: or `HFR_DATA_ROOT`). Status: self-test passes on
a synthetic series; the real-data render is pending the next time the drive is
plugged in.

Top panel: rotary power spectrum of the hourly surface current at the
best-covered cells of each region over the held-out window (Mar-Jun 2024),
Welch with 30-day Hann segments. The current is treated as the complex series
u + iv, so clockwise rotation (solid) and counterclockwise (dashed) are
separated; in the northern hemisphere inertial oscillations are clockwise
only, tides elliptical, sub-inertial flow roughly balanced. Vertical lines mark
M2, K1, O1 and each region's inertial frequency. Right axis: the fraction of
variance each target filter passes, computed analytically: the 25-point boxcar
applied once (|H|^2, with its side lobes visible past 1 cpd) and the order-2
Butterworth at 18 h and 48 h applied forward and backward (|H|^4).

Bottom panel: ten days of one cell's u, raw and under the three filters. Point
of the figure: the energy the altimeter cannot see is the tidal lines and the
clockwise inertial bump; the 18-h filter leaks the diurnal and inertial band,
the 48-h filter is clean but damps real two-day flow, and the 25-h mean sits
between with about 5 % loss at 8-day periods. That is why 25 h is the headline
target and the other two are reported as tiers.

Files in this folder are not numbered; numbering is assigned in the paper.
