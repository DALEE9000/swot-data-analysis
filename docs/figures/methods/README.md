# Method diagrams

Six schematic figures, one per method, drawn from the constants of the
methods rather than from data. Regenerate with

```bash
python scripts/methods_diagrams.py
```

Each figure is also in `.svg` for slides and the paper. The mathematics
behind them is in `papers/manuscript/briefing.pdf`; the numbers quoted in the
captions come from `experiments/holdout/` and `docs/research-log.md`
(14 to 16 November 2026).

## `method_rows` — what one training row contains

Every experiment in the project trains the same kind of row. The satellite
side is a 3 × 3 stencil of SWOT Level-3 cells centred on one radar cell at
one pass: eight fields per cell (mean dynamic topography, sea surface height
anomaly, geostrophic velocity, its anomaly, and the ERA5 wind interpolated to
the cell), 72 columns in all. The centre cell is the one being predicted;
the eight neighbours give the network the spatial context that a single
height value lacks, which is how it can see gradients and curvature of the
height field. An engineered block (speed, direction, vorticity of the
stencil) is derived from the same columns. Experiments add further columns
(wind lags, the slab current, the Coriolis parameter, season, the hour
offset) without touching the 72. The learned model is the evolved champion:
a convolutional path over the stencil feeding a multilayer perceptron,
trained with a masked squared error so that rows with one missing component
still count, early stopping on the last 10 % of passes. It predicts both
components of the current at the centre cell, and the truth it is scored
against is the HF radar current at that cell.

## `method_daily_target` — Phase 1, the daily-target model

One row per cell per pass. The target is the radar current averaged over the
25 hours centred on the pass, the field's standard definition of a
"daily" surface current. The drawing shows why that choice matters: the
hourly radar current inside the window oscillates at the inertial period
(18 to 21 hours at these latitudes) and at the tidal periods, and the box
average removes most of that oscillation. In spectral terms the 25-hour
boxcar has its first null at 0.96 cycles per day, which sits on the inertial
band at 35 to 40°N. The target therefore keeps the slow geostrophic flow the
height field can see and discards most of the motion that the wind drives.
This is the model behind the time-transfer and space-transfer results.

## `method_wind_history` — Phase 2, wind history on the daily target

Same rows, same target, more columns: the ERA5 wind read at the pass time
minus each of eight lags from 3 hours to 8 days, plus the Coriolis parameter
and the season as sines and cosines of the day of year, in four feature
sets. The axis is drawn as the square root of the lag so that the short
lags do not pile up. The result was a clean negative: no change in R²
inside the bootstrap intervals on either coast with three seeds. The
response function, obtained by perturbing one lag at a time by 2 m/s and
averaging the change in the prediction (a finite-difference Jacobian), is
nonetheless physical: 1.2 % of the wind speed at 30 to 40° to the right of
the wind, 70 to 80 % of it at lag zero. The network had learned the Ekman
drift; the lagged part had nothing left to explain because the daily target
had averaged it away.

## `method_hourly_target` — Experiment 3, the hourly target

The same pass-time stencil now generates 25 rows per cell per pass, one for
each hour from 12 hours before the pass to 12 hours after (nine of the 25
are used, at three-hour spacing, with the non-zero offsets drawn for one
base row in eight). Each row is asked for the detided hourly radar current
at its own hour, and carries three new kinds of input: the offset in hours
from the pass, so the network knows how old the snapshot is; the ERA5 wind
at eleven lags ending at the target hour (0, 3, 6, ..., 24 hours, then 36
and 48; two columns each), so the wind history is the history relative to
the hour being predicted rather than relative to the pass, and nothing from
after that hour enters. The three-hour steps give about six samples per
inertial period of 18 to 21 hours, enough to fix the phase of the
oscillation; the coarser 36 and 48 hour lags cover the slab's four-day
memory without adding collinear columns; and the slab-layer current at the target hour.
The slab model (lower left) is a mixed layer of depth H driven by the wind
stress and rotated by the Coriolis force with a four-day damping time,
integrated hour by hour from the ERA5 wind; its two columns give the
network the phase of the inertial oscillation directly. Detiding (lower
right) fits five tidal constituents (M2, S2, N2, K1, O1) by least squares
to each radar cell's training-year record and subtracts the fitted
sinusoids from every hour, so the target is the current minus its
predictable tide. With the wind at the target hour, skill is flat from
−12 to +12 hours where the snapshot alone decays, which is the paper's
mechanism for turning sparse passes into a continuous product.

**Nine and eleven are different things.** Nine is the number of target
hours per pass, the offsets at which a row asks for the current: −12, −9,
−6, −3, 0, +3, +6, +9, +12 hours from the pass, one row each (the full design
has 25, every hour from −12 to +12, hence "25 rows per pass"). Eleven is the
number of wind lags inside each row, the times at which the ERA5 wind is
read: 0, 3, 6, 9, 12, 15, 18, 21, 24, 36 and 48 hours before that row's own
target hour, two columns each. For one pass the model is asked nine
questions, and to answer each it is handed eleven readings of the wind that
preceded that question's hour. In the diagram the nine are the pink squares
on the time axis and the eleven are the orange triangles inside each row's
window.

## `method_hourly_to_daily` — the hourly route as a daily product

Takes the nine hourly predictions of one pass, averages them, and scores
the average against the 25-hour-mean target on the same rows as the daily
champion, three seeds on each side. This is the direct answer to "does
using the hours individually improve the daily number": yes on the West
Coast (u 0.39 against 0.35, intervals apart), ties or gains elsewhere,
never a loss. The orange ring marks the check: the prediction at the pass
hour alone, scored against the daily mean, explains 14 % of its variance
on the West Coast and 64 % on the East and Gulf. The hourly model is
predicting real sub-daily motion that cancels over the day, and there is
far more of it on the West Coast, consistent with the rotary spectra.

## `method_holdout` — the evaluation protocol

Every model is fitted on SWOT science cycles 001 to 016 (July 2023 to June
2024) and scored on cycles 017 to 058 (June 2024 to August 2026), so every
number in the project is out of time. Confidence intervals are a block
bootstrap over passes, because the rows of one pass share the same
satellite swath and are not independent. Transfer in space is tested by
training on one continent and scoring the other two, and within the United
States by leaving one region out. The time test passes (13 of 16 regions
show no drift); the space test fails (u collapses, v survives, worse than
calibrated geostrophy), and the rule the paper draws is to train on every
network it intends to predict.
