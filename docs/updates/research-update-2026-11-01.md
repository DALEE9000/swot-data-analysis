# Research update, 1 November 2026

Covers 30 and 31 October and the morning of 1 November. Previous update:
`research-update-2026-10-29.md` (with its 30 October addendum). Raw material
in `docs/research-log.md`; figures in `docs/figures/` with an index in
`docs/figures/README.md`.

## What was done

**Calval table (30 October).** The baselines ladder was run on the 1-day-repeat
campaign (`evolve_uswc_v1`, US West Coast, April to July 2023, 81 inputs
including GOES SST) with a 7-day buffer, since the record ends three weeks after
training and a 30-day buffer leaves nothing. The evaluation script was
generalised to single-region runs, DUACS was pulled for the calval window, and
two rungs were added to both ladders: a per-grid-point climatology (the
training-period mean at each radar cell) and a per-cell linear model
(cell-specific calibration slope and wind response, partially pooled toward the
global fit with a prior worth five rows).

**Figure set and protocol (30 October).** The ladder, RMSE and tier figures now
use the headline protocol (SWOT-valid rows, 30-day buffer, cycles 15-16). A
two-panel figure puts the West Coast ladder on the science orbit beside the same
ladder on the calval orbit.

**Repository cleanup (31 October).** Twenty-one top-level folders became
thirteen; 8 GB of duplicate cache recovered as hard links; 5.6 GB of old output
archived to Z: and removed locally; a third party's AVISO password found
hard-coded in four tracked files, removed in favour of environment variables
(owner to rotate; it remains in git history). README layout redrawn to the real
tree with a scripts index; a no-new-top-level-folders rule in CLAUDE.md.

**Replication (31 October).** `docs/replication.md` sets out three levels:
regenerate figures from tracked results (minutes), rebuild data and retrain the
champion from public sources (days; the radar aggregate is the one hop without
a one-line script, now documented with an HFRNet THREDDS recipe), and rerun the
evolutionary search (not replayable; audit trail in `database.jsonl`).
`requirements-lock.txt` pins the environment. A Zenodo manifest with SHA-256
for the ten archive files is ready; the deposit needs the account holder.

**Pipeline map and figure library (31 October).** `docs/pipeline-map.md`
draws sources to mirrors to rows to model to tables in three Mermaid charts.
Figure files lost their numbers (numbering happens in the paper) and
`docs/figures/README.md` lists every figure with what it shows and which script
makes it.

**Rotary-spectrum figure (31 October, data pending).**
`scripts/hfr_rotary_spectrum.py` computes the rotary power spectrum of the
hourly radar current (clockwise and counterclockwise separated) with the three
target-filter responses in a panel beneath it and a ten-day time strip. The
self-test on a synthetic series passes; the real render waits for the hourly
archive on Z:. One fact from the self-test worth a sentence in the paper: the
25-hour running mean passes 94.5 % of 8-day variance, a 5 % damping of
week-scale flow.

**Calval retrains on sister regions.** The calval champion was retrained with
three seeds on the East Coast and Hawaii calval data. Unbuffered three-seed
means: usegc 0.63/0.50, ushi 0.69/0.59. The three-region buffered table is
running at the time of writing; its ladder and benchmark grids will go into
`docs/figures/calval/`, separate from the science set.

## The numbers

Headline protocol (25-h target, SWOT-valid rows, buffered), per-region R² u/v.

| Model | usegc science | uswc science | uswc calval |
|---|---|---|---|
| per-grid-point climatology | 0.37 / 0.55 | -0.01 / -0.98 | 0.20 / 0.06 |
| climatology + wind | 0.45 / 0.56 | | 0.17 / 0.18 |
| SWOT calibrated + wind (5 par.) | 0.48 / 0.56 | 0.28 / 0.11 | 0.04 / 0.26 |
| per-cell linear | 0.48 / 0.59 | 0.25 / 0.11 | 0.38 / 0.36 |
| DUACS calibrated + wind | 0.53 / 0.58 | | 0.11 / 0.29 |
| ridge, 72 stencil columns | 0.54 / 0.59 | | 0.10 / 0.14 |
| AlphaEvolve champion | 0.57 / 0.64 | 0.21 / 0.11 | 0.43 / 0.56 |

Science: n about 180,000 rows per region, 95 % block-bootstrap intervals of
roughly plus or minus 0.03 on the East Coast and wider on the West. Calval
uswc: 117,018 rows, 25 passes, N_eff about 550.

## What the numbers say

**On the 21-day orbit, a linear model with the same inputs gets nearly all the
skill.** The champion's margin over ridge on the identical 72 columns is three
to five hundredths on the East Coast and nothing elsewhere. About half the East
Coast skill is location alone. The West Coast is a tie near 0.2/0.1 for every
model from five parameters upward. The physical reading: geostrophy is linear in
the height gradient and the Ekman response is linear in the wind at fixed
stratification, so a snapshot of height and wind gives a linear model almost
everything it can have. Nonlinearity would have to enter through what the
snapshot lacks: wind history, stratification (SST, season), spatial context
beyond three cells.

**Under daily sampling the network pulls away.** Calval West Coast: 0.43/0.56
against 0.38/0.36 for the per-cell linear model and 0.04/0.26 for the global
one, a gap of 0.2 in v. Two confounds remain: 84 visits per cell instead of 12,
and SST as an input. One science West Coast run with SST separates them.

**Climatology must be the reference in the paper.** Reporting R² against the
held-out mean overstates skill on the East Coast; the per-cell training mean is
the honest floor, and the paper should report skill above it.

**The target filter is defensible.** The tier study (truth_tiers figure) shows
the 25-hour mean removes nearly all the variance the altimeter cannot see;
48-hour low-pass adds three hundredths, 18-hour costs more than that. The
rotary spectrum will show why in one panel.

## Discussion of 1 November: the goal

The stated aim is a model with R² above 0.9 that is far ahead of the
baselines. Against the 25-hour radar current that number is almost certainly
above the ceiling the data allow. With RMSE 0.08 m/s at R² 0.57 the East Coast
target's standard deviation is about 0.12 m/s; radar 25-hour means carry 0.05 to
0.08 m/s of measurement error in the literature, which caps a perfect predictor
between about 0.6 and 0.85 before physics. The multi-resolution radar targets
already built give a direct estimate of that ceiling (disagreement between 2-km
and 6-km 25-hour means at the same point and time).

The version of "far ahead" the field recognises is scale-dependent: DUACS has no
skill below about 100 km, and a SWOT-based model that reaches 0.4 or 0.5 at 15
to 80 km against DUACS near zero is the result the SWOT community wants from a
radar validation. The regime version is the West Coast, where everything fails
at 0.2 and a model reaching 0.5 would mean real new physics (upwelling, wind
history) was learned.

## Pending and next

- Three-region calval table: done the same day, see the addendum below.
- Rotary-spectrum render on real data: done 2 November, see the second addendum.
- Paired bootstrap of champion minus ridge on the same passes, to test the
  three-hundredths East Coast gap directly; regime decomposition of where the
  gain lives (distance from coast, wind speed, Gulf Stream edge).
- Ceiling estimate from the multi-resolution radar targets, per region.
- Scale-dependent skill (coherence or band-filtered R²) against DUACS.
- Science West Coast run with SST (cheap; settles the calval confound).
- Phase 2 inputs in order: wind history (2 to 3 days), previous SWOT pass,
  SST and gradient, DUACS as large-scale prior, wider stencil; then a fresh
  evolution with the buffered per-region fitness over the richer inputs.
- Housekeeping for David: Zenodo deposit, AVISO credential rotation, Recycle
  Bin, `AVISO_USER`/`AVISO_PW` in the environment.

## Addendum, 1 November 2026, afternoon: the three-region calval table

Headline protocol (25-h target, SWOT-valid rows, 7-day buffer), R² u/v with 95 %
block-bootstrap intervals. Figures in `docs/figures/calval/`.

| Model | usegc (209,834 rows, 26 passes) | uswc (117,018, 25) | ushi (1,240, 12) |
|---|---|---|---|
| per-grid-point climatology | 0.41 / 0.59 | 0.20 / 0.06 | 0.90 / 0.73 |
| SWOT calibrated + wind (5 par.) | 0.37 / 0.60 | 0.04 / 0.26 | -0.02 / 0.03 |
| per-cell linear | 0.59 / 0.65 | 0.38 / 0.36 | 0.87 / 0.69 |
| DUACS calibrated + wind | 0.47 / 0.64 | 0.11 / 0.29 | 0.05 / 0.06 |
| ridge, 72 stencil columns | 0.52 / 0.65 | 0.10 / 0.14 | 0.73 / 0.56 |
| AlphaEvolve champion | 0.62 / 0.62 (retrained) | 0.43 / 0.56 (campaign) | 0.72 / 0.60 (retrained) |

What this changes. The calval advantage seen on the West Coast does not carry
to the East Coast: there the retrained champion (0.62/0.62, CI about plus or
minus 0.07 in u, 0.04 in v) sits on top of the per-cell linear model (0.59/0.65)
and ridge (0.52/0.65), within the intervals. Hawaii is a tiny domain (1,240
rows) where a steady flow makes location alone worth 0.90/0.73 and nothing with
SWOT or DUACS in it beats the per-cell linear fit. So the network's clear win is
specific to the West Coast calval run, which is the one it was evolved on; on
the two sister regions it ties the strongest linear baseline. That sharpens the
earlier reading: with snapshot inputs, linear is enough almost everywhere, on
both orbits. The West Coast calval margin (0.2 in v over per-cell linear) is
now the single result that needs explaining, and the SST-on-science-orbit run
is the experiment that explains it.

## Addendum, 2 November 2026: the two jobs that needed the hourly archive

**Rotary spectrum, real data** (`docs/figures/hfr_target/hfr_rotary_spectrum`).
Thirty fully covered cells per region, March to June 2024. Both coasts show the
same skeleton: a red continuum falling from multi-day periods, the semidiurnal
M2 line as the tallest peak, the diurnal pair K1 and O1, and a clockwise-only
bump at the local inertial frequency (1.27 cycles per day at 39.9 N on the East
Coast, 1.16 at 35.5 N on the West). The differences are physical. On the East
Coast the counterclockwise power exceeds the clockwise at periods longer than
about five days, which is the Gulf Stream meander band; on the West Coast the
diurnal lines are far stronger and clockwise, the sea-breeze and diurnal-tide
response of the California shelf. The middle panel explains the tier results.
The 25-hour running mean has its first null at 0.96 cycles per day, almost on
top of K1 and O1, passes nothing at M2, and leaks about 5 % in a side lobe near
1.5 cycles per day, which is just above the inertial band. The 18-hour
Butterworth passes roughly half of the diurnal and inertial energy and a sixth
of M2, so a model scored against it is scored against noise it cannot predict.
The 48-hour filter is clean above 0.7 cycles per day but also damps the
two-to-five-day band where the wind-driven signal lives.

**Calval truth tiers.** Raw, 18-hour and 48-hour tiers built for all three
calval regions from the 2023 hourly archive; the rebuilt 25-hour mean matches
the stored target exactly in every region (r = 1.0000). Scored on the rows
finite in every tier (usegc 54,128 rows, 29 %; uswc 67,200, 58 %), R² u/v:

| Truth | usegc champion | usegc per-cell linear | uswc champion | uswc per-cell linear |
|---|---|---|---|---|
| raw hourly | 0.20 / 0.31 | 0.36 / 0.38 | 0.28 / 0.39 | 0.27 / 0.34 |
| 18-h low-pass | 0.44 / 0.47 | 0.65 / 0.61 | 0.41 / 0.48 | 0.36 / 0.41 |
| 25-h mean (target) | 0.52 / 0.54 | 0.62 / 0.72 | 0.47 / 0.50 | 0.43 / 0.42 |
| 48-h low-pass | 0.54 / 0.54 | 0.66 / 0.72 | 0.47 / 0.50 | 0.43 / 0.43 |

Two things to note. The ordering of tiers is the same as on the science orbit
(raw lowest, 18-h below the target, 48-h at or a few hundredths above it), so
the target choice is defensible on both orbits. And on the East Coast common
subset, which is the 29 % of rows inside the densest radar coverage, the
per-cell linear model beats the retrained champion by 0.10 in u and 0.18 in v,
a larger gap than the full-row table showed (there the two tied). The
network's East Coast calval parity is therefore subset-dependent: it holds
on the full set and fails where the radar is best. Hawaii has 127 rows common
to every tier and is omitted from the tier figure and greyed in the grid.
