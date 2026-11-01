# Research update, 29 October 2026

SWOT-to-surface-velocity project (SWOTxAI). Phase 1 of the preprint plan:
evaluation protocol, baselines, and what they say about the model. Numbers
below are in `experiments/evolve/evolve_all_science_v1/baselines/results.md`;
figures in `docs/figures/science/` (README there explains each one);
full narrative in `docs/research-log.md`.

## What was done

Every result in the project until last week was scored on raw model output
against the radar with a random 80/20 row split (the leaky protocol found in
July) or, for the evolution campaign, on a temporal split with no buffer, no
baselines, and uncertainties from row counts. This phase replaced that with
the protocol the field expects and re-scored the existing champion model
without retraining it.

- **Protocol.** Temporal holdout (SWOT science cycles 13 to 16, late March to
  mid June 2024) with a 30-day buffer after the last training pass; rows
  restricted to radar cells where SWOT L3 geostrophy is defined; 95 percent
  confidence intervals from a block bootstrap that resamples whole SWOT
  passes; an effective sample size from the same bootstrap.
- **Baselines, all on identical rows.** Training-period mean; DUACS L4
  geostrophy (the conventional multi-satellite maps, pulled from Copernicus
  Marine), raw and after a per-region linear calibration fitted on training
  rows; SWOT L3 geostrophy, raw and calibrated; each of those plus a
  least-squares fit to instantaneous ERA5 wind (five parameters per
  component); a ridge regression on the full 72-column stencil, the linear
  model with exactly the network's inputs.
- **Truth tiers.** The same predictions scored against four versions of the
  radar truth: raw hourly, an 18-hour low-pass, the project's 25-hour running
  mean, and a 48-hour low-pass, the latter two built from the hourly HFRNet
  record and verified to reproduce the stored target exactly.
- **Data fixes.** A pass-pairing bug (SWOT and radar passes matched by list
  position rather than time) was fixed; it affected only training rows in
  Puerto Rico. The Gulf of Alaska and Great Lakes radars were dropped from
  all SWOT-skill results: SWOT L3 geostrophy is defined at 2 percent and
  0 percent of their cells.

## The numbers

Held-out R² for u / v on SWOT-valid rows, all test cycles. Brackets are the
95 percent interval on u for the evolved network.

| | East & Gulf Coast | West Coast | Puerto Rico | Hawaii |
|---|---|---|---|---|
| DUACS geostrophy, calibrated | 0.46 / 0.54 | 0.09 / −0.15 | 0.16 / 0.18 | 0.15 / 0.47 |
| DUACS + instantaneous wind | 0.51 / 0.56 | 0.24 / 0.19 | 0.18 / 0.07 | 0.20 / 0.55 |
| SWOT L3 geostrophy, calibrated | 0.39 / 0.51 | 0.10 / −0.35 | 0.04 / 0.07 | 0.16 / 0.20 |
| SWOT + instantaneous wind | 0.44 / 0.53 | 0.24 / 0.11 | 0.06 / −0.14 | 0.14 / 0.21 |
| Ridge on 72 stencil columns | 0.52 / 0.56 | 0.24 / 0.12 | 0.08 / −0.18 | 0.02 / 0.21 |
| Evolved network (pooled) | 0.53 / 0.61 [0.44, 0.61] | 0.24 / 0.12 [0.19, 0.30] | 0.13 / −0.08 [0.02, 0.18] | 0.09 / 0.16 [−0.51, 0.21] |

Effective sample sizes for the network's errors: about 500 to 850 on the East
and Gulf Coast, 400 to 700 on the West Coast, under 100 in Puerto Rico and
Hawaii, from 9,000 to 430,000 rows. Roughly five to ten independent samples
per pass.

## What the numbers say

1. **The radar truth is already detided.** The targets are 25-hour running
   means, close to Tchonang's 48-hour tier. Against raw hourly currents the
   East Coast skill drops from 0.62 to 0.31; the gap is sub-daily variance
   no snapshot model reaches, and the 18-hour tier shows about half of it is
   near-inertial.

2. **Raw SWOT L3 geostrophic velocity is a worse predictor than the mean at
   the radar.** It correlates with the radar (r = 0.63 to 0.72 in the east,
   0.32 in the west) but carries 1.2 to 1.4 times its amplitude; regression
   slopes are 0.25 to 0.53. Likely small-scale slope noise the 20 to 30 km
   L3 filter leaves in, consistent with Tchonang's 70-km optimum. Any
   "better than SWOT" number has to be stated against a calibrated
   geostrophy.

3. **DUACS matches or beats SWOT L3 at the radar.** Calibrated DUACS plus
   instantaneous wind equals the evolved network on West Coast u, beats it on
   West Coast v and in Puerto Rico and Hawaii, and trails it only on the
   East and Gulf Coast, by 0.02 to 0.05 with overlapping intervals on u.
   The wide swath, as currently used (centre-cell level features, 3-by-3
   stencil, instantaneous wind), adds little over the nadir maps at coastal
   radar cells. DUACS has no nadir gap, no swath edges, and a longer
   averaging window; and SWOT L3 is cross-calibrated to DUACS during
   processing, so the two are not independent at large scales.

4. **Nonlinearity buys little.** On the West Coast a five-parameter linear
   model matches the 45-hour evolved network exactly. On the East Coast the
   network adds about 0.09 over the five-parameter model and 0.02 to 0.05
   over the ridge with identical inputs. Wind carries most of the reachable
   West Coast signal (about 19 percent RMSE reduction on v from wind alone).

5. **The buffer costs nothing.** Buffered scores equal or slightly exceed
   unbuffered ones everywhere, so the pooled 0.45 was forward skill, not
   boundary leakage. A six-month tidal-state buffer is impossible with 16
   cycles and is stated as a limitation.

## What this changes

- The paper's claim cannot be "SWOT-based ML beats the conventional maps at
  the coast". The defensible contribution has to be found in a band where
  nadir maps cannot compete (sub-100-km high-passed skill) or by features
  that use SWOT's resolution rather than its level: smoothed geostrophic
  pairs, slope and curvature fields, and wind history. Adding DUACS as an
  input so the model learns SWOT's increment over it is the cleanest
  framing.
- The next experiment (Phase 2) is wind history: eight days of lagged ERA5
  winds, Coriolis and season, with a perturbation-derived response function
  to compare with Ubelmann et al. (2025). Phase 1 makes this the critical
  test: the reachable West Coast signal is ageostrophic and the model has
  only instantaneous wind.
- No further evolution campaign until the fitness matches this protocol
  (25-hour target, SWOT-valid rows, 30-day buffer, per-region mean, Gulf of
  Alaska and Great Lakes excluded) and the new inputs exist. Evolving on the
  current inputs would rediscover the current ceiling.
- Headline numbers are per region with intervals, East Coast and West Coast
  foremost; Puerto Rico and Hawaii do not support a skill claim.

## Still running or pending

- Done since the first draft: the three-seed region-only retrain (seeds
  differ by under 0.01 in R² on the two large regions, by up to 0.14 in
  Hawaii; pooled training wins the East Coast, Puerto Rico and Hawaii,
  region-only wins the West Coast by 0.03 on u, all inside the intervals)
  and the fitness redefinition in the evolution harness.
- Calval-phase table (1-day repeat orbit) with the same baselines.

## Questions for discussion

- Is the high-passed band the right place to look for SWOT's increment, or
  is the coastal radar footprint simply the wrong test bed for a swath
  altimeter whose advantage is offshore?
- Should the paper be framed as a validation of what SWOT L3 adds over
  DUACS at coastal radars (including the negative result), with the
  wind-history mechanism as the positive finding?

## Addendum, 30 October 2026

Two baselines were added after this update was written, and they change
how the table above should be read.

- **Per-grid-point climatology.** The training-period mean current at each
  radar cell, with no satellite or wind input, scores 0.37 / 0.55 on the
  East and Gulf Coast. The evolved network's increment over climatology
  plus wind is about 0.12 on u and 0.09 on v, not the 0.57 the headline
  R² suggests. R² against the held-out mean credits a model for knowing
  the mean field; the paper should also report skill relative to a spatial
  climatology. On the West Coast the eight-month training mean has the
  wrong sign in the test season (upwelling reversal), so it is a poor
  climatology there and a multi-year radar mean would be fairer.
- **Calval (1-day repeat) table, West Coast.** With daily revisits the
  picture inverts: the calval champion reaches 0.43 / 0.56 against 0.04 /
  0.26 for the 5-parameter linear model and 0.38 / 0.36 for a cell-specific
  linear model. Under 21-day sampling the network adds nothing over linear
  models on the same coast; under daily sampling it adds 0.05 on u and
  0.20 on v over the best linear model. Whether that is the sampling
  density or the SST input the calval run had is a one-run test (science
  West Coast with SST). Calval retrains for the East Coast and Hawaii are
  running. Tables: `experiments/evolve/evolve_uswc_v1/baselines/results.md`
  (7-day buffer, since the record ends three weeks after training).
