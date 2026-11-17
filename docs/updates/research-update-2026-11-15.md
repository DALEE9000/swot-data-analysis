# Research update, 15 November 2026

Covers 2 to 15 November. Previous update: `research-update-2026-11-01.md`.
Raw material in `docs/research-log.md`; figures in `docs/figures/holdout/`
with an index in `docs/figures/README.md`; the same fortnight written up
with the mathematics as a LaTeX briefing in `papers/manuscript/briefing.pdf`,
and the paper scaffold in `papers/manuscript/main.pdf`.

## Key insights, in the order they should change what we do

1. **The target decides what the inputs can teach.** The 25-hour-mean radar
   current, the field's standard truth, has a boxcar response whose first null
   (0.96 cycles per day) sits on the inertial band at 35 to 40°N. On that
   target eight days of wind history added nothing to the champion (Phase 2,
   13 November). On the detided hourly current the same wind history is the
   largest gain of the study (14 November). Averaging the truth had averaged
   away the physics. Every future input experiment should be run on the
   hourly target first.

2. **One SWOT pass is good for a whole day, if the wind is current.** With the
   wind at the target hour, its two-day history and a slab-layer inertial
   current as inputs, skill is flat from 12 hours before the pass to 12 hours
   after (uswc 0.29 / 0.37 in u / v, usegc 0.51 / 0.43), where the model with
   only the pass-time wind decays by a quarter to a third. The decay of a
   snapshot was never the height field going stale; it was the wind input.
   This is the mechanism by which sparse passes become a continuous product,
   and the reason the whole-ocean plan is feasible.

3. **The hourly route improves the daily number.** Averaging the nine hourly
   predictions of a pass and scoring against the 25-hour mean, three-seed
   ensembles on both sides, beats the daily champion on the West Coast
   (u 0.39 against 0.35, intervals apart; v 0.56 against 0.52) and in v on the
   East and Gulf (0.65 against 0.61), ties elsewhere, never loses. The hourly
   model's pass-hour prediction alone predicts the daily mean poorly on the
   West Coast (u 0.14) and well on the East and Gulf (0.64): the West Coast
   surface current has far more sub-daily motion that cancels over the day,
   which matches the rotary spectra.

4. **Transfer in time holds; transfer in space does not.** Trained on year
   one, the recipe keeps its skill over two more years on 13 of 16 regions
   with drift slopes indistinguishable from zero. Trained on networks it has
   not seen, it is worse than calibrated geostrophy, with u blowing up and v
   surviving, inside a continent as much as across continents. The Coriolis
   parameter as an input does not rescue it. The all-19 model recovers every
   region it has seen. Rule for the paper: train on every network you intend
   to predict; the spatial generalisation claim is withdrawn.

5. **The network's wind response is physical, and on the hourly target it
   resolves the inertial oscillation.** On the daily target: 1.2 % of the wind
   speed at 28 to 41° to the right, 70 to 80 % of it instantaneous, in
   agreement with a least-squares fit on the same rows (1.1 % at 40°, 0.9 % at
   37°), lower than the drogued-drifter literature, not higher. On the hourly
   target: 1.3 to 1.5 % at 18 to 25° right, and the response vector rotates
   clockwise with lag, right of the wind at 0 to 3 h, left at 15 to 18 h, right
   again by 24 h, one turn in the local inertial period. Coriolis and season as
   inputs add +0.01 inside the intervals on both targets.

6. **Detide the target, do not expect the model to learn the tide.** Five
   harmonics fitted on the training year remove 2 to 4 % of the hourly
   variance on the West Coast and 7 to 8 % on the East and Gulf; the undetided
   control loses exactly that much u skill on the East and Gulf and nothing on
   the West Coast. Neither the height snapshot nor the wind carries tidal
   phase.

7. **Seed spread sets what a difference means.** In-region ±0.01 in R² on the
   daily target and ±0.005 on the hourly target (three seeds each);
   out-of-region ±0.2, itself a diagnostic of extrapolation. A three-seed
   ensemble gains 0.005 to 0.01 over one seed, which is exactly the size of the
   Coriolis-and-season "gain", so that stays a null. Any claimed gain under
   0.02 is noise until reseeded.

8. **Drifters cannot answer the inertial question in the SWOT era yet.** The
   GDP hourly product ends October 2022; the 6-hourly product runs to June
   2025 and would give order 140,000 matched rows worldwide, enough to score
   an open-ocean inference, not to train one. Hourly velocities would have to
   be derived from raw GPS fixes or the Copernicus in-situ product.

## What was done

**Data (4 to 11 November).** Thirteen radar networks added from the European
HFR node and IMOS ACORN; the US record refetched from NCEI after HFRNet's
server was retired; every record extended to its full length; SWOT cycles
017 to 058 downloaded for all nineteen regions; ERA5 winds to August 2026;
25-hour targets for the new cycles; a frozen evaluation set of 16 million
rows (11 million clean) across 58 cycles; daily radar movies and radar-beside-
SWOT movies for every region and continent. Disk moves to the archive drive
with junctions; chunked freezes for the two large regions.

**Experiment 1, the holdout (11 to 13 November).** Seven training sets plus
four US leave-one-region-out sets plus three seeds; scored with block
bootstrap over passes; figures `skill_through_time`, `skill_through_time_quarterly`,
`transfer_grid`, `transfer_map`, `loro_us`.

**Experiment 2, wind history on the daily target (12 to 13 November).** Lagged
winds, Coriolis and season as extra columns on the frozen rows; three feature
sets; LightGBM control; finite-difference response function; figures
`wind_history_gain`, `wind_response`.

**Experiment 3, the hourly target (14 November).** `scripts/hourly_targets.py`
(detided hourly radar at the holdout cells) and `scripts/hourly_eval.py`
(rows at nine offsets, wind lags and slab current at the target hour, three
feature sets, an undetided control, scoring by offset, the averaged-hourly
daily comparison); figures `hourly_skill_offset`, `hourly_tide_effect`,
`hourly_wind_response`; seeds 43 and 44 for both wind models overnight 14/15
November, all tables as three-seed ensembles.

**Writing.** `papers/manuscript/main.tex` is the paper scaffold (JGR Oceans
section plan, key points, figure slots, every open decision marked);
`papers/manuscript/briefing.tex` is the fortnight with the mathematics
(pooled R², block bootstrap, the hourly-versus-daily variance identity, the
harmonic detiding fit, the slab model and its exact hourly step, the
perturbation Jacobian); `refs.bib` holds the literature; the baselines
equations stay single-sourced in `docs/methods-equations/`. Build with
`bash papers/manuscript/build.sh`.

## Open items

- Sensitivity of the hourly result to the slab depth H and to using all 25
  offsets instead of nine.
- The hourly experiment on the other seventeen networks (needs hourly targets
  per region; usegc-sized builds need the archive drive).
- Drifter route: 6-hourly GDP product (free, a few hundred MB) for scoring an
  open-ocean inference, or hourly velocities derived from raw fixes.
- OSSE with LLC4320 for skill where there is no truth.
- A daily model trained on uswc + usegc alone, as the exact control for the
  averaged-hourly comparison.
