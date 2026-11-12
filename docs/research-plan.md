# Research plan: next steps toward the SSV preprint

Written 2026-10-28. Dates are targets, not deadlines; update them as work
lands. Each phase ends with a checkpoint that should be recorded in
`docs/research-log.md`. The reasoning behind the ordering is in the
2026-10-23 paper notes (`papers/NOTES.md`, "The five methods I would explore
first") and the 2026-10-27 discussion summarized at the bottom of this file.

Status key: `[ ]` not started, `[~]` in progress, `[x]` done.

---

## Phase 1 — Lock the evaluation protocol (2026-10-28 to 2026-11-18)

Goal: every number in the paper is scored the same way, against the right
truth, with baselines. No new features and no new models in this phase. The
reason this comes first is that it changes every existing result, so any
feature experiment run before it would have to be rescored anyway.

### Week of 2026-10-28 — temporal holdout everywhere
- [~] Make `step_evaluate` use the buffered temporal holdout (2026-10-28: done
      outside the pipeline in `scripts/baselines_eval.py`; `step_evaluate`
      itself still random-splits — decide whether to change working code) (last 20 percent
      of cycles, 30-day gap between the end of training and the start of
      test). The registry R² of about 0.8 is a leaked number and must not
      reach the paper.
- [x] Rescore the pooled champion `gen031_00` under the new protocol
      (2026-10-28, `experiments/evolve/evolve_all_science_v1/baselines/`;
      RF/MLP production models not rescored — no saved weights; the ridge
      and solo twin stand in).
- [x] Drop glna (Great Lakes) and gak from all SWOT-skill results (decided
      2026-10-29; both have ~no SWOT geostrophic coverage at radar cells).
      Still to do: remove them from the pooled preset before the Phase-2 retrain.
- [x] Pass-pairing bug fixed in `step_flatten` (time-matched, <= 1 h;
      2026-10-29); prvi/ushi rebuilt; held-out rows unchanged. It has no geostrophic
      regime at these scales and no full-stencil cells to animate.

### Week of 2026-11-04 — three tiers of HFR truth
**Correction (2026-10-28):** the existing targets are already 25-h running
means (≈ a 48-h tier with sinc leakage), so "raw hourly" needs a re-pull of
the hourly record (local only for usegc). Decide whether the tiers are worth
the download before doing this week.
- [x] (2026-10-29, `scripts/hfr_tiers.py`, outside the pipeline) raw hourly, 18-hour low-passed,
      and 48-hour low-passed targets (Butterworth order 2, zero-phase,
      applied to the full hourly record before interpolation to SWOT times).
      The 18-hour filter removes most of the tidal band and keeps
      near-inertial motion; the 48-hour filter removes both.
- [x] Mask HDOP > 1.25 (already applied at source by HFRNet at 1.0; no
      HDOP variable survives) and and cells with less than 50 percent temporal coverage
      (Kachelein 2024).
- [x] Report skill on all three tiers side by side (2026-10-29; truth_tiers). The gap between raw and
      18-hour R² is approximately the non-phase-locked tidal floor that no
      snapshot model can reach.

### Week of 2026-11-11 — baselines and the field's metrics
- [x] At every HFR matchup point compute: DUACS L4 geostrophic velocity,
      SWOT L3 ugos/vgos, and SWOT geostrophic plus an instantaneous Ekman
      term. (2026-10-28: SWOT geos, geos+Ekman, calibrated geos, geos+wind,
      ridge, wind-only, train-mean done; DUACS pulled 2026-10-29 and scored —
      it matches or beats SWOT L3 at the radar; see log Finding 5.) (DUACS is a one-time download; estimate the volume and cost
      before pulling anything.)
- [x] Per component and per region: RMSE in m/s, Nash–Sutcliffe R², percent
      RMSE change against each baseline, and R² on the residual
      HFR − ugos (2026-10-28; science mission).
- [x] Separate tables for the calval and science phases (2026-11-01: usegc and ushi calval added, figures in docs/figures/calval; 2026-10-30: calval uswc
      table in `experiments/evolve/evolve_uswc_v1/baselines/`, 7-day buffer;
      usegc and ushi calval retrains running; climatology and per-cell linear
      rungs added to both tables). At 21-day sampling DUACS tends to beat SWOT
      geostrophy (Tchonang 2026), and a reviewer will check.
- [x] Block bootstrap over (cycle, pass) snapshots for confidence intervals;
      report effective degrees of freedom per region (2026-10-28: N_eff in
      the hundreds; see research log).

### Also in Phase 1
- [x] Rerun `scripts/solo_vs_pooled.py --seeds 42,43,44` (2026-10-29:
      done for uswc, usegc, prvi, ushi; baselines table and figures now
      carry the three-seed mean. gak/glna stay at one seed: dropped from
      SWOT-skill results, so the negative-transfer claim is not made).
- [x] Redefine the evolve fitness on the new protocol (2026-10-29:
      `--fitness per_region_buffered --regions uswc,usegc,prvi,ushi`;
      25-h target, SWOT-valid rows, 30-day buffer, per-region mean).
      Headline protocol decided the same day.

Checkpoint (2026-11-18): one table of champion/RF/MLP versus the three
baselines, on three truth tiers, calval and science separately, with
bootstrap intervals. This table is the spine of the paper.

---

## Phase 1b — Out-of-time and out-of-region evaluation (2026-11-11 to 2026-11-18)

Added 2026-11-11, after the data campaign of Nov 7-10 extended every region
to SWOT science cycles 001-058 (Jul 2023 to Oct 2026) and added six European
and seven Australian radar regions with ERA5 winds and DUACS for the whole
window. The Phase 1 tables hold out cycles 13-16 with a 30-day buffer because
one year was all there was; now 42 cycles exist that no decision ever saw.

Protocol: the champion is a training recipe, not weights, so it is retrained
on year one (cycles 001-016, everything the evolutionary search was allowed
to see) and scored once on 017-058. The search is NOT rerun: its inner split
(train 001-012, fitness 013-016) already used year one the only way it can be
used, and rerunning it would roll the dice on a different recipe without
making the holdout any cleaner. Seven training sets of the same recipe give
the spatial axis for free, since the model takes 72 stencil columns and no
coordinates: US, Europe, Australia, all nineteen, and leave-one-continent-out
(not-US, not-Europe, not-Australia). Baselines (calibrated geostrophy with
and without a fitted wind response, DUACS likewise, cell climatology, train
mean) are fitted on each region's year-one rows. Rows after the ERA5 record
(2026-08-31) are excluded rather than scored with a stale wind.

- [x] Freeze every region's rows for all cycles (2026-11-11; usegc and uswc
      need `--chunk 10`, the whole record does not fit in commit memory).
- [x] Seven retrains, one seed each (2026-11-11, 2.7 h of GPU in total).
- [ ] Seeds 43, 44 for us_y1, not_us_y1, all_y1.
- [x] Score; `experiments/holdout/results.md` (2026-11-11).
- [x] Figures: `docs/figures/holdout/` (2026-11-11).
- [x] Second search not justified: no drift to fix (research log 2026-11-11 22:10).
- [x] `docs/replication.md` updated (2026-11-11); paper methods notes pending.
- [ ] Leave-one-region-out within a continent (train on five US regions, score the sixth).
- [ ] Latitude-aware inputs (sin(latitude), or velocities and winds scaled by f, or
      per-region standardisation) and a rerun of not_us_y1.

Result (2026-11-11): the recipe transfers in time (own-continent champion beats
calibrated geostrophy in 13 of 16 scored regions over two further years, level
with DUACS plus wind in the US and Europe, drift of a few points at most) and
fails in the leave-one-continent-out sense (negative R² in the US and most of
Australia); training on all 19 regions recovers the in-region skill at no cost
in the US and Europe. Details and the table: research log 2026-11-11 22:10.

Checkpoint (2026-11-18): the transfer grid with intervals and the drift
figure. If skill holds across years and continents the paper's claim becomes
"a recipe found on one year of one coastline transfers"; if it does not, the
failure pattern by latitude and tidal regime is the finding.

---

## Phase 2 — The wind-history experiment (2026-11-18 to 2026-12-16)

Goal: one experiment that yields a mechanism, not only a skill number.
Ubelmann 2025 shows that the wind-driven surface current depends on about
eight days of wind history through a latitude- and season-dependent
response function; instantaneous wind only captures the steady Ekman part.
Our grouped importances already rank wind second for v. This is the
experiment most likely to move skill and to produce a figure about the ocean
rather than about the network.

### Week of 2026-11-18 — features
- [ ] In `interp_sources`, sample ERA5 u10/v10 at the SWOT observation time
      minus each lag in {0, 3, 6, 12, 18, 24, 48, 96, 192} hours. Stencil
      k = 1 for wind (it is smooth at 0.25°): 18 columns.
- [ ] Add the Coriolis parameter f (or sin of latitude) and sin/cos of
      day-of-year so the model can learn the inertial phase and the seasonal
      mixed-layer change.
- [ ] Fold in the cheap extras from the paper notes in the same retrain: a
      70-km Gaussian geostrophic pair (Tchonang's CCS optimum), Rossby
      number, SSH Laplacian, cyclogeostrophic speed.

### Weeks of 2026-11-25 and 2026-12-02 — retrain and ablate
- [ ] Retrain RF, MLP, and the champion form with and without the lagged
      winds, under the Phase 1 protocol.
- [ ] Stacked explained-variance figure: DUACS, then SWOT geostrophy, then
      plus instantaneous wind, then plus wind history, then residual, for u
      and v separately.

### Week of 2026-12-09 — the response function
- [ ] Perturb one wind lag at a time and plot the velocity response against
      lag. Overlay on Ubelmann 2025 Fig. 8. A damped inertial oscillation
      with a seasonal amplitude change is the result to look for.
- [ ] Expect a stronger, more downwind, less rotated response than the
      drogued-drifter literature: HFR measures the top metre, where Stokes
      drift and wind slip live. State this in the paper before a reviewer
      does.

Checkpoint (2026-12-16): ΔR² table for the wind-history run and the
response-function figure.

---

## Phase 3 — Write the preprint (2026-12-16 to 2027-01-20)

Three figures carry the paper: the baselines table (Phase 1), the stacked
explained-variance bar (Phase 2), and the response-function overlay
(Phase 2). Everything else is supporting material.

- [ ] Draft against `docs/preprint-style-checklist.md`.
- [ ] Novelty claim, scoped: the first multi-region, temporally held-out,
      wind- and SST-ablated validation of ML total-velocity fields against
      HFR networks across both SWOT orbit phases, against DUACS and
      geostrophic baselines. Cite Zhou et al. 2025 and C. Wang et al. 2025
      explicitly; do not claim "first".
- [ ] State the temporal-holdout leakage finding (2026-07-21) in the
      abstract or methods.
- [ ] Disclosure list from `papers/NOTES.md` (L3 version, filter scales,
      HDOP mask, effective HFR depth, ERA5 and GOES products, DUACS
      version, split protocol and buffer).
- [ ] Target: JGR Oceans or GRL. Decide with advisor by 2027-01-06.

---

## Phase 4 — If Phase 2 finishes early, or for paper two

### Helmholtz output head (Berlinghieri 2023)
Output a streamfunction Ψ and a potential Φ on the stencil footprint and
differentiate to (u, v) with a fixed finite-difference layer; stronger L2
penalty on Φ to encode that vorticity dominates. Report vorticity and
divergence skill against HFR-derived fields. Expect ζ skill high and δ skill
low; that gap is the balanced-versus-unbalanced result and should be framed
as a finding. Also the natural entry point for an SQG depth extension
(Carli 2024) in a later paper.

### SST as gradients and tendency (Lenain 2026, GOFLOW)
Replace SST level with log|∇SST| and the ±1-hour tendency, plus a cloud
fraction column. Answers the question GOFLOW invites ("why do you need SWOT
at all?") with a number: the marginal gain of SWOT features over an SST-only
model. Needs GOES frames one hour either side of each pass; estimate the
pull before touching S3.

### The AlphaEvolve methods paper
Split off from the oceanography paper. The findings already exist in the
research log: the leakage discovery, evolution optimizing the systems layer
once the modeling layer saturated, wildcards buying the two plateau breaks,
the mutator grafting onto the champion and collapsing lineage diversity.
Venue: an AI-for-science workshop or JAMES.

### Deferred to separate papers
SQG projection of Ψ to depth; pretraining on LLC4320 and fine-tuning on HFR;
Lagrangian or navigation-based validation (Biferale 2019).

---

## Why this order (summary of the 2026-10-27 discussion)

Model skill is not the bottleneck; measurement is. Every result so far is
scored against raw hourly HFR, with an unbuffered split and no baselines.
Kachelein 2024 shows roughly half of the M2-band HFR energy in the
California Current is not phase-locked to the astronomical tide, so raw
hourly HFR carries a floor no snapshot model can reach; and the field's
currency is "percent better than DUACS against independent in-situ
velocity", which we cannot yet state. Fixing the protocol changes every
number, so it comes first. The wind-history experiment comes second because
it is cheap, the importances already point at it, and it can produce a
physical response function rather than only a skill delta. The Helmholtz
head and SST gradients are good ideas that cost more and are not needed for
the first paper. The evolution harness is a separate contribution whose
defense would distract an oceanography reviewer, so it becomes paper two.
