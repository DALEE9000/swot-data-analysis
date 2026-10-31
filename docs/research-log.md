# Research Log

Dated notes on findings, experiments, and decisions for the SWOTxAI project.
Convention: newest entry first; one `##` section per date; record *why* alongside
*what*. Machine-readable experiment details live in `experiments/registry/` and
`experiments/evolve/*/database.jsonl` — this log is for the human narrative.

---

## 2026-10-31 — Repository cleanup plan

### 15:20 Group E leftovers; bathymetry and orbit data checked

David: "finish up with E" and "we don't need the bathymetry? or orbit data?"

Swept README, CLAUDE.md, .gitignore, pyproject and docs for paths to the moved
folders. CLAUDE.md still said `src/swot/` was used only by the example
notebooks; corrected to name the three modules the pipeline imports
(`download_swaths.find_swaths`, `data_loaders`, `swot_utils`). Ten `.gitignore`
rules for folders that no longer exist at the top level were removed and the
comment block rewritten to point at `archive/`.

`orbit_data/` is required: `config.py`, `batch.py`, `build_science_colocations.py`
and `find_swaths` read the two orbit shapefile zips to select passes. The 84
tracked files in `ne_10m_bathymetry_all/` (35 MB) are read only by
`src/swot/plotting_scripts.py`, which nothing imports, and that module
downloads and unzips the Natural Earth archive itself when the folder is
missing. The folder is dead weight in the working tree; untracking it would not
shrink clones because it is already in history. Left in place pending David's
call.

### 15:00 Cleanup group B: loose folders archived to Z:

David: "Z is plugged in."

The WD Elements disk was still flagged Offline by Windows from the 2026-10-30
eject (the flag persists across unplugging); an elevated `Set-Disk -IsOffline
$false` brought Z: back. Robocopy then copied the five large untracked folders
and the two RF animation folders to
`Z:\data_repository\columbia_oceanography\SWOT-data-analysis\archive\`
(old_movies, SWOT_hf_plotting, data as data_era5_grib, notebooks,
animations_rf/{uswc,usegc}; 87 files, 5.6 GB, zero failures). Byte totals and
file counts were compared per folder before anything was removed; all six
matched. The local copies went to the Recycle Bin rather than a hard delete,
so they are recoverable until it is emptied, and C: frees the 5.6 GB at that
point. The small folders (examples, images, scratch_notebooks) and the stray
`experiments/duacs_pull.log` moved into the gitignored `archive/` locally.

Top level is now 13 folders, down from 21: src, scripts, tests, docs, papers,
orbit_data, ne_10m_bathymetry_all, experiments, cache, frames, animations,
archive, plus .streamlit. Every group of `docs/cleanup-plan.md` is executed.

### 14:00 Cleanup groups A, C, D, E executed

David: "i've reconnected the Z drive. do A and D. what are the loose folders in
B?" then "and do C and E".

**Group D first (credentials).** The grep for the hard-coded AVISO SFTP password
found four text sites, not one: `src/swot/download_swaths.py` (function default
and a module-level dict), `src/swot/thredds.py`, and
`scripts/build_science_colocations.py`, which had the password as the fallback
of an `os.environ.get`. All four now read `AVISO_USER` and `AVISO_PW` from the
environment with no fallback. The README sentence pointing at the file for
credentials now names the two variables. Compiled `.pyc` files carrying the
literal were purged with every `__pycache__` in the tree. The password remains
in two commits of git history (`53e3c8c`, `0316896`); the owner of the account
should rotate it. History was not rewritten.

**Group A.** Deleted `build/`, root and test `__pycache__`, `src/swotxai.egg-info`
(editable install still imports), the empty `experiments/akns/`, the stale July
frame set `frames/usegc/ann/ann_20260718_112506_3e1298/`, and 224 compiled
candidate caches under `experiments/evolve/`. The five cache twin pairs
(`swot_features_science_<rid>_1_16.pkl` with and without the
`_evolve_gen031_00` run_id suffix) were not byte-identical: 1,700 of 38 MB
differ in gak, starting at byte 105. Loading both and comparing every array
showed the content identical for gak, glna, uswc, usegc; the byte difference is
pickle key order. Rather than delete one of each pair, the suffixed file was
replaced by an NTFS hard link to its twin, so both cache keys still resolve and
about 8 GB came back. prvi and ushi have only the suffixed file and were left
alone. The RF animation folders (57 MP4s under `animations/{uswc,usegc}/rf/`,
id001 to id016 and earlier) were not deleted: they are a complete record of the
spring random-forest work, and the plan's "id012 to id016" item would have cut
a sequence in half. Moved to Group B as "archive to Z:".

**Group C.** `git mv figures docs/figures/regions`; `docs/poster` to
`docs/figures/poster` with the two output-path strings changed in
`scripts/poster_figures.py` and `scripts/champion_importance.py`, then
`poster_figures.py` rerun to confirm it writes to the new folder (it does; the
importance JSON and plots moved with the folder and need no rerun);
`docs/research-update-2026-10-29.md` to `docs/updates/` with the research-log
link updated; `swot_documentation/` to `papers/handbooks/`;
`docs/poster/importance_run.log` to the evolve run folder it belongs to;
`gen_netrc.py` removed from git (NASA Earthdata helper copied from NeurOST,
nothing imported it, PO.DAAC is not a data source here).

**Group E.** README "File hierarchy" block redrawn to the actual tree (the old
one listed `pipeline.py` and other files that no longer exist) and a "Scripts
index" table added with one row per script and the folder it writes to;
`CLAUDE.md` gains a "Repository layout" rule (no new top-level folders, figures
under `docs/figures/<topic>/`, memos under `docs/updates/`, scratch under
`archive/`); `.gitignore` gains `/archive/` ready for Group B.

### 10:30 Cleanup plan drafted, nothing moved

David: "this repo has so much sprawl. multiple figures, nested folders...can
you create a plan to clean up."

Surveyed the tree (sizes, git tracking, which code reads which folder) and
wrote `docs/cleanup-plan.md`. No file was moved or deleted. Findings that
shaped the plan:

- 21 top-level entries, 9 of them output or scratch folders nothing reads
  (old_movies 2.3 GB, SWOT_hf_plotting 1.3 GB, data 1.1 GB of raw ERA5 GRIB
  already converted, notebooks, scratch_notebooks, images, examples, build,
  swot_documentation). Figures live in three places (figures/,
  docs/figures/baselines/, docs/poster/).
- cache/ holds 22 GB; five `*_evolve_gen031_00.pkl` files are the same size as
  their un-suffixed twins (about 8 GB), left by the champion animation run.
  To be confirmed byte-identical with cmp before deletion.
- experiments/ (86 GB) is internally consistent per region and is the
  pipeline's own cache_path layout; left alone except an empty akns/ and a
  stray log.
- Security: `src/swot/download_swaths.py` has a third party's AVISO SFTP
  username and password hard-coded and tracked in git. Flagged as the first
  thing to fix, with a note to have the owner rotate it.

Plan structure: Group A regenerable deletions, Group B moves of untracked
folders into a gitignored archive/ (3.6 GB better sent to Z:), Group C tracked
moves with the two script path strings that change (poster_figures.py,
champion_importance.py), Group D credential removal, Group E README layout
block, a scripts index table, and a CLAUDE.md rule against new top-level
folders. Output folders cache/, frames/, animations/ stay in place because the
pipeline writes there and moving them means editing working code.

## 2026-10-30 — Calval table, climatology and per-cell baselines

### 12:30 — Calval (1-day repeat) baselines table; two new rungs on the ladder
**Prompted:** "do the last item for phase 1, but explain what that is" and
"do we not have calval for the other regions too?"
**What calval is:** SWOT's first six months (Apr-Jul 2023) flew a 1-day
repeat over a few ground tracks; only uswc, usegc, gak and ushi radars sit
under them (glna/prvi have no calval passes). The existing calval evolve
run is `evolve_uswc_v1` (uswc only, 9 features incl. GOES SST, 81 stencil
columns, champion `gen020_02`, pooled fitness 0.361). Daily revisits mean
~84 training days per radar cell instead of ~12 cycles.
**Done (code, additive):** `scripts/baselines_eval.py` generalised to
single-region runs (features/stencil/cycle range read from the run's
`data_meta.json`; training rows from the run's own `data.npz`; sister
regions can be added from `solo/<rid>/`); `sample_duacs` concatenates every
`duacs_l4_{rid}_*.nc`; `scripts/duacs_pull.py --start/--end`;
`scripts/solo_vs_pooled.py --regions` accepts any preset region so the
calval champion can be retrained on usegc and ushi (running, 3 seeds; usegc
frozen at 1,784,455 train / 478,167 test rows). DUACS pulled for Mar-Aug
2023 over uswc, usegc, ushi (Copernicus, free, 270 MB). README updated.
**Protocol note:** calval training ends 2023-06-19 and the record ends
2023-07-10, so a 30-day buffer leaves zero held-out passes; the calval table
uses a 7-day buffer (25 of 39 passes, 117,018 SWOT-valid rows). Truth tiers
are not available (Z: drive offline), so the table is 25-h target only.
**Two new baselines, both missions:** (1) `cell_mean`, the training-period
mean (u, v) at each radar grid point: pure climatology, the score a model
gets for knowing only where it is. (2) `cell_linear_geos_wind`, a
cell-specific linear fit on (1, g_u, g_v, W_u, W_v), partially pooled
toward the global fit with a prior worth 5 rows (predictors standardised):
the cell-specific calibration slope and wind response a network could learn
from repeat tracks. Local fits cover 73 % of calval rows (median 71 rows per
cell) and 43-70 % of science rows (median 7-10 rows per cell, so the prior
dominates there).
**Results, headline protocol (R² u / v):**
- Calval uswc (7-d buffer): climatology 0.20/0.06; SWOT cal + wind
  0.04/0.26; DUACS cal + wind 0.11/0.29; ridge (81 cols) 0.10/0.14;
  per-cell linear 0.38/0.36; calval champion 0.43/0.56 [0.36,0.47] /
  [0.51,0.59]. N_eff ~550.
- Science usegc: climatology 0.37/0.55; climatology + wind 0.45/0.56;
  SWOT cal + wind 0.48/0.56; per-cell linear 0.48/0.59; DUACS cal + wind
  0.53/0.58; ridge 0.54/0.59; champion 0.57/0.64.
- Science uswc: climatology -0.01/-0.98 (sign flips: training Aug-Mar,
  test Apr-Jun, upwelling-season reversal); per-cell linear 0.25/0.11;
  champion 0.21/0.11.
- prvi, ushi: intervals too wide; DUACS + wind best in both.
**Reading (Findings 7-9):**
7. *Climatology carries most of the East Coast score.* A lookup of the
   training mean at each cell reaches 0.37/0.55; the champion's increment
   over climatology + wind is ~0.12 on u and ~0.09 on v. Nash-Sutcliffe R²
   against the held-out mean credits a model for knowing the mean field,
   so the paper must also report skill relative to the spatial
   climatology (the anomaly-skill convention). Caveat: this climatology is
   an eight-month, seasonally biased mean, not a multi-year one; on the
   West Coast it is worse than useless because the mean flow reverses
   between the training and test seasons.
8. *Under daily sampling the network earns its keep; under 21-day sampling
   it does not.* On calval uswc the champion beats every linear model by
   0.3 and the per-cell linear model by 0.05 on u and 0.20 on v; the
   per-cell rung shows that location-conditional linear response explains
   most of u, and that what remains is nonlinear and mostly in v. On
   science uswc the same region gives champion = ridge = 5-parameter
   linear = per-cell linear within 0.03. The difference is not the
   region, it is the number of visits per cell (84 vs 12) and, possibly,
   the SST feature the calval run had and the science run did not.
   Separating those two is a one-run experiment: science uswc with SST.
9. *The calval result is the strongest positive result in the project so
   far,* but it rests on one region, 25 held-out passes after the buffer,
   and a 3-week test window. The usegc and ushi calval retrains (running)
   decide whether it generalises.
**Figures:** all seven regenerated with the two new rungs (fig1/fig2 are
now 10-row ladders; fig6 has 15 model rows). Plan item ticked; figures
README table extended.

### 13:30 — fig7 (two orbits, one radar); fig1-3 switched to the headline protocol
**Done:** `fig_two_orbits()` added to `scripts/baselines_figures.py`
(`fig7_uswc_two_orbits`): science uswc (30-d buffer, pooled champion) beside
calval uswc (7-d buffer, calval champion), full ladder incl. the new
climatology and per-cell rungs. fig1, fig2, fig3 now default to
`buffer30d/swot_valid` so the figures match the headline protocol decided
on 2026-10-29 (they had been drawn on all four test cycles); captions
updated. **Why:** the paper's cleanest contrast is the same radar under two
sampling densities, and the figures should show the protocol the text
quotes. **Retrain progress:** usegc calval seed 42 scored 0.635/0.507 on its
unbuffered test rows; seeds 43-44 and ushi follow.

## 2026-10-29 — Truth tiers, DUACS baseline, and the paper figures

### 08:40 — Predictors written onto the figures
**Prompted:** "make it clear on the figures too". **Done:** in
`scripts/baselines_figures.py` the ladder labels (fig1, fig2) are now two
lines, model name then right-hand side and parameter count (e.g. "SWOT +
instantaneous wind / 1, g_u, g_v, W_u, W_v (5 par.)"; "ridge on 3x3
stencil / 72 stencil cols, L2 (73 par.)"; champion rows "72 stencil cols,
nonlinear"); fig3 legend names the predictors; fig6 row labels are
"model: predictors". A shared caption line (RHS_NOTE) states the target
(HFR 25-h u, v at the pass point and time) and the symbol key. fig3's
legend moved below the panels (it had grown wide enough to squeeze the
first panel). All seven figures re-rendered and inspected; README updated.
**Why:** a reader of the ladder has to see that the target never changes
and only the predictors grow, or the "what does nonlinearity buy" reading
of the ridge-to-MLP gap is not available from the figure alone.

### 08:20 — "What is regressed against what" table
**Prompted:** "so what is regressed against what?" then "can we make that
clear". **Done:** new section in `docs/figures/baselines/README.md` with
one row per `results.md` model key: the predictors on the right-hand side,
the number of fitted parameters per component, and a one-line note (e.g.
`geos_cal+wind_lsq` = 1, ug, vg, Wu, Wv; `ridge72_solo` = the 72 stencil
columns; `geos+wind_lsq` fits wind to the residual HFR - g with geostrophy
at slope 1). `results.md` header now points to that section (one-line
change in `render_md`, re-rendered with `--render-only`). **Why:** the
ladder only means something if a reader can see that the target is fixed
and only the right-hand side grows from row to row; that was implicit in
the code and nowhere in prose.

### 08:00 — fig6: heatmap of the full benchmark grid
**Prompted:** "holy crap so many numbers in this results md, why" ... "can
you make a visualization of all the different benchmark 'tests' you
create in the 1000 rows?" **Done:** `fig_benchmark_grid()` added to
`scripts/baselines_figures.py` (additive; runs at the end of the script),
writing `docs/figures/baselines/fig6_benchmark_grid_{u,v}.{png,svg}`: six
region panels x 13 models x (5 truth tiers x 4 protocols), tile colour and
label = R², headline column boxed, unscored cells grey. 892 scored cells
for u. README entry added. **Why:** the thousand-row table is the full
factorial of region x tier x protocol x model; a single page that shows
the whole grid makes two things checkable at a glance that the table
hides: the ladder ordering is stable across protocol columns on the two
large regions (the conclusions do not hinge on the protocol decision),
and the small regions flip sign between adjacent columns. Not a paper
figure; a supplement or an internal audit view.

### 07:30 — Three-seed region-only retrain finished; baselines and figures regenerated
**Prompted:** follow-through on "ok what else to do in this first phase";
the multi-seed `solo_vs_pooled.py --seeds 42,43,44` run launched then
(regions uswc, usegc, prvi, ushi) has finished.
**Done:** `scripts/baselines_eval.py --bootstrap 1000` rerun; it found and
averaged the three seeds per region (`solo/<rid>/seed4{2,3,4}/pred.npz`),
so the `champion_solo` row in `baselines/results.{json,md}` is now a
three-seed mean on every protocol and tier. `scripts/baselines_figures.py`
rerun; the "region-only retrain" bars in fig1 to fig3 carry that mean.
Figures README, plan checkbox and the research update edited to say so.
Per-seed scores stay in `solo/comparison.md` (log: `solo/multiseed_run.log`).
**Results (headline protocol, 25-h target, SWOT-valid rows, 30-day buffer;
R² u / v, pooled vs region-only mean):** uswc 0.206/0.112 vs 0.240/0.103;
usegc 0.571/0.643 vs 0.563/0.635; prvi 0.138/-0.053 vs 0.058/-0.143;
ushi 0.172/0.129 vs 0.177/0.139. Seed spread (solo, unbuffered test set):
uswc u 0.235 to 0.240, usegc within 0.01, prvi u 0.033 to 0.056, ushi u
-0.138 to 0.003 and v 0.087 to 0.182.
**Reading:** (1) Seed noise is small where the data are large (uswc, usegc:
under 0.01) and comparable to the pooled-vs-solo gap where they are small
(ushi spread 0.14 on u). So the one-seed row was already representative
for the two regions the paper leads with; Puerto Rico and Hawaii remain
too small for a pooled-vs-solo verdict either way. (2) Pooled training
beats region-only in usegc, prvi and ushi and loses in uswc by 0.03 on u,
every difference inside the bootstrap interval. Transfer across regions is
a small positive at best, not a selling point; the paper should report
pooled as the default and this comparison as a robustness check. (3) The
region-only West Coast number (0.240) equals the ridge on the same 72
columns (0.217 [0.16, 0.27]) within CI, which reinforces Finding 4: on the
West Coast the network adds nothing the linear model does not.
**Phase 1 remaining:** calval-phase table.

### 06:30 — Headline protocol decided; evolve fitness redefined to it; research update written
**Prompted:** "yes do 25-hour buffered now. just make the advisor memo a
'research update' with the date." **Decision (David):** the paper's
headline protocol is the 25-h detided HFR target, SWOT-valid rows (finite
centre-cell L3 geostrophy), 30-day buffer after the last training pass,
per-region R² with block-bootstrap CIs; the 18-h tier reported beside it;
gak and glna excluded.
**Done (code, additive):** (1) `step_flatten` now writes `entry["time"]`
(the SWOT pass time) into every flatten entry. (2) `prepare_pooled_data`
replays per-row pass times for train and test rows (`_row_times`; legacy
caches without the key fall back to the SWOT pass list by position, exact
for unthinned cycles), saves `test_time` (int64 s) in `data.npz` and
`region_train_end` in `data_meta.json`. (3) New
`harness.protocol_fitness()` = mean over regions of per-region
mean(R²_u, R²_v) on rows with finite centre-cell SWOT geostrophy and pass
time >= train_end + buffer; regions with < 50 such rows are skipped. (4)
`EvolveSettings.fitness` ∈ {pooled (legacy), per_region_buffered};
`scripts/evolve.py --fitness per_region_buffered --fitness-buffer-days 30
--regions uswc,usegc,prvi,ushi` (the `--regions` subset applies to both
the pooled data prep and the fitness). Candidates record `fitness_pooled`
and the full `fitness_protocol` breakdown alongside the new `fitness`.
**Verification:** `protocol_fitness` applied to the existing champion with
per-row times reconstructed from the baselines replay reproduces
`baselines_eval.py`'s buffer30d/swot_valid champion numbers to 1e-6 in
all four regions (uswc 0.206/0.112, usegc 0.571/0.643, prvi 0.138/−0.053,
ushi 0.172/0.129; protocol fitness of the current champion = 0.2398 vs
its pooled 0.4458). A mock-mutator smoke run on a prvi+ushi pool
(`--fitness per_region_buffered`, CPU, 1 generation) froze `test_time`
and `region_train_end` and scored candidates with the protocol
(fitness 0.021 vs pooled −0.009 for the same predictions); smoke dir
removed. **Why per-region mean rather than pooled:** the pooled R² was
93 % usegc+uswc rows and rewarded between-region variance (07-23
decomposition); the per-region mean selects for mappings that transfer,
matches how the paper reports skill, and makes the fitness a number the
paper can quote. Caveat: prvi/ushi contribute equal weight with far
fewer rows and wide CIs, so a noisier fitness — acceptable for selection,
and the per-region breakdown is stored for inspection.
**Also:** `docs/updates/research-update-2026-10-29.md` — the dated research
update for the advisor (Phase-1 table, the five findings, what changes,
open questions).

### 05:00 — Champion provenance made explicit in the figures and tables
**Prompted:** "what is the scored MLP in the figures?" / "MLP meaning?" /
"can we make clearer where the MLP is from somewhere". **Done:** the two
network rows are now labelled "AlphaEvolve champion MLP, pooled training"
and "..., region-only retrain" in fig1/fig2, the fig1 and fig3 captions
name candidate gen031_00 of run evolve_all_science_v1 and say the pooled
row uses the campaign predictions while the retrain row is the same code
trained on one region; `docs/figures/baselines/README.md` gained a section
"Where the AlphaEvolve champion MLP comes from" (search setup, code path,
model form in words, what each row is, why the production RF/MLP are
absent); `results.md` header states the same. Rendering note: cmr10 has no
underscore glyph, so identifiers in captions go through `\mathtt{}`.

### 04:30 — Re-pairing results: prvi loses 4.4 % of training rows, held-out rows unchanged
**Done:** rebuilt the prvi and ushi flatten caches with time-paired passes
(prvi: 6 + 8 SWOT passes in cycles 004/005 had no HFR field within 1 h
and were skipped; ushi: 5 skipped), re-froze the solo datasets, retrained
the solo twins (seed 42), re-ran the full baselines evaluation with the
replay now calling the pipeline's own `_pair_by_time` (so the evaluation's
row reconstruction can no longer drift from the pipeline), regenerated
the five figures. **Before/after:** prvi training rows 168,021 → 160,683;
test rows identical in both regions (byte-equal `Y_test`), so every
held-out number for the pooled champion and the SWOT/DUACS baselines is
unchanged. ushi's dataset is byte-identical — the single radar field in
its thinned cycle 004 matched the same pass under both rules, and cycle
005 had no radar at all. The only moved numbers are the solo twin and the
training-row-fitted baselines on prvi: solo twin −0.008/−0.055 → 
+0.056/−0.092 (u/v, SWOT-valid rows), ridge 0.079/−0.179 → 0.076/−0.183,
calibrated geostrophy 0.033/0.070 → 0.037/0.070 — all inside the
bootstrap intervals. **Conclusion:** the pairing bug contaminated ~4 % of
prvi's training rows and nothing else that reaches the paper; the fix is
a correctness fix with no effect on the Phase-1 findings. The pooled
champion's training set still contains those ~7 k mispaired prvi rows
(0.2 % of 3.8 M); it will be retrained on clean data in Phase 2.
Mispaired artifacts kept for provenance under `*.mispaired_20261029` and
`solo/_mispaired_20261029/`; delete when no longer wanted.

### 03:30 — Pass-pairing bug fixed in `step_flatten`; gak and glna dropped from SWOT-skill results
**Prompted:** "ok fix it. i thought we drop great lakes? and gak?" after the
10-28 discussion of thresholds. **Decision:** gak and glna are dropped from
every SWOT-skill claim and from the next pooled training (no SWOT L3
geostrophy at their radar cells; 10-28 Finding 3). The pairing fix still
goes in because prvi and ushi carry the same bug and it is a pipeline
correctness fix.
**Done:** `src/swotxai/pipeline/steps_ml.py` — new `_pair_by_time()`
replaces the positional `zip(hfr_list, swot_list)`: each SWOT pass is
paired with the HFR entry nearest in time and used only if the gap is
<= 1 h (`_PAIR_MAX_HOURS`); unmatched passes are skipped and counted in
the progress message; entries without a time coordinate fall back to
positional pairing so legacy pickles still load. Self-check in the
scratchpad: a thinned HFR list re-pairs correctly and the skipped pass is
reported. **Thresholds, with reasons:** time 1 h — the radar is hourly and
a pass crosses a box in ~1 min, so a true match is <= 0.5 h by
construction and anything beyond 1 h is a shifted pair; the build's 36 h
is a data-loading window, never a match tolerance. Space — unchanged:
linear interpolation from the 6-km radar grid to the SWOT grid returns
NaN unless all four surrounding radar cells are valid, i.e. an implicit
one-radar-cell rule, consistent with Kachelein's 15–30 km decorrelation
scales; disclose in the methods that the finer SWOT grid oversamples the
radar (why uncertainties come from the pass-level block bootstrap, not row
counts). Optional refinement for the raw hourly tier only: nearest-
neighbour sampling instead of linear, to avoid a small spatial smoothing.
**Scope of the damage:** all mispaired cycles (glna 003–011, prvi 004/005,
ushi 004/005) are *training* cycles; the held-out cycles 13–16 were clean
in every region (verified by the replay), so the Phase-1 evaluation rows
were already correct; only some prvi/ushi training rows were wrong (and a
small share of the pooled champion's training set — not retrained now;
the Phase-2 pooled retrain, which also drops gak/glna, will use fixed data).
**Rebuild launched:** old prvi/ushi flatten caches renamed
`*.mispaired_20261029`, old solo artifacts moved to
`solo/_mispaired_20261029/`; `solo_vs_pooled.py --regions prvi,ushi`
rebuilds the flatten with time pairing, re-freezes the solo datasets and
retrains the solo twin (seed 42). Then `baselines_eval.py` reruns for the
two regions (training-row baselines and the solo twin change; the
champion's held-out predictions do not).

### 02:00 — Phase 1 extended: four truth tiers and the DUACS column
**Prompted:** David pointed the project at the hourly HFRNet files on his
external drive (`Z:\data_repository\columbia_oceanography\HFR\Code\Data`)
and created a Copernicus Marine account; "backup other data i have to here
as well". **Done:** (1) `scripts/hfr_tiers.py` builds, for every held-out
row, the raw hourly radar field at the hour nearest the pass, an 18-h and a
48-h zero-phase Butterworth (order 2, `filtfilt`) low-pass from the
continuous hourly record (±10 days of padding, gaps ≤ 6 h interpolated,
cells with < 80 % coverage within ±2 cutoff periods masked), and a rebuilt
25-h running mean. **Verification:** the rebuilt 25-h mean reproduces the
stored target with r = 1.0000 and zero RMS difference on prvi, uswc and
usegc (ushi r = 0.999, 45 of 12,806 rows differ — the target came from a
slightly different file set), which validates the whole chain from hourly
files → rolling mean → interpolation → frozen `Y_test`. (2)
`scripts/duacs_pull.py` fetched DUACS L4 (reprocessed, 1/8°, daily;
`cmems_obs-sl_glo_phy-ssh_my_allsat-l4-duacs-0.125deg_P1D`) for all six
regional boxes, July 2023 – June 2024, 550 MB total, free; all 366 days
came from the reprocessed series so no NRT fill was needed. (3)
`scripts/baselines_eval.py` gained the DUACS baselines (raw, calibrated on
training rows, calibrated + instantaneous wind; the training-row metadata
is replayed and verified against the solo `Y_train` exactly) and a
truth-tier loop that scores every model against all four truths on the
common subset of rows where every tier is finite. (4) Backup of
`experiments/` and `cache/` (~80 GB) to the external drive via robocopy in
copy-newer mode (never deletes).

**Finding 5 — DUACS matches or beats SWOT L3 at the radar.** On SWOT-valid
held-out rows, calibrated DUACS geostrophy vs calibrated SWOT geostrophy
(R²_u/R²_v): usegc 0.46/0.54 vs 0.39/0.51; uswc 0.09/−0.15 vs 0.10/−0.35;
prvi 0.16/0.18 vs 0.03/0.07; ushi 0.15/0.47 vs 0.16/0.20. Adding
instantaneous wind to DUACS (5 linear parameters) gives usegc 0.51/0.56,
uswc 0.24/0.19, prvi 0.19/0.07, ushi 0.20/0.55. **The evolved SWOT-based
champion beats DUACS + wind only on usegc (0.53/0.61 vs 0.51/0.56, u
inside the CI overlap), ties it on uswc u (0.24 vs 0.24) and loses on
uswc v (0.12 vs 0.19), prvi v and ushi v.** Reading for the paper: at
coastal radar cells, the wide-swath L3 geostrophic velocity as currently
used (centre-cell level features, 3×3 stencil, instantaneous wind) adds
little over the conventional nadir-altimeter maps, and the DUACS maps have
no nadir gap, no swath edges and a longer effective averaging window.
This is the honest Phase-1 headline and it reshapes the paper's claim:
the contribution has to be shown in a band where DUACS cannot compete
(sub-100-km high-passed skill, Jensen's logic) or with features that use
SWOT's resolution rather than its level (70-km smoothed pair, slope and
curvature fields, Phase 2 wind history). Caveat: SWOT L3 is cross-
calibrated to DUACS maps during processing, so the two are not
independent at large scales; and DUACS at 1/8° is smooth, which is itself
an advantage against a 25-h-mean truth.

**Finding 6 — skill vs the truth's timescale (common rows).** usegc
champion: raw 0.31/0.30 → 18-h 0.53/0.48 → 25-h 0.62/0.63 → 48-h
0.64/0.63. The ~0.3 gap between raw hourly and the detided tiers is the
sub-daily (tidal + near-inertial) floor no snapshot model reaches; the
18-h tier, which keeps near-inertial motion, sits halfway, so roughly half
of that floor is near-inertial — exactly the band the Phase-2 wind history
targets. uswc is flat and low across tiers (0.22 → 0.28 → 0.29 → 0.30 for
u; v ≈ 0.03–0.09): the floor is not the West Coast's problem; the
reachable slow signal is simply small. prvi and ushi curves are flat or
fall with more smoothing. Note the common-row subset (27 % of usegc rows,
the well-covered cells) scores higher than the full table — report both.

**Figures for the paper:** `docs/figures/baselines/` — fig1 skill ladder
with block-bootstrap CIs; fig2 % RMSE change vs calibrated SWOT geostrophy;
fig3 skill vs truth tier; fig4 geostrophy-overshoot joint densities (slope
0.25–0.53, σ ratio 1.2–1.4); fig5 coverage map (fraction of passes with
SWOT geostrophy per grid point; the nadir gap and swath edges show as
diagonal bands). `README.md` alongside describes each figure and how to
read it. `scripts/baselines_figures.py` regenerates all five.

**Open decisions carried forward:** pairing-bug fix location; whether to
evaluate in a sub-100-km high-passed band (needs along-swath spectra —
Phase 2 adjacent); which truth tier becomes the headline (recommend 25-h
target for continuity with the training, 18-h reported alongside).

---

## 2026-10-28 — Dated research plan written

### 17:30 — docs/glossary.md
**Prompted:** "in general, list a glossary of acronyms in my research
notes." **Done:** `docs/glossary.md` — ~130 acronyms harvested by regex from
papers/NOTES.md, the research log, the research plan and the preprint
checklist, defined and grouped (altimetry/SWOT, dynamics, ground truth,
other inputs, statistics, ML, venues, project). Products name their owner
for citation.

### 16:30 — Phase 1 results: the baselines table (science, cycles 13–16, 25-h detided HFR)
**Done:** full six-region run of `scripts/baselines_eval.py` (B = 1000
block bootstrap over (cycle, pass)); replay verified exactly in all six
regions; usegc pairing confirmed clean (so the pairing bug is confined to
glna/prvi/ushi and only to cycles the HFR build thinned). Added after the
first pass: a *calibrated* geostrophy baseline (per-component a + b·ugos
fitted on training rows) and a calibrated-geostrophy-plus-wind linear
model (5 parameters per component), because raw L3 geostrophy turned out
to be a worse predictor than the mean (below), which makes "% better than
raw SWOT" meaningless. Table: `experiments/evolve/evolve_all_science_v1/
baselines/results.md` (full JSON alongside).

**Result 1 — raw SWOT L3 geostrophy overshoots the radar everywhere.**
On SWOT-valid held-out rows, R² of ugos/vgos used directly as a prediction:
uswc −1.17/−0.99, usegc −0.15/+0.06, prvi −0.44/−2.26, ushi −0.88/−1.33.
Yet it is correlated with the radar (Pearson r: uswc 0.32/0.34, usegc
0.63/0.72, prvi 0.32/0.29, ushi 0.42/0.49). The mismatch is amplitude and
offset: the L3 velocity has ~1.3× the radar's standard deviation (uswc
0.147 vs 0.114 m/s for u) and misses the mean alongshore flow (uswc v mean:
radar −0.139 m/s, geostrophy −0.056). Least-squares slopes of radar on
geostrophy are 0.25–0.30 (uswc, prvi, ushi) and 0.46–0.53 (usegc) — the
attenuation a regression applies to a noisy regressor. Reading: at coastal
radar cells the 20–30-km-filtered L3 velocity is dominated by small-scale
SSH-slope noise (velocity is a derivative), consistent with Tchonang's
finding that 70-km smoothing is the CCS optimum against in situ velocity.
Calibrated geostrophy explains R² 0.10/−0.35 (uswc; v negative because a
single slope can't fix the offset structure), 0.39/0.51 (usegc), 0.03/0.07
(prvi), 0.16/0.20 (ushi).

**Result 2 — the champion vs linear physics, SWOT-valid rows, all test
cycles (R²_u/R²_v; %ΔRMSE vs calibrated geostrophy):**
- usegc: calibrated geos 0.39/0.51 → +wind linear 0.44/0.53 → ridge on
  72 stencil cols 0.52/0.56 → **champion 0.53/0.61** (−12.4 %/−11.0 %
  RMSE), solo twin 0.52/0.60. 95 % CI on the champion: u [0.44, 0.61],
  v [0.54, 0.67]. Genuine, significant skill; the nonlinear model adds
  ~0.09 over the 5-parameter linear and ~0.05 over the linear stencil.
- uswc: calibrated geos 0.10/−0.35 → +wind linear **0.24/0.11** → ridge
  0.24/0.12 → champion 0.24/0.12 (−8.2 %/−19.1 %), solo twin 0.27/0.09.
  CI u [0.19, 0.30], v [0.03, 0.19]. **A five-parameter linear model
  (calibrated geostrophy + instantaneous wind) matches the 45-hour evolved
  champion on the West Coast.** Nonlinearity and the spatial stencil add
  nothing there; wind adds most of what there is (v: −18.5 % RMSE from
  wind alone). This is the strongest argument yet for the wind-history
  experiment (Phase 2): the West Coast signal the model can reach is
  ageostrophic, and the only wind information it has is instantaneous.
- prvi: everything is near zero (champion 0.13/−0.08; CI_u [0.02, 0.18],
  CI_v includes 0 only on the low side). ushi: champion 0.09/0.16 with
  CIs spanning zero (u [−0.51, 0.21]). Neither supports a skill claim.
- gak / glna: no SWOT coverage (10-28 14:00 entry); champion CIs span
  zero or are deeply negative; omit from SWOT-skill claims.

**Result 3 — the 30-day buffer does not cost skill.** With the buffer
(cycles 15–16 only; 44–92 passes), usegc champion rises to 0.57/0.64 and
uswc stays 0.21/0.11; every region's buffered numbers sit inside the
unbuffered CIs. Training-adjacent cycles are not inflating the holdout
at this scale; the pooled 0.4458 is a forward-skill number, not a
leakage artefact. (Kachelein's ~6-month tidal-state window remains
untestable with 16 cycles — state as a limitation.)

**Result 4 — effective sample sizes are in the hundreds, not the
hundreds of thousands.** Block-bootstrap N_eff for the champion's errors:
uswc ≈ 700 (u) / 410 (v) from 429,643 rows and 67 passes; usegc ≈ 490 /
860 from 405,248 rows and 138 passes; prvi ≈ 90 / 65; ushi ≈ 40 / 30.
Roughly 5–10 independent samples per pass. Any significance claim in the
paper must use these; a naive N would make every 0.01 difference
"significant". The champion-vs-ridge gap on usegc (0.53 vs 0.52 for u)
is inside the CI; the champion-vs-calibrated-geostrophy gap is not.

**Caveats recorded in the table header:** R²_res is relative to Var(HFR −
raw L3 geos), which is inflated by the overshoot, so it reads high;
calibration coefficients are per region and fitted on training rows
only; "SWOT-valid" rows are 59–71 % of radar rows in the four usable
regions (the rest are swath edges / nadir gap / coastal mask where the
champion predicts from MDT + wind alone — scored separately in the
"all_rows" blocks, where the champion is 0.22/0.05 on uswc and 0.46/0.53
on usegc).

**What this changes in the plan:** (1) the paper's headline must be
per-region on usegc and uswc with CIs, not the pooled 0.4458; (2) Phase 2
(wind history) is now the critical experiment, because on uswc the
champion is exactly a linear geostrophy-plus-wind model; (3) a 70-km
geostrophic pair is worth adding to the same retrain to test whether the
L3 overshoot is scale noise; (4) DUACS and the 18-h/48-h tiers still
need downloads and approval.

### 14:00 — Phase 1 begun: baselines evaluation script + three findings from mapping the data
**Prompted:** "i agree with first establishing baselines. can you work on that
and log major things that you do?"
**Done:** `scripts/baselines_eval.py` (new, additive; pipeline untouched).
It scores the pooled-science champion `gen031_00` and its solo-trained twin
(from `solo_vs_pooled.py`) against the baselines the field reports — SWOT
L3 geostrophy (centre stencil cell), geostrophy + classical Ekman (3 % of
wind speed, 45° right), geostrophy + a least-squares instantaneous wind
response fitted on the region's training rows, a ridge on all 72 stencil
columns (the linear-skill reference), a wind-only linear fit, and the
training-period mean — all on the SAME held-out rows (the frozen evolve
test split, cycles 13–16), under two temporal protocols (all test rows;
rows ≥ 30 days after the last training pass) and two row sets (every row;
rows with finite centre-cell SWOT + ERA5). Metrics: R² (Nash–Sutcliffe with
the held-out mean), RMSE in m/s, % RMSE change vs geostrophy, R² on the
residual HFR − ugos. 95 % CIs from a block bootstrap over (cycle, pass)
snapshots (B = 1000), plus an effective-N estimate (iid variance of the
mean error / block-bootstrap variance of it). Self-check in `--selftest`
(metric identities vs sklearn; N_eff ≈ N for independent blocks and ≈ number
of blocks for block-correlated errors). Run command in the README.
**Design choice — no retraining:** the frozen `data.npz` holds no per-row
metadata, so the script replays `step_flatten` + `concat_for_ann` row
construction from the cached `swot_features` and `hfr_target` pickles
(dict order, list order, row-major grid, rows kept when either target is
finite) and asserts the replayed HFR values equal `Y_test` exactly. Verified
on gak and glna; the gak champion numbers reproduce the 08-24 log entry
(−0.74 / 0.19). Full six-region run launched in the background.

**Finding 1 — the HFR truth is already a 25-hour running mean, not raw
hourly.** The colocation build (`scripts/build_science_colocations.py`)
applies a centred 25-h rolling mean before interpolating to the pass grid
(documented in the README's HFR-target section; the 10-28 research plan
wrongly assumed raw hourly). Consequence for the paper: the 0.4458 was
measured against a crudely detided target. A 25-h boxcar has its first
spectral null at 25 h and sits at ~0.5 % amplitude at M2, so semidiurnal and
diurnal tides are essentially gone; near-inertial motion at 18–20 h is
attenuated to ~18 % amplitude (~3 % variance); the half-power point is
~56 h. So the existing target is closest to Tchonang's 48-h tier, with
sinc side-lobe leakage rather than a flat Butterworth passband. The
three-tier plan needs the continuous hourly record around each pass, which
exists locally only for usegc (`HFR/Code/Data/aggregated_hfr/usegc/*.nc4`,
57 GB, 1 km); other regions would need a THREDDS re-pull (free, but sized
and approved first). Kachelein's non-phase-locked floor is therefore
already mostly removed from the target; the paper should say "25-h detided"
everywhere and drop the raw-hourly tier from the plan.

**Finding 2 — a SWOT↔HFR pass-pairing bug in small regions.** `step_flatten`
pairs passes with `zip(hfr_list, swot_list)` by list position, but the HFR
build drops passes with no HFR sample within ±36 h, so when a cycle loses
HFR passes the lists shift. Mapped by the exploration pass: glna cycles
003/004/005/009/010/011, prvi 004/005, ushi 004/005 have unequal list
lengths; paired HFR-minus-SWOT time offsets reach +351 h (glna) and
+447 h (prvi); prvi cycle 005 pairs 0–2 are on different grids. uswc and
gak are clean (all pairs within ±0.5 h); usegc is being checked by the
replay. The evaluation script drops pairs with |Δt| > 2 h or mismatched
grids from scoring and reports the clean fraction. Not fixed in the
pipeline (working code; needs David's decision — the fix is to match on
pass time, not position, in `step_flatten` or in the HFR build).

**Finding 3 — gak and glna have (almost) no SWOT geostrophic coverage at
HFR cells.** Centre-cell `ugos_filtered` is finite on only 2 % of gak's
held-out rows and 0 % of glna's (uswc/usegc 69 %, prvi 59 %, ushi 71 %;
MDT 65 % / 0 %; ERA5 100 %). The L3 filtered geostrophic product is masked
where the radar lives in those two regions (coast / swath-edge / nadir
gap). So every gak and glna result in the project — including the 08-24
"pooling actively harmed gak" finding — is about rows where the SWOT
columns are NaN→0-imputed and the model sees only MDT and wind. First
numbers (B = 300): on gak, a wind-only linear fit scores v R² 0.40 and u
−0.02; the pooled champion −0.74 / 0.19; the solo twin −0.06 / 0.36; a
ridge on the stencil 0.07 / 0.38. The pooled champion's gak failure is
negative transfer of a mapping that expects SWOT inputs onto rows that
have none — not a statement about SWOT skill in the Gulf of Alaska. For
the paper: report gak/glna as "no SWOT coverage" and drop them from any
SWOT-skill claim; the cross-region story should be told on uswc, usegc,
prvi, ushi.

**Also noted:** the 30-day buffer with 21-day cycles means scoring on
cycles 15–16 only (cycle 13 starts ≈ 0 days after training ends, cycle 14
≈ 21 days); Kachelein's ~6-month tidal-state leakage window cannot be met
with 16 cycles and should be stated as a limitation. HFR source files are
HDOP-filtered at 1.0 by HFRNet (gak attrs: "Removed 9 solutions exceeding
HDOP threshold of 1"), stricter than Kachelein's 1.25, so the HDOP mask is
effectively done at source; no HDOP variable survives in the targets.
**Not done (needs download + approval):** DUACS L4 baseline; hourly HFR
for the 18-h/48-h tiers outside usegc.

### 10:30 — docs/research-plan.md
**Prompted:** "can we create an outline of next research steps, dated?
somewhere in this project directory" (following the 2026-10-27 discussion of
research direction based on papers/NOTES.md and the project state).
**Done:** `docs/research-plan.md` — four phases with target dates and
checkboxes. Phase 1 (to 2026-11-18): lock the evaluation protocol (buffered
temporal holdout in `step_evaluate`, three HFR truth tiers raw/18 h/48 h
with HDOP and coverage masks, DUACS/ugos/Ekman baselines with
Nash-Sutcliffe, percent-vs-DUACS and residual R2, calval and science
separately, block bootstrap over (cycle, pass) for effective N, drop glna,
multi-seed solo-vs-pooled, redefine evolve fitness on the new protocol).
Phase 2 (to 2026-12-16): lagged ERA5 wind history {0..192 h} plus f and
day-of-year plus the cheap derived features, and a perturbation-derived
response function to overlay on Ubelmann 2025 Fig. 8. Phase 3 (to
2027-01-20): draft the preprint around three figures. Phase 4: Helmholtz
head, SST gradients, and the AlphaEvolve methods paper as paper two.
**Why this order:** measurement, not model skill, is the current bottleneck.
Every existing number is scored on raw hourly HFR with an unbuffered split
and no baselines; Kachelein's non-phase-locked tidal floor and the field's
"percent better than DUACS" convention mean the protocol change rewrites
every result. The wind experiment follows because it is cheap and can yield
a physical mechanism (a learned response function), the differentiator
against Zhou 2025 and Wang 2025. Also on 2026-10-27, at David's request,
CLAUDE.md gained a project-only rule to explain research and science in
full prose rather than terse fragments.

---

## 2026-10-23 — Paper notes for the preprint

### 14:00 — papers/NOTES.md: structured notes on all 15 PDFs in papers/
**Prompted (from the oceanscout session):** "look in the papers in
/SWOT-data-analysis/papers and provide notes of insightful things for my SSV
project from each paper." **Done:** five parallel readers, three papers each,
every page read (PDFs are owner-password-locked; text extracted with PyMuPDF,
sources untouched). Per paper: what it does, numbers, methods to borrow,
direct SSV implications, caveats, verbatim quotes with page numbers,
products to match/disclose. A synthesis section at the top ranks the
actionable items.

**Why it matters / headline findings:**
- Two mislabeled files: "Sinha 2024" is Sinha & Abernathey 2021 (pure OSSE,
  no drifters); "Manucharyan 2026" is the 2027 Annual Review, not the 2021
  JAMES OSSE paper.
- Two direct competitors surfaced via the review: Zhou et al. 2025 GRL
  (SWOT + drifters ML velocity) and C. Wang et al. 2025 GRL (SWOT + HFR
  balanced/unbalanced). Novelty claim must be scoped to multi-region,
  temporal-holdout, wind/SST-ablated HFR-network validation, not "first."
- Biggest feature gap: Ubelmann 2025 shows the wind-driven gain needs an
  ~8-day lagged wind history; instantaneous ERA5 leaves most of the inertial
  band unexplained. Second: GOFLOW gets r² ~0.7 from log|grad SST| and its
  hourly change; SST level is probably the wrong representation.
- Kachelein 2024 gives a target-noise floor: 48% of M2-band HFR energy in
  the CCS is non-phase-locked (64% after FES2014). Report skill on raw and
  18-h/48-h low-passed HFR; mask HDOP > 1.25.
- Tchonang 2026: L3 filtered ugos ≈ 20–30 km Gaussian; CCS optimum vs in
  situ is 70 km. Add a 70-km geostrophic pair as features.
- Berlinghieri 2023 motivates a (Phi, Psi) Helmholtz output head; the
  current u/v feature asymmetry is the frame-dependence it warns about.

---

## 2026-10-16 — Preprint style and formatting checklist

### 18:30 — docs/preprint-style-checklist.md added
David asked (from the oceanscout session): "compile a comprehensive list of
stylistic, formatting things i need to do in my SSV preprint so it maximizes
likelihood of getting accepted to a top tier oceanography journal", encoded in
a markdown file. No preprint draft exists yet, so the checklist is written
against the journal rules rather than against a manuscript.

Sources: AMS formatting page (fetched directly), AGU rules via the agutex
template and third-party restatements (agu.org blocks scrapers; numbers marked
"verify"), AGU Open Research guidance from data.agu.org, section outlines and
availability statements extracted from the exemplar PDFs in `papers/`, the
ml-ssv literature review gaps, and an oceanscout `ask` on genre conventions
(110 sources; saved at `../oceanscout/answers/2026-10-16-what-do-the-most-cited-...md`).

Why it matters for the paper: the field's currency is "percent better than
DUACS against independent in situ velocity"; representativeness disclosures
(product version, orbit phase, filter scale, platform depth, wind and tide
corrections) are now reviewer expectations; bootstrap significance and
Lagrangian skill are the two differentiators almost nobody reports. The
temporal-holdout leakage finding (2026-07-21) must be stated in the abstract.

---

## 2026-09-26 — Poster figures: fitness trajectory + champion architecture

### (later) — Poster iteration: importance styling + simplified architecture
**Prompted:** color/style requests on the importance chart (rectangular bars
kept, teal/orange palette, value labels, ΔR² definition caption) and "this
architecture diagram is still too verbose for my poster... maybe have it go
side to side." **Done:** `champion_architecture_simple` — horizontal 7-box
version (inputs → physics features → ridge baseline → CNN/MLP residual pair
→ average+baseline → (u,v)), one idea per box, evolution provenance as the
single caption line. The detailed vertical diagram is retained for the paper.

### (later) — Champion feature importances (grouped permutation)
**Prompted:** "let's get a graph of the feature importances too." **Done:**
`scripts/champion_importance.py` — the champion has no importance API
(free-form candidate code), so: retrain once on the frozen pooled data (same
params, 750 s), then predict the temporal holdout plus 8 copies in which one
base feature's 9 stencil columns are row-shuffled *together* (grouped, shared
permutation — per-cell shuffling would leave the other 8 stencil copies
carrying the signal). Importance = drop in held-out R² per component. All
9 prediction sets (11.7 M rows) came from the single training call.
Checkpointed to `docs/poster/champion_importance.json`; figure alongside.
**Results (ΔR²_u / ΔR²_v):** ugos 0.870/0.054 — u skill is overwhelmingly
carried by the geostrophic u field; MDT 0.236/0.363 (largest for v; makes
sense pooled: the mean dynamic topography encodes which regime/current
system a row sits in); vgos 0.216/0.243 (both components — consistent with
cross-component coupling in the flow-regime features); ugosa 0.197/0.034;
ERA5 v10 0.100/0.130 (wind matters more for v, the weaker geostrophic
component); vgosa 0.084/0.032; ERA5 u10 0.062/0.044; **SSHA ≈ 0.01/0.01 —
near-zero: the level anomaly itself is redundant once its gradients are in
via the gos/gosa fields and the champion's own spatial-derivative features.**
That last point is a paper-worthy negative result: SWOT's information enters
through slopes, not levels. Caveat for the poster: permutation importance on
correlated features (gos fields are SSHA derivatives) splits credit
arbitrarily between correlates — state importances as "reliance," not
causal contribution.

**Prompted:** "i need to make a scientific poster with these results — fitness
graph from the app, but on a white background, and best model architecture in
a clear diagram" (+ "make sure the graphs have latex font"). **Done:** new
`scripts/poster_figures.py` → `docs/poster/` (PNG 300 dpi + SVG for print):
(1) `fitness_evolve_all_science_v1` — all 138 scored candidates of the pooled
run, chronological, viridis-colored by generation with the best-so-far
envelope; seed (0.294) and champion (`gen031_00`, 0.4458) direct-labeled.
Same content as the app's Evolve-tab plot but publication-styled (white
surface, Computer Modern). (2) `champion_architecture_gen031_00` — flow
diagram drawn from the champion's actual source (72-col input → 152-col
engineered features → ridge baseline → ConvWideDeep + HeteroMLP residual
ensemble → (u,v), training recipe footer). Deliberately NOT the app's regex
heuristic diagram, which has a documented false-positive history (07-21
"residual connections" mislabel). **Typesetting note:** LaTeX look via
matplotlib mathtext with the cm fontset rather than `usetex` — cmr10 lacks
glyphs for `—`/`_`/unicode math, so identifiers render through `\mathtt{}`
and symbols through mathtext; this keeps the figure build dependency-free
and fast while matching Computer Modern output.

---

## 2026-08-24 — Champion animations per region + pooled-vs-solo comparison design

### (afternoon) — Two new post-processing scripts for the pooled-science champion
**Prompted:** "1) create animations of each region using the BEST model
produced from the AlphaEvolve implementation; 2) [my advisor] suggests it is
worth comparing this model against individual models (of the same form)
trained in each region — how to proceed?"

**Done (1):** `scripts/animate_evolve_champion.py`. The evolve harness never
persists champion weights (candidates are train-and-predict subprocesses that
only emit held-out predictions), so the script retrains `gen031_00` exactly
once on the run's frozen pooled `data.npz` with the candidate's own
`params.json` (seed 42, 60-epoch cap, 3240 s budget — deterministic up to GPU
nondeterminism), and in the same call predicts every full-stencil grid row of
every cycle/pass in all six regions (enumerated from each region's cached
flatten, byte-identical to what the animation plotter builds). A tiny lookup
"model" (content-hash of the feature matrix → stored prediction slice) plugs
into the backend-agnostic `swotxai.models.predict` interface, so the existing
`step_inference` + `step_animate` pipeline runs unmodified — panel gating
(SST excluded, as in the run), output paths
(`animations/{region}/ann/evolve_gen031_00`), and frame styling all inherited.
Predictions are checkpointed (`anim_preds.npz`) so reruns skip the ~1 h
retrain. Animation titles carry the champion's *per-region* held-out metrics
from the run database, not the pooled 0.4458 — honest per-region skill on
screen. **Why this design:** don't touch working pipeline code; one training
instead of six; the hash lookup avoids fragile call-order coupling with the
plotter.

**Done (2):** `scripts/solo_vs_pooled.py` — the advisor comparison. For each
region in the pool: freeze that region's own dataset via the existing
`harness.prepare_data` (same temporal 80/20 split the pooled prep used, so
the held-out rows are IDENTICAL to the rows behind the pooled champion's
per-region breakdown), train the same `candidate.py` with the same params
via the run's own `run_candidate` subprocess, score with the same
`score_predictions`. Output: `solo/comparison.{json,md}` — per region,
pooled r2_u/r2_v vs solo r2_u/r2_v and the winner. Supports `--seeds a,b,c`
(mean across seeds) since single-seed deltas on small regions are noise.
**Fairness notes for the paper:** same wall-clock budget per model means the
solo models get far more epochs per row (budget is a cap; early stopping
active on both sides, so this is the "each side trains to convergence within
the same compute envelope" comparison, which is the operationally relevant
one). The champion's hyperparameters were evolved *for pooled data* — a solo
model of the same form is a transfer, so a solo win in a region is strong
evidence pooling hurt there; a pooled win is weaker evidence pooling helped
(a per-region evolve could still beat it). Expected axis of interest: usegc
(93 % of pooled skill mass) should be ~tied; small regions (gak, glna, prvi,
ushi) test whether cross-region transfer beats tiny-sample solo training;
glna is the physical outlier and likely solo-wins.

**Launched:** both scripts chained in one detached job (animations first —
GPU serialized), logs at `experiments/evolve/evolve_all_science_v1/
{anim_run,solo_run}.log`. Also added `.gitignore` cover for the new
regenerable `.npz` artifacts and README commands for both scripts.

### (late evening) — Pooled vs region-solo results (seed 42): interaction is
### gak-shaped, not uniform
**Result** (full table in `experiments/evolve/evolve_all_science_v1/solo/
comparison.md`; same candidate code/params, identical per-region temporal
holdouts):
- **Big regions are ties.** uswc: solo 0.137 vs pooled 0.133 mean R²
  (components split: solo better on u, pooled on v). usegc: pooled 0.494 vs
  solo 0.486. With ~1.7–1.9 M rows each, a region trains itself; pooling
  neither transfers skill in nor drags it down. The pooled 0.4458 headline
  is therefore not an artifact of borrowing between the big regions.
- **gak is the decisive case: pooling actively harmed it.** Solo on 2,394
  rows: −0.06/0.36 (u/v). Pooled: −0.74/0.19. The universal row-weighted
  mapping is simply wrong for gak's regime, and its 2 k rows can't bend the
  shared weights — the negative-transfer signature. A 2-minute solo model
  beats the 45-hour pooled champion there.
- **Mid-size regions get a small positive transfer.** ushi (34 k rows):
  pooled 0.081 vs solo 0.026. prvi (168 k rows): pooled 0.005 vs solo
  −0.050. Both effects are small and near the noise floor, but both point
  the same way: below ~200 k rows and above gak's scale, sharing weights
  helps slightly.
- **glna fails both ways** (−1.13 pooled vs −0.98 solo on 110 test rows) —
  no learnable geostrophic signal, as expected.
**Reading for the paper:** pooling is not uniformly good or bad — it's
neutral where data is plentiful, mildly positive at intermediate scale, and
harmful where the regime differs (gak). The `--fitness per-region` idea from
07-23 would push evolution toward transferable mappings; a cheaper fix is
dropping glna and either dropping gak or giving candidates a per-region loss
reweighting. Caveat: single seed; gak/glna/ushi deltas should be re-run with
`--seeds 42,43,44` before quoting (the script resumes and only trains the
new seeds).

### (evening) — Animations done: 5/6 regions; glna structurally unanimatable
**Result:** champion retrain + prediction of all 24,577,534 animation rows
took 1165 s — far under the 3240 s budget, because the champion's own
GPU-staging innovation (the throughput fix that won gen031) makes its
training converge and early-stop quickly. Animations delivered for uswc,
usegc, gak, prvi, ushi under `animations/{region}/ann/evolve_gen031_00/`,
titles carrying each region's own held-out R² from the run database.
**Finding — glna cannot be animated at k=3:** the Great Lakes box is so
small that *every* grid cell sits within stencil reach of a swath edge, so
the animation plotter's full-stencil rule (drop any row with a NaN feature)
leaves zero predictable cells; all 16 cycles skip and `step_animate` raises
"No frames were generated." Training was unaffected (the pooled prep keeps
rows with NaN stencil cells; the loss masks them) — glna contributes ~450
train rows but has no visualizable full-stencil area. This is a second,
independent reason (after the no-geostrophic-regime argument from 07-23) to
drop glna from the pool in future runs. Also: science-mission passes are
dense — uswc has ~343 (cycle, pass) blocks over 16 cycles (~21 passes per
cycle) vs calval's 2 — but animations render only passes [0, 1] per cycle
(`build_frame_dicts` default), so the MP4s show a thin slice of what the
champion actually predicted; the full 24.6 M-row prediction set is
checkpointed in `candidates/gen031_00/anim_preds.npz` if a denser
visualization is ever wanted. Script hardened mid-run (per-region try/except
+ explicit skip for regions with no full-stencil rows); the in-flight process
ran the old code, crashed on glna after gak, and prvi/ushi were finished by
an immediate checkpoint-resumed relaunch (no retrain, CPU-only) that ran
concurrently with the solo-comparison GPU job.

**Prompted:** "push this code to github, add to .gitignore anything you see
fit, but keep the logs public." **Done:** pushed the AlphaEvolve subsystem
(`src/swotxai/evolve/`, `scripts/evolve.py`, Evolve tab), this research log,
and the full evolution provenance of both campaigns — databases with every
candidate's idea/metrics/lineage, run logs, and all 224 generated
`candidate.py` modules (~4 MB) — to github.com/DALEE9000/swot-data-analysis.
Gitignored: the frozen `data.npz` arrays (0.4–1.5 GB, regenerable from the
pipeline) and per-candidate `predictions.npz` dumps. **Why:** the candidate
code + idea trail *is* the paper's evidence base; publishing it makes the
evolution auditable end-to-end.

---

## 2026-07-23 — Pooled-science campaign complete: 0.4458, and what "pooled skill" really means

### 20:07 — `evolve_all_science_v1` finished (launched 2026-07-21 23:05, ~45 h wall)
**Setup:** pooled science mission (uswc+usegc+gak+glna+prvi+ushi,
time-interleaved), 8 features (SST deselected by David in the new launcher),
k=3; 35 gens × 4 children, Fable via claude-cli, 60-min candidate budget;
first run with diagnostics-guided prompts + wildcard slots active.
**Result:** champion `gen031_00` — **fitness 0.4458** (r2_u 0.4055, r2_v
0.4861) vs seed 0.2943 = **1.51×**. 141 candidates, 138 trained OK; the only
3 failures were a network outage + laptop sleep on 07-23 (~14:37–15:37) that
the loop absorbed as rejected slots and continued (negative-timeout signature
in the log = system slept mid-`claude` call). 28 record-setting candidates;
progress was still nonzero at the end (last record gen031).
**Wildcards earned their slots:** 34 wildcard candidates averaged slightly
lower fitness than refiners (0.413 vs 0.423 — by design, they buy
information) but set 3 of the records, including the two decisive
plateau-breaks (gen012_03, gen022_03 → the 0.4436 lineage). The
honor-the-parent instruction held: no champion-grafting observed in ideas.
**Champion's closing innovation was *throughput*, not modeling:** it
diagnosed that parents were wall-clock-bound on CPU fancy-indexing + PCIe
transfer of every 4096-row batch (~3.4M×160 matrix), and restructured the
input pipeline so the LR schedule/early stopping actually get to operate
within the budget. The evolution optimized the *systems* layer once the
modeling layer saturated — a finding in itself.

### Key finding — decomposing pooled R² (prompted by David asking how overall
skill can be 0.4458 when per-region skill is poor). Champion per-region
(r2_u/r2_v): usegc 0.46/0.53, uswc 0.22/0.05, gak −0.74/0.19, glna
0.12/−2.37, prvi 0.08/−0.07, ushi 0.02/0.14. Measured decomposition:
uswc+usegc hold 93% of held-out rows; a region-mean-only baseline (zero
within-region skill) scores pooled R² 0.053 (u) / 0.124 (v) purely from
between-region variance; pooled R² = between-region bonus + variance-weighted
(not row-weighted) average of per-region R²s — usegc dominates via rows ×
variance × skill. Honest reading: strong genuine East/Gulf-coast skill,
modest uswc-u, near-zero transfer to small regions; glna (Great Lakes, no
geostrophic regime at these scales) is physically misplaced in the pool.
→ Candidate next steps: `--fitness per-region` (mean of per-region R²s,
  optionally excluding glna) to select for transferable skill; per-region
  loss reweighting inside candidates; compare champion vs calval champion
  cross-mission.

---

## 2026-07-21 (evening) — Evolve v2: pooled regions, architecture summaries, diagnostics-guided mutation

*(From here on, entries follow the logging convention David set at ~22:05:
date+time, what he prompted, what was done, and why — raw material for the
research paper.)*

### ~19:30–21:00 — Pooled multi-region evolve support
**Prompted:** "is there no way of combining all the regions in science or
cal/val at once?" **Done:** added `--pooled {calval,science}` to
`scripts/evolve.py` (`prepare_pooled_data` in `evolve/harness.py`): reuses the
existing `multiregion.py` machinery conceptually but concatenates per region
*before* pooling, so every region gets its own temporal 80/20 split; per-region
test slices are stored and every candidate now records a per-region R²
breakdown. Verified all local mirrors exist for both missions (no S3).
**Why:** cross-region generalization is a stronger test of whether evolved
techniques are physics or regional quirks; per-region splits prevent one
region's date range from leaking into another's holdout. **Finding:** likely
latent bug in the app's existing pooled path — `run_multiregion` drops `"df"`
from flattened entries (memory optimization, commit 17cf8ad) but ANN's
`concat_for_ann` requires it → pooled+ANN in the app likely raises KeyError;
pooled+RF unaffected. Not fixed pending David's decision. Memory estimate for
pooled calval: ~2.5–4 M rows ≈ 1–1.4 GB arrays, ~4–5 GB per candidate process
— comfortable in 64 GB; science pooled is *smaller* (16 cycles vs ~105).

### ~21:40–22:00 — Per-candidate architecture summaries in the Evolve tab
**Prompted:** "can you provide a summary and even a diagram of the model
architecture and hyperparameters involved for each candidate" (+ follow-up:
include the fixed run config, e.g. stenciling). **Done:** new
`evolve/inspect.py` — heuristic (regex/pattern) extraction of hyperparameters
and components from candidate source, rendered in the tab as a table +
Graphviz flow diagram (input → feature engineering → ridge/MLP/trees →
combine), plus a "Run configuration" expander (mission, region, cycles,
stencil k, features, rows, epoch cap). **Why:** candidates are free-form code;
a structured view makes the evolution legible. Labeled "heuristic — the diff
is ground truth." **Correction found while explaining the champion:** the
detector's "Residual/skip connections" label was a false positive (matched the
word "skipped" in a comment); the champion's MLP is a plain feedforward stack —
its "residual" is the statistical kind (predicting ridge errors), not ResNet
skips. Pattern tightened.

### ~22:05–22:25 — Diagnostics-guided mutation + wildcard exploration (v2)
**Prompted:** "it converged around 0.36, how do i improve the score? what can
i do to prompt the LLM to improve the performance?" then "can you encode these
features in the alphaevolve code?" **Done:** (1) `harness.diagnostics()` — every
candidate's held-out error is now binned by forecast horizon (early/late
test), flow regime (true-speed terciles), and observation quality (stencil
validity), stored in the DB and injected into the parent block of every
mutation prompt, with system-prompt guidance to attack the weakest bin.
(2) Wildcard slots — every Nth candidate (`--wildcard-every`, default 4) gets
an explore-don't-refine prompt (structurally new predictive core; plumbing
reuse allowed; "a lower score is acceptable — this slot buys information").
(3) Honor-the-assigned-parent instruction added, since run 1 showed the
mutator systematically grafting onto the champion, collapsing lineage
diversity (the gen 8–13 plateau). **Why:** a scalar fitness gives the proposer
almost no gradient; diagnostic bins turn mutation into targeted debugging.
Wildcards are the prompting equivalent of island models. Verified end-to-end
on the synthetic smoke set (wildcard fired at the scheduled index; 11
diagnostic keys per candidate). Also added the mandatory research-log rule to
CLAUDE.md at David's request.

### 22:25–22:40 — Evolve run launcher in the app (features/stencil/gens/children)
**Prompted:** "i want a UI option to change the stenciling and features in
evolve, with number of generations and children toggable too." **Done:**
(1) `scripts/evolve.py` gained `--features` and `--stencil-k` overrides
(re-validated through `SWOTConfig`; warns-and-ignores if the run's dataset is
already frozen, since datasets freeze at first launch by design). (2) The
Evolve tab gained a "Launch a new run" panel: run name, mutator (+CLI model),
data domain (config region / pooled calval / pooled science), feature
multiselect, stencil k, generations, children, epoch cap, timeout, wildcard
cadence. It spawns `scripts/evolve.py` as a **detached** Windows process
(survives app restarts, identical to a CLI launch, resumable), logging to the
run's `full_run.log`; the tab's existing monitor picks it up. Guards: run-name
validation, running-run collision block, and — for `claude-api` — a shown
dollar estimate with an explicit approval checkbox before `--yes` is passed
(the CLI's interactive cost gate can't prompt from a detached process).
**Why:** feature-set and stencil ablations were identified as the next
experiments; a form + button lowers the cost of running many. Verified:
override validation rejects unknown features; a detached mock run completed
end-to-end; headless app executes clean.

### ~22:45 — Pooled rows now time-interleaved across regions
**Prompted:** "so when i train on pooled science, the weights optimize across
all regions?" (conceptual question; answering it exposed a prep flaw).
**Done/why:** yes — one weight set, row-weighted loss over all regions'
stacked rows, and candidates are region-blind (no region label or lat/lon in
the features), which forces a universal mapping. But the pooled prep had
concatenated regions as sequential blocks, so candidates' evolved
temporal-tail validation / recency weighting ("later rows = later in time")
would have degenerated to "last region in the list." Fixed: regions are now
interleaved in 20 time-chunks (chunk i of each region covers the same
fraction of the shared mission window), so global row order ≈ chronological
across regions; per-region test rows are tracked as span lists and the
per-candidate region breakdown uses them. `row_order` documented in
data_meta so the mutator LLM knows. Unit-tested: row-index↔time correlation
0.999 after interleave; spans recover each region's rows exactly. Caveat
noted: pooled loss is row-weighted, so uswc dominates — watch the per-region
R² spread per candidate.

### ~22:55 — First UI-launched run crashed on Windows encoding; fixed + relaunched
**Prompted:** "is it running or did it hit an error?" (David had launched
`evolve_all_science_v1` from the new panel: pooled science, 35×4, claude-cli/
Fable, 8 features — SST deselected — k=3). **What happened:** the detached
process died during region 1/6 data prep with `UnicodeEncodeError` — Windows
gives redirected stdout the cp1252 codepage, and a pipeline progress message
contains "→". **Fix:** `scripts/evolve.py` now reconfigures stdout/stderr to
UTF-8 at startup; the app launcher and the candidate sandbox both set
`PYTHONIOENCODING=utf-8` as well (an LLM-written candidate printing unicode
could have crashed a run the same way). Relaunched with the reconstructed
command; region caches built before the crash are reused. **Lesson for the
paper's reproducibility notes:** long-running evolutionary harnesses need
locale-independent I/O — the crash came from a *log message*, not the science.

### ~22:50 — Region picker in the Evolve launcher
**Prompted:** "i want the region to be in the config UI on evolve." **Done:**
`scripts/evolve.py` gained `--region {uswc,usegc,gak,ushi,glna,prvi}` +
`--mission {calval,science}` (applies the region's preset paths/bbox/cycles
via `presets.config_overrides`, rejecting untrainable combos like
prvi·calval); the launcher's "Data domain" dropdown now lists every trainable
region×mission (10 single-region options) plus the two pooled modes.
Pre-checked that every region×mission has complete local mirrors (swot/hfr/
era5/goes) so no launch can touch S3. **Why:** removes the config.yaml-editing
step for cross-region evolve experiments; each domain still freezes its own
dataset per run name. Verified: gak·calval preset resolves correct paths;
prvi·calval rejected; headless app clean.



**Run:** `evolve_uswc_v1` — LLM-guided code evolution of the ANN training module
(mutator: Claude Fable 5 via `claude -p`, $0 API; 20 generations × 4 children;
uswc / calval cycles 474–578; all 9 features, stencil k=3; 1,018,355 train /
273,146 test rows × 81 inputs; candidates screened at 60-epoch cap on the
GTX 1650 Ti, ~5–7 min each).

**Final result (run completed 2026-07-21 10:14, ~11 h wall, 81/81 candidates
trained successfully, $0 API):** champion `gen020_02` — **fitness 0.3609**
(r2_u 0.320, r2_v 0.402) vs the **seed's 0.1700** = **2.12×** the production
architecture's forward-in-time skill, discovered autonomously overnight.
Progression of bests: 0.170 (seed) → 0.284 (gen1) → 0.306 (gen2) → 0.311
(gen4) → 0.329 (gen6) → 0.3339 (gen7) → *six-generation plateau* → 0.3502
(gen14, the MLP+tree hybrid breakthrough) → 0.3609 (gen20, still climbing at
the end — the run was stopped by schedule, not convergence; resuming with more
generations is likely to keep paying). Per-generation mean fitness rose 0.17 →
0.35 — the whole population improved, not just the champion. The champion's
final innovation: recency-weighted refit of the ridge *baseline* itself via a
leak-free "tail-stacked" ensemble interface it had built two generations
earlier.

**Techniques the winning lineage discovered/stacked** (each kept only after
surviving the temporal holdout; full code + per-candidate "idea" text in
`experiments/evolve/evolve_uswc_v1/`):

1. **Ridge residualization** — fit an exact linear ridge first; train the MLP
   on residuals only. (Linear part carries the quasi-geostrophic signal.)
2. **Physics-derived features** — wind pseudo-stress (|W|·W components from
   era5_u/v) and current-speed magnitudes computed inside the model.
3. **Missingness as signal** — per-base-feature stencil validity fractions
   (fraction of non-NaN cells in each k² block) appended as features; encodes
   position relative to swath edges.
4. **Temporal-tail validation** — early stopping validated on the *most recent*
   training rows instead of a random split (first idea Fable proposed, gen 1;
   matches the temporal test).
5. Component-balanced masked **Huber** loss (vs seed's masked MSE);
   recency-weighted fine-tune pass; small seed ensemble within time budget;
   EMA/Polyak + checkpoint weight averaging. Gaussian input-noise augmentation
   was tried and measurably regressed → dropped.

**Observed emergent behavior:** the mutator routinely overrides its assigned
parent and grafts onto the best program on the leaderboard ("I build on the
strongest verified program rather than the weaker assigned parent") —
accelerated convergence, reduced lineage diversity.

**Key methodological finding — legacy evaluation is inflated by leakage.**
The pipeline's `step_evaluate` pools *all* cycles (`training_percentage=1.0`)
and takes a random 20% test split, but training used the first 80% of cycles —
so most "test" rows were in the training set; and even a clean random split
lets the model interpolate smooth ocean fields. This explains registry R² ≈ 0.8
vs. 0.17 for the same architecture on a strict *temporal* holdout (last 20% of
cycles, never seen). Registry numbers are best read as "in-period fit";
temporal-holdout numbers as "forward skill on new passes."
→ TODO: consider fixing `step_evaluate` (or adding a `r2_*_holdout` metric)
   pipeline-wide; do not compare evolve fitness to registry R² directly.

**Infrastructure built (2026-07-20→21):** `src/swotxai/evolve/` + `scripts/evolve.py`
+ app "Evolve" tab. Agent-agnostic mutators: `--mutator claude-api | claude-cli |
codex-cli | mock` (+`--cli-model`); CLI mutators run on subscriptions ($0/token).
Candidates run in sandboxed subprocesses (AST import whitelist; AWS_*/ANTHROPIC_*
env stripped; hard timeout); fitness scored by the trusted parent from predictions
only; append-only DB → resumable. Anthropic API console org got auto-banned before
first credit purchase (appeal filed via support.claude.com) — irrelevant in
practice since `--mutator claude-cli` (Claude subscription) is the working path.

**Addendum (same day):** evolve now supports pooled multi-mission-wide runs —
`--pooled calval` (uswc+usegc+gak+ushi) / `--pooled science` (+glna, prvi):
per-region temporal splits stacked into one frozen dataset, fitness = pooled
mean R², per-region breakdown recorded per candidate. All local mirrors for
both missions verified present. **Possible latent bug found while wiring it:**
`multiregion.run_multiregion` pops `"df"` from flattened entries to save
memory (commit 17cf8ad), but the ANN path's `concat_for_ann` requires
`entry["df"]` — pooled + ANN in the app likely crashes (KeyError) since that
commit; pooled + RF unaffected. Not fixed (needs a decision on rebuilding the
joint X without df); evolve's pooled prep sidesteps it by concatenating before
dropping anything.

**Next steps:**
- [ ] When the run completes: retrain champion at full budget (200 epochs);
      also score it under the legacy random-split protocol for an
      apples-to-apples number vs the registry (~0.8 expected).
- [ ] Read the champion code end-to-end; consider porting the robust wins
      (ridge residualization, validity fractions, temporal-tail validation,
      Huber) back into `models/ann/` as config-gated options.
- [ ] Feature-set ablations as separate evolve runs (e.g. no-SST) — feature
      sets are frozen per run by design.
- [ ] Optional A/B: same run with `--mutator codex-cli` (needs `npm i -g
      @openai/codex`) to compare mutator quality.
- [ ] Revisit lineage diversity (mutator ignores assigned parents) if plateaus
      persist — e.g. instruct it to honor the parent, or add islands.

---

## 2026-07-20 — AlphaEvolve subsystem designed and built

Decided at design time: full code mutation (not just hyperparameters), LLM as
proposer, CLI-owns-execution + Streamlit read-only viewer, ANN backend first.
Fitness deliberately defined on a temporal holdout to make memorization
unrewarding for an LLM optimizing against the metric. Verified with a
mock-mutator smoke test on synthetic data (`experiments/evolve/smoke_mock`,
disposable) before any real run.

---

# Project history (reconstructed from the git record, 2026-05 → 2026-07)

*Backfilled 2026-07-21 from 46 commits. The repo works in bursts with large
checkpoint commits, so dates mark when work landed, not necessarily when it
was done.*

**2026-05-01 — Project born.** Initial commit: SWOT L3 altimetry → subsurface
velocity ML pipeline with random-forest backend, Streamlit GUI, HFR radar as
ground truth; legacy `src/swot/` orbit/swath tooling carried in from earlier
notebook work.

**2026-05-03 → 05-07 — The GPU week.** Rapid iteration on training/inference
backends: cuML GPU backend added (PR #1), an XGBoost refactor tried and
reverted, dual-GPU parallel u/v training tried and reverted to sequential,
cuML FIL inference fought through several "invalid configuration argument"
fixes before settling on auto-detect-cuML-with-sklearn-CPU-fallback, plus
LightGBM CUDA as an alternative GPU path. Same week: pipeline re-architected
around *template flattened data* (runs share a flatten cache — the ancestor of
today's `_flat_stem()` sharing), ERA5/GOES moved to preprocessed pkl artifacts
with S3 paths, vast.ai/nohup remote-training workflow documented.

**2026-05-08 — Train/test split scare.** "fixed critical error in training
percentage", `held_out` argument added to the concat helpers, then "restoring
original evaluation logic." Notable in hindsight: the `held_out` (temporal
tail) machinery built this day is exactly what the 2026-07-21 evolve harness
uses for honest fitness — while the "restored" random-split evaluation is the
one now known to leak (see 2026-07-21 entry).

**2026-05 → 2026-07 — Uncommitted growth + data campaign.** Two months of work
accumulated outside git, landing later as one checkpoint. Per the data-side
records: the July colocation campaign built the two-mission dataset (science +
calval × all HFR networks, 25 h targets at every resolution) on
`s3://swot-ai-ssv` — and streaming ~4 TB from S3 (~$290) taught the
never-stream-repeatedly / local-mirrors-only rule now codified in CLAUDE.md.

**2026-07-18 — The big checkpoint.** One 48-file, +5,776-line commit
("two-mission data campaign + app upgrades") followed by structural cleanups:
`SWOTxAI/` flattened into the repo root, `docs/model-data.md` added,
"local-first everywhere" (closed the ERA5/GOES S3-streaming gap), pooled-
training memory halved via float32 flattening. This checkpoint is where the
modern shape of the repo landed: partitioned rf/ann model backends (PyTorch
MLP with masked-MSE joint u/v training), the batch runner, the experiment
registry (`experiments/registry/{rf,ann}/`), and the five-tab app.

**2026-07-20/21 — AlphaEvolve.** See the dated entries above. First run
doubled temporal-holdout skill over the production architecture overnight and
exposed the legacy evaluation leakage — arguably the project's two most
consequential findings since the dataset itself.
