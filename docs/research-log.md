# Research Log

Dated notes on findings, experiments, and decisions for the SWOTxAI project.
Convention: newest entry first; one `##` section per date; record *why* alongside
*what*. Machine-readable experiment details live in `experiments/registry/` and
`experiments/evolve/*/database.jsonl` — this log is for the human narrative.

---

## 2026-11-12 — Finishing Phase 1b: seed spread and leave-one-region-out

### 11:15 David: "i need a really simple explanation of this written somewhere"

David asked why the bars in `loro_us` do not sit at the mean of the seed dots. The bar scores the seed-averaged prediction (an ensemble), the dots score each seed alone; averaging predictions cancels the seeds' independent errors, so the ensemble beats a typical single seed wherever the seeds disagree. A plain-language section was added at the end of `docs/glossary.md` covering seeds, ensembles against single-seed scores (with a toy example), how to read R² and %ΔRMSE, y1, LORO and cell climatology, plus LORO and y1 as glossary entries. The figure title now says the same in one line.

### 11:05 David: "are there figures that demonstrate these results"

The grid averages u and v, which hides the shape of the failure, so `docs/figures/holdout/loro_us` was added (`fig_loro` in `holdout_eval.py`): grouped bars of R² per component for uswc, usegc, prvi, ushi under the five training sets that matter, single-seed values as white dots. It shows in one glance that the leave-one-region-out bar drops below zero in u for three of four regions while staying near zero or positive in v, that the in-training dots sit on top of each other, and that the out-of-region dots scatter.

### 05:35 Results: the recipe does not transfer between US regions either; seed noise is small

Chain finished 05:31 (10 retrains, 5.2 h of GPU; `SWOT_L3/science/holdout_loro.log`). Results rescored into `experiments/holdout/results.{json,md}` and the four figures redrawn; `transfer_grid` now carries a "US w/o own region" column.

**Leave-one-region-out (LORO) within the US.** Each of the four scorable US regions, predicted by the champion recipe trained on year one of the other five US regions (R² u / v, %ΔRMSE against calibrated geostrophy; own-continent model in brackets):

| held out | US without it | US (own) | all 19 | not US |
|---|---|---|---|---|
| uswc | −0.36 / +0.01 (+23 / +9 %) | +0.29 / +0.41 | +0.28 / +0.41 | −1.50 / +0.28 |
| usegc | −0.43 / +0.35 (+44 / +9 %) | +0.46 / +0.53 | +0.45 / +0.54 | +0.17 / +0.09 |
| prvi | +0.12 / 0.00 (−1 / +1 %) | +0.18 / +0.07 | +0.17 / +0.06 | −0.10 / −0.50 |
| ushi | −0.01 / +0.03 (+6 / −1 %) | +0.20 / +0.06 | +0.17 / +0.02 | −0.01 / −0.43 |

So the answer to the question the experiment was built to ask is the sobering one. Holding out a single US network while keeping the hemisphere, the latitude band and the processing chain the same still destroys the skill: the held-out region is predicted worse than calibrated geostrophy in three of four cases, and the failure has the same shape as the cross-continent one, u collapsing to a negative R² while v keeps some skill. prvi is the exception, where the held-out model is about level with geostrophy, but prvi has little skill to lose. The recipe is therefore learning something specific to each network (or to each region's regime, which within the US is confounded with the network: the California Current, the Gulf Stream shelf, Puerto Rico and Hawaii are four different oceans), not a general map from SWOT geostrophy and wind to surface current. Adding a region to the training set fixes that region (the `all` column) without hurting the others, so the practical rule for the paper is "train on every network you intend to predict", and the claim of spatial generalisation is withdrawn rather than qualified.

**Seed spread.** Three seeds (42, 43, 44) for `us_y1`, `all_y1` and `not_us_y1`; single-seed R² is stored as `per_seed_r2` in `results.json` and printed in `results.md`. For the in-training models the spread is ±0.01 in R² (uswc us_y1 u: 0.282, 0.285, 0.284; usegc us_y1 v: 0.521, 0.520, 0.532), so the differences between `us_y1` and `all_y1` in the US (0.00 to 0.03) are within seed noise, as suspected. For the out-of-region model the spread is an order of magnitude larger (uswc not_us_y1 u: −1.35, −1.58, −1.71; usegc not_us_y1 v: +0.13, +0.14, −0.04), which is itself informative: a model extrapolating off its training manifold is unstable from seed to seed, while one interpolating is not. The three-seed mean prediction is now what the tables and figures show for those three variants; the headline US numbers moved by at most 0.01.

**What changes in the plan.** Phase 1b is closed except for the latitude-aware inputs, which move to the start of Phase 2. The LORO result sharpens what that experiment can show: if a Coriolis-scaled input does not rescue the within-US transfer, the missing information is not a hemisphere sign but the regional regime itself, and the paper should say so.

### 00:25 David: "can we finish phase 1b? ... let's do 1 and 2 for now."

Two of the three open Phase 1b items launched as one unattended chain (log `SWOT_L3/science/holdout_loro.log`); the third, latitude-aware inputs, moves to the start of Phase 2 where the Coriolis parameter is a planned input anyway.

**Why these two.** Every number in the 2026-11-11 table rests on one training run per training set, so a gap of a point or two between, say, `us_y1` and `all_y1` cannot yet be told from initialisation noise. Seeds 43 and 44 for `us_y1`, `all_y1` and `not_us_y1` give a three-seed mean prediction (the scorer averages every `pred_seed*.npz` in the variant folder, unchanged) and, new, a per-seed single-run R² stored as `per_seed_r2` in `results.json` and printed under each region's table in `results.md`, so the spread is on the record. The leave-one-continent-out failure confounds two things: the model never saw the physical regime, and it never saw the particular radar network. Four new training sets (`us_wo_<rid>_y1`: the five other US regions, held-out region scored) separate them. If uswc is predictable from the other five US networks, the continent failure is about regime (latitude, hemisphere, shelf), not about memorising a network; if it fails inside the US too, the recipe is learning network-specific structure. Only uswc, usegc, prvi and ushi are held out, because gak and glna have no clean test rows to score.

**Changes.** `holdout_eval.py`: `LORO` dict merged into `VARIANTS`; `_variant_preds` returns the per-seed list; `score_all` records `per_seed_r2`; `write_md` prints it; `transfer_grid` gets one column "US w/o own region" (each US row shows its own held-out model) instead of four nearly empty columns. GPU estimate: four LORO retrains at about 30 min, six seed retrains (2 x 30, 2 x 35, 2 x 10 min), about 4.5 h in all, then rescore and figures. Disk: each new variant writes a 5-6 GB `data.npz` into `experiments/holdout/`, which is a junction to Z: since last night, so C: stays at 269 GB free.

## 2026-11-11 — Radar targets for cycles 017-058; the out-of-time evaluation

### 01:40 David: "do those three steps. what do i do after that?"

Three steps from the status report: the 25-hour radar targets (colocation) for cycles 017-058, ERA5 Sep-Oct 2026, and commit/push. The commit went out as 6baba3a. The NCAR mirror still ends at Aug 2026, so ERA5 has nothing to fetch yet; the README line reruns it. The colocation could not use `build_science_colocations.py --phase hfr`: it reads the radar from the S3 aggregates, which end in 2024/2025 anyway, and it skips any target that exists. `build_hfr_multires.py` reads the local hourly archives on Z:, so it got an append mode (cycles already in the pkl are kept; only cycles present in the SWOT pkl and absent from the target are colocated) and six new default-resolution targets for the US regions (uswc/usegc/glna/prvi/ushi 6 km, gak 2 km) that are saved to `hfr_science_<rid>.pkl` as well as the per-resolution name. Three streams by continent from 01:43. Australia finished 02:03 (159 to 530 colocated passes per region over the 42 new cycles; the "skipped" passes have no radar within 36 h); glna's empty cycles line up with the Great Lakes winter shutdowns and the NCEI archive ending Aug 2026. Europe and the US streams were still running at 02:20; usegc, with 60 passes per cycle on per-hour files, is the long tail.

The answer to "what after that": the extended set makes three things possible that the Phase 1 tables could not do, a true out-of-time test (train year one, score 42 later cycles), a spatial generalization test across continents, and the wind-history experiment at nineteen latitudes. David: "ok once that is done, proceed with headline evaluation. come up with some intuitive and clear visualization to portray the results. i agree that spatial (and temporal) generalization is important, and i would like to test that too. so we retrain alphaevolve?"

### 22:10 Results of the out-of-time and out-of-region evaluation

The chain finished at 21:51 (uswc froze in six chunks, the seven retrains took 32, 9, 5, 35, 11, 35 and 35 minutes in the order us, eu, au, all, not-us, not-eu, not-au, which is training time scaling with rows under the recipe's early stopping; scoring with 1000 block-bootstrap draws took four minutes). Three regions have no clean test rows and are skipped: glna (the lake record ends with the winter shutdowns and the NCEI archive), gak (93 rows; the radar record there stops at cycle 036) and eu_gbight (its radar record ends in 2023, so the region is training-only). The test window ends on 2026-08-31 for every region because the ERA5 mirror ends there; the US radar archives end at cycle 055 as well. Two things to keep in mind when reading the numbers: uswc and usegc hold 4.4 M and 4.2 M of the 11 M scored rows, so any pooled statement is a statement about those two; and the calibrated-geostrophy reference is broken for au_cof (R² of -15.65 for u from a year-one calibration on a small, noisy set), so au_cof's percent-RMSE columns mean nothing and its R² is what to read.

R² (Nash-Sutcliffe, mean of u and v) on cycles 017-058, every model trained or fitted on cycles 001-016:

| region | rows | calibrated geostrophy | DUACS calibrated + wind | champion, own continent | champion, all 19 | champion, other continents |
|---|---|---|---|---|---|---|
| uswc | 4.4 M | 0.13 | 0.35 | 0.34 | 0.34 | -0.55 |
| usegc | 4.2 M | 0.38 | 0.46 | 0.49 | 0.50 | 0.15 |
| prvi | 293 k | 0.07 | 0.21 | 0.11 | 0.11 | -0.42 |
| ushi | 96 k | 0.06 | 0.13 | 0.13 | 0.10 | -0.17 |
| eu_biscay | 135 k | 0.07 | 0.19 | 0.21 | 0.23 | 0.14 |
| eu_galicia | 258 k | 0.04 | 0.37 | 0.39 | 0.36 | 0.20 |
| eu_wmed | 507 k | 0.24 | 0.29 | 0.30 | 0.24 | 0.11 |
| eu_tirlig | 120 k | 0.21 | 0.24 | 0.25 | 0.28 | 0.20 |
| eu_norway | 255 k | 0.25 | 0.33 | 0.48 | 0.42 | -0.10 |
| au_rot | 15 k | 0.12 | 0.37 | 0.24 | 0.25 | 0.04 |
| au_turq | 349 k | 0.09 | 0.14 | 0.13 | 0.07 | -0.05 |
| au_cof | 27 k | (broken) | n/a | 0.21 | 0.30 | -0.85 |
| au_newc | 65 k | 0.11 | 0.22 | 0.24 | 0.12 | -0.03 |
| au_sag | 51 k | -0.01 | 0.45 | 0.34 | 0.10 | 0.11 |
| au_nwa | 44 k | 0.11 | 0.31 | 0.09 | 0.03 | -0.18 |
| au_corl | 194 k | 0.04 | 0.11 | 0.05 | 0.03 | -0.17 |

Full tables with intervals and percent-RMSE columns are in `experiments/holdout/results.md`; `results.json` has the per-cycle numbers.

Out of time, the recipe holds. Retrained on year one and scored on the next two years, the own-continent champion beats calibrated geostrophy in 13 of the 16 scored regions, and in the US and Europe it sits level with or above the strongest baseline (DUACS calibrated with a fitted wind response): uswc 0.34 against 0.35, usegc 0.49 against 0.46, eu_galicia 0.39 against 0.37, eu_norway 0.48 against 0.33. The percent-RMSE gains against calibrated geostrophy are the same size as in the Phase 1 tables (uswc -13 percent, usegc -9, eu_galicia -20, eu_norway -17). Drift over the two years is small. The per-cycle median gain of the own-continent champion in the second test year (cycles 034-058) against the first (017-033): uswc -10.7 then -13.8 percent, usegc -9.6 then -9.1, ushi -6.5 then -3.7, eu_galicia -20.9 then -14.5, eu_norway -19.4 then -16.9, eu_wmed -5.3 then -3.1; prvi, eu_tirlig and the Australian shelf regions sit near zero in both years. Where there is any erosion it is a few points, not a collapse, and the calibrated baselines erode by the same amount in the same regions, which says the drift is in the data (radar coverage, SWOT reprocessing state) rather than in the network.

Out of region, in the leave-one-continent-out sense, it fails, and the failure has a shape. Trained on Europe and Australia and applied to uswc, v keeps an R² of 0.25 (about the calibrated-plus-wind baseline) while u collapses to -1.35; trained on the US and applied to Europe, skill stays positive everywhere (0.07 to 0.20) but well below the own-continent 0.21 to 0.48; trained on Australia alone (509 k rows, the smallest set) and applied anywhere else it is negative across the board. One component blows up in magnitude while the other survives, and the damage grows with the distance between the target regime and the training mix. The reading: the model takes 72 raw columns in metres per second and no coordinates, so it cannot know where it is, but the feature distributions tell it anyway, and the same surface slope maps to a different velocity at a different latitude (the 1/f factor applies to the wind-driven response too), while the European and Australian shelves are far more tidal than the US west coast. A network tuned to the US range of those relationships extrapolates on Norwegian or Australian inputs and extrapolation in a nonlinear model gives magnitude blow-ups. It is a coverage problem, not a capacity one: the all-19 model recovers the own-continent skill in the US and Europe at no cost (uswc 0.34, usegc 0.50, eu_galicia 0.36, eu_norway 0.42), so one network can hold these regimes at once as long as it has seen them. In Australia the all-19 model loses to the own-continent one in four of seven regions (au_newc 0.12 against 0.24, au_sag 0.10 against 0.34), which is what one expects when seven small regions are 8 percent of the training rows.

The honest claim is therefore narrower than "a recipe found on one coastline transfers": the recipe transfers in time, with two more years and little drift, and it transfers in space only when the training set covers the target's regime. Australia is the weak spot either way: the regions are small and DUACS plus wind beats the champion in au_rot, au_nwa and au_sag, which also means those regions are the right place to test whether a latitude-aware input fixes the transfer.

Next, in order of cost: a leave-one-region-out within a continent (train on five US regions, score the sixth) to separate "never saw this regime" from "never saw this continent"; then latitude-aware or latitude-invariant inputs (add sin(latitude), or scale velocities and winds by f, or standardise each region's columns by its own year-one statistics) and a rerun of the not-US variant to see whether the u blow-up goes away; seeds 43 and 44 for us_y1, all_y1 and not_us_y1 so the headline numbers carry a seed spread. The second search (fitness on 017-030, holdout 031-058) is not justified by the drift curve: the recipe did not lose skill with time, so a time-aware search has nothing obvious to fix.

Figure fixes on the way to this entry: per-cycle dates had been written with seconds read as days (year 4.7 million; the scorer now converts through `datetime64[s]`); a list of `np.datetime64` scalars plots as raw integers, so the time axis takes a datetime64 array and python dates; the cmr10 font has no superscript two, no underscore and no "<", so those go through mathtext; the own-continent outlines in the grid had compared a continent name to a region list and never drew; map numbers collided in the European and Australian clusters and were dropped (the grid carries them); lines in the drift figure no longer bridge radar gaps.

### 23:40 David: "is there a way we can denoise this RMSE change plot? is that an honest move?" then "yes do 2 plus 3"

The per-cycle drift figure is noisy because each 21-day cycle holds a few dozen passes (a handful in prvi, au_rot, au_sag), so the ratio of two RMSEs computed on small samples jumps around; that is sampling noise and summarising it is honest as long as the summary is a statistic one can defend and the raw points stay visible. Added `skill_through_time_quarterly`: the scorer now keeps per-pass sums of squared error per model (`per_pass` in `results.json`), which is all that pooled RMSE over any window and a bootstrap over passes need. Each panel shows the percent RMSE change against calibrated geostrophy pooled by calendar quarter with a 95 percent interval from 500 resamples of passes, the single-cycle values as faint dots, and for the own-continent champion, the all-regions champion and DUACS plus wind a linear trend fitted to the per-cycle values weighted by rows, with the slope in percent per year and its bootstrap interval printed in the panel. Reading: uswc and usegc have slopes of about -1 to +1 percent per year with intervals straddling zero, so no erosion over two years; prvi shows a small positive slope for all three (+1.6 to +5.3), the baselines included, which says the reference moved rather than the model; eu_galicia's +7 to +9 per year comes from a record that thins out in 2026 (two passes per quarter at the end) and the interval reaches zero; the Australian panels with wide intervals (au_rot, au_sag, au_newc) have too few passes per quarter for the slope to mean anything, and the figure says so by its error bars. The per-cycle figure stays as the raw view.

### 23:20 David: "create a bibliography of data sources in bibtex format, find a good place to save all of it"

`docs/data-sources.bib`: one entry per dataset or service the pipeline reads (SWOT L3 LR SSH Expert v3.0 with its CNES DOI and the Fu et al. 2024 GRL paper; the AVISO orbit shapefiles; the NCEI HF radar RTV archive, which has no collection DOI, the radial-velocity accession's DOI is noted so nobody cites it for the totals; the European HFR Node with the JERICO-NEXT QC report it points to; IMOS ACORN with the IMOS citation form; ERA5 as the CDS dataset DOI plus Hersbach et al. 2020 plus the NCAR AWS mirror actually read; the DUACS reprocessed and near-real-time products; GOES ABI L2 SST). Access dates are the mirror-build days from the replication guide. Two DOIs copied from provider catalogues are marked "verify" (NCAR ds633.0, the DUACS NRT product). `docs/replication.md` section 2 now points to the file. Lives in `docs/` because the data story is part of the replication guide; `papers/` holds the literature PDFs, which have no bib yet.

### 22:40 David: "we are running low on hard drive space once again. think about what we can move over to Z"

C: had 21 GB free of 952. The audit: `experiments/<rid>/flattened/` held 190 GB of flatten caches (usegc 109 GB, uswc 51 GB, including today's six chunk pickles per region and the July and October caches), `experiments/holdout/` held 55 GB of which 37 GB were seven identical copies of the 16 M-row test matrix (one `data.npz` per retrain variant), `cache/` holds 50 GB of interp_sources pickles, and the two biggest radar pkls existed twice (`hfr_science_<rid>.pkl` and its `_6km` twin, 8.1 GB and 3.4 GB). Done tonight: every `flattened/` folder and `experiments/holdout/` moved to the same relative path under Z: with a junction left behind (robocopy /MOVE, log `Z:\...\SWOT-data-analysis\logs\move_to_z_20261111.log`), the twin pkls replaced by NTFS hard links (11.5 GB back at once), and a 2.2 GB temp file from the crashed freeze deleted. The move finished at 23:35 (usegc 38 min, uswc 16 min, holdout 21 min); C: went from 21 GB free to 268 GB free. `cache/` and the model weights under `experiments/<rid>/{ann,rf}/` are the next candidates. The rule now lives in `CLAUDE.md` ("DISK SPACE"): before any job that writes more than about 1 GB, decide whether the output belongs on Z:, and keep the junction mechanism so scripts never change.

### 18:34 David: "yes do steps 1-5. keep everything recorded and logged."

usegc froze in six chunks by 16:32: 8.3 M rows over cycles 001-055 (the NCEI archive ends before cycle 056), 6.0 M of them in the test cycles; the process peaked near 69 GB of commit even chunked, so Windows grew the page file to 25 GB on its own. With 18 of 19 regions frozen and David's go, one chained run started at 18:34: freeze uswc (chunked), the seven year-one retrains in order us_y1, eu_y1, au_y1, all_y1, not_us_y1, not_eu_y1, not_au_y1 (seed 42, GPU, one at a time), then scoring with 1000 block-bootstrap draws and the three figures. Log `SWOT_L3/science/holdout_run.log`. Expected finish around 02:00 on Nov 12. Results entry follows when it lands.

### 15:50 usegc does not fit; chunked flatten

All nineteen colocations finished at 15:27 (usegc last: 3,150 colocated passes over cycles 017-058, 8.5 GB pkl; the per-cycle checkpoint save made the last cycles take 13 minutes each). The usegc freeze then died in flatten with `_ArrayMemoryError: Unable to allocate 2.58 MiB` while nothing else heavy was running: the SWOT and radar pkls plus 60 passes times 58 cycles of 72-column frames exceed the machine's 72 GB commit limit by themselves. The preset pkls load whole whatever the cycle range, so the relief is in the freeze: shared steps run once from their caches, then `step_flatten` runs on ten cycles at a time (`--chunk 10`; each chunk gets its own flatten cache name through `cycles_start`/`cycles_end`) and the rows are extracted and the chunk freed before the next. Output identical to the one-shot path; only peak memory changes.

### 11:35 Resume after the memory kills; one NCEI file quirk

Overnight the three jobs died from the Windows commit limit (59 GB held by browsers and chat apps, not by us). David said "continue" at 10:29 with 51 GB of commit free, so the US colocation resumed from its per-cycle checkpoint (ushi at cycle 040) and the four remaining European freezes ran beside it; all six European regions were frozen by 10:55 (eu_wmed is the biggest at about one million rows). ushi and prvi colocated and froze by 11:27. The uswc colocation then crashed at cycle 026 with `ValueError: coordinate 'time_run' not present in all datasets.`: some NCEI hourly files carry a `time_run` scalar coordinate (the model run time of the near-real-time product) and others do not, and `xr.concat` refuses to join them. `slab_mean` in `build_hfr_multires.py` now drops every non-index coordinate (`reset_coords(drop=True)`) before the concat; only u, v on (time, lat, lon) are needed for the 25-hour mean. Resumed from the cycle-025 checkpoint.

### 02:00 Retrain the champion, do not rerun the search

The question came up three times in different forms and the answer is worth recording. The champion `gen031_00` is a training recipe; every use retrains it from scratch (about 27 min on the GPU). The search ranked recipes by training them on cycles 001-012 and scoring on 013-016, so it already used year one the only way a search can: with a scoring set carved out of it. Rerunning the search "on 1-16" would be the same procedure with different dice (wall-clock budgets and the LLM mutator make it nondeterministic) and would produce a second champion of unknown relation to the published one. The evaluation instead refits the chosen recipe on all sixteen cycles of year one, the standard refit-after-selection step, and scores it once on 017-058, which no decision in the chain ever saw. The only search worth running later uses the new data for selection: train 001-016, fitness 017-030, holdout 031-058; that costs GPU days and belongs after the drift curve is seen.

Design of `scripts/holdout_eval.py` (new): freeze every region's rows for all cycles through the pipeline's own shared steps and flatten (cache names carry `_1_58`, so nothing stale is reused; the glna freeze reproduces the old evolve test rows exactly, 109 rows for cycles 13-16); seven retrains of the recipe on year one (US, Europe, Australia, all, and leave-one-continent-out), training rows in chronological order so the recipe's internal validation slice is the latest 10 percent; baselines fitted on the same year-one rows per region; scoring as in `baselines_eval.py` (clean pairing, SWOT-valid rows, block bootstrap over cycle and pass, percent RMSE change against calibrated geostrophy), plus per-cycle skill for the drift figure. One pipeline trap found on the way: `interp_to_swot` takes the nearest ERA5 hour with no tolerance, so passes after the end of the ERA5 record (2026-08-31) would silently get the last hour's wind; those rows are flagged and excluded. DUACS daily maps for Jul 2024 to Oct 2026 were pulled for all nineteen regions from Copernicus (free; the tail comes from the near-real-time product, recorded per day in the file). Figures planned: skill through time per region with the training year shaded, the 19-by-10 transfer grid, and a world map of out-of-continent R².

## 2026-11-09 — ERA5 winds for the 13 new regions; last two movies and the SWOT-era cuts

### 22:52 David: "yes do those three tasks. go / and do ERA5 winds"

The three render steps that the memory kill of Nov 8 had interrupted were restarted as one sequential chain (usegc re-render from 2024-07-01, then the North America continent from the same date, then `--assemble-only --since 2023-04-01` for every region and continent). Logs `render_S6.log`, `render_S7.log`, `render_since.log` and `chain3.log` on Z: under `archive/hfr_anim/`.

ERA5: `build_era5_winds.py --mission science` for the six European and seven Australian regions, from the public NCAR mirror `s3://nsf-ncar-era5` (AWS Open Data, anonymous, egress paid by the bucket owner, so $0; the estimate given before launch was ~36 GB streamed). Window Jul 2023 to Jul 2024, the same window the US pkls have, i.e. science cycles 001-016.

### 23:15 ERA5 was 20 minutes into the first month with nothing written

The script opened each monthly file lazily through `s3fs` and subset it once per region. The NCAR files are 1.5 GB per variable per month, gzip-compressed HDF5 with chunks of 27 h x 139 lat x 277 lon, so every region pulled its own chunk columns for the whole month and the thirteen regions (clustered in two places on the globe, so largely the same chunks) re-fetched the same bytes with no cache between them. At the ~40 MB/s the connection gives, the first month alone was heading past 30 minutes. Fix: one `fs.cat()` GET per file into memory (`io.BytesIO`, 1.5 GB), then `xr.open_dataset(engine="h5netcdf")` on the buffer and all thirteen subsets locally. Months then took ~2 minutes each; the 13 months finished at 23:46 and the thirteen pkls were written (8 to 60 MB each, 9528 hours, 10x10 to 24x24 cells). The process exited 139 (segfault in the h5netcdf/s3fs teardown after its own "done." line); the pkls check out: hourly axis without gaps, longitude monotone after the prime-meridian stitch for the western Med (-1.0 to 4.0), winds within +-23 m/s, no NaNs.

Still open: the same winds for cycles 017-058 (Aug 2024 to Oct 2026, 27 more months for all 19 regions, ~80 GB more streamed, $0) so the new cycles can be used as test rows with the full feature set. Not launched; needs David's word.

### 03:15 (2026-11-10) The campaign is complete

Chain finished at 03:11: usegc movie (5423 frames, 60 MB, 23:45), North America (6127 frames, 55 MB, 02:12; 858 frames each redone from 2024-07-01 so the SWOT panel runs to Nov 2026), then the 22 SWOT-era cuts (1315 frames each, 2023-04-01 to 2026-11-05, 13 to 71 MB; 03:11). Every HFR+SWOT product now exists: 19 region movies, 3 continent movies, 22 `_since2023` cuts, 3.1 GB on Z: behind the `docs/figures/hfr_animations` junction. The continent re-render ran at 7 to 9 s per frame instead of 4.6 because the ERA5 stream shared the CPU; sequential jobs would have been faster in total.

What the campaign leaves in place: the full radar records for 19 regions (US 2006/2012 to Aug 2026 from NCEI, Europe 2016/2019 to Oct 2026, Australia 2009 to Nov 2026) as hourly archives on Z:, every SWOT science cycle 001-058 regridded into each region's pkl, ERA5 10 m winds Jul 2023 to Aug 2026 for every region, and the movies above. Not done: the colocation `hfr` phase (25-hour radar targets) for cycles 017-058, so the new cycles are not yet training or test rows; Sep-Oct 2026 ERA5 (mirror lag); COSYNA radar after Jun 2023 (the node hangs on those months).

### 23:55 David: "yes get all that ERA5 too"

The NCAR mirror ends at Aug 2026 (listing checked), so the extension is Aug 2024 to Aug 2026, 25 months, 19 regions; Sep and Oct 2026 are not on the mirror yet and the same command picks them up when they appear. `build_era5_winds.py` got a `--months START END` override and an append mode: when a region's pkl exists only the requested months absent from its time axis are fetched, cut, concatenated onto the existing record and saved, so the US pkls were extended in place rather than rebuilt. Launched 23:58 next to the North America render (log `SWOT_L3/science/era5_ext.log`).

### 00:57 (2026-11-10) MemoryError on the usegc concat; per-month disk cache

uswc saved (Jul 2023 to Aug 2026, 27,792 hours, 791 MB), then the usegc concat failed with `Unable to allocate 1.15 GiB` while 21 GB of RAM was still free: the process held every region's 25 months in memory on top of the two 1.5 GB file buffers, the North America render held ~25 GB, and the Windows commit limit (RAM plus page file) was what ran out. Everything streamed for the other 18 regions was lost. Fix: every (region, month) subset is pickled to `experiments/<rid>/era5/_months/` the moment it is cut, the final concat reads one region at a time and deletes its month files after the save, and a month whose regions are all cut already is not downloaded again, so a crash costs one month and the run resumes. Rerun 01:00 to 02:10 (log `era5_ext2.log`; ~2 min per month alone, ~5 min while the render was on the CPU). All 19 science pkls now span 2023-07-01T00 to 2026-08-31T23 hourly without gaps or NaNs (checked), 16 MB (glna) to 2.46 GB (usegc). Both ERA5 runs exited 139 after their own "done." line, a segfault in the h5netcdf/s3fs teardown, harmless.

## 2026-11-07 — Full radar records, every SWOT cycle, HFR+SWOT animations per region and continent

### 01:35 What David asked and the plan

David's answer to "how much more can we fetch?" was: do (1) refetch the US gaps and (2) extend Europe and Australia, fetch all SWOT cycles, save to Z:, and make animations of HFR and SWOT across all times, per region and per continent. Everything here is free HTTP or the AVISO SFTP account; nothing touches S3.

Server inventory before starting: the European node answered again (it had timed out the evening before) and its aggregated datasets run from 2019; IMOS ACORN holds Rottnest 2010-2026, Turquoise Coast and South Australia 2009-2026, Coffs 2012-2026, Newcastle 2017-2026, Coral Coast 2019-2026, North West Shelf 2021-2026 (Capricorn Bunker and Bonney Coast end in 2017 and have no region config, so they were left out). AVISO's `l3_lr_ssh/v3_0/Expert` has two folders: `reproc` with science cycles 001-032 (plus the calval 474-578) and `forward` with 032-058, cycle 058 ending 2026-10-24; granules are only 11 MB, so the 90 s per granule seen in the first campaign is SFTP latency, not volume. HFRNet's THREDDS server resolved in DNS but refused every connection, so the US part waits.

### 01:40 Radar fetches launched

Three background jobs write to the Z: archive: European node networks in two streams (EUSKOOS, Galicia, Ibiza, DeltaEbro, ICATMAR in one; TirLig, ARPAS, Vestlandet in the other, followed by COSYNA restricted to 2019-01 .. 2023-06 because German Bight requests from Oct 2023 are what hung the node last time), each 2019-01 .. 2026-10, and the seven active IMOS sites 2009-01-01 .. 2026-11-06. Rates at launch: Biscay months in 15 s, TirLig months (30 MB) in 45 s, IMOS about 12 files per second, so Europe takes a few hours and Australia about a day. `hfr_convert_networks.py` got `--start/--end` so the monthly archives can be extended over the new months afterwards (it skips months already converted).

### 01:50 SWOT cycles 017-058 for all 19 regions

`build_science_colocations.py` now lists both the reproc and forward folders per cycle (reproc wins when a pass is in both), takes `--cycles 017-058`, and a `--manifest-tag` so parallel workers keep separate manifests and temp granules; the first launch without the per-worker temp file had four processes overwriting one `_tmp_granule.nc`, caught within minutes and relaunched. The pkl phase changed from "exists, skip" to append mode: an existing pkl is loaded and only cycles with local segments that it lacks are regridded and added, which is what lets the US regions (whose cycle 001-016 segments were deleted after the S3 upload) extend without re-downloading. Four workers over cycles 017-027, 028-038, 039-048, 049-058; 184 unique passes per cycle across the 19 regions; measured 65 s per granule per worker, so the 7,700 granules take about a day and a half.

### 02:05 HFR gap fetcher for HFRNet (ready, server down)

`fetch_hfrnet.py` compares the local archive's hours per calendar month with the server's time axis and plans (a) months with no local hours at all and (b) everything after the last local hour; partly present months are left alone so no hour is duplicated when the hourly readers concatenate. Output is one zlib month file in the archive's own naming (`<rid>_6km_Resolution_Hourly_RTV_best_YYYYMM01_YYYYMMDD.nc4`, u/v/hdop/number_of_sites), fetched a day per DAP request. From the frame census the empty months after each network matured are: uswc 2015-04, 2015-06, 2016-06, 2017-06; usegc 2025-01 .. 2025-06; prvi 2023-10 (its 2010-2013 and 2017-18 holes look like real outages and are also planned, cheap to try); ushi 2013-01, 2022-03..05, 2022-08..09, 2023-10; glna 2023-10, 2024-09..10, 2025-02; gak none. Plus the tails: every US archive ends between Apr and Jul 2025 and the server runs to the present. A background probe retries the server every ten minutes and the fetch starts when it answers.

### 02:10 HFR and SWOT side by side

`animate_hfr_swot.py`: left panel the daily-mean radar current, right panel the SWOT L3 geostrophic current (`ugos_filtered`, `vgos_filtered` from the regridded pkls, science and calval) on the swaths that crossed the box that day, same cmocean speed scale 0-1 m/s and the same arrow scale so the eye compares them directly. One frame per calendar day over the union of the two records, blank panels where a source is missing; when no pass fell on a day the most recent one up to ten days old is drawn and the title gives its age, because a region sees SWOT only every few days and a mostly-empty panel reads as broken. Radar is reduced once to a daily-mean cache per region and year (NetCDF on Z:, about 1 MB per year for the small regions) that is rebuilt only when the archive gets newer files, and a rebuild deletes that year's frames for the region and its continent so they re-render; new SWOT cycles are handled by `--redo-from`. Continents (North America, Europe, Australia) reuse the same figure with every region's grid drawn on one map, panels stacked when the box is wide. A Rottnest test frame on 2023-08-03 showed the two ascending/descending passes next to the radar field, about 1 s per frame. Four render jobs are running (uswc with the Australian regions; usegc with the European ones; prvi, glna; ushi, gak), continents after all caches exist. The region frames made yesterday under `archive/hfr_anim_frames/` are untouched; this is a second product.

### 02:10 (2026-11-08) C: at 98 %: movies and SWOT pkls moved to Z:

David saw C: down to 19.5 GB. The repo's `experiments/` holds 88 GB (pkls, colocations, targets) and `cache/` 14 GB, both from before; the day added 2 GB of movies, and the pending pkl append would add about 25 GB (usegc's 3.3 GB pkl grows with 42 more cycles). Moved `docs/figures/hfr_animations/` and the 19 `experiments/<region>/swot_cycles/` folders (8.5 GB) to the Z: mirror of the repo and left directory junctions in their place, so every path in the scripts and the app still resolves; deleted the 3.4 GB of Nov 5-6 HFR-only frames under `archive/hfr_anim_frames/` on C: (David: do not save frames; the daily MP4s exist, the coverage table in docs/model-data.md stays as computed). C: free went from 19.5 to 33 GB.

### 01:30 (2026-11-08) SWOT-era cuts

David watched a movie and saw no SWOT: at 24 fps the prvi film is 3 min 55 s for 2010-2026 and SWOT exists only from 2023, so the right panel is empty for the first 3 min 10 s and the comparison year is a 15 s stretch near the end (and blank again after Jul 2024 until the new cycles are appended). `animate_hfr_swot.py --since 2023-04-01` now also writes `<name>_hfr_swot_since2023.mp4` from the frames on or after that day; with `--assemble-only` it writes only the cuts. Queued to run after the Jul-2024-onward re-render.

Overnight (Nov 7-8): all radar fetches finished (Europe 10 networks, IMOS 7 sites at ~500k hourly files, NCEI 47k files for the US gaps and the Apr-2025 to Aug-2026 tails); all converted; the six US and seven Australian region movies re-rendered on the full records; North America and Europe continents done; Australia continent rendering. SWOT: 26 of 42 cycles complete by 01:00 with twelve workers (four more added at 23:40 on the tail of each range); the download end triggers the pkl append (three parallel jobs) and five re-render jobs for every frame from Jul 2024 on. Three processes (the North America render and the Coffs and South Australia converters) died silently at 18:33 on Nov 7 with nothing in their logs or the system log; all were resumed from where they stopped, and the monitor line now counts live python processes.

### 12:20 HFRNet is gone; the US record lives at NCEI

The Scripps HFRNet THREDDS server never answered (its DNS resolves, connections time out) because CORDC retired it: IOOS now points real-time totals at NDBC's THREDDS (`dods.ndbc.noaa.gov/thredds/catalog/hfradar.html`) and the CoastWatch and PacIOOS ERDDAPs mirror that, but all of them hold only the last three months (Aug to Nov 2026). The full record is NOAA NCEI's HF radar RTV archive, plain HTTPS directory listings by year, month and region, one NetCDF per hour from June 2006 to Aug 2026 (the archive lags about three months). Most hours up to mid-2025 exist twice, an SIO file (the HFRNet product, ~280 KB for the West Coast) and an NDBC one (~120 KB); from Aug 2025 there is one producer under a new `rtv-<region>-6km-uwls_v1r0_...` naming. `fetch_hfrnet.py` was rewritten for it: same plan logic (empty local months plus everything after the last local hour), files saved in the archive's own one-per-hour naming so every reader picks them up, SIO preferred, 25-hour averages skipped, eight download threads. The plan: uswc 24 months, usegc 19, gak 13, glna 20, prvi 33, ushi 21, about 95,000 files and 25 GB; two streams started at 12:22. April 2015, the first uswc hole, has its full 1,440 files on NCEI, so those holes were fetch gaps in our archive, not outages.

### 06:40 Continent maps: arrows by pixels, not by grid

The first North America continent frames were black blobs: arrow spacing was "one in every grid/30 cells" and arrow length "1 m/s = a sixth of the panel height", both fine for one region filling a panel and wrong when six regions are specks on a hemisphere. Both are now in pixels: arrows every ~30 px from the panel's degrees-per-pixel and the grid spacing, length 170 px per m/s on region panels (what they had) and 60 px per m/s with a thinner shaft on continent panels. Region movies already rendered are unchanged by construction. Job A (uswc plus the seven Australian regions on their current archives), C (prvi, glna) and D's ushi are done; usegc, gak and the six European regions are rendering; North America restarted with the fix.

Fetch notes from the morning: EUSKOOS Aug 2024 and Galicia Dec 2025 came back as whole-record files (250 MB and 1 GB, the node glitch already handled by the converter's time slice); COSYNA publishes 20-minute maps, so its "hourly" month files hold ~2,170 steps; the node's records per network are EUSKOOS and Galicia from 2019, Ibiza and DeltaEbro from 2019, ICATMAR from Apr 2023, TirLig from Aug 2016 (fetched too), ARPAS Nov 2022 to Jul 2025, Vestlandet from Oct 2023, COSYNA from 2019. The IMOS fetcher died once on a connection reset inside a file body (its retry only wrapped the request), fixed and resumed from Turquoise Coast; Rottnest alone is 82,586 hour files with 2,568 missing days over 2010-2026.

Disk note: C: is at 98 % (23 GB free), so the cache and frames went to Z: via `HFR_ANIM_ROOT`; the 1.6 GB of 800 px frames parked yesterday under `archive/hfr_anim_frames_100dpi/` are still waiting for David to delete.

## 2026-11-06 — Radar animations re-rendered: sharper, Computer Modern

### 11:20 200 dpi frames, project font, higher-rate encoder

David: "the image quality is quite blurry, can we sharpen it? ... can we
use LaTeX font in the animations". Two causes of the blur: frames were 800
pixels wide (100 dpi) and the shared `assemble_animation` helper encodes at
imageio quality 8. The script now renders at 200 dpi (1600 pixels wide, the
height rounded to a multiple of 16 pixels so ffmpeg does not resize) and
writes the MP4 itself with libx264 at quality 10, yuv420p, which players
and browsers decode without a plugin. Font: the same Computer Modern
settings the paper figures use (cmr10 serif, cm mathtext, no unicode
minus). cmr10 has no degree glyph, so the gridline labels use cartopy's
longitude and latitude formatters with a mathtext degree sign; cmr10 also
has no vertical bar, which rendered as a dash in the title, so the title
separators are commas. The first set of frames was moved aside to
`archive/hfr_anim_frames_100dpi/` (1.6 GB, can be deleted) and all nineteen
regions relaunched as four jobs at 11:20; the German Bight test took 0.9 s
per frame at the new size, so about three hours for the US records.

14:10 Re-render finished, 2 h 50 min for all nineteen regions, same frame
counts as the first pass, no errors. MP4s total 935 MB (US West Coast alone
450 MB at 1600 px and quality 10, against 112 MB before); frames 3.4 GB in
`archive/hfr_anim_frames/`.

19:15 David: "the videos wont play...". ffprobe showed the cause: imageio
quality 10 means lossless x264, and lossless x264 switches to the High
4:4:4 Predictive profile, which Windows Media Player, browsers and
hardware decoders refuse. The writer now passes CRF 18 in the standard
High profile (visually lossless, plays everywhere, about half the size).
All nineteen MP4s re-encoded from the cached frames with
`--assemble-only`, no re-rendering.

19:40 David asked how many dates are missing elsewhere and whether we keep
a record of the spans. We did not; only the archive filenames carried year
ranges. Added `--coverage` to the animation script (days with data per
region from the frame cache, gaps of a week or more listed) and wrote the
table into `docs/model-data.md` under a new section. Findings: the three
western Med, Ligurian and Galicia archives are complete for Jul 2023 to Jul
2024; Norway misses one day; the long US records miss 60 (gak) to 1,211
(prvi, mostly 2010 to 2013 and Sep 2017 to Feb 2018) days. Two gap patterns
point at the archive rather than the radars: October 2023 is absent in
prvi, ushi and glna together, and uswc misses whole calendar months (Apr
and Jun 2015, Jun 2016, Jun 2017) while usegc misses Jan to Jun 2025. These
are probably HFRNet files never fetched; October 2023 sits inside the
science training window, so refetching it is worth doing before the next
scoring pass.

## 2026-11-05 — Daily animations of every radar region

### 16:10 One script, nineteen regions, daily-mean frames

David: "can you make animations of every single hfr region we have in the
world for whatever times we have?" Built `scripts/animate_hfr_regions.py`.
It walks each region's hourly archive on Z: (the six US HFRNet 6 km
archives over their full records, 2010 or 2012 to 2024/2025, and the
thirteen converted European and IMOS archives over July 2023 to July 2024),
averages each calendar day, and draws one frame per day that has any finite
cell: speed as a sequential colour field fixed at 0 to 1 m/s for every
region so the continents compare, arrows thinned to about thirty across the
grid, Natural Earth coastline for orientation, cell count in the title.
Days with no data are skipped rather than drawn empty, so the date in the
title jumps over radar outages. Frames are cached as PNGs under the
gitignored `archive/hfr_anim_frames/<region>/` and skipped when present, so
a hibernation or crash costs nothing; the MP4 is assembled with the
project's existing `assemble_animation` helper at 24 frames per second.

Why daily means rather than hourly: the US archives are up to fifteen years
of hourly fields (Hawaii alone is 117,709 one-hour files), which would be
130,000 frames per region; a day per frame gives three to four minutes of
video for the long US records and about fifteen seconds for the new regions,
and the daily mean removes the tidal and inertial signal the 25-h target
also removes, so the animation shows the same low-frequency circulation the
models are scored on. Rendering reuses one figure per region and only
updates the colour array, arrow vectors and title, which keeps a frame
under a second. Two layout bugs found on the German Bight test: the figure
height used a cosine-corrected aspect, but PlateCarree draws degrees square,
so the map sat in a tall box of white space; and the pixel height must be a
multiple of 16 or ffmpeg resizes the video. Both fixed before the full run.

Launched as four parallel processes (Hawaii; Puerto Rico; US West Coast,
Gulf of Alaska and Europe; US East-Gulf, Great Lakes and Australia) at
16:06, about 29,000 frames in all, estimated two hours. Outputs
`docs/figures/hfr_animations/<region>_daily.mp4`, listed in the figure
library; MP4s are gitignored by the repo-wide rule.

16:20 First five minutes showed Hawaii and Puerto Rico at about 15 s per
day: the one-file-per-hour archives were being opened with
`open_mfdataset`, which on the USB drive with four processes contending
took 4.8 s per day against 1.0 s for a plain per-file open and `concat`.
Switched to the latter and relaunched those two jobs (cached frames kept).
Measured per-frame costs with the contention: a 14-day US West Coast file
loads in 17 s, the daily resample in 1 s, one frame renders in 0.85 s.
Revised estimate: about three hours for the long US records.

18:50 All nineteen animations finished, 25,870 frames, 248 MB of MP4, 1.6 GB
of cached PNG frames. Two more problems on the way: Hawaii skipped three
days in September 2016 whose hourly files are not readable NetCDF (three of
5,001 days, left out); and the Gulf of Alaska job died with a MemoryError
because its archive is in year-long files (9,116 hours by 223 by 413 cells,
3 GB per variable) and the reader loaded and then sorted a whole file with
four jobs sharing memory. The reader now opens multi-day files lazily and
reads one day at a time, which also dropped the per-frame cost to 0.6 s.
Frame counts and spans: uswc 4,521 (2012-01 to 2024-12), usegc 4,729 (2012-01
to 2025-07), prvi 4,423 (2010-01 to 2025-06), ushi 5,001 (2010-07 to
2025-06), gak 2,754 (2017-11 to 2025-07), glna 677 (2022-08 to 2025-04);
Europe 296 to 397 days each except German Bight 56; Australia 287 to 386
except Coffs 47 and South Australia 194 (radar record Aug 2023 to Feb 2024
only, a gap the colocation count of 96 of 161 passes had already hinted at).

## 2026-11-04 — New radar networks: Europe and Australia

### 11:40 Downloads started for fourteen new regions

David, after the data-coverage discussion: "i would like to add more data from
other radar networks. i think adding point measurements from drifters adds
some complication", then "get whatever SWOT we need" and a time estimate
before disconnecting the machine. Agreed on both points: drifters would need
a second colocation framework (moving points, 15-m drogue, wind slip, no cell
for the per-cell rungs), while other radar networks reuse the whole pipeline
and turn the regional table into a cross-continent test of the same method.

Catalogue check (free, no login): the European HFR node serves hourly
gridded totals as one aggregated THREDDS dataset per network with a NetCDF
subset service, 2019 to present; IMOS ACORN serves per-hour files of about
125 KB per site, 2023 and 2024 present. Chosen regions (bounding boxes from
the catalogue grid bounds plus 0.5 degree): eu_biscay (EUSKOOS), eu_galicia,
eu_wmed (Ibiza, Ebro, Catalan), eu_tirlig (Ligurian/Tyrrhenian, Tuscany,
Sardinia), eu_gbight (COSYNA), eu_norway (Vestlandet); au_rot, au_turq,
au_cof, au_newc, au_cbg, au_sag, au_bonc, au_nwa. `find_swaths` gives 82
unique science passes over the fourteen boxes, so about 1,300 granules over
cycles 1 to 16. Added the boxes to `REGIONS` in
`build_science_colocations.py` (additive; no `HFR_SOURCES` entry, the HFR
for these comes from local files). Wrote `scripts/fetch_hfr_eu.py` (NCSS,
one file per network per month, Jul 2023 to Jul 2024, v3 with v2.2 fallback)
and `scripts/fetch_hfr_imos.py` (day-catalogue listing then fileServer, eight
threads, missing days recorded). Both skip existing files, so a dropped
connection costs nothing.

Launched three background jobs at 11:41: the SWOT download phase for the six
European regions followed by the eight Australian ones (AVISO SFTP, logs in
`SWOT_L3/science/download_eu.log` and `download_au.log`), the European HFR
fetch, and the IMOS fetch (logs beside them). Estimate given: 200 to 330 GB
of SWOT granules over SFTP, 8 to 14 hours, kept on disk only as about 6 GB
of region segments; about 3 GB and 30 minutes for Europe; about 9 GB and one
to two hours for IMOS. Cost zero, nothing touches S3. Disk on C: is 35 GB
free, enough for the segments and the radar files. AVISO credentials are now
read from the gitignored `.env` at the repo root at launch time only.

First European subset request failed with HTTP 400: the node's NCSS grid
service rejects `accept=netcdf4` ("Format netcdf4 is not supported for
Grid data request"); switched to `accept=netcdf` (NetCDF-3), a month of
EUSKOOS is 3.4 MB, relaunched 11:45. IMOS runs at about nine files per
second, so one site-year takes about a quarter of an hour.

12:05 David: "actually i will work from home today. can you save this data to
the Z drive, which is now connected?" Disk was again flagged Offline;
brought online with elevated `Set-Disk`. Stopped the three jobs, deleted
`.part` files, checked every saved SWOT segment opens (all six did), moved
`SWOT_L3/` to `Z:\...\SWOT-data-analysis\SWOT_L3` and left a directory
junction at the repo path so the colocation script needs no change; moved the
radar folders to the Z: archive beside the HFRNet ones and relaunched the two
fetch scripts with `HFR_DATA_ROOT` pointing there. Everything resumed from
where it stopped.

12:45 IMOS finished in about 50 minutes, all on Z:. Files per site over
July 2023 to July 2024 (8,760 possible): Rottnest 5,294, Turquoise Coast
8,291, Newcastle 7,029, North West Shelf 6,232, South Australia Gulfs
2,991, Coffs Harbour 418 (dark most of the year). Capricorn Bunker and
Bonney Coast returned nothing: their THREDDS folders end in 2017, so both
regions are dropped. The catalogue also has CORL, "Coral Coast" (WA,
28 to 32 S, 112 to 116 E, 2019 to present), a wider grid spanning the
Rottnest and Turquoise Coast footprints; added as `au_corl` and queued.

14:55 European node: eight networks complete (Basque, Galicia, Ibiza, Ebro,
Catalan, Ligurian/Tyrrhenian, Sardinia; Tuscany is access-restricted, HTTP
401, dropped). German Bight requests then began timing out at ten minutes,
first October then November 2023, and by 14:50 the server answered nothing at
all, not even its catalogue page, so the fetch was stopped and a watcher left
running that probes the catalogue every five minutes and resumes German
Bight and Norway when it returns. Whether our long monthly requests wedged
the server or it fell over on its own is unknown; a one-day request also
hung, so it was not the window size.

20:30 Server back at 20:05 after five hours. The German Bight v3 dataset
hangs on any request from October 2023 onward, even a six-hour window,
while its catalogue entry claims coverage to the present; the v2.2 dataset
ends in October 2022. So German Bight keeps July to September 2023 only
(SWOT cycles 1 to 4) and is parked; retry another day, possibly a broken
aggregation on their side. Norway's record starts 2023-10-09 (the earlier
months returned HTTP 400 "does not intersect actual time range"), so its
fetch runs from October 2023; 400 is now a permanent skip like 401/404.

2026-11-05 09:00 Europe finished at 03:46 (620 granules, about 1,030
segments, 16 hours). Australia started 03:47, 33 unique passes over seven
regions, about 45 minutes per cycle. At 08:38 the SFTP transfer silently
stalled mid-granule (temp file frozen at 4.5 MB for twenty minutes, process
alive, no exception, so the reconnect logic never fired); killed and
relaunched at 09:00, resumed from cycle 7 with nothing lost. If this recurs
the fix is a socket timeout or keepalive on the paramiko transport.

12:05 Loader written: `scripts/hfr_convert_networks.py` turns the two
foreign layouts into the HFRNet-style monthly archive that
`build_hfr_multires.py` already reads (dims time/lat/lon, u/v, range-dated
filenames), so the target builder needed only thirteen new rows in its
table. Findings from the files: European totals are on regular 1-D grids
(2 km Ligurian/Sardinia/German Bight, 3 km western Med, 5 km Basque/Norway,
6 km Galicia) with QC flags 1 and 4 only; IMOS CODAR sites (Rottnest,
Coffs, South Australia, North West Shelf) are regular 1-D, but the WERA
sites (Turquoise Coast, Newcastle, Coral Coast) are curvilinear 2-D grids,
so those are nearest-cell binned through a KD-tree onto a regular grid at
native spacing; flags 1 and 2 kept. Regions with several networks (western
Med: Ibiza, Ebro, Catalan; Ligurian: TirLig, Sardinia) are binned to the
finest native spacing and averaged where they overlap. One surprise: the
Basque July 2024 subset request had returned the network's entire 2019 to
2026 record (250 MB), so the converter now clips every month to its
calendar bounds. SWOT pkl phase run for the six European regions; ERA5
and DUACS region tables extended (ERA5 subset also fixed for a box that
crosses the prime meridian, the western Med); ERA5 not pulled, it is an
S3 read and needs David's approval.

12:50 European regions through the colocation chain: SWOT pkls (116 to 231
pass entries per region), converted archives, and 25-h radar targets via
`build_hfr_multires.py` with thirteen new rows in its table. Colocated passes:
Biscay 114 of 116, Galicia 126, western Med 172, Ligurian/Tyrrhenian 231,
Norway 156 of 201 (radar record starts October 2023), German Bight 20 of 180
(only July to September 2023 radar, see above). The builder names outputs
`hfr_science_<region>_<suffix>.pkl` where the suffix repeats part of the
region name (it splits on the first underscore); copied each to the default
`hfr_science_<region>.pkl` the pipeline and `baselines_eval.py` expect rather
than touch the builder. DUACS L4 pulled from Copernicus for all thirteen
regions (3 to 27 MB each). Thirteen regions added to `presets.REGION_META`
so `config_overrides` resolves them through the local mirror; GOES stays
unset for them. Australian chain armed to run pkls and targets when the SWOT
download and archive conversion finish.

13:40 SWOT download finished for all thirteen regions: Australia 293
granules in just under ten hours (one stall), 818 segments (Rottnest 83,
Turquoise Coast 86, Coffs 93, Newcastle 108, South Australia 161, North West
Shelf 116, Coral Coast 171); Europe 620 granules, about 1,030 segments. The
whole SWOT store for the new regions is 606 MB on Z:, against roughly 170 GB
transferred. Australian pkl and target steps running from the armed chain.

14:05 Australian regions through the chain. Pkls 83 to 171 pass entries;
colocated passes: Rottnest 68 of 83, Turquoise Coast 80 of 86, Coffs 9 of 93
(radar up four months only), Newcastle 95 of 108, South Australia 96 of 161,
North West Shelf 116 of 116, Coral Coast 158 of 171. All thirteen regions now
have SWOT pkl, 25-h radar target and DUACS in the local mirror; Coffs Harbour
and German Bight are too thin to score and will be reported as such. The only
missing input is ERA5 wind, which the ladder needs on every row.

(The loader and per-region pkl and target phases named here as still to
build were done the next morning; see 12:05 to 14:05 above. ERA5 remains the
one missing input.)

## 2026-11-03 — Methods notes: R² and the climatology rung

### 09:30 Written explanation of pooled R² and the per-cell climatology, for review

David: "what do you mean reference is climatology", "how to compute
climatology? just demeaning?", "i still don't get it", "wait i thought we
measure point by point, not the entire region!", "so climatology is still
pooled over region", "so we get a climatology per cell?", "and how does
climatology compare to r squared. please also add these to research notes
clearly so i can review". Z: drive ejected at the start of the exchange.

New `docs/methods-notes.md` (indexed in the README tree), a plain-prose
companion to the glossary, with two sections: how the project computes R²
(one pooled number per region and component, regional held-out mean as the
reference, predictions per row but score per list) and what the climatology
rung is (per-cell training-period mean as a zero-input model, scored with the
same pooled R²; worked two-cell example with the arithmetic; the two ways it
differs from R²'s own reference, per-cell versus regional and training versus
held-out; the day-to-day share as model R² minus climatology R²; the skill
score and per-cell R² as alternative displays; the results table with the
day-to-day share per region). The sticking point in the conversation was
pooled versus per-cell scoring: predictions are per row, R² is per region, so
a model is credited for reproducing the between-cell mean flow, and the
climatology rung exists to make that credit visible. Future method questions
get a section in the same file.

### 11:00 One equation per ladder rung, from the code

David, after walking through climatology, climatology + wind and SWOT
calibrated + wind: "i think we should list equations for each regression
no?" Added section 3 to `docs/methods-notes.md`: a table with the prediction
equation and parameter count for all fifteen rungs, read directly from
`baselines_eval.py` (`fit_linear` least squares with intercept, u and v fitted
separately; `cell_linear` ridge-toward-global shrinkage with lambda 5 written
out; `fit_ridge72` standardised with NaN to zero and alpha 1; Ekman rule 3 %
at 45 degrees). Section 2 gained the full ladder tables with a
champion-minus-climatology row. Point of the exercise: the paper's methods
section can be lifted from this table, and the reader can see that every
linear rung is the same least-squares fit with a different input set.

### 11:40 LaTeX version of the equations

David: "let's get these equations down in latex format somewhere" and "so all
of these are regressions?" No manuscript exists yet, so
`docs/methods-equations/methods-equations.tex` is a standalone article
(pdflatex, amsmath, booktabs) whose body between the BEGIN BODY / END BODY markers can be
`\input` into the paper: notation, pooled R² and RMSE with the block
bootstrap, the skill score against climatology, every ladder rung as a
numbered equation (fixed rules, climatology, the seven least-squares rungs,
the per-cell shrinkage, ridge, the champion), and the truth tiers with the
25-h boxcar. Given its own folder at David's request and compiled with
TeX Live 2024 (two passes, no errors); the PDF is tracked beside the source,
aux and log files ignored. Answer to
the regression question, also in the notes: four rungs are fixed rules with
no parameters, climatology is a lookup, every rung from climatology + wind
to DUACS + wind is the same ordinary least squares with a different input
set, per-cell linear and ridge are penalised least squares, and the champion
is the one nonlinear fit.

## 2026-11-02 — Z: drive jobs: rotary spectrum on real data, calval truth tiers

### 11:10 Rotary spectrum rendered; calval tiers for three regions; tier guards in the figures

David: "connected my Z drive. what tasks required it, the rotary power spectrum?"

The WD Elements disk came up Offline again (Windows keeps the flag after a
safe eject); brought online with an elevated `Set-Disk -Number 1 -IsOffline
$false`.

**Rotary spectrum** (`scripts/hfr_rotary_spectrum.py --regions usegc,uswc
--start 2024-03-01 --end 2024-06-30`, 30 fully covered cells per region, about
six minutes, mostly reading the archive). Written to
`docs/figures/hfr_target/`. Reading of the figure is in the 2 November
addendum to the research update: M2 dominant on both coasts, clockwise-only
inertial bump at 1.27 cpd (usegc, 39.9 N) and 1.16 cpd (uswc, 35.5 N),
counterclockwise excess at periods over five days on the East Coast (Gulf
Stream), strong clockwise diurnal lines on the West Coast. The filter panel
shows the 25-h boxcar's first null at 0.96 cpd and a 5 % side lobe near 1.5
cpd, the 18-h Butterworth leaking half the diurnal/inertial band, the 48-h
filter damping the 2-5 day band.

**Calval tiers.** `hfr_tiers.py --run evolve_uswc_v1 --regions
ushi,uswc,usegc` built raw, 18-h and 48-h tiers from the 2023 archive; the
rebuilt 25-h mean matches the stored target exactly (r = 1.0000, rms 0.0000) in
all three regions, so the tier files sample the same hours and cells the
pipeline trained on. `baselines_eval.py --run evolve_uswc_v1 --regions
uswc,usegc,ushi --buffer-days 7` re-scored the ladder on every tier (results
in `experiments/evolve/evolve_uswc_v1/baselines/`). Numbers in the update
addendum. Finding worth keeping: on the East Coast common subset (29 % of
rows, the densest radar coverage) the per-cell linear model beats the
retrained champion by 0.10/0.18, where the full-row table had them tied. The
champion's parity there is subset-dependent.

**Figure guards.** Hawaii has 127 rows finite in every tier; R² on that subset
is meaningless (train-mean R² of -57). `fig_tiers` now drops regions with
under 1,000 common rows and says so in the caption; the benchmark grid greys
tier blocks under the same rule. The tiers figure also takes the champion
label from the per-mission table and the caption's training-period wording
from the mission, so the calval version no longer says "72 stencil cols" or
"cycles 1-12". Calval figures regenerated; figure index, science README,
replication task table updated.

## 2026-11-01 — Research update, calval on three regions, figures split by mission

### 14:30 Calval table for usegc and ushi; figure folders split into science / calval / hfr_target

David: "can we do all that now for calval?", "we should separate the figures
into science cycle and calval", and "can you provide an update of what was
done since the last research update".

**Eval bug.** The first three-region run silently scored only uswc: for a
single-region run `evaluate()` overwrote the requested region list with the
run's own region. Changed so a non-pooled run keeps the requested sister
regions whenever their solo freeze (`solo/<rid>/data.npz`) exists, own region
first. Relaunched; usegc and ushi replayed and verified against their frozen
Y_test exactly.

**Numbers** (25-h target, SWOT-valid, 7-day buffer, R² u/v): usegc retrained
champion 0.62/0.62 against per-cell linear 0.59/0.65, ridge 0.52/0.65,
climatology 0.41/0.59, 5-parameter linear 0.37/0.60 (intervals about plus or
minus 0.07 in u); uswc campaign champion 0.43/0.56 against per-cell linear
0.38/0.36 (unchanged); ushi, only 1,240 rows and 12 passes, climatology
0.90/0.73, per-cell linear 0.87/0.69, ridge 0.73/0.56, champion 0.72/0.60. The
West Coast calval margin does not generalise: on the East Coast the network
ties the strongest linear baseline, and in Hawaii a steady flow makes location
alone the best model. The one place the network clearly wins is the run it was
evolved on. Written into the 1 November research update as an addendum.

**Figures by mission.** `docs/figures/baselines/` became `docs/figures/science/`
(git mv, references repointed); `baselines_figures.py` now reads the run's
mission from `data_meta.json` and writes to `docs/figures/<mission>/`, with
per-mission protocol key (buffer30d or buffer7d), caption text, champion labels
and provenance note, regions narrowed to those present, panel widths scaled to
the region count, tier columns hidden when no tier has data, and the
science-only figures (overshoot, coverage, two orbits) skipped for calval. For
calval the champion is one row per region: campaign predictions for uswc, the
retrained twin for usegc and ushi, labelled as such. Ladder x-range extended to
1.0 (Hawaii climatology at 0.90 was clipped), the ">=" glyph replaced by "at
least" (cmr10 lacks it), the legend moved to the empty upper-left so it cannot
sit on the champion row. The rotary-spectrum figure writes to
`docs/figures/hfr_target/` since it belongs to neither orbit.
`docs/figures/README.md` rewritten with one section per folder.

**Research update** `docs/updates/research-update-2026-11-01.md` written,
covering 30 and 31 October (calval table, climatology rungs, cleanup,
replication, pipeline map, figure library, rotary spectrum) and the 1 November
discussion of the goal: an R² above 0.9 against the 25-hour radar is above the
measurement ceiling; the field-recognised version of a large margin is
scale-dependent skill against DUACS below 100 km, or the West Coast regime.

## 2026-10-31 — Repository cleanup plan

### 19:40 Rotary-spectrum figure: colours, split panels, legend placement

David: "can we get some colors on that rotary spectral plot", "i need some help
interpreting it too", "lord this is hard to get", "so what's the purpose of
having two vertical axes", "try splitting it", "also watch for collisions
between the legend and plot", and "no keep the frequency per cycle" when a
period axis was offered.

Changes to `scripts/hfr_rotary_spectrum.py`: filters coloured consistently in
every panel (target 25-h mean teal, Butterworth 18-h amber, 48-h violet;
regions ink and magenta); the twin right-hand axis replaced by a separate
middle panel for the filter responses sharing the frequency axis, so reading
is "find a spike on top, drop down, read the fraction passed"; tide and
inertial labels moved inside the top panel; the time-strip legend placed below
its panel so it cannot cross the tidal swings; frequency (cycles per day) kept
as the only horizontal axis per David's preference, with the axis label
spelling out slow-left, fast-right. Self-test re-run and inspected after each
change. The interpretation that landed, for the paper's methods paragraph: the
hourly radar current is mostly tide and inertial motion the altimeter cannot
see; the spectrum shows where that energy sits; the 25-hour mean removes it
while keeping the multi-day flow, with the two Butterworth cutoffs bracketing
it (18 h leaks the diurnal band, 48 h removes real two-day flow).

### 19:00 Figure library without numbers; rotary-spectrum figure (script ready, data pending)

David: "yes i want to do the power spectrum. should i do a rotary power spectrum
here comparing our options? you also need not number the figures, i will do
that in the paper. just create a library of them and a glossary so i know
which is what." Earlier in the thread: "should we make animations of the hfr
under butterworth?" (answered: no for the paper; one spectrum figure instead).

**Numbering dropped.** The eight baselines figures were renamed from
`figN_name` to `name` (git mv, both PNG and SVG), the save calls in
`baselines_figures.py` changed to match, and every reference in the README,
the figure README, the pipeline map, the replication guide and the plan
updated. `docs/figures/README.md` is the new library index: one table per
topic folder (baselines, poster, regions) with what each figure shows, what it
is for, and the script that regenerates it, plus the colour and font
conventions. Numbering now happens only in the paper.

**Rotary spectrum, why.** For a vector current the rotary spectrum (complex
series u + iv, negative frequencies clockwise, positive counterclockwise) is
the right diagnostic: inertial oscillations are clockwise-only in the northern
hemisphere and show as a clean one-sided peak at the local inertial period;
tidal ellipses appear on both sides; the sub-inertial flow is roughly balanced.
The target filters act on u and v with the same real coefficients, so one
response curve per filter applies to both rotary sides. The figure therefore
shows where the energy the altimeter cannot see lives and how much of it each
filter passes, against the continuum they keep.

**Script.** `scripts/hfr_rotary_spectrum.py` reuses `hfr_tiers.hourly_files`
and the baselines figure style. Welch with 30-day Hann segments and 50 %
overlap, averaged over the 30 best-covered cells per region (coverage >= 90 %,
gaps linearly interpolated), held-out window Mar-Jun 2024 by default. Filter
power responses computed analytically: boxcar 25-point |H|^2 (applied once),
Butterworth order 2 |H|^4 (filtfilt applies it twice). A `--selftest` builds a
synthetic series (red noise, CW-dominant M2 ellipse, K1, clockwise inertial at
35 N, white noise) and asserts: the inertial peak is clockwise-only (ratio > 5),
M2 is CW-dominant, the boxcar passes < 1 % at M2 and K1, the 48-h Butterworth
< 0.1 % at M2, the 18-h leaks more K1 than the 48-h, and all three pass > 90 %
of 8-day variance. The last threshold was first set at 95 % and failed on the
boxcar, which passes 94.5 % at 8 days; that is a real property of the running
mean (it damps week-scale flow by about 5 %) and is now stated in the figure
notes rather than hidden. Rendered self-test figure in `archive/selftest/`.
`hfr_tiers.py` now also honours `HFR_DATA_ROOT`.

**Pending.** The real-data render needs the hourly archive on Z:. Command in
the README; output `docs/figures/baselines/hfr_rotary_spectrum.{png,svg}`;
entry already in the figure README and library index marked pending.

### 17:30 Pipeline map

David, after a run of questions on where the 25-hour mean and the Butterworth
tiers sit relative to training: "i think i need a visual map of the entire
pipeline somewhere."

Wrote `docs/pipeline-map.md`: three Mermaid flowcharts (sources to local
mirrors; mirrors to flattened rows, data.npz and the evolve harness;
predictions to the baselines table and figures) plus reading notes. Mermaid
over a drawn figure because it renders on GitHub and in VS Code with nothing
to maintain, and can be exported to SVG for the paper later. The questions
that prompted it are answered by the map's structure: the target is fixed at
stage 1 (25-hour running mean at pass time), the evolve harness bypasses the
pipeline's train step and owns training inside each candidate, and the
Butterworth tiers enter only at stage 3 as alternative yardsticks for the same
predictions. README points to it from the Replication section and the file
hierarchy.

### 16:40 Replication gaps closed: lock file, HFRNet recipe, archive manifest, guide

David: "do all of the four things too. make a note that these replication
tasks should be updated as i'm progressing with the research in project-level
claude.md."

1. `requirements-lock.txt` at the repo root: `pip list --format=freeze` of the
   swotopentoolkit env (276 pins; Python 3.13.1; torch 2.13.0+cu126). A first
   attempt with `pip freeze` produced 167 `@ file:///` conda-build references
   that pip cannot install elsewhere, so the list form was used instead; the
   header says some conda-installed packages may resolve to different builds.
2. HFRNet route. Our hourly radar files came as handed-over NetCDFs screened at
   HDOP 1.0; HFRNet's public THREDDS "best" aggregations are screened at 1.25,
   so a public rebuild has slightly more cells. The recipe (an xarray/OPeNDAP
   pull per region, resolution and year into a folder named like ours, then
   `aggregate_hfr.py --phase build --keep-local` and `build_hfr_multires.py`)
   is in `docs/replication.md` 2.3. The two HFR scripts had our machine's path
   hard-coded; both now honour `HFR_DATA_ROOT` with the old default kept. The
   colocation script's `HFR_SOURCES` dict (S3 paths) was left as is and the
   guide says to repoint it; that code opens the archive inside a retry
   closure and changing it was not worth the risk today.
3. Data archive manifest: ten files (both runs' `data.npz`, `data_meta.json`,
   champion `candidate.py`, `params.json`, `predictions.npz`), sizes and
   SHA-256, about 1.95 GB, in `docs/replication.md` 2.7 with the Zenodo deposit
   steps. The deposit itself needs David's account and is pending; the DOI slot
   is marked.
4. `docs/replication.md`: the three levels with commands, the expected
   headline numbers, where Level 2 can diverge (HDOP, row-level rebuild
   differences, GPU jitter), the Level 3 caveat that the search is not
   replayable and the audit trail that replaces replay, and a task table.
   README gains a "Replication" section pointing there. CLAUDE.md gains a
   "Replication tasks (keep current)" rule so the guide and lock file are
   updated in the same session as any change to runs, protocols or data.

### 16:00 Replication: three levels, and what the paper must say

David: "how would someone replicate this research?" then "make a note of these
three levels in my research notes."

Everything below the commit `7b4baa3` (pushed to GitHub) is the state being
described.

**Level 1, regenerate the paper's numbers and figures from the repository.**
`results.json` for the science and calval baselines runs, the solo-versus-pooled
comparisons, and the per-seed metadata are tracked. `scripts/baselines_figures.py`
reproduces fig1 to fig7 from them with no data download and no GPU. The research
log records every protocol decision in order; `docs/figures/baselines/README.md`
says what each model regresses against what. Fully covered today; a reviewer can
do it in an afternoon.

**Level 2, rebuild the data and retrain the champion and the baselines.**
Possible from public sources, with effort. SWOT L3 Expert v3 from AVISO over
SFTP (free account; `scripts/build_science_colocations.py`, credentials from
`AVISO_USER`/`AVISO_PW`); pass selection from the tracked orbit shapefiles.
ERA5 winds and GOES SST from anonymous public S3 (`build_era5_winds.py`,
`build_goes_sst.py`). DUACS from Copernicus Marine, free account
(`duacs_pull.py`). The weak link is the HF radar: the project reads hourly
aggregates built from raw HFRNet files that live only locally and on Z:, with
the aggregates in a private S3 bucket. A replicator must pull the networks from
the HFRNet THREDDS server and run `aggregate_hfr.py` and `build_hfr_multires.py`;
the README names the source but has no turnkey command for that first hop. After
the mirrors exist, the pipeline, 25-hour detiding, stencil flattening and
temporal split are deterministic code. The champion is retrainable without the
search: its `candidate.py`, `params.json` and the pooled-data recipe are tracked,
and `solo_vs_pooled.py` / `animate_evolve_champion.py` retrain it with seeds
42, 43, 44. Budget: about 100 GB disk, a modest GPU, a few days, mostly
downloads.

**Level 3, rerun the evolutionary search.** Not replicable in the strict sense,
and the paper should say so. The mutator is an LLM, the search is stochastic, a
second run yields a different champion with a different architecture and
fitness. What is replicable is the procedure and the claim: LLM-guided search
over the training code with the buffered per-region fitness finds a model that
beats the linear ladder by the reported margins. The tracked `database.jsonl`
holds every candidate, its idea, metrics and parent, so the champion's lineage
is auditable even though the search cannot be replayed. Present it as artifact
plus provenance plus harness, which is the standard framing for this class of
result.

**Gaps to close before a preprint (not yet done).** (1) A pinned environment:
`pyproject.toml` has unversioned dependencies and torch+CUDA is installed by
hand; a `pip freeze` lock and the Python version remove the most common failure.
(2) A documented HFRNet-to-aggregate command so the radar side has the one-line
recipe the SWOT and wind sides have. (3) A DOI'd archive (Zenodo) of the frozen
`data.npz` (about 1.4 GB) and champion predictions so Level 2 can start from the
exact rows rather than a rebuild that may differ by a few rows. (4) A short
`docs/replication.md` walking the three levels and stating the Level 3 caveat.

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
