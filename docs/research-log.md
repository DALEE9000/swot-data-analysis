# Research Log

Dated notes on findings, experiments, and decisions for the SWOTxAI project.
Convention: newest entry first; one `##` section per date; record *why* alongside
*what*. Machine-readable experiment details live in `experiments/registry/` and
`experiments/evolve/*/database.jsonl` — this log is for the human narrative.

---

## 2026-07-24 — Code + evolution provenance published to GitHub

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
