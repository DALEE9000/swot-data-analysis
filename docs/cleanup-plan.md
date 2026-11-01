# Repository cleanup plan

Drafted 2026-10-31. Status: Groups A, C, D, E executed 2026-10-31 (see research log). Group B executed 2026-10-31: large folders and RF animations copied to `Z:\data_repository\columbia_oceanography\SWOT-data-analysis\archive\` (verified by byte count) and the local copies sent to the Recycle Bin; small folders moved to `archive/`. Cache twins were content-identical (byte diff was dict key order) and were replaced by NTFS hard links rather than deleted, so both cache keys still resolve. Each step says what
it touches, whether git tracks it, and what has to change to keep the code
working. Steps are grouped by risk so they can be approved and run one group
at a time.

## Where the sprawl is

Disk, from the repo root (MB):

| Folder | MB | Tracked | What it is |
|---|---|---|---|
| experiments/ | 85,978 | evolve/ and registry/ only | S3 mirrors, flattened pickles, model weights, DUACS, evolve runs |
| cache/ | 22,084 | no | step pickles (regenerable); about 8 GB are duplicate pairs |
| old_movies/ | 2,263 | no | six swath GIFs from 2025 |
| SWOT_hf_plotting/ | 1,303 | no | one notebook, one merged nc4, two test PNGs |
| data/ | 1,150 | no | raw ERA5 GRIB for usegc 2023 (already built into experiments/usegc/era5) |
| notebooks/ | 858 | no | 4 notebooks plus one GOES nc |
| animations/ | 464 | no | MP4 output, includes rf/ runs id012..id016 from July |
| papers/ | 115 | no | PDF library plus NOTES.md |
| frames/ | 42 | no | PNG frames, including one stale run ann_20260718_112506 |
| ne_10m_bathymetry_all/ | 35 | yes, 84 files | shapefiles, used only by src/swot/plotting_scripts.py |
| images/ | 30 | no | four early inference PNGs |
| examples/, swot_documentation/, scratch_notebooks/ | 44 | no | tutorial notebooks, two PDFs, scratch |
| docs/ | 13 | 2 of 33 files | logs, plan, glossary, baselines figures, poster figures |
| build/, __pycache__/, src/swotxai.egg-info/ | 1 | no | build leftovers |

Twenty-one top-level entries. Nine are output or scratch folders that no code
reads. Three sets of figures live in three places (`figures/`,
`docs/figures/science/`, `docs/poster/`). Two folders hold notebooks
(`notebooks/`, `scratch_notebooks/`) plus a third notebook inside
`SWOT_hf_plotting/`.

## Target layout

```
SWOT-data-analysis/
├── README.md  CLAUDE.md  pyproject.toml  config.yaml  app.py
├── src/swotxai/            pipeline, models, evolve (unchanged)
├── src/swot/               legacy helpers still imported by the pipeline (unchanged, credentials removed)
├── scripts/                15 entry points (unchanged; README gains a one-table index)
├── tests/
├── docs/
│   ├── research-log.md  research-plan.md  glossary.md  model-data.md  preprint-style-checklist.md
│   ├── updates/            research-update-YYYY-MM-DD.md (dated memos)
│   └── figures/
│       ├── baselines/      Phase-1 figure set (unchanged)
│       ├── poster/         was docs/poster/ (one path string in two scripts)
│       └── regions/        was figures/ (4 region-overview PNGs)
├── papers/                 PDF library plus NOTES.md; gains handbooks/ (was swot_documentation/)
├── orbit_data/  ne_10m_bathymetry_all/   static assets (unchanged)
├── experiments/            data mirrors and runs (unchanged layout; stray files tidied)
├── cache/  frames/  animations/           pipeline outputs, gitignored (pruned, not moved)
└── archive/                gitignored; everything below moves here untouched
    ├── old_movies/  SWOT_hf_plotting/  images/  scratch_notebooks/  notebooks/  examples/
    └── data_era5_grib/     was data/
```

Output folders (`cache/`, `frames/`, `animations/`) stay where they are because
the pipeline writes to those paths. Moving them means editing working code, and
the gain is small once they are pruned.

## Group A: zero-risk deletions (regenerable or empty)

| Path | MB | Why safe |
|---|---|---|
| build/, __pycache__/ (root), src/swotxai.egg-info/, tests/__pycache__/ | 1 | build artefacts, gitignored, recreated by pip and pytest |
| experiments/akns/ | 0 | three empty folders from an abandoned region |
| cache/swot_features_science_*_evolve_gen031_00.pkl (5 files) | about 8,000 | same size as the un-suffixed file beside each one; confirm byte-identical with `cmp` before deleting. Re-animating the champion regenerates them. |
| frames/usegc/ann/ann_20260718_112506_3e1298/ | small | frames of a superseded July run |
| animations/*/rf/ (id012..id016, test1) | about 50 | July RF runs not referenced in any doc or log |
| experiments/evolve/*/candidates/*/__pycache__ | small | compiled candidates |

Check for the cache twins, run before any deletion (prints nothing if identical):

```bash
cmp cache/swot_features_science_usegc_1_16.pkl cache/swot_features_science_usegc_1_16_evolve_gen031_00.pkl
```

## Group B: move untracked loose folders into archive/ (no code change)

All untracked, no script or module references them. Add `/archive/` to
`.gitignore` and move:

| From | To | Note |
|---|---|---|
| old_movies/ | archive/old_movies/ | 2.3 GB; better: copy to Z: and delete locally |
| SWOT_hf_plotting/ | archive/SWOT_hf_plotting/ | 1.3 GB nc4; same |
| data/ | archive/data_era5_grib/ | raw GRIB already converted to experiments/usegc/era5 |
| images/ | archive/images/ | |
| scratch_notebooks/ | archive/scratch_notebooks/ | |
| notebooks/ | archive/notebooks/ | README line 167 to update |
| examples/ | archive/examples/ | README lines 135 and 172 to update |
| experiments/duacs_pull.log | delete, or experiments/uswc/duacs/ | one stray log at the experiments root |
| docs/poster/importance_run.log | experiments/evolve/evolve_all_science_v1/ | run log filed with figures |

Z: is offline (ejected 2026-10-30). The 3.6 GB of GIFs and nc4 should go there
rather than stay on C:.

## Group C: tracked moves and small path edits (one commit)

| Step | Touches | Change |
|---|---|---|
| `git mv figures docs/figures/regions` | 4 tracked PNGs | README line 170 |
| `mv docs/poster docs/figures/poster` | 10 untracked files | `scripts/poster_figures.py` line 10 and its output dir; `scripts/champion_importance.py` line 11 and its json path; README lines 341 and 346 |
| `mkdir docs/updates; mv docs/research-update-2026-10-29.md docs/updates/` | 1 untracked md | links in research-plan.md and research-log.md |
| `mv swot_documentation papers/handbooks` | 2 PDFs, untracked | README line 173; `.gitignore` already ignores /papers/ |
| `git rm gen_netrc.py` (done) | 1 tracked file, nothing imported it | Earthdata login helper copied from NeurOST; delete if PO.DAAC pulls now go through local mirrors |

## Group D: security, do first regardless of the rest

`src/swot/download_swaths.py` lines 93 and 265 contain a hard-coded SFTP
username and password for the AVISO server, belonging to a third party, and the
file is tracked in git. Remove both literals and read them from `.netrc` or an
environment variable. The history still holds them, so also tell the
credential's owner to rotate it. Rewriting history is not worth it for a private
repo once the password is dead.

## Group E: documentation that prevents the next sprawl

- README "Repository layout" block (lines 130 to 175) redrawn to match the target layout above.
- README gains a table of the 15 scripts, one line each, with the folder each writes to. Three scripts have no README mention today: `build_era5_winds.py`, `build_goes_sst.py`, `repartition_usegc1km.py`.
- One rule added to CLAUDE.md: new figures go under `docs/figures/<topic>/`, dated memos under `docs/updates/`, notebooks and one-off output under `archive/`. No new top-level folders.

## Not changing

- `experiments/<region>/{swot_cycles,hfr_target,era5,goes,flattened,ann,rf,duacs}`: this is the pipeline's cache_path layout. Consistent across regions.
- `experiments/evolve/<run>/{candidates,solo,baselines,...}`: run provenance, tracked, referenced throughout the research log.
- `src/swot/`: three pipeline modules import it. Legacy but live.
- `ne_10m_bathymetry_all/`: already in history, so untracking frees no clone size.
- `cache/` large non-duplicate pickles (about 14 GB): rebuilding the usegc science features takes hours.

## Order and effort

1. Group D (10 minutes; contact the credential's owner about rotation).
2. Group A after the `cmp` check (5 minutes, frees about 8 GB).
3. Group B with Z: reconnected (15 minutes, frees 3.6 GB from C:).
4. Groups C and E together as one commit (30 minutes), then rerun `scripts/poster_figures.py` to confirm the new output path.
