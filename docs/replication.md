# Replicating this research

Written 2026-10-31 against commit `7b4baa3`. Three levels, from "check the
paper's numbers" to "rerun the search". Each level says what it needs, what
command does it, and where it can diverge from what we got. The task list at
the end is kept current as the research moves (project rule in CLAUDE.md).

## Level 1: regenerate every number and figure from the repository

Needs: a clone, Python, matplotlib. No data download, no GPU, about ten minutes.

Everything the figures read is tracked: `results.json` and `results.md` under
`experiments/evolve/evolve_all_science_v1/baselines/` (science, 21-day orbit)
and `experiments/evolve/evolve_uswc_v1/baselines/` (calval, 1-day orbit), the
solo-versus-pooled `comparison.json`, and per-seed `meta.json`/`params.json`.

```bash
pip install -r requirements-lock.txt --extra-index-url https://download.pytorch.org/whl/cu126
pip install -e .
python scripts/baselines_figures.py --run evolve_all_science_v1   # Phase-1 figure set -> docs/figures/science/
python scripts/poster_figures.py    --run evolve_all_science_v1   # fitness scatter, architecture -> docs/figures/poster/
```

What to read alongside: `docs/research-log.md` (every protocol decision, in
order, with the reason), `docs/figures/science/README.md` (what each model
regresses against what, per figure), `docs/glossary.md`.

## Level 2: rebuild the data, retrain the champion and the baselines

Needs: free AVISO+ and Copernicus Marine accounts, about 100 GB of disk, a GPU
with 4 GB or more, a few days of wall time (mostly downloads). Costs nothing
in cloud fees if you work from local disk; the project's own S3 bucket is
private and billed to us, so do not point scripts at it.

### 2.1 Environment

`requirements-lock.txt` is the exact `pip list` of the environment behind the
results (Python 3.13.1, torch 2.13.0+cu126). `pyproject.toml` carries the loose
ranges if the lock does not resolve on your platform.

### 2.2 Inputs and where they come from

| Input | Source | Account | Script | Lands in |
|---|---|---|---|---|
| SWOT L3 LR SSH Expert v3 (science cycles 1-16; calval 474-578) | AVISO+ SFTP `swot_products/l3_karin_nadir/l3_lr_ssh/v3_0/Expert/reproc` | free AVISO+ login, exported as `AVISO_USER`, `AVISO_PW` | `scripts/build_science_colocations.py` | `experiments/{region}/swot_cycles/` |
| Pass selection | orbit shapefiles, tracked in `orbit_data/` | none | same script (`find_swaths`) | |
| HF radar hourly surface currents | NOAA NCEI HF radar RTV archive (HFRNet THREDDS at Scripps retired 2026), see 2.3 | none | `scripts/aggregate_hfr.py`, `scripts/build_hfr_multires.py`, colocation script | `experiments/{region}/hfr_target/` |
| HF radar, European node (10 networks, 6 regions `eu_*`) | `thredds.hfrnode.eu` NCSS, totals v3 (v2.2 fallback) | none | `scripts/fetch_hfr_eu.py` | `HFR_DATA_ROOT/eu_<net>_hourly/` |
| HF radar, IMOS ACORN (8 sites, regions `au_*`) | `thredds.aodn.org.au` fileServer, `gridded_1h-avg-current-map_QC` | none | `scripts/fetch_hfr_imos.py` | `HFR_DATA_ROOT/imos_<site>_hourly/` |
| ERA5 10-m winds | NCAR mirror `s3://nsf-ncar-era5` (anonymous; lags ~2 months) | none | `scripts/build_era5_winds.py` (append mode with `--months`) | `experiments/{region}/era5/`, science Jul 2023 to Aug 2026 for all 19 regions |
| GOES-16/18 SST (calval runs only) | `s3://noaa-goes16`, `s3://noaa-goes18` (anonymous) | none | `scripts/build_goes_sst.py` | `experiments/{region}/goes/` |
| DUACS L4 daily geostrophy (baseline only) | Copernicus Marine `SEALEVEL_GLO_PHY_L4_MY_008_047` | free, via `copernicusmarine login` | `scripts/duacs_pull.py` | `experiments/{region}/duacs/` |

Regions: `uswc`, `usegc`, `prvi`, `ushi` carry the science results; `gak` and
`glna` have no SWOT coverage at their radar cells and are dropped from the
fitness; calval exists for `uswc`, `usegc`, `ushi` only. Bounding boxes and
default radar resolution per region: `src/swotxai/presets.py`.

### 2.3 HF radar: the one hop without a one-line script

Our hourly files were handed over as per-hour and per-chunk NetCDFs under
`HFR/Code/Data/<region>_<res>_Resolution_hourly/` and were screened at HDOP 1.0.
The public route is HFRNet's THREDDS server, whose "best" hourly RTV
aggregations are screened at HDOP 1.25, so expect slightly more cells than we
had. Pull one region and resolution per year with xarray over OPeNDAP, drop the
files into a folder named like ours, then run the two project scripts:

```python
# pull_hfrnet.py (illustrative; check the exact dataset name in the catalog at
# https://hfrnet-tds.ucsd.edu/thredds/catalog.html -> HFR/<REGION>/<res>/hourly/RTV/)
import xarray as xr, pathlib
region, res, code, name = "uswc", "6km", "USWC", "US_West_Coast"
url = (f"https://hfrnet-tds.ucsd.edu/thredds/dodsC/HFR/{code}/{res}/hourly/RTV/"
       f"HFRADAR_{name}_{res}_Resolution_Hourly_RTV_best.ncd")
out = pathlib.Path("HFR/Code/Data") / f"{region}_{res}_Resolution_hourly"; out.mkdir(parents=True, exist_ok=True)
ds = xr.open_dataset(url)[["u", "v"]]
for year in range(2012, 2025):
    ds.sel(time=str(year)).to_netcdf(out / f"{region}_{res}_{year}.nc4")
```

```bash
# where the raw folders live (default is our machine's path)
export HFR_DATA_ROOT=/path/to/HFR/Code/Data
# full-period aggregate per (region, resolution); --phase build skips the S3 upload
python scripts/aggregate_hfr.py --targets uswc_6km --phase build --keep-local
# 25-h running-mean colocation targets at every resolution for the held-out rows
python scripts/build_hfr_multires.py --targets uswc_6km
```

The colocation script reads the default-resolution aggregate for each region
from the `HFR_SOURCES` dict at the top of `scripts/build_science_colocations.py`;
point those entries at your aggregates. We can also share our aggregates on
request (tens of GB), which removes the HDOP difference.

### 2.4 Build the colocations and features

```bash
export AVISO_USER=... AVISO_PW=...
python scripts/build_science_colocations.py --regions uswc,usegc,prvi,ushi --mission science --keep-local
python scripts/build_science_colocations.py --regions uswc,usegc,ushi        --mission calval  --keep-local
python scripts/build_era5_winds.py
python scripts/build_goes_sst.py          # calval only
python scripts/duacs_pull.py              # science window
python scripts/duacs_pull.py --regions uswc,usegc,ushi --start 2023-03-01T00:00:00 --end 2023-08-31T23:59:59
```

Phases inside the colocation script are resume-safe; skip its upload phase
(`--phase`, see the docstring). Everything downstream (regridding, 25-hour
detiding of the radar, 3x3 stencil flattening, the temporal 80/20 split with
the 30-day buffer) is deterministic code in `src/swotxai/`.

### 2.5 Retrain the champion without rerunning the search

The champion is ordinary tracked code: `candidates/gen031_00/candidate.py` and
`params.json` (science, pooled over four regions, 72 input columns) and
`candidates/gen020_02/` (calval `uswc`, 81 columns including SST). The frozen
train/test arrays `data.npz` are not tracked (1.5 GB and 0.43 GB). Either take
them from the data archive in 2.7, or rebuild them: `scripts/evolve.py` writes
`data.npz` as its first step before any candidate runs.

```bash
python scripts/solo_vs_pooled.py --run evolve_all_science_v1 --seeds 42,43,44   # retrain per region, 3 seeds
python scripts/baselines_eval.py --run evolve_all_science_v1 --bootstrap 1000  # full table, science
python scripts/baselines_eval.py --run evolve_uswc_v1 --buffer-days 7          # calval table
python scripts/baselines_figures.py --run evolve_all_science_v1
```

Expected headline (25-h target, SWOT-valid rows, 30-day buffer, per-region R²
u/v): `usegc` champion 0.57/0.64 against 0.37/0.55 for per-cell climatology and
0.54/0.59 for ridge-72; `uswc` all models tie near 0.2/0.1. Calval `uswc`:
0.43/0.56 against 0.38/0.36 per-cell linear. Seed-to-seed spread is a few
hundredths; bootstrap intervals are in `results.json`.

### 2.6 Where Level 2 can diverge

- HDOP screening 1.25 versus 1.0 changes which radar cells exist.
- A rebuilt `data.npz` can differ by a handful of rows if a granule is missing
  or a radar outage edge lands differently; the archive in 2.7 removes this.
- GPU nondeterminism in torch gives seed-level jitter of order 0.01 in R².

### 2.7 Data archive (DOI pending)

To be deposited on Zenodo under the paper's title. Files, sizes and SHA-256 as
of 2026-10-31 (paths relative to `experiments/evolve/`):

| File | Size | SHA-256 |
|---|---|---|
| `evolve_all_science_v1/data.npz` | 1,510,770,794 | `d9c59d2b56e5a87c4d5041dde3a460b40ba8211444742368aa8d33ce74f0dc3c` |
| `evolve_all_science_v1/data_meta.json` | 7,001 | `b1acaf4f2d2e1da85a55b1d7962de1fc9ce20bf6b82ab4cfcdfad312b7c3995d` |
| `evolve_all_science_v1/candidates/gen031_00/candidate.py` | 24,967 | `e610df7998b6bb6753f3d0d8a117ab0cdd9d781e9bd453c8904e75a4a9b44d6e` |
| `evolve_all_science_v1/candidates/gen031_00/params.json` | 73 | `19f8d794cffa1e72aae4932b352d6868953d9289f2dedf2c44213b2efdb58203` |
| `evolve_all_science_v1/candidates/gen031_00/predictions.npz` | 9,436,001 | `120457f2b769dd464085ae857f9d5ab38fa6fc6ff70ddc3cc3f59bc3f14d1a48` |
| `evolve_uswc_v1/data.npz` | 428,779,334 | `e9fe74f2b54ba89ed06afa4a76128f61ef219d6a3e23160da413e29563845a91` |
| `evolve_uswc_v1/data_meta.json` | (small) | `2e30a4a48cb579bdec11c37865cf87f1e95011522488ee7ab30cf6e6e846f56c` |
| `evolve_uswc_v1/candidates/gen020_02/candidate.py` | 38,841 | `3aedac4b499d42a0b422cf91b64f8a05c4270f3970f63ade41d7ffdd39adda1d` |
| `evolve_uswc_v1/candidates/gen020_02/params.json` | 73 | `4db48b316c212272ec0494df23026d152ab1bcac9e234a8d67a12370ba0a95aa` |
| `evolve_uswc_v1/candidates/gen020_02/predictions.npz` | 1,969,214 | `3ca0e0004b57b89a9eb3f805537846fe075c437f347343cbe170b7cacfc73834` |

About 1.95 GB in total, well under Zenodo's 50 GB per record. `data.npz` keys:
`X_train, Y_train, X_test, Y_test, test_time`; `data_meta.json` names the
features, stencil, regions, row spans and per-region training end dates.
Deposit steps (manual, needs the account holder): create the record, upload
these ten files, add `requirements-lock.txt` and the commit hash to the
description, reserve the DOI, then write the DOI here and in the README. Verify
each upload with `sha256sum` against this table.

## Level 3: rerun the evolutionary search

Not replicable in the strict sense, and the paper says so. The mutator is a
large language model (`--mutator claude-cli`, Claude Code CLI), the search is
stochastic, and a second run yields a different champion with a different
architecture and fitness. What is replicable is the procedure and the claim:
LLM-guided search over the training code with the buffered per-region fitness
finds a model that beats the linear ladder by the margins in the tables.

```bash
python scripts/evolve.py --name my_run --pooled science --regions uswc,usegc,prvi,ushi \
    --fitness per_region_buffered --fitness-buffer-days 30 --mutator claude-cli
```

Audit trail of our run: `experiments/evolve/evolve_all_science_v1/database.jsonl`
holds every candidate (idea, metrics, parent), `state.json` the loop state,
`full_run.log` the run log, and every `candidate.py` is tracked. The champion's
lineage can be followed even though the search cannot be replayed. Note that
`evolve_all_science_v1` itself was run with the earlier pooled-R² fitness; the
buffered per-region protocol was adopted on 2026-10-29 for evaluation and for
future searches (research log, that date).

## Replication task list (keep current)

| Task | Status | Note |
|---|---|---|
| Pinned environment | done 2026-10-31 | `requirements-lock.txt` |
| HFRNet to aggregate recipe | done 2026-10-31 | section 2.3; `HFR_DATA_ROOT` override in the two HFR scripts |
| Data archive with DOI | manifest ready; deposit pending | section 2.7, needs the Zenodo account holder |
| This document | done 2026-10-31 | update when protocols, runs or data change |
| Calval tables for `usegc`, `ushi` | done 2026-11-01 | `baselines_eval.py --run evolve_uswc_v1 --regions uswc,usegc,ushi --buffer-days 7`; figures in `docs/figures/calval/` |
| Rerun search under the buffered fitness | not started | Phase 2 of `docs/research-plan.md` |
| Rotary-spectrum figure on real radar data | done 2026-11-02 | `scripts/hfr_rotary_spectrum.py --regions usegc,uswc --start 2024-03-01 --end 2024-06-30` (needs Z: or `HFR_DATA_ROOT`) |
| New radar networks: Europe (6 regions) and Australia (7 regions), science orbit | SWOT, radar, DUACS downloaded and Europe colocated 2026-11-05; ERA5 built 2026-11-09/10 (Jul 2023 to Aug 2026, all 19 regions) | `fetch_hfr_eu.py`, `fetch_hfr_imos.py`, then `build_science_colocations.py --regions eu_*,au_* --phase download`; 82 passes, ~1,300 granules; HFR-target and pkl phases still need a loader for the non-HFRNet file layout |
| Daily-mean animations of every radar region (19 regions, full archives) | done 2026-11-05 | `scripts/animate_hfr_regions.py`; MP4s gitignored, regenerate from the hourly archives on Z: |
| Full radar records for Europe (2019-present) and Australia (2009-present), US HFRNet gaps and 2025-2026 tail | done 2026-11-08 (EU node, IMOS, NCEI for the US; the Scripps HFRNet server is retired) | README "Full radar records" block: `fetch_hfr_eu.py`, `fetch_hfr_imos.py`, `hfr_convert_networks.py --start 2009-01 --end 2026-10`, `fetch_hfrnet.py --go` |
| SWOT science cycles 017-058 (forward stream from 033) for all 19 regions | done 2026-11-08 (segments on Z: via the `experiments/<rid>/swot_cycles` junctions; appended to all 19 pkls) | `build_science_colocations.py --phase download --cycles ... --manifest-tag ...`, then `--phase pkl --cycles 001-058` (append mode, existing cycles kept) |
| HFR and SWOT side-by-side animations, per region and per continent | done 2026-11-10: 22 full-record movies plus 22 `_since2023` cuts on Z: (`docs/figures/hfr_animations` junction) | `scripts/animate_hfr_swot.py` (README block); daily cache and frames on Z: under `archive/hfr_anim/` |
| Calval truth tiers (raw, 18-h, 48-h) for `uswc`, `usegc`, `ushi` | done 2026-11-02 | `hfr_tiers.py --run evolve_uswc_v1 --regions ushi,uswc,usegc`, then the calval `baselines_eval.py` line above |
