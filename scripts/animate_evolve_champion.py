"""Animate every region with an evolve champion candidate.

The evolve harness only keeps champion predictions on the held-out test rows,
so this script (1) retrains the champion once on the run's frozen pooled
data.npz with the exact params.json the run used, (2) predicts every
full-stencil grid row of every cycle/pass in every region, and (3) drives the
existing step_inference + step_animate pipeline with a lookup "model" that
serves those precomputed predictions — the animation panels, gating, and
output paths are untouched pipeline code.

All reads are local (pipeline caches / local mirrors) — never S3.
Resumable: the trained predictions are checkpointed to
experiments/evolve/{run}/candidates/{cid}/anim_preds.npz; delete it to force
a retrain.

Usage:
    python scripts/animate_evolve_champion.py --run evolve_all_science_v1
    # candidate defaults to the run's highest-fitness "ok" candidate
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import swotxai  # noqa: F401  (torch-before-numpy DLL ordering)
import numpy as np

from swotxai.config import load_config
from swotxai.presets import config_overrides

EVOLVE_ROOT = REPO_ROOT / "experiments" / "evolve"


def _cb(step, frac, msg):
    print(f"[{step}] {msg}", flush=True)


def _key(arr: np.ndarray) -> str:
    return f"{hashlib.sha1(arr.tobytes()).hexdigest()}:{arr.shape[0]}"


class _LookupModel:
    """Serves precomputed champion predictions through the backend-agnostic
    swotxai.models.predict interface, keyed by the content of the feature
    matrix plotter passes in (byte-identical to what phase 1 enumerated)."""

    _swotxai_kind = "ann"  # routes swotxai.models.predict to .predict()

    def __init__(self, table: dict, col: int):
        self.table = table
        self.col = col

    def predict(self, X):
        arr = X.to_numpy(dtype=np.float32) if hasattr(X, "to_numpy") else np.asarray(X, dtype=np.float32)
        if not len(arr):
            return np.zeros(0, dtype=np.float32)
        return self.table[_key(arr)][:, self.col]


def _best_candidate(run_dir: Path) -> str:
    best, best_fit = None, -np.inf
    with open(run_dir / "database.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            fit = (rec.get("metrics") or {}).get("fitness")
            if rec.get("status") == "ok" and fit is not None and fit > best_fit:
                best, best_fit = rec["candidate_id"], fit
    if best is None:
        raise SystemExit("No successful candidate in database.jsonl")
    print(f"Champion: {best} (fitness {best_fit:.4f})")
    return best


def _candidate_record(run_dir: Path, cid: str) -> dict:
    with open(run_dir / "database.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("candidate_id") == cid:
                return rec
    return {}


def _region_config(base, rid: str, mission: str, meta: dict, run_id: str):
    cfg = replace(
        base, **config_overrides(rid, mission),
        model="ann", features=list(meta["features"]), stencil_k=int(meta["stencil_k"]),
        run_id=run_id, experiment_id="", frame_dir="", animation_output="",
    )
    cfg.ensure_output_paths()
    return cfg


def _enumerate_blocks(flat: dict):
    """Yield the full-stencil feature blocks in the exact form plotter builds
    them (df rows with no NaN in any column, as float32)."""
    for key in flat:
        for entry in flat[key]:
            df = (entry or {}).get("df")
            if df is None:
                continue
            arr = df[~df.isna().any(axis=1)].to_numpy(dtype=np.float32)
            if len(arr):
                yield arr


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="evolve run name under experiments/evolve/")
    ap.add_argument("--candidate", default=None, help="candidate id (default: best ok candidate)")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--regions", default=None, help="comma-separated subset (default: all in the run)")
    args = ap.parse_args()

    run_dir = EVOLVE_ROOT / args.run
    meta = json.loads((run_dir / "data_meta.json").read_text(encoding="utf-8"))
    cid = args.candidate or _best_candidate(run_dir)
    cand_dir = run_dir / "candidates" / cid
    params = json.loads((cand_dir / "params.json").read_text(encoding="utf-8"))
    mission = meta["mission"]
    regions = meta.get("regions") or [meta["region"]]
    if args.regions:
        regions = [r for r in args.regions.split(",") if r in regions]
    run_id = f"evolve_{cid}"
    base = load_config(args.config)

    from swotxai.pipeline.orchestrator import run_shared_steps
    from swotxai.pipeline.steps_ml import step_flatten, step_inference
    from swotxai.pipeline.steps_viz import step_animate

    preds_path = cand_dir / "anim_preds.npz"
    index_path = cand_dir / "anim_index.json"

    if preds_path.exists() and index_path.exists():
        print("Reusing checkpointed champion predictions.")
        pred = np.load(preds_path)["pred"]
        index = json.loads(index_path.read_text(encoding="utf-8"))
    else:
        # Phase 1: enumerate every animation row across all regions.
        blocks, index = [], []
        for rid in regions:
            rcfg = _region_config(base, rid, mission, meta, run_id)
            shared = run_shared_steps(rcfg, _cb, True)
            flat = step_flatten(rcfg, shared["hfr_interp_data"], shared["swot_features"], _cb, True)
            n0 = len(blocks)
            for arr in _enumerate_blocks(flat):
                index.append({"rid": rid, "key": _key(arr), "n": int(arr.shape[0])})
                blocks.append(arr)
            print(f"[{rid}] {len(blocks) - n0} pass blocks, "
                  f"{sum(b.shape[0] for b in blocks[n0:]):,} rows")
            del shared, flat
            gc.collect()
        X_anim = np.concatenate(blocks)
        blocks.clear()
        gc.collect()
        if X_anim.shape[1] != meta["n_inputs"]:
            raise SystemExit(f"Column mismatch: animation rows have {X_anim.shape[1]} "
                             f"inputs, run data has {meta['n_inputs']}")
        print(f"Total animation rows: {len(X_anim):,} x {X_anim.shape[1]}")

        # Phase 2: one champion training run (exact run params), predict all rows.
        data = np.load(run_dir / "data.npz")
        spec = importlib.util.spec_from_file_location("evolve_candidate", cand_dir / "candidate.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        t0 = time.time()
        print(f"Training champion {cid} (budget {params['time_budget_s']:.0f}s)...")
        pred = np.asarray(
            module.train_and_predict(data["X_train"], data["Y_train"], X_anim, params),
            dtype=np.float32)
        print(f"Champion trained + predicted in {time.time() - t0:.0f}s")
        del X_anim, data
        gc.collect()
        np.savez_compressed(preds_path, pred=pred)
        index_path.write_text(json.dumps(index), encoding="utf-8")

    table, off = {}, 0
    for it in index:
        table[it["key"]] = pred[off:off + it["n"]]
        off += it["n"]
    model_u, model_v = _LookupModel(table, 0), _LookupModel(table, 1)

    per_region = (_candidate_record(run_dir, cid).get("metrics") or {}).get("per_region", {})

    # Phase 3: standard inference + animation per region. One region failing
    # (e.g. glna: grid too small for any full-stencil row, so the plotter has
    # nothing to predict) must not kill the remaining regions.
    outputs = []
    for rid in regions:
        if not any(it["rid"] == rid for it in index):
            print(f"[{rid}] SKIPPED: no full-stencil rows (every cell within "
                  f"stencil reach of a swath edge) — nothing to animate")
            continue
        try:
            rcfg = _region_config(base, rid, mission, meta, run_id)
            shared = run_shared_steps(rcfg, _cb, True)
            flat = step_flatten(rcfg, shared["hfr_interp_data"], shared["swot_features"], _cb, True)
            swot_dict, hfr_dict = step_inference(
                rcfg, model_u, model_v, shared["swot_regridded"],
                shared["hfr_interp_data"], flat, _cb, True)
            paths = step_animate(
                rcfg, swot_dict, hfr_dict, shared.get("cycle_data", {}),
                shared["swot_regridded"], shared["hfr_interp_data"],
                per_region.get(rid, {}), _cb)
            outputs.extend(paths)
            print(f"[{rid}] animations: {paths}")
            del shared, flat, swot_dict, hfr_dict
            gc.collect()
        except Exception as exc:  # noqa: BLE001 — keep animating other regions
            print(f"[{rid}] FAILED: {type(exc).__name__}: {exc}")
            gc.collect()

    print("\nAll animations:")
    for p in outputs:
        print(f"  {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
