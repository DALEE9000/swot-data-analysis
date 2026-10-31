"""Pooled champion vs region-solo models of the same form.

For each region in an evolve run's pool, freeze that region's own dataset
(same temporal 80/20 split the pooled prep used), train the champion
candidate code on region-only data with the run's exact params (same seed,
epoch cap, wall budget), and score it on the region's held-out rows — the
SAME rows the pooled champion's per-region breakdown was scored on. Output:
a comparison table pooled-vs-solo per region.

All reads local; the only cost is GPU time. Resumable: finished
(region, seed) predictions are kept under
experiments/evolve/{run}/solo/{rid}/seed{N}/ and skipped on rerun.

Usage:
    python scripts/solo_vs_pooled.py --run evolve_all_science_v1
    python scripts/solo_vs_pooled.py --run evolve_all_science_v1 --seeds 42,43,44
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import swotxai  # noqa: F401
import numpy as np

from swotxai.config import load_config
from swotxai.presets import config_overrides

EVOLVE_ROOT = REPO_ROOT / "experiments" / "evolve"


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
    return best


def _candidate_record(run_dir: Path, cid: str) -> dict:
    with open(run_dir / "database.jsonl", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            if rec.get("candidate_id") == cid:
                return rec
    return {}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True)
    ap.add_argument("--candidate", default=None, help="default: best ok candidate")
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--regions", default=None, help="comma-separated subset")
    ap.add_argument("--seeds", default="42", help="comma-separated seeds (default: the run's 42)")
    args = ap.parse_args()

    run_dir = EVOLVE_ROOT / args.run
    meta = json.loads((run_dir / "data_meta.json").read_text(encoding="utf-8"))
    cid = args.candidate or _best_candidate(run_dir)
    cand_py = run_dir / "candidates" / cid / "candidate.py"
    run_params = json.loads((run_dir / "candidates" / cid / "params.json").read_text(encoding="utf-8"))
    mission = meta["mission"]
    regions = meta.get("regions") or [meta["region"]]
    if args.regions:                      # any preset region; lets a single-region run (e.g. calval uswc)
        regions = args.regions.split(",")   # be retrained on its sister regions of the same mission
    seeds = [int(s) for s in args.seeds.split(",")]

    pooled = (_candidate_record(run_dir, cid).get("metrics") or {}).get("per_region", {})
    base = load_config(args.config)
    solo_root = run_dir / "solo"

    from swotxai.evolve.harness import prepare_data, score_predictions

    results: dict[str, dict] = {}
    for rid in regions:
        rcfg = replace(
            base, **config_overrides(rid, mission),
            model="ann", features=list(meta["features"]), stencil_k=int(meta["stencil_k"]),
        )
        rdir = solo_root / rid
        print(f"\n=== {rid}: freezing region dataset ===", flush=True)
        data_path = prepare_data(rcfg, rdir, train_fraction=meta.get("train_fraction", 0.8))
        Y_test = np.load(data_path)["Y_test"]
        rmeta = json.loads((rdir / "data_meta.json").read_text(encoding="utf-8"))
        print(f"[{rid}] {rmeta['n_train']:,} train / {rmeta['n_test']:,} test rows", flush=True)

        per_seed = []
        for seed in seeds:
            sdir = rdir / f"seed{seed}"
            sdir.mkdir(parents=True, exist_ok=True)
            pred_path = sdir / "pred.npz"
            if not pred_path.exists():
                params = dict(run_params, seed=seed)
                (sdir / "params.json").write_text(json.dumps(params), encoding="utf-8")
                print(f"[{rid}] training seed {seed} "
                      f"(budget {params['time_budget_s']:.0f}s)...", flush=True)
                proc = subprocess.run(
                    [sys.executable, "-m", "swotxai.evolve.run_candidate",
                     str(cand_py), str(data_path), str(pred_path),
                     str(sdir / "meta.json"), str(sdir / "params.json")],
                    cwd=REPO_ROOT, timeout=run_params["time_budget_s"] * 2 + 600)
                run_meta = json.loads((sdir / "meta.json").read_text(encoding="utf-8"))
                if proc.returncode != 0 or run_meta.get("status") != "ok":
                    print(f"[{rid}] seed {seed} FAILED: {run_meta.get('error')}", flush=True)
                    continue
            pred = np.load(pred_path)["pred"]
            m = score_predictions(pred, Y_test)
            m["seed"] = seed
            per_seed.append(m)
            print(f"[{rid}] seed {seed}: r2_u {m['r2_u']:.4f}  r2_v {m['r2_v']:.4f}", flush=True)

        if per_seed:
            results[rid] = {
                "n_train": rmeta["n_train"], "n_test": rmeta["n_test"],
                "seeds": per_seed,
                "solo": {k: float(np.mean([s[k] for s in per_seed]))
                         for k in ("rmse_u", "r2_u", "rmse_v", "r2_v", "fitness")},
                "pooled": pooled.get(rid, {}),
            }

    out = solo_root / "comparison.json"
    out.write_text(json.dumps({
        "run": args.run, "candidate": cid, "seeds": seeds, "results": results,
    }, indent=2), encoding="utf-8")

    lines = [
        f"# Pooled vs region-solo — {args.run} champion `{cid}`",
        "",
        f"Same candidate code, same params (epoch cap {run_params['max_epochs']}, "
        f"budget {run_params['time_budget_s']:.0f}s), seeds {seeds}. Scored on each "
        "region's identical temporal holdout.",
        "",
        "| region | rows (train) | pooled r2_u | solo r2_u | pooled r2_v | solo r2_v | winner (mean r2) |",
        "|---|---|---|---|---|---|---|",
    ]
    for rid, r in results.items():
        p, s = r["pooled"], r["solo"]
        pm = (p.get("r2_u", np.nan) + p.get("r2_v", np.nan)) / 2
        sm = (s["r2_u"] + s["r2_v"]) / 2
        winner = "pooled" if pm >= sm else "solo"
        lines.append(
            f"| {rid} | {r['n_train']:,} | {p.get('r2_u', float('nan')):.3f} | {s['r2_u']:.3f} "
            f"| {p.get('r2_v', float('nan')):.3f} | {s['r2_v']:.3f} | {winner} ({pm:.3f} vs {sm:.3f}) |")
    md = solo_root / "comparison.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\nWrote {out}\n{md}\n")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    sys.exit(main())
