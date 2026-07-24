"""AlphaEvolve-style evolutionary search over the ANN training code.

Resumable CLI: re-running with the same --name picks up where it stopped
(completed candidates are skipped via the append-only database). Monitor a
run live in the app's Evolve tab.

Examples:
    # Free smoke test (no API calls) on the configured region's cached data:
    python scripts/evolve.py --config config.yaml --name test_mock --mock-llm \
        --generations 2 --children 2

    # Real run (prints a dollar estimate and asks for confirmation first):
    python scripts/evolve.py --config config.yaml --name evolve_uswc_v1 \
        --generations 10 --children 4 --budget-usd 10

All data reads are local (pipeline cache / local mirrors) — this script never
touches S3. The only paid resource is the Claude API, gated below.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

# Detached/redirected stdout defaults to the ANSI codepage on Windows, which
# cannot encode characters the pipeline prints (e.g. "→") — force UTF-8.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import swotxai  # noqa: F401  (torch-before-numpy DLL ordering)

from swotxai.config import load_config
from swotxai.evolve.database import EvolveDB
from swotxai.evolve.harness import prepare_data
from swotxai.evolve.loop import EvolveSettings, run_evolution
from swotxai.evolve.mutator import (
    DEFAULT_MODEL, PRICES, ClaudeMutator, CLIAgentMutator, MockMutator,
    estimate_cost_usd,
)

EVOLVE_ROOT = REPO_ROOT / "experiments" / "evolve"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default="config.yaml", help="pipeline config YAML (data domain)")
    p.add_argument("--name", required=True, help="evolve run name (dir under experiments/evolve/)")
    p.add_argument("--generations", type=int, default=10)
    p.add_argument("--children", type=int, default=4, help="candidates per generation")
    p.add_argument("--budget-usd", type=float, default=10.0, help="hard API spend cap")
    p.add_argument("--mutator", default="claude-api",
                   choices=["claude-api", "claude-cli", "codex-cli", "mock"],
                   help="who proposes candidates: claude-api (Anthropic API, per-token billing), "
                        "claude-cli (local `claude -p`, Claude subscription), "
                        "codex-cli (local `codex exec`, ChatGPT subscription), "
                        "mock (free hyperparameter jitter)")
    p.add_argument("--llm-model", default=DEFAULT_MODEL, choices=sorted(PRICES),
                   help="model for --mutator claude-api")
    p.add_argument("--cli-model", default=None,
                   help="model override for CLI mutators (claude-cli: passed as "
                        "`claude -p --model X`, e.g. 'fable' or 'opus'; "
                        "codex-cli: passed as `codex exec -m X`)")
    p.add_argument("--mock-llm", action="store_true", help="alias for --mutator mock")
    p.add_argument("--train-max-epochs", type=int, default=60,
                   help="epoch cap per candidate (screening speed; production uses 200)")
    p.add_argument("--timeout-min", type=float, default=30.0, help="per-candidate wall-clock limit")
    p.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--yes", action="store_true", help="skip the interactive cost confirmation")
    p.add_argument("--data-npz", default=None,
                   help="use a pre-built data.npz instead of running the pipeline (testing)")
    p.add_argument("--pooled", default=None, choices=["calval", "science"],
                   help="pool ALL trainable regions of a mission into one dataset "
                        "(per-region temporal splits; per-region metrics recorded)")
    p.add_argument("--wildcard-every", type=int, default=4,
                   help="every Nth candidate is a structural explorer (prompted to try a "
                        "fundamentally different architecture instead of refining; 0 = off)")
    p.add_argument("--region", default=None,
                   choices=["uswc", "usegc", "gak", "ushi", "glna", "prvi"],
                   help="use this region's preset (paths, bbox, cycles) instead of "
                        "config.yaml's region; requires --mission")
    p.add_argument("--mission", default=None, choices=["calval", "science"],
                   help="mission for --region (glna/prvi are science-only)")
    p.add_argument("--features", default=None,
                   help="comma-separated feature list overriding config.yaml "
                        "(e.g. 'mdt,ssha_filtered,ugos_filtered'); ignored if the run's "
                        "dataset is already frozen")
    p.add_argument("--stencil-k", type=int, default=None, choices=[1, 3, 5, 7],
                   help="stencil size overriding config.yaml; ignored if the run's "
                        "dataset is already frozen")
    return p.parse_args()


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _load_config_with_overrides(args):
    """Load config.yaml, then apply --region/--mission preset paths and
    --features / --stencil-k (re-validated by SWOTConfig)."""
    from dataclasses import replace

    config = load_config(args.config)
    if args.region:
        from swotxai.presets import MISSION_REGIONS, config_overrides

        mission = args.mission or config.mission
        if args.region not in MISSION_REGIONS.get(mission, []):
            raise SystemExit(
                f"--region {args.region} has no trainable HFR target for mission "
                f"'{mission}' (valid: {MISSION_REGIONS.get(mission)})")
        config = replace(config, **config_overrides(args.region, mission))
    overrides: dict = {"model": "ann"}
    if args.features:
        overrides["features"] = [f.strip() for f in args.features.split(",") if f.strip()]
    if args.stencil_k:
        overrides["stencil_k"] = args.stencil_k
    return replace(config, **overrides)


def main() -> int:
    args = parse_args()
    run_dir = EVOLVE_ROOT / args.name
    run_dir.mkdir(parents=True, exist_ok=True)
    db = EvolveDB(run_dir)

    mutator_kind = "mock" if args.mock_llm else args.mutator

    # ---- Cost gate (MANDATORY before any paid API call) ---------------------
    n_calls = args.generations * args.children
    if mutator_kind == "mock":
        log("Mock mutator: $0.00 API cost.")
    elif mutator_kind in ("claude-cli", "codex-cli"):
        log(f"{mutator_kind}: $0.00 per-token cost — runs on your existing "
            f"{'Claude' if mutator_kind == 'claude-cli' else 'ChatGPT'} subscription "
            f"(counts toward its usage limits; ~{n_calls} agent calls planned).")
    else:
        est = estimate_cost_usd(args.llm_model, n_calls)
        already = db.spent_usd
        print(
            f"\n=== API COST ESTIMATE ===\n"
            f"  model:            {args.llm_model}\n"
            f"  planned calls:    {n_calls} ({args.generations} generations x {args.children} children)\n"
            f"  est. per call:    ~9k input + ~5k output tokens\n"
            f"  ESTIMATED TOTAL:  ${est:.2f}"
            + (f"  (already spent in this run: ${already:.2f})" if already else "") + "\n"
            f"  hard budget cap:  ${args.budget_usd:.2f} (loop stops before exceeding it)\n"
            f"  compute:          local GPU only; no S3 access anywhere.\n"
        )
        if not args.yes:
            answer = input("Proceed with API spending? Type 'yes' to continue: ").strip().lower()
            if answer != "yes":
                print("Aborted — no API calls were made.")
                return 1

    # ---- Data ---------------------------------------------------------------
    if args.data_npz:
        data_npz = Path(args.data_npz)
        data_meta = {}
        meta_path = data_npz.parent / "data_meta.json"
        if meta_path.exists():
            data_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    elif args.pooled:
        from swotxai.evolve.harness import prepare_pooled_data
        config = _load_config_with_overrides(args)
        log(f"Preparing pooled {args.pooled} arrays (all trainable regions, cached/local)...")
        data_npz = prepare_pooled_data(config, args.pooled, run_dir)
        data_meta = json.loads((run_dir / "data_meta.json").read_text(encoding="utf-8"))
    else:
        config = _load_config_with_overrides(args)
        log("Preparing frozen train/test arrays (shared steps + flatten, all cached/local)...")
        data_npz = prepare_data(config, run_dir)
        data_meta = json.loads((run_dir / "data_meta.json").read_text(encoding="utf-8"))

    # A run's dataset is frozen at first launch — warn if overrides differ.
    if args.features or args.stencil_k:
        want_feats = ([f.strip() for f in args.features.split(",") if f.strip()]
                      if args.features else None)
        if want_feats and sorted(data_meta.get("features", [])) != sorted(want_feats):
            log(f"WARNING: run already frozen with features={data_meta.get('features')} "
                f"— requested {want_feats} ignored. Use a new --name to change them.")
        if args.stencil_k and data_meta.get("stencil_k") != args.stencil_k:
            log(f"WARNING: run already frozen with stencil_k={data_meta.get('stencil_k')} "
                f"— requested {args.stencil_k} ignored. Use a new --name to change it.")
    log(f"Data: {data_meta.get('n_train', '?')} train / {data_meta.get('n_test', '?')} test rows, "
        f"{data_meta.get('n_inputs', '?')} inputs ({data_npz})")

    # ---- Run ----------------------------------------------------------------
    if mutator_kind == "mock":
        mutator = MockMutator()
    elif mutator_kind == "claude-cli":
        cmd = ["claude", "-p"]
        if args.cli_model:
            cmd += ["--model", args.cli_model]
        mutator = CLIAgentMutator(cmd, label=f"claude-cli:{args.cli_model or 'default'}")
    elif mutator_kind == "codex-cli":
        cmd = ["codex", "exec", "--skip-git-repo-check"]
        if args.cli_model:
            cmd += ["-m", args.cli_model]
        mutator = CLIAgentMutator(
            cmd + ["-"], label=f"codex-cli:{args.cli_model or 'default'}",
            output_file_flag="--output-last-message",
        )
    else:
        mutator = ClaudeMutator(args.llm_model)
    settings = EvolveSettings(
        name=args.name, generations=args.generations, children=args.children,
        budget_usd=args.budget_usd,
        llm_model=(args.llm_model if mutator_kind == "claude-api" else mutator_kind),
        train_max_epochs=args.train_max_epochs, timeout_min=args.timeout_min,
        seed=args.seed, device=args.device, wildcard_every=args.wildcard_every,
        data_meta=data_meta,
    )
    best = run_evolution(db, data_npz, mutator, settings, log=log)

    if best is None:
        log("No successful candidates.")
        return 1
    log(f"DONE. Best: {best.candidate_id} fitness {best.fitness:.4f} "
        f"(r2_u {best.metrics['r2_u']:.4f}, r2_v {best.metrics['r2_v']:.4f}); "
        f"API spent ${db.spent_usd:.2f}.")
    log(f"Best code: {db.candidates_dir / best.candidate_id / 'candidate.py'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
