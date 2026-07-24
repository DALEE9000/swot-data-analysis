"""The evolution controller: seed -> propose -> screen -> train -> score -> select.

Designed for long unattended runs on this machine (single GPU, may hibernate):
candidates run sequentially, every result is appended to the JSONL database
immediately, and re-running with the same --name resumes where it stopped.
The API budget is a hard cap checked before every LLM call.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from swotxai.evolve.database import CandidateRecord, EvolveDB
from swotxai.evolve.harness import diagnostics, score_predictions
from swotxai.evolve.mutator import estimate_cost_usd
from swotxai.evolve.sandbox import code_violations, run_candidate


@dataclass
class EvolveSettings:
    name: str
    generations: int = 10
    children: int = 4
    budget_usd: float = 10.0
    llm_model: str = "claude-opus-4-8"
    train_max_epochs: int = 60
    timeout_min: float = 30.0
    seed: int = 42
    device: str = "cuda"
    parent_top_k: int = 3
    explore_prob: float = 0.2   # chance of picking a random (non-top) parent
    wildcard_every: int = 4     # every Nth candidate is a structural explorer (0 = off)
    data_meta: dict = field(default_factory=dict)


def _select_parent(db: EvolveDB, settings: EvolveSettings, rng: random.Random) -> CandidateRecord:
    ok = db.successful()
    top = db.top_k(settings.parent_top_k)
    explorers = [r for r in ok if r not in top]
    if explorers and rng.random() < settings.explore_prob:
        return rng.choice(explorers)
    # Weight top-k by rank (best gets highest weight).
    weights = [settings.parent_top_k - i for i in range(len(top))]
    return rng.choices(top, weights=weights, k=1)[0]


def _select_inspiration(db: EvolveDB, parent: CandidateRecord) -> CandidateRecord | None:
    for r in db.top_k(3):
        if r.candidate_id != parent.candidate_id:
            return r
    return None


def evaluate_candidate(db: EvolveDB, rec: CandidateRecord, code: str,
                       data_npz: Path, Y_test: np.ndarray,
                       settings: EvolveSettings, log,
                       X_test: np.ndarray | None = None) -> CandidateRecord:
    """Write, screen, run, and score one candidate; appends to the DB."""
    cand_dir = db.candidate_dir(rec.candidate_id)
    (cand_dir / "candidate.py").write_text(code, encoding="utf-8")

    violations = code_violations(code)
    if violations:
        rec.status = "rejected"
        rec.error = "; ".join(violations[:10])
        db.append(rec)
        log(f"  {rec.candidate_id}: REJECTED ({rec.error[:120]})")
        return rec

    params = {"seed": settings.seed, "device": settings.device,
              "max_epochs": settings.train_max_epochs,
              "time_budget_s": settings.timeout_min * 60 * 0.9}
    result = run_candidate(cand_dir / "candidate.py", data_npz, cand_dir,
                           params, timeout_s=settings.timeout_min * 60)
    rec.train_seconds = round(result.get("train_seconds", 0.0), 1)

    if result["status"] != "ok":
        rec.status = result["status"]
        rec.error = result["error"][:2000]
        db.append(rec)
        log(f"  {rec.candidate_id}: {rec.status.upper()} ({rec.error.splitlines()[0][:120] if rec.error else ''})")
        return rec

    pred = np.load(result["predictions_path"])["pred"]
    rec.metrics = score_predictions(pred, Y_test)
    # Diagnostic bins: fed back into the next mutation prompt so the proposer
    # can target the weakest regimes instead of optimizing a blind scalar.
    if X_test is not None:
        try:
            rec.metrics["diagnostics"] = diagnostics(pred, Y_test, X_test)
        except Exception:  # noqa: BLE001 — diagnostics are advisory, never fatal
            pass
    # Pooled runs: per-region breakdown (fitness stays the pooled score).
    # region_test_slices values are lists of [start, end) spans (regions are
    # time-interleaved in chunks); a legacy flat [start, end] pair also works.
    slices = (settings.data_meta or {}).get("region_test_slices") or {}
    if slices:
        per_region = {}
        for rid, spans in slices.items():
            if spans and not isinstance(spans[0], (list, tuple)):
                spans = [spans]
            idx = np.concatenate([np.arange(s, e) for s, e in spans])
            per_region[rid] = {k: v for k, v in
                               score_predictions(pred[idx], Y_test[idx]).items()
                               if k != "fitness"}
        rec.metrics["per_region"] = per_region
    rec.fitness = rec.metrics["fitness"]
    rec.status = "ok"
    db.append(rec)
    return rec


def run_evolution(db: EvolveDB, data_npz: Path, mutator, settings: EvolveSettings,
                  log=print) -> CandidateRecord | None:
    """Run (or resume) the evolutionary search. Returns the best candidate."""
    rng = random.Random(settings.seed)
    _data = np.load(data_npz)
    Y_test, X_test = _data["Y_test"], _data["X_test"]
    del _data
    t_start = time.time()

    db.save_state(running=True, settings={
        "generations": settings.generations, "children": settings.children,
        "budget_usd": settings.budget_usd, "llm_model": settings.llm_model,
        "train_max_epochs": settings.train_max_epochs,
        "timeout_min": settings.timeout_min, "seed": settings.seed,
    }, data_meta=settings.data_meta)

    try:
        # --- Generation 0: the seed (current production ANN code). -----------
        if not any(r.candidate_id == "gen000_00" for r in db.records):
            seed_code = (Path(__file__).parent / "seed_candidate.py").read_text(encoding="utf-8")
            rec = CandidateRecord(candidate_id="gen000_00", generation=0, parent_id=None,
                                  idea="Seed: port of the production MLP + train_ann.")
            log("gen000_00: evaluating seed candidate...")
            rec = evaluate_candidate(db, rec, seed_code, data_npz, Y_test, settings, log,
                                     X_test=X_test)
            if rec.ok:
                log(f"  seed fitness {rec.fitness:.4f} "
                    f"(r2_u {rec.metrics['r2_u']:.4f}, r2_v {rec.metrics['r2_v']:.4f}, "
                    f"{rec.train_seconds:.0f}s)")
            else:
                log("Seed candidate failed — aborting (fix the environment before evolving).")
                return None

        if not db.successful():
            log("No successful ancestor to evolve from — aborting.")
            return None

        done = {r.candidate_id for r in db.records}
        n_total = settings.generations * settings.children
        n_done_evolved = len([r for r in db.records if r.generation > 0])

        for gen in range(1, settings.generations + 1):
            for child in range(settings.children):
                cid = f"gen{gen:03d}_{child:02d}"
                if cid in done:
                    continue

                # Budget gate BEFORE each API call (hard cap; only paid
                # mutators — CLI/subscription and mock proposers cost $0).
                projected = db.spent_usd + estimate_cost_usd(settings.llm_model, 1)
                if getattr(mutator, "paid", False) and projected > settings.budget_usd:
                    log(f"Budget cap reached (${db.spent_usd:.2f} spent, "
                        f"${settings.budget_usd:.2f} cap) — stopping.")
                    return db.best()

                # Wildcard slots: deterministic by candidate index, so resume
                # keeps the same schedule.
                idx = (gen - 1) * settings.children + child
                wildcard = (settings.wildcard_every > 0
                            and idx % settings.wildcard_every == settings.wildcard_every - 1)

                parent = db.best() if wildcard else _select_parent(db, settings, rng)
                inspiration = _select_inspiration(db, parent)
                history = db.top_k(5) + [r for r in db.records[-5:] if r not in db.top_k(5)]

                try:
                    idea, code, usage = mutator.propose(
                        parent, db.code_of(parent.candidate_id), history,
                        inspiration,
                        db.code_of(inspiration.candidate_id) if inspiration else None,
                        settings.data_meta,
                        wildcard=wildcard,
                    )
                except Exception as exc:  # noqa: BLE001 — API/parse failure is a data point
                    rec = CandidateRecord(candidate_id=cid, generation=gen,
                                          parent_id=parent.candidate_id,
                                          status="rejected", error=f"proposer failed: {exc}"[:500],
                                          llm=getattr(exc, "usage", {}) or {})
                    db.append(rec)
                    log(f"  {cid}: proposer failed ({exc})")
                    continue

                if wildcard:
                    usage = {**usage, "wildcard": True}
                rec = CandidateRecord(
                    candidate_id=cid, generation=gen, parent_id=parent.candidate_id,
                    inspiration_ids=[inspiration.candidate_id] if inspiration else [],
                    idea=("[wildcard] " + idea) if wildcard else idea, llm=usage,
                )
                rec = evaluate_candidate(db, rec, code, data_npz, Y_test, settings, log,
                                         X_test=X_test)

                n_done_evolved += 1
                best = db.best()
                elapsed = time.time() - t_start
                rate = elapsed / max(n_done_evolved, 1)
                eta_min = rate * (n_total - n_done_evolved) / 60
                fit = f"{rec.fitness:.4f}" if rec.fitness is not None else rec.status
                log(f"  {cid}: fitness {fit} | best {best.fitness:.4f} ({best.candidate_id}) "
                    f"| spent ${db.spent_usd:.2f} | ETA {eta_min:.0f} min")
                db.save_state(spent_usd=round(db.spent_usd, 4),
                              best_id=best.candidate_id, best_fitness=best.fitness,
                              n_candidates=len(db.records))

        return db.best()
    finally:
        best = db.best()
        db.save_state(running=False, spent_usd=round(db.spent_usd, 4),
                      best_id=best.candidate_id if best else None,
                      best_fitness=best.fitness if best else None,
                      n_candidates=len(db.records))
