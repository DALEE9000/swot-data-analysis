"""Append-only candidate database for an evolve run.

One evolve run = one directory under experiments/evolve/{name}/ containing:
    database.jsonl   one record per candidate (append-only -> resumable)
    state.json       run settings, budget spent, running flag
    data.npz         the frozen train/test arrays for this run
    candidates/      gen{g:03d}_{i:02d}/candidate.py + stdout.log + result files
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class CandidateRecord:
    candidate_id: str            # e.g. "gen003_01"
    generation: int
    parent_id: str | None        # None for the seed
    inspiration_ids: list[str] = field(default_factory=list)
    status: str = "pending"      # ok | failed | timeout | rejected
    idea: str = ""               # LLM's one-paragraph description of the change
    metrics: dict = field(default_factory=dict)   # r2_u, r2_v, rmse_u, rmse_v, fitness
    fitness: float | None = None
    train_seconds: float | None = None
    error: str = ""
    llm: dict = field(default_factory=dict)       # model, input_tokens, output_tokens, cost_usd
    recorded_at: str = ""

    @property
    def ok(self) -> bool:
        return self.status == "ok" and self.fitness is not None


class EvolveDB:
    def __init__(self, run_dir: str | Path):
        self.run_dir = Path(run_dir)
        self.db_path = self.run_dir / "database.jsonl"
        self.state_path = self.run_dir / "state.json"
        self.candidates_dir = self.run_dir / "candidates"
        self.candidates_dir.mkdir(parents=True, exist_ok=True)
        self.records: list[CandidateRecord] = self._load()

    def _load(self) -> list[CandidateRecord]:
        records = []
        if self.db_path.exists():
            with open(self.db_path, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    d = json.loads(line)
                    d.pop("ok", None)
                    records.append(CandidateRecord(**d))
        return records

    def append(self, rec: CandidateRecord) -> None:
        rec.recorded_at = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.records.append(rec)
        with open(self.db_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(rec)) + "\n")

    def candidate_dir(self, candidate_id: str) -> Path:
        d = self.candidates_dir / candidate_id
        d.mkdir(parents=True, exist_ok=True)
        return d

    def code_of(self, candidate_id: str) -> str:
        return (self.candidates_dir / candidate_id / "candidate.py").read_text(encoding="utf-8")

    # -- selection helpers ------------------------------------------------

    def successful(self) -> list[CandidateRecord]:
        return [r for r in self.records if r.ok]

    def best(self) -> CandidateRecord | None:
        ok = self.successful()
        return max(ok, key=lambda r: r.fitness) if ok else None

    def top_k(self, k: int) -> list[CandidateRecord]:
        return sorted(self.successful(), key=lambda r: r.fitness, reverse=True)[:k]

    # -- state -------------------------------------------------------------

    def load_state(self) -> dict:
        if self.state_path.exists():
            return json.loads(self.state_path.read_text(encoding="utf-8"))
        return {}

    def save_state(self, **updates) -> dict:
        state = self.load_state()
        state.update(updates)
        state["updated_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        tmp = self.state_path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(state, indent=2), encoding="utf-8")
        tmp.replace(self.state_path)
        return state

    @property
    def spent_usd(self) -> float:
        return sum(r.llm.get("cost_usd", 0.0) for r in self.records)
