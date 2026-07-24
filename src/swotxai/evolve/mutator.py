"""Candidate proposers: the Claude API mutator and a free mock mutator.

The LLM sees the parent candidate's full code + metrics, a leaderboard of
prior attempts, and one inspiration candidate, and returns a complete
replacement module. Full-module rewrites (rather than diffs) are deliberate:
candidates are ~150 lines, and rewrites avoid diff-application bugs.

Cost control: every call's token usage is converted to dollars via PRICES and
returned to the loop, which enforces the run budget before each call.
"""
from __future__ import annotations

import random
import re
import textwrap

# USD per million tokens (input, output). Used for pre-run estimates and
# per-call cost accounting; keep in sync with platform pricing.
PRICES = {
    "claude-opus-4-8": (5.00, 25.00),
    "claude-sonnet-5": (3.00, 15.00),
    "claude-haiku-4-5": (1.00, 5.00),
}
DEFAULT_MODEL = "claude-opus-4-8"

# Conservative per-call token estimates for the upfront cost gate.
EST_INPUT_TOKENS = 9_000
EST_OUTPUT_TOKENS = 5_000


def estimate_cost_usd(model: str, n_calls: int) -> float:
    price_in, price_out = PRICES.get(model, PRICES[DEFAULT_MODEL])
    per_call = EST_INPUT_TOKENS / 1e6 * price_in + EST_OUTPUT_TOKENS / 1e6 * price_out
    return per_call * n_calls


SYSTEM_PROMPT = textwrap.dedent("""\
    You are an expert ML researcher evolving a PyTorch training module for an
    oceanographic regression task, in the style of AlphaEvolve: you receive a
    parent program and its measured score, and you propose ONE improved
    variant of the full module.

    ## The task the module solves
    Predict subsurface ocean velocity (u, v) from SWOT satellite altimetry
    features, validated against HFR radar ground truth.
    - X_train: float32 (n_train, d). Columns are k*k spatial-stencil copies of
      base features (sea-surface height anomaly, geostrophic velocities, MDT,
      optionally ERA5 winds and SST), feature-major. NaNs mark stencil padding
      at swath edges.
    - Y_train: float32 (n_train, 2) = (u, v) in m/s; NaN marks an invalid
      component (a row may have only one valid target).
    - The test split is TEMPORALLY held out (later cycles), so generalization
      across time is what scores well — memorization does not.
    - Score = mean of R^2(u) and R^2(v) on the held-out split, computed by the
      harness from your returned predictions. Higher is better.

    ## Hard constraints (violations are rejected without being run)
    - Define exactly: train_and_predict(X_train, Y_train, X_test, params) ->
      np.ndarray of shape (len(X_test), 2), finite everywhere.
      params keys: seed (int), device ("cuda"/"cpu"), max_epochs (int),
      time_budget_s (float soft wall-clock budget — check it in training loops).
    - Imports ONLY from: numpy, torch, math, random, time, copy, typing,
      dataclasses, collections, itertools, functools, warnings, sklearn, scipy.
    - No file I/O, no network, no os/sys/subprocess, no open/exec/eval.
    - Hardware: one 4 GB GPU (GTX 1650 Ti), 64 GB RAM. Keep the full dataset on
      CPU and move batches to the device; respect max_epochs and time_budget_s.
    - Seed all randomness from params["seed"].

    ## What to change
    Make one focused, well-motivated change per generation (architecture,
    loss, feature handling, optimization schedule, ensembling within budget,
    NaN treatment, target weighting, ...). Avoid kitchen-sink rewrites; build
    on what measurably worked in the history you are shown.

    IMPROVE THE ASSIGNED PARENT. The parent shown below was chosen by the
    selection strategy to maintain lineage diversity — do NOT swap it for a
    different leaderboard program, even a higher-scoring one, unless the task
    explicitly says so. Graft proven ideas from the inspiration INTO the
    parent instead.

    ## Using the diagnostics
    When a diagnostic breakdown is provided (error by forecast horizon, flow
    regime, and stencil validity), target the weakest bins: a late-test
    deficit suggests drift/recency handling; a fast-regime deficit suggests
    front/eddy dynamics the features miss; an edge-stencil deficit suggests
    missingness handling. Say in your <idea> which weakness you are attacking.

    ## Output format (exactly this, nothing else)
    <idea>One short paragraph: what you changed and why it should raise held-out R^2.</idea>
    ```python
    # the complete replacement module, including the contract docstring
    ```
""")


def _history_table(records) -> str:
    if not records:
        return "(no prior candidates)"
    lines = ["| id | gen | parent | fitness | r2_u | r2_v | idea |",
             "|----|-----|--------|---------|------|------|------|"]
    for r in records:
        idea = (r.idea or "").replace("\n", " ")[:120]
        fit = f"{r.fitness:.4f}" if r.fitness is not None else r.status
        m = r.metrics or {}
        lines.append(f"| {r.candidate_id} | {r.generation} | {r.parent_id or '-'} | {fit} "
                     f"| {m.get('r2_u', float('nan')):.4f} | {m.get('r2_v', float('nan')):.4f} | {idea} |")
    return "\n".join(lines)


WILDCARD_INSTRUCTION = textwrap.dedent("""\
    ## WILDCARD GENERATION — explore, don't refine
    This slot is reserved for structural diversity. Do NOT refine the parent
    or the leaderboard champion. Propose a STRUCTURALLY DIFFERENT approach
    that the lineage has not tried — examples of admissible directions (pick
    one, or invent your own): reshape each base feature's k*k stencil block
    into a tiny 2D image and use a small CNN; a joint heteroscedastic head
    predicting per-row uncertainty and weighting the loss by it; gated
    mixture-of-experts over flow regimes; a wide-and-deep split (linear path
    + deep path trained jointly end-to-end); target decomposition into speed
    and direction. Reuse the parent's proven data plumbing (feature
    augmentation, temporal validation, ridge residualization) freely, but the
    core predictive structure must be new. A lower score than the champion is
    acceptable — this slot buys information, not rank.""")


def _diagnostics_block(parent) -> str:
    diag = (parent.metrics or {}).get("diagnostics") or {}
    if not diag:
        return ""
    lines = [f"  {k}: {v}" for k, v in diag.items()]
    return "## Parent diagnostic breakdown (held-out bins)\n" + "\n".join(lines)


def build_user_prompt(parent, parent_code: str, history, inspiration=None,
                      inspiration_code: str | None = None, data_meta: dict | None = None,
                      wildcard: bool = False) -> str:
    parts = []
    if data_meta:
        parts.append(f"## Dataset\n{data_meta}")
    parts.append(f"## Leaderboard (best and recent attempts)\n{_history_table(history)}")
    if wildcard:
        parts.append(WILDCARD_INSTRUCTION)
    if inspiration is not None and inspiration_code and not wildcard:
        parts.append(
            f"## Inspiration: {inspiration.candidate_id} "
            f"(fitness {inspiration.fitness:.4f})\nIdea: {inspiration.idea}\n"
            f"```python\n{inspiration_code}\n```"
        )
    m = parent.metrics or {}
    role = "Reference program (reuse its plumbing, replace its predictive core)" if wildcard \
        else "Parent to improve"
    parts.append(
        f"## {role}: {parent.candidate_id} (fitness {parent.fitness:.4f}, "
        f"r2_u {m.get('r2_u', float('nan')):.4f}, r2_v {m.get('r2_v', float('nan')):.4f}, "
        f"train {parent.train_seconds or 0:.0f}s)\n"
        f"```python\n{parent_code}\n```"
    )
    diag = _diagnostics_block(parent)
    if diag:
        parts.append(diag)
    parts.append("Propose the next candidate. Remember: <idea>...</idea> then one ```python block.")
    return "\n\n".join(parts)


def parse_response(text: str) -> tuple[str, str]:
    """Extract (idea, code) from a model response; raises ValueError if absent."""
    idea_m = re.search(r"<idea>(.*?)</idea>", text, re.DOTALL)
    idea = idea_m.group(1).strip() if idea_m else ""
    code_blocks = re.findall(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if not code_blocks:
        raise ValueError("no ```python code block in response")
    return idea, code_blocks[-1].strip() + "\n"


class ClaudeMutator:
    """Proposes candidates via the Claude API. Requires ANTHROPIC_API_KEY
    (or an `ant auth login` profile) in the parent process only — the
    credential is stripped from candidate subprocess environments."""

    paid = True  # real dollars per call -> budget-gated by the loop

    def __init__(self, model: str = DEFAULT_MODEL):
        import anthropic

        self.model = model
        self.client = anthropic.Anthropic()

    def propose(self, parent, parent_code, history, inspiration=None,
                inspiration_code=None, data_meta=None, wildcard=False) -> tuple[str, str, dict]:
        """Returns (idea, code, usage) — usage has token counts + cost_usd."""
        user_prompt = build_user_prompt(parent, parent_code, history,
                                        inspiration, inspiration_code, data_meta,
                                        wildcard=wildcard)
        with self.client.messages.stream(
            model=self.model,
            max_tokens=16000,
            thinking={"type": "adaptive"},
            system=[{"type": "text", "text": SYSTEM_PROMPT,
                     "cache_control": {"type": "ephemeral"}}],
            messages=[{"role": "user", "content": user_prompt}],
        ) as stream:
            response = stream.get_final_message()

        text = "".join(b.text for b in response.content if b.type == "text")
        usage = self._usage(response)
        if response.stop_reason == "refusal":
            raise RuntimeError("model declined the request (stop_reason=refusal)")
        idea, code = parse_response(text)
        return idea, code, usage

    def _usage(self, response) -> dict:
        u = response.usage
        price_in, price_out = PRICES.get(self.model, PRICES[DEFAULT_MODEL])
        input_total = (u.input_tokens
                       + (u.cache_creation_input_tokens or 0)
                       + (u.cache_read_input_tokens or 0))
        # Approximation: cache writes bill ~1.25x, reads ~0.1x; close enough
        # for budget enforcement, and always >= the read-discounted truth.
        cost = ((u.input_tokens + 1.25 * (u.cache_creation_input_tokens or 0)
                 + 0.1 * (u.cache_read_input_tokens or 0)) / 1e6 * price_in
                + u.output_tokens / 1e6 * price_out)
        return {"model": self.model, "input_tokens": int(input_total),
                "output_tokens": int(u.output_tokens), "cost_usd": round(cost, 4)}


class CLIAgentMutator:
    """Agent-agnostic proposer: pipes the prompt to a local CLI coding agent
    and parses its reply. Runs on the agent's own auth/subscription — no API
    credits required. Works with any command that reads a prompt on stdin and
    prints (or writes to a file) a plain-text response.

    Presets live in scripts/evolve.py:
      claude-cli : ["claude", "-p"]                       (Claude subscription)
      codex-cli  : ["codex", "exec", "--skip-git-repo-check", "-"]
                   + --output-last-message <file>          (ChatGPT subscription)
    """

    paid = False  # subscription-metered, not per-token billed

    def __init__(self, command: list[str], label: str,
                 output_file_flag: str | None = None, timeout_s: float = 900.0):
        import shutil

        self.model = label
        self.output_file_flag = output_file_flag
        self.timeout_s = timeout_s
        exe = shutil.which(command[0])
        if exe is None:
            raise RuntimeError(
                f"'{command[0]}' not found on PATH — install the CLI (and log in) first."
            )
        self.command = [exe] + list(command[1:])

    def _invoke(self, prompt: str) -> str:
        import subprocess
        import tempfile
        from pathlib import Path

        cmd = list(self.command)
        out_file = None
        if self.output_file_flag:
            fd, tmp = tempfile.mkstemp(suffix=".md", prefix="evolve_llm_")
            import os as _os
            _os.close(fd)
            out_file = Path(tmp)
            # Keep a trailing "-" (read-prompt-from-stdin positional) last.
            insert_at = len(cmd) - 1 if cmd and cmd[-1] == "-" else len(cmd)
            cmd[insert_at:insert_at] = [self.output_file_flag, str(out_file)]
        try:
            proc = subprocess.run(
                cmd, input=prompt, capture_output=True, text=True,
                encoding="utf-8", errors="replace", timeout=self.timeout_s,
            )
            if proc.returncode != 0:
                raise RuntimeError(
                    f"{self.model} exited {proc.returncode}: {(proc.stderr or proc.stdout)[-800:]}"
                )
            if out_file is not None:
                return out_file.read_text(encoding="utf-8", errors="replace")
            return proc.stdout
        finally:
            if out_file is not None:
                out_file.unlink(missing_ok=True)

    def propose(self, parent, parent_code, history, inspiration=None,
                inspiration_code=None, data_meta=None, wildcard=False) -> tuple[str, str, dict]:
        """Returns (idea, code, usage); usage carries the label and $0 cost."""
        prompt = SYSTEM_PROMPT + "\n\n" + build_user_prompt(
            parent, parent_code, history, inspiration, inspiration_code, data_meta,
            wildcard=wildcard,
        )
        last_err = None
        for _attempt in range(2):
            text = self._invoke(prompt)
            try:
                idea, code = parse_response(text)
                return idea, code, {"model": self.model, "input_tokens": 0,
                                    "output_tokens": 0, "cost_usd": 0.0}
            except ValueError as exc:
                last_err = exc
                prompt += ("\n\nREMINDER: your previous reply was unparseable. Output "
                           "EXACTLY <idea>...</idea> followed by ONE ```python code block "
                           "containing the complete module, and nothing else.")
        raise ValueError(f"{self.model} produced no parseable candidate: {last_err}")


class MockMutator:
    """Free offline proposer for smoke tests: jitters seed hyperparameters."""

    model = "mock"
    paid = False

    def __init__(self, rng: random.Random | None = None):
        self.rng = rng or random.Random(0)

    def propose(self, parent, parent_code, history, inspiration=None,
                inspiration_code=None, data_meta=None, wildcard=False) -> tuple[str, str, dict]:
        hidden = self.rng.choice(["(128, 128)", "(256, 256, 128)", "(512, 256)", "(256, 128, 64)"])
        dropout = self.rng.choice(["0.0", "0.05", "0.1", "0.2"])
        lr = self.rng.choice(["3e-4", "1e-3", "3e-3"])
        code = parent_code
        code = re.sub(r"hidden=\([\d, ]+\)", f"hidden={hidden}", code)
        code = re.sub(r"dropout=[\d.]+(?=\))", f"dropout={dropout}", code)
        code = re.sub(r"lr=[\d.e-]+", f"lr={lr}", code)
        idea = f"[mock] hidden={hidden}, dropout={dropout}, lr={lr}"
        return idea, code, {"model": "mock", "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}
