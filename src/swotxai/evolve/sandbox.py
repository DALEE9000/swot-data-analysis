"""Static safety checks and subprocess execution for evolve candidates.

Two containment layers, honoring the repo's hard no-S3 rule:

1. Static AST screen — candidate code may only import a whitelist of numeric
   libraries; file/network/OS modules and dangerous builtins are rejected
   before anything runs.
2. Subprocess isolation — candidates train in a child process with cloud
   credentials stripped from the environment and a hard wall-clock timeout,
   so a hung or crashed candidate cannot take down the evolve loop (or reach
   S3 even if the static screen were bypassed).
"""
from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ALLOWED_IMPORTS = {
    "numpy", "torch", "math", "random", "time", "copy", "typing",
    "dataclasses", "collections", "itertools", "functools", "warnings",
    "sklearn", "scipy",
}

FORBIDDEN_BUILTINS = {
    "open", "exec", "eval", "compile", "__import__", "input", "breakpoint",
    "globals", "vars", "memoryview",
}

_SECRET_ENV_MARKERS = ("TOKEN", "SECRET", "PASSWORD", "CREDENTIAL", "API_KEY")
_SECRET_ENV_PREFIXES = ("AWS_", "ANTHROPIC_", "OPENAI_")


def code_violations(code: str) -> list[str]:
    """Return a list of rule violations; empty list means the code passes."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e}"]

    violations = []
    has_contract = False
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                root = alias.name.split(".")[0]
                if root not in ALLOWED_IMPORTS:
                    violations.append(f"forbidden import: {alias.name}")
        elif isinstance(node, ast.ImportFrom):
            root = (node.module or "").split(".")[0]
            if node.level == 0 and root not in ALLOWED_IMPORTS:
                violations.append(f"forbidden import: from {node.module}")
            elif node.level > 0:
                violations.append("forbidden relative import")
        elif isinstance(node, ast.Name) and node.id in FORBIDDEN_BUILTINS:
            violations.append(f"forbidden builtin: {node.id}")
        elif isinstance(node, ast.Attribute) and node.attr in ("__globals__", "__builtins__", "__subclasses__"):
            violations.append(f"forbidden attribute access: {node.attr}")
        elif isinstance(node, ast.FunctionDef) and node.name == "train_and_predict":
            has_contract = True

    if not has_contract:
        violations.append("missing required function train_and_predict(X_train, Y_train, X_test, params)")
    return violations


def sandbox_env() -> dict:
    """Copy of os.environ with cloud credentials and secrets stripped."""
    env = {}
    for k, v in os.environ.items():
        upper = k.upper()
        if upper.startswith(_SECRET_ENV_PREFIXES):
            continue
        if any(m in upper for m in _SECRET_ENV_MARKERS):
            continue
        env[k] = v
    # Torch/conda DLL quirk on this machine: torch must tolerate duplicate OpenMP.
    env["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # Candidate stdout is redirected to a log file; force UTF-8 so a unicode
    # print inside evolved code can't crash the run (Windows defaults to cp1252).
    env["PYTHONIOENCODING"] = "utf-8"
    # Make swotxai importable in the child even without an editable install.
    src_root = str(Path(__file__).resolve().parents[2])
    env["PYTHONPATH"] = src_root + os.pathsep + env.get("PYTHONPATH", "") if env.get("PYTHONPATH") else src_root
    return env


def run_candidate(
    candidate_py: Path,
    data_npz: Path,
    out_dir: Path,
    params: dict,
    timeout_s: float,
) -> dict:
    """Execute a candidate in a subprocess; return a result dict.

    Result keys: status ("ok"|"failed"|"timeout"), error, train_seconds,
    predictions_path (npz with array "pred", present when status == "ok").
    """
    # Absolute paths throughout: the child runs with cwd=out_dir, so relative
    # inputs (e.g. experiments/evolve/...) would otherwise resolve wrongly.
    candidate_py = Path(candidate_py).resolve()
    data_npz = Path(data_npz).resolve()
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_path = out_dir / "predictions.npz"
    meta_path = out_dir / "run_meta.json"
    params_path = out_dir / "params.json"
    log_path = out_dir / "stdout.log"
    params_path.write_text(json.dumps(params), encoding="utf-8")
    for stale in (pred_path, meta_path):
        stale.unlink(missing_ok=True)

    cmd = [
        sys.executable, "-m", "swotxai.evolve.run_candidate",
        str(candidate_py), str(data_npz), str(pred_path), str(meta_path), str(params_path),
    ]
    t0 = time.time()
    try:
        with open(log_path, "w", encoding="utf-8") as log:
            proc = subprocess.run(
                cmd, stdout=log, stderr=subprocess.STDOUT,
                env=sandbox_env(), timeout=timeout_s,
                cwd=str(out_dir),
            )
    except subprocess.TimeoutExpired:
        return {"status": "timeout", "error": f"exceeded {timeout_s:.0f}s wall-clock limit",
                "train_seconds": time.time() - t0}

    elapsed = time.time() - t0
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    else:
        meta = {}

    if proc.returncode != 0 or meta.get("status") != "ok":
        tail = ""
        try:
            tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        except OSError:
            pass
        return {"status": "failed",
                "error": meta.get("error") or f"exit code {proc.returncode}\n{tail}".strip(),
                "train_seconds": meta.get("train_seconds", elapsed)}

    return {"status": "ok", "error": "",
            "train_seconds": meta.get("train_seconds", elapsed),
            "predictions_path": str(pred_path)}
