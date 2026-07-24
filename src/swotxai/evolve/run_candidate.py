"""Subprocess entry point: train one evolve candidate and write predictions.

Usage (invoked by sandbox.run_candidate, not by hand):
    python -m swotxai.evolve.run_candidate candidate.py data.npz pred.npz meta.json params.json

Importing swotxai first matters on this machine: the package __init__ handles
the torch-before-numpy DLL ordering. The candidate module is loaded from its
file path and must define train_and_predict(X_train, Y_train, X_test, params).
"""
from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import swotxai  # noqa: F401  (DLL ordering side effect)
import numpy as np


def _load_candidate(path: Path):
    spec = importlib.util.spec_from_file_location("evolve_candidate", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main() -> int:
    candidate_py, data_npz, pred_path, meta_path, params_path = (Path(p) for p in sys.argv[1:6])
    params = json.loads(params_path.read_text(encoding="utf-8"))
    meta = {"status": "failed", "error": "", "train_seconds": 0.0}

    try:
        data = np.load(data_npz)
        X_train, Y_train, X_test = data["X_train"], data["Y_train"], data["X_test"]

        module = _load_candidate(candidate_py)
        t0 = time.time()
        pred = module.train_and_predict(X_train, Y_train, X_test, params)
        meta["train_seconds"] = round(time.time() - t0, 1)

        pred = np.asarray(pred, dtype=np.float32)
        if pred.shape != (len(X_test), 2):
            raise ValueError(f"predictions have shape {pred.shape}, expected {(len(X_test), 2)}")
        if not np.isfinite(pred).all():
            raise ValueError("predictions contain non-finite values")

        np.savez_compressed(pred_path, pred=pred)
        meta["status"] = "ok"
    except Exception as exc:  # noqa: BLE001 — any candidate failure is data
        meta["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

    return 0 if meta["status"] == "ok" else 1


if __name__ == "__main__":
    sys.exit(main())
