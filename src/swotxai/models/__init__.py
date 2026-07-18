"""Model backends for the SWOTxAI pipeline.

Each backend lives in its own fully partitioned package:

- ``swotxai.models.rf``  — random forest (sklearn / cuML / LightGBM)
- ``swotxai.models.ann`` — artificial neural network (PyTorch MLP)

A backend package exposes a ``steps`` module implementing the per-job ML
steps with a uniform interface:

    step_train(config, flattened, cb, use_cache) -> (model_u, model_v)
    step_evaluate(config, model_u, model_v, flattened, cb) -> metrics dict

``model_u`` / ``model_v`` are any objects usable with ``predict()`` below.
"""
from __future__ import annotations

import importlib

from swotxai.config import AVAILABLE_MODELS


def get_backend(model: str):
    """Return the ``steps`` module for the requested model backend."""
    if model not in AVAILABLE_MODELS:
        raise ValueError(f"Unknown model backend {model!r}. Available: {AVAILABLE_MODELS}")
    return importlib.import_module(f"swotxai.models.{model}.steps")


def predict(model, X):
    """Backend-agnostic prediction dispatch.

    ANN wrappers tag themselves with ``_swotxai_kind = "ann"`` so this check
    never imports torch on the RF path (and vice versa: the RF dispatcher
    only lazily imports cuML / LightGBM when those model types are present).
    """
    if getattr(model, "_swotxai_kind", None) == "ann":
        return model.predict(X)
    from swotxai.models.rf.training import predict as rf_predict
    return rf_predict(model, X)
