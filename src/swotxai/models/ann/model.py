from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn


def resolve_device(device: str = "auto") -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


ACTIVATIONS = {
    "silu": nn.SiLU,
    "relu": nn.ReLU,
    "gelu": nn.GELU,
    "tanh": nn.Tanh,
}


class MLP(nn.Module):
    """Plain MLP trunk with a 2-unit head predicting (u, v) jointly.

    Predicting both components with one network lets the hidden layers
    share the flow representation, unlike the RF backend which fits two
    independent forests."""

    def __init__(self, n_inputs: int, hidden: tuple[int, ...] = (256, 256, 128),
                 dropout: float = 0.1, activation: str = "silu"):
        super().__init__()
        act = ACTIVATIONS[activation.lower()]
        layers: list[nn.Module] = []
        d = n_inputs
        for h in hidden:
            layers += [nn.Linear(d, h), nn.LayerNorm(h), act(), nn.Dropout(dropout)]
            d = h
        layers.append(nn.Linear(d, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ANNRegressor:
    """Inference wrapper: standardization + NaN imputation + batched forward.

    Inputs are standardized with the training-set mean/scale and NaNs
    (stencil padding at swath edges) are imputed to 0 — i.e. the feature
    mean — so the network sees the same distribution at train and test
    time."""

    _swotxai_kind = "ann"

    def __init__(self, net: MLP, mean: np.ndarray, scale: np.ndarray, meta: dict):
        self.net = net
        self.mean = np.asarray(mean, dtype=np.float32)
        self.scale = np.asarray(scale, dtype=np.float32)
        self.meta = meta  # features, stencil_k, hidden, dropout, training history, ...

    @property
    def n_features_in_(self) -> int:
        return len(self.mean)

    def _prepare(self, X) -> np.ndarray:
        X = np.asarray(X, dtype=np.float32)
        X = (X - self.mean) / self.scale
        return np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    @torch.no_grad()
    def predict_uv(self, X, batch_size: int = 65536, device: str = "auto") -> np.ndarray:
        """Predict both components; returns array of shape (n, 2) = (u, v)."""
        dev = resolve_device(device)
        self.net.to(dev).eval()
        Xp = self._prepare(X)
        out = np.empty((len(Xp), 2), dtype=np.float32)
        for i in range(0, len(Xp), batch_size):
            xb = torch.from_numpy(Xp[i:i + batch_size]).to(dev)
            out[i:i + batch_size] = self.net(xb).cpu().numpy()
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.net.state_dict(),
            "mean":       self.mean,
            "scale":      self.scale,
            "meta":       self.meta,
        }, path)

    @classmethod
    def load(cls, path: str | Path) -> "ANNRegressor":
        ckpt = torch.load(path, map_location="cpu", weights_only=False)
        meta = ckpt["meta"]
        net = MLP(
            n_inputs=meta["n_inputs"],
            hidden=tuple(meta["hidden"]),
            dropout=meta["dropout"],
            activation=meta.get("activation", "silu"),
        )
        net.load_state_dict(ckpt["state_dict"])
        return cls(net, ckpt["mean"], ckpt["scale"], meta)


class ANNComponentView:
    """Single-component view onto a joint ANNRegressor.

    The pipeline's evaluate / inference code was written around a
    (model_u, model_v) pair; two views over the same network keep that
    interface without duplicating weights."""

    _swotxai_kind = "ann"

    def __init__(self, base: ANNRegressor, component: str):
        self.base = base
        self.component = component
        self._idx = {"u": 0, "v": 1}[component]

    @property
    def n_features_in_(self) -> int:
        return self.base.n_features_in_

    def predict(self, X) -> np.ndarray:
        return self.base.predict_uv(X)[:, self._idx]
