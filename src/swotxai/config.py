from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

# All SWOT L3 variables available for use as model input features
AVAILABLE_FEATURES = [
    "mdt",
    "ssha_filtered",
    "ugos_filtered",
    "vgos_filtered",
    "ugosa_filtered",
    "vgosa_filtered",
    "era5_u",
    "era5_v",
    "SST",
]

# Default feature set matching the notebook
DEFAULT_FEATURES = [
    "mdt",
    "ssha_filtered",
    "ugos_filtered",
    "vgos_filtered",
    "ugosa_filtered",
    "vgosa_filtered",
    "era5_u",
    "era5_v",
    "SST",
]

# Model backends
AVAILABLE_MODELS = ("rf", "ann")

# Artifact names that belong to a specific model backend and are therefore
# routed into a per-model subdirectory by cache_path().
_MODEL_ARTIFACTS = {"inference", "rf_u", "rf_v", "rf_meta", "ann_model"}

# Activation functions available for the ANN backend
ANN_ACTIVATIONS = ("silu", "relu", "gelu", "tanh")

# Legacy config.yaml keys → new partitioned names (pre-ANN configs)
_LEGACY_KEYS = {
    "n_estimators":   "rf_n_estimators",
    "max_depth":      "rf_max_depth",
    "use_gpu":        "rf_use_gpu",
    "use_lgbm":       "rf_use_lgbm",
    "sklearn_n_jobs": "rf_n_jobs",
}


@dataclass
class SWOTConfig:
    # --- Data sources ---
    swot_path: str = "s3://swot-ai-ssv/SWOT_L3/calval/Expert_reproc_v3_uswc_calval"
    hfr_path: str = ""
    era5_path: str = ""
    era5_pkl_path: str = ""     # S3 (or local) path to a pre-processed ERA5 pkl; auto-saved on first load
    goes_nc_path: str | None = None  # optional; GOES panels skipped if None

    # --- Domain ---
    sw_corner: list[float] = field(default_factory=lambda: [-127.0, 37.5])
    ne_corner: list[float] = field(default_factory=lambda: [-123.0, 42.5])

    # --- Mission phase ---
    mission: str = "calval"  # "calval" or "science"
    sph_calval_path: str = "orbit_data/sph_calval_swath.zip"
    sph_science_path: str = "orbit_data/sph_science_swath.zip"

    # --- Model selection ---
    model: str = "rf"  # "rf" (random forest) or "ann" (neural network)

    # --- Shared ML settings (used by every backend) ---
    features: list[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))
    stencil_k: int = 3
    random_state: int = 42

    # --- Random-forest hyperparameters (swotxai.models.rf) ---
    rf_n_estimators: int = 50
    rf_max_depth: int = 15
    rf_n_jobs: int = -1       # -1 = all cores; overridden per-job in batch mode
    rf_use_gpu: bool = True   # use cuML RandomForest (requires NVIDIA GPU + RAPIDS)
    rf_use_lgbm: bool = False # use LightGBM GPU RandomForest (fallback if cuML unavailable)

    # --- ANN hyperparameters (swotxai.models.ann) ---
    ann_hidden_layers: list[int] = field(default_factory=lambda: [256, 256, 128])
    ann_activation: str = "silu"  # one of ANN_ACTIVATIONS
    ann_dropout: float = 0.1
    ann_lr: float = 1e-3
    ann_weight_decay: float = 1e-4
    ann_batch_size: int = 4096
    ann_max_epochs: int = 200
    ann_patience: int = 15       # early-stopping patience (epochs without val improvement)
    ann_val_fraction: float = 0.1
    ann_device: str = "auto"     # "auto" | "cuda" | "cpu"

    # --- Animation ---
    cycles_start: int = 474
    cycles_end: int = 578
    frame_dir: str = ""          # auto-derived in __post_init__ if blank
    animation_output: str = ""   # auto-derived as {frame_dir}/animation if blank
    fps: int = 8
    dpi: int = 150

    # --- Caching ---
    cache_dir: str = "cache"
    run_id: str = ""

    # Unique per-run identifier. Minted by the orchestrator at run start
    # (never persisted to YAML); also used to name animation outputs when
    # frame_dir / animation_output are left blank.
    experiment_id: str = ""

    # --- Preset pkl paths (skip load_swot+regrid / load_hfr+interp_hfr) ---
    swot_pkl_path: str | None = None
    hfr_pkl_path: str | None = None

    # --- Experiment region: an HFR network id ("uswc", "usegc", "gak",
    # "akns", "glna") or None ---
    # When set, flattened / inference / weights are routed into
    # experiments/{region}/…  instead of cache_dir.
    # Model-specific artifacts live under a per-model subdirectory
    # (…/{region}/rf/… or …/{region}/ann/…).
    region: str | None = None

    def __post_init__(self):
        if self.mission not in ("calval", "science"):
            raise ValueError(f"mission must be 'calval' or 'science', got {self.mission!r}")
        if self.model not in AVAILABLE_MODELS:
            raise ValueError(f"model must be one of {AVAILABLE_MODELS}, got {self.model!r}")
        unknown = [f for f in self.features if f not in AVAILABLE_FEATURES]
        if unknown:
            raise ValueError(f"Unknown features: {unknown}. Available: {AVAILABLE_FEATURES}")
        if self.stencil_k % 2 == 0:
            raise ValueError(f"stencil_k must be odd, got {self.stencil_k}")
        if self.rf_n_jobs == 0:
            raise ValueError("rf_n_jobs cannot be 0; use -1 for all cores or a positive integer")
        if not self.ann_hidden_layers or any(int(h) <= 0 for h in self.ann_hidden_layers):
            raise ValueError(f"ann_hidden_layers must be positive integers, got {self.ann_hidden_layers}")
        self.ann_activation = self.ann_activation.lower()
        if self.ann_activation not in ANN_ACTIVATIONS:
            raise ValueError(f"ann_activation must be one of {ANN_ACTIVATIONS}, got {self.ann_activation!r}")
        if not 0.0 <= self.ann_dropout < 1.0:
            raise ValueError(f"ann_dropout must be in [0, 1), got {self.ann_dropout}")
        if not 0.0 < self.ann_val_fraction < 0.5:
            raise ValueError(f"ann_val_fraction must be in (0, 0.5), got {self.ann_val_fraction}")
        if self.ann_device not in ("auto", "cuda", "cpu"):
            raise ValueError(f"ann_device must be 'auto', 'cuda' or 'cpu', got {self.ann_device!r}")
        if self.run_id and not re.match(r'^[A-Za-z0-9_\-]+$', self.run_id):
            raise ValueError(f"run_id must contain only letters, digits, underscores, or hyphens. Got: {self.run_id!r}")
    def ensure_output_paths(self) -> None:
        """Fill blank frame_dir / animation_output.

        Called by the orchestrator after the experiment_id is minted, so
        unnamed animation outputs are named after the experiment's unique id.
        """
        ident = self.experiment_id or self.run_id or "run"
        if not self.frame_dir:
            base = f"frames/{self.region}/{self.model}" if self.region else f"frames/{self.model}"
            self.frame_dir = f"{base}/{ident}"
        if not self.animation_output:
            base = f"animations/{self.region}/{self.model}" if self.region else f"animations/{self.model}"
            self.animation_output = f"{base}/{ident}"

    def _file_stem(self, name: str, ident: str | None = None) -> str:
        """Build a fully-qualified file stem: {name}_{mission}_{region}_{start}_{end}_{ident}"""
        parts = [name, self.mission]
        if self.region:
            parts.append(self.region)
        parts.append(f"{self.cycles_start}_{self.cycles_end}")
        ident = self.run_id if ident is None else ident
        if ident:
            parts.append(ident)
        return "_".join(parts)

    def _flat_stem(self) -> str:
        """Flattened-specific stem keyed by what changes the matrix — stencil_k
        and the feature list (order-sensitive) — never run_id or model, so
        backends and runs with identical inputs share the cache."""
        import hashlib
        parts = ["flattened", self.mission]
        if self.region:
            parts.append(self.region)
        parts.append(f"{self.cycles_start}_{self.cycles_end}")
        parts.append(f"k{self.stencil_k}")
        feat_key = hashlib.sha1(",".join(self.features).encode()).hexdigest()[:6]
        parts.append(f"f{feat_key}")
        return "_".join(parts)

    def cache_path(self, name: str) -> Path:
        # Flattened data is model-agnostic and shared between backends.
        if name == "flattened":
            stem = self._flat_stem()
            if self.region:
                return Path("experiments") / self.region / "flattened" / f"{stem}.pkl"
            return Path(self.cache_dir) / f"{stem}.pkl"

        suffix = ".pkl"
        if name in ("rf_u", "rf_v"):
            suffix = ".joblib"
        elif name == "ann_model":
            suffix = ".pt"

        # Model-specific artifacts are partitioned under a per-model directory.
        # With no run_id they are keyed by the run's unique experiment_id, so
        # unnamed runs never collide (shared data stems keep run_id-only keys
        # and stay reusable across runs).
        if name in _MODEL_ARTIFACTS:
            stem = self._file_stem(name, ident=self.run_id or self.experiment_id)
            if self.region:
                base = Path("experiments") / self.region / self.model
                if name == "inference":
                    return base / self.mission / f"{stem}{suffix}"
                return base / "weights" / f"{stem}{suffix}"
            return Path(self.cache_dir) / self.model / f"{stem}{suffix}"

        return Path(self.cache_dir) / f"{self._file_stem(name)}{suffix}"


def load_config(path: str | Path) -> SWOTConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    # Migrate pre-partition key names (n_estimators → rf_n_estimators, ...)
    for old, new in _LEGACY_KEYS.items():
        if old in data and new not in data:
            data[new] = data.pop(old)
        else:
            data.pop(old, None)
    data.pop("experiment_id", None)  # per-run, never loaded from file
    return SWOTConfig(**data)


def save_config(config: SWOTConfig, path: str | Path) -> None:
    data = asdict(config)
    data.pop("experiment_id", None)  # per-run, never persisted
    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)


def default_config() -> SWOTConfig:
    return SWOTConfig()
