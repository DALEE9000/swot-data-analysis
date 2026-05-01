from __future__ import annotations

import re
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import yaml

# All SWOT L3 variables available for use as RF features
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


@dataclass
class SWOTConfig:
    # --- Data sources ---
    swot_path: str = "s3://swot-ai-ssv/SWOT_L3/calval/Expert_reproc_v3_uswc_calval"
    hfr_path: str = ""
    era5_path: str = ""
    goes_dir: str | None = None  # optional; GOES panels skipped if None

    # --- Domain ---
    sw_corner: list[float] = field(default_factory=lambda: [-127.0, 37.5])
    ne_corner: list[float] = field(default_factory=lambda: [-123.0, 42.5])

    # --- Mission phase ---
    mission: str = "calval"  # "calval" or "science"
    sph_calval_path: str = "orbit_data/sph_calval_swath.zip"
    sph_science_path: str = "orbit_data/sph_science_swath.zip"

    # --- RF features ---
    features: list[str] = field(default_factory=lambda: list(DEFAULT_FEATURES))

    # --- RF hyperparameters ---
    stencil_k: int = 3
    n_estimators: int = 50
    max_depth: int = 15
    random_state: int = 42
    sklearn_n_jobs: int = -1  # -1 = all cores; overridden per-job in batch mode

    # --- Animation ---
    cycles_start: int = 474
    cycles_end: int = 578
    frame_dir: str = ""          # auto-derived in __post_init__ if blank
    animation_output: str = ""   # auto-derived as {frame_dir}/animation if blank
    fps: int = 4
    dpi: int = 150

    # --- Caching ---
    cache_dir: str = "cache"
    run_id: str = ""

    # --- Preset pkl paths (skip load_swot+regrid / load_hfr+interp_hfr) ---
    swot_pkl_path: str | None = None
    hfr_pkl_path: str | None = None

    # --- Experiment region: "uswc" | "usegc" | None ---
    # When set, flattened / inference / weights are routed into
    # SWOTxAI/code/experiments/{region}/…  instead of cache_dir.
    region: str | None = None

    def __post_init__(self):
        if self.mission not in ("calval", "science"):
            raise ValueError(f"mission must be 'calval' or 'science', got {self.mission!r}")
        unknown = [f for f in self.features if f not in AVAILABLE_FEATURES]
        if unknown:
            raise ValueError(f"Unknown features: {unknown}. Available: {AVAILABLE_FEATURES}")
        if self.stencil_k % 2 == 0:
            raise ValueError(f"stencil_k must be odd, got {self.stencil_k}")
        if self.sklearn_n_jobs == 0:
            raise ValueError("sklearn_n_jobs cannot be 0; use -1 for all cores or a positive integer")
        if self.run_id and not re.match(r'^[A-Za-z0-9_\-]+$', self.run_id):
            raise ValueError(f"run_id must contain only letters, digits, underscores, or hyphens. Got: {self.run_id!r}")
        if not self.frame_dir:
            base = f"SWOTxAI/frames/{self.region}" if self.region else "SWOTxAI/frames"
            self.frame_dir = f"{base}/{self.run_id}" if self.run_id else base
        if not self.animation_output:
            base = f"SWOTxAI/animations/{self.region}" if self.region else "SWOTxAI/animations"
            self.animation_output = f"{base}/{self.run_id}" if self.run_id else f"{base}/animation"

    def _file_stem(self, name: str) -> str:
        """Build a fully-qualified file stem: {name}_{mission}_{region}_{start}_{end}_{run_id}"""
        parts = [name, self.mission]
        if self.region:
            parts.append(self.region)
        parts.append(f"{self.cycles_start}_{self.cycles_end}")
        if self.run_id:
            parts.append(self.run_id)
        return "_".join(parts)

    def cache_path(self, name: str) -> Path:
        stem = self._file_stem(name)
        if self.region:
            base = Path("SWOTxAI/code/experiments") / self.region
            if name == "flattened":
                return base / "flattened" / f"{stem}.pkl"
            if name == "inference":
                return base / self.mission / f"{stem}.pkl"
            if name in ("rf_u", "rf_v"):
                return base / "weights" / f"{stem}.joblib"
            if name == "rf_meta":
                return base / "weights" / f"{stem}.pkl"
        return Path(self.cache_dir) / f"{stem}.pkl"


def load_config(path: str | Path) -> SWOTConfig:
    with open(path) as f:
        data = yaml.safe_load(f)
    return SWOTConfig(**data)


def save_config(config: SWOTConfig, path: str | Path) -> None:
    with open(path, "w") as f:
        yaml.dump(asdict(config), f, default_flow_style=False, sort_keys=False)


def default_config() -> SWOTConfig:
    return SWOTConfig()
