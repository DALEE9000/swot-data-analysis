"""Region/mission preset definitions — single source of truth.

Consumed by the Streamlit app (sidebar preset picker) and by
swotxai.multiregion (pooled all-region training). Every preset points at the
pre-built artifacts on s3://swot-ai-ssv with raw sources as fallback.
"""
from __future__ import annotations

S3 = "s3://swot-ai-ssv"

# rid -> (label, default-resolution HFR archive under HFR/, sw_corner, ne_corner)
REGION_META = {
    "uswc":  ("US West Coast",      "uswc/uswc_6km_Resolution_hourly_2012_2024.nc4",
              [-126.8, 31.2], [-116.6, 49.9]),
    "usegc": ("US East-Gulf Coast", "usegc/usegc_6km_Resolution_hourly_2012_2025.nc4",
              [-97.7, 22.8], [-68.0, 44.4]),
    "gak":   ("Gulf of Alaska",     "gak/gak_2km_Resolution_hourly_2017_2025.nc4",
              [-131.2, 53.6], [-129.8, 54.8]),
    "akns":  ("Alaska North Slope", "akns/akns_6km_Resolution_Hourly_RTV_best_2010_2024.nc4",
              [-162.7, 70.0], [-154.7, 73.1]),
    "glna":  ("Great Lakes",        "glna/glna_6km_Resolution_hourly_2022_2025.nc4",
              [-85.4, 45.3], [-84.2, 46.4]),
    "prvi":  ("Puerto Rico / USVI", "prvi/prvi_6km_Resolution_hourly_2010_2025.nc4",
              [-68.4, 15.8], [-63.8, 19.6]),
    "ushi":  ("Hawaii",             "ushi/ushi_6km_Resolution_hourly_2010_2025.nc4",
              [-159.4, 19.3], [-154.3, 22.1]),
}

# Regions with a non-empty HFR target per mission (akns radars have been dark
# since 2022 — no ground truth in either window; glna/prvi have no calval
# passes). These are the trainable sets the pooled mode uses.
MISSION_REGIONS = {
    "science": ["uswc", "usegc", "gak", "glna", "prvi", "ushi"],
    "calval":  ["uswc", "usegc", "gak", "ushi"],
}

# GOES SST exists for every trainable region (akns alone is out of
# geostationary view, and it isn't trainable anyway).
_GOES_REGIONS = {"uswc", "usegc", "gak", "glna", "prvi", "ushi"}

_MISSION_CYCLES = {"science": (1, 16), "calval": (474, 578)}


def _pkl_name(rid: str, mission: str) -> str:
    if mission == "science":
        return f"swot_expert_reproc_v3_{rid}_science.pkl"
    return f"swot_expert_reproc_v3_calval_{rid}_474_578.pkl"


def preset_entry(rid: str, mission: str) -> dict:
    """All preset fields for one (region, mission)."""
    label, hfr_file, sw, ne = REGION_META[rid]
    c0, c1 = _MISSION_CYCLES[mission]
    entry = {
        "swot_pkl": f"{S3}/experiments/{rid}/swot_cycles/{_pkl_name(rid, mission)}",
        "swot_path": f"{S3}/SWOT_L3/{mission}/Expert_reproc_v3_{rid}_{mission}",
        "hfr_pkl": f"{S3}/experiments/{rid}/hfr_target/hfr_{mission}_{rid}.pkl",
        "hfr_path": f"{S3}/HFR/{hfr_file}",
        "era5_pkl": f"{S3}/experiments/{rid}/era5/era5wind_{mission}_{rid}_10m.pkl",
        "sw_corner": sw,
        "ne_corner": ne,
        "mission": mission,
        "cycles_start": c0,
        "cycles_end": c1,
        "region": rid,
    }
    if rid in _GOES_REGIONS:
        entry["goes_nc"] = f"{S3}/experiments/{rid}/goes/goes_sst_{mission}_{rid}.nc"
    return entry


def build_presets() -> dict:
    """Ordered preset dict for the app sidebar: pooled entries first, then
    per-region science, then per-region calval."""
    presets = {}
    for mission in ("science", "calval"):
        presets[f"ALL regions — pooled ({mission})"] = {
            "multi": True,
            "mission": mission,
            "regions": list(MISSION_REGIONS[mission]),
            # cosmetic bbox spanning the constituent regions (config requires one)
            "sw_corner": [-162.7, 15.8],
            "ne_corner": [-63.8, 55.0] if mission == "calval" else [-63.8, 73.1],
            "cycles_start": _MISSION_CYCLES[mission][0],
            "cycles_end": _MISSION_CYCLES[mission][1],
            "region": f"all_{mission}",
        }
    for rid, (label, *_rest) in REGION_META.items():
        presets[f"{label} (science)"] = preset_entry(rid, "science")
    for mission_rid in ("uswc", "usegc", "gak", "ushi"):
        label = REGION_META[mission_rid][0]
        presets[f"{label} (calval)"] = preset_entry(mission_rid, "calval")
    return presets


def config_overrides(rid: str, mission: str) -> dict:
    """SWOTConfig kwargs for one (region, mission) — used by multiregion."""
    p = preset_entry(rid, mission)
    return {
        "swot_pkl_path": p["swot_pkl"],
        "swot_path": p["swot_path"],
        "hfr_pkl_path": p["hfr_pkl"],
        "hfr_path": p["hfr_path"],
        "era5_pkl_path": p["era5_pkl"],
        "goes_nc_path": p.get("goes_nc"),
        "sw_corner": p["sw_corner"],
        "ne_corner": p["ne_corner"],
        "mission": mission,
        "cycles_start": p["cycles_start"],
        "cycles_end": p["cycles_end"],
        "region": rid,
    }
