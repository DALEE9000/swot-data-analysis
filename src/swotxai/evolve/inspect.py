"""Heuristic candidate inspector: summarize an evolved module's architecture.

Candidates are free-form code, so this extracts a best-effort structured view
for the Evolve tab: hyperparameters (regex over common assignment patterns),
detected components (keyword/API scan), and a Graphviz DOT flow diagram.
Always presented in the UI as "auto-detected" — the diff is ground truth.
"""
from __future__ import annotations

import re

_HP_PATTERNS = {
    "hidden layers":  r"hidden\s*=\s*[\(\[]\s*([\d,\s]+?)\s*,?\s*[\)\]]",
    "dropout":        r"dropout\s*=\s*([\d.]+)",
    "learning rate":  r"\blr\s*=\s*([\d.eE+-]+)",
    "weight decay":   r"weight_decay\s*=\s*([\d.eE+-]+)",
    "batch size":     r"\b(?:bs|batch_size)\s*=\s*(\d+)",
    "patience":       r"\bpatience\s*=\s*(\d+)",
    "val fraction":   r"n_val\s*=\s*max\(1,\s*int\(n\s*\*\s*([\d.]+)\)",
    "ridge alpha":    r"(?:Ridge\(|alpha\s*=\s*)([\d.eE+-]+)",
    "huber delta":    r"delta\s*=\s*([\d.]+)",
    "EMA decay":      r"(?:ema_decay|decay)\s*=\s*(0\.9[\d]*)",
}

# (label, regex over the source, category)
_COMPONENTS = [
    ("Ridge linear baseline (residualized)", r"Ridge\(|ridge",             "structure"),
    ("Gradient-boosted trees",               r"HistGradientBoosting|GradientBoosting|lightgbm", "structure"),
    ("MLP trunk",                            r"nn\.Linear",                "structure"),
    ("LayerNorm",                            r"LayerNorm",                 "structure"),
    ("BatchNorm",                            r"BatchNorm",                 "structure"),
    ("Residual/skip connections",            r"skip_conn|shortcut|\bx = x \+|\breturn x \+", "structure"),
    ("Seed ensemble",                        r"for .*seed.* in|n_members|member_preds|seeds\s*=", "structure"),
    ("Masked MSE loss",                      r"masked_mse",                "training"),
    ("Huber loss",                           r"[Hh]uber",                  "training"),
    ("Component-balanced loss",              r"balanc",                    "training"),
    ("Temporal-tail validation",             r"tail|temporal.*val|val.*temporal", "training"),
    ("Recency weighting / fine-tune",        r"recency|recent",            "training"),
    ("EMA / weight averaging",               r"\bEMA\b|ema_|polyak|weight_averag|checkpoint_averag", "training"),
    ("Input noise augmentation",             r"noise",                     "training"),
    ("Validity-fraction features",           r"validity|valid_frac|isfinite.*mean",  "features"),
    ("Wind pseudo-stress features",          r"pseudo.?stress|\bstress\b", "features"),
    ("Speed/magnitude features",             r"speed|magnitude|hypot|sqrt.*\*\*2", "features"),
    ("Stencil spread/statistics features",   r"spread|std.*stencil|stencil.*std", "features"),
    ("NaN-native handling (no imputation)",  r"NaNs intact|nan.*native|missing.*nativ", "features"),
]

_ACTIVATIONS = ["SiLU", "GELU", "ReLU", "Tanh", "Mish", "ELU"]
_OPTIMIZERS = ["AdamW", "Adam", "SGD", "RMSprop"]
_SCHEDULERS = ["ReduceLROnPlateau", "OneCycleLR", "CosineAnnealing", "StepLR"]


def summarize_candidate(code: str) -> dict:
    """Best-effort structured summary of an evolved candidate module."""
    hp: dict[str, str] = {}
    for label, pat in _HP_PATTERNS.items():
        m = re.search(pat, code)
        if m:
            hp[label] = m.group(1).strip()

    components: dict[str, list[str]] = {"structure": [], "training": [], "features": []}
    for label, pat, cat in _COMPONENTS:
        if re.search(pat, code):
            components[cat].append(label)

    acts = [a for a in _ACTIVATIONS if re.search(rf"nn\.{a}\b|\b{a}\(\)", code)]
    opts = [o for o in _OPTIMIZERS if re.search(rf"optim\.{o}\(", code)]
    scheds = [s for s in _SCHEDULERS if s in code]
    if acts:
        hp["activation"] = ", ".join(acts)
    if opts:
        hp["optimizer"] = ", ".join(opts)
    if scheds:
        hp["LR schedule"] = ", ".join(scheds)

    hidden = []
    if "hidden layers" in hp:
        try:
            hidden = [int(x) for x in hp["hidden layers"].split(",") if x.strip()]
        except ValueError:
            hidden = []

    return {"hyperparams": hp, "components": components, "hidden": hidden,
            "loc": len(code.splitlines())}


def architecture_dot(summary: dict, n_inputs: int | None = None, theme: dict | None = None) -> str:
    """Graphviz DOT flow diagram from a candidate summary."""
    t = theme or {}
    ink = t.get("ink", "#dbe7f3")
    accent = t.get("accent", "#3fb6b2")
    surface = t.get("surface_alt", "#101f33")
    grid = t.get("grid", "#24405e")

    def node(nid, label, shape="box", color=surface):
        return (f'{nid} [label="{label}", shape={shape}, style="rounded,filled", '
                f'fillcolor="{color}", color="{grid}", fontcolor="{ink}", fontname="Segoe UI"];')

    comps = summary["components"]
    lines = [
        "digraph G {",
        'rankdir=TB; bgcolor="transparent";',
        f'edge [color="{grid}", fontcolor="{ink}", fontname="Segoe UI", fontsize=10];',
        node("X", f"Input\\n{n_inputs or '?'} features (stencil x base)"),
    ]
    prev = "X"

    feat = comps["features"]
    if feat:
        label = "Feature engineering\\n" + "\\n".join(f"+ {f}" for f in feat)
        lines.append(node("FE", label))
        lines.append(f"{prev} -> FE;")
        prev = "FE"

    has_ridge = any("Ridge" in c for c in comps["structure"])
    has_trees = any("boosted" in c for c in comps["structure"])
    has_mlp = any("MLP" in c for c in comps["structure"])
    has_ens = any("ensemble" in c.lower() for c in comps["structure"])

    if has_ridge:
        lines.append(node("RIDGE", "Ridge linear baseline\\n(exact fit"
                          + (f", alpha={summary['hyperparams']['ridge alpha']}" if "ridge alpha" in summary["hyperparams"] else "")
                          + ")", color=surface))
        lines.append(f"{prev} -> RIDGE;")

    if has_mlp:
        hidden = summary["hidden"] or []
        mlp_label = "MLP trunk\\n" + (" -> ".join(str(h) for h in hidden) if hidden else "(layers n/a)")
        extras = [c for c in comps["structure"] if c in ("LayerNorm", "BatchNorm", "Residual/skip connections")]
        if extras:
            mlp_label += "\\n(" + ", ".join(extras) + ")"
        if "activation" in summary["hyperparams"]:
            mlp_label += f"\\nact: {summary['hyperparams']['activation']}"
        lines.append(node("MLP", mlp_label, color=accent + "33" if len(accent) == 7 else surface))
        src = prev
        edge_label = ' [label="residuals"]' if has_ridge else ""
        lines.append(f"{src} -> MLP{edge_label};")

    if has_trees:
        lines.append(node("GBT", "Gradient-boosted trees\\n(per component, NaN-native)"))
        edge_label = ' [label="residuals"]' if has_ridge else ""
        lines.append(f"{prev} -> GBT{edge_label};")

    combine_srcs = [n for n, on in (("RIDGE", has_ridge), ("MLP", has_mlp), ("GBT", has_trees)) if on]
    if len(combine_srcs) > 1 or has_ens:
        label = "Combine"
        if has_ens:
            label += "\\n(seed ensemble)"
        if any("EMA" in c for c in comps["training"]):
            label += "\\n(EMA weights)"
        lines.append(node("SUM", label, shape="ellipse"))
        for s in combine_srcs:
            lines.append(f"{s} -> SUM;")
        prev = "SUM"
    elif combine_srcs:
        prev = combine_srcs[0]

    lines.append(node("OUT", "(u, v) predictions", shape="ellipse", color=accent))
    lines.append(f"{prev} -> OUT;")
    lines.append("}")
    return "\n".join(lines)
