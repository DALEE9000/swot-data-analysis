"""Experiment registry — automatic, append-only record of every completed run.

Each pipeline run that reaches completion is recorded under a per-model
registry directory, so RF and ANN experiment histories stay partitioned:

    experiments/registry/{rf,ann}/experiments.jsonl   (full records)
    experiments/registry/{rf,ann}/experiments_summary.csv (flat view)

A record captures everything fed into the run — the full config (data inputs,
domain, features, stenciling, model hyperparameters), the resulting metrics
(RMSE / R² / feature importances), library versions, and per-step timings —
under a unique experiment id.
"""
from __future__ import annotations

import csv
import json
import platform
import uuid
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

from swotxai.config import SWOTConfig

REGISTRY_ROOT = Path("experiments/registry")

_SUMMARY_COLUMNS = [
    "experiment_id", "recorded_at", "model", "run_id", "region", "mission",
    "cycles_start", "cycles_end", "stencil_k", "n_features", "features",
    "hyperparameters", "rmse_u", "rmse_v", "r2_u", "r2_v",
]


def _library_versions(model: str) -> dict:
    env = {"python": platform.python_version()}
    try:
        import sklearn
        env["scikit_learn"] = sklearn.__version__
    except ImportError:
        pass
    if model == "ann":
        try:
            import torch
            env["torch"] = torch.__version__
            env["cuda_available"] = torch.cuda.is_available()
        except ImportError:
            pass
    else:
        try:
            import lightgbm
            env["lightgbm"] = lightgbm.__version__
        except ImportError:
            pass
    return env


def _hyperparameters(config: SWOTConfig) -> dict:
    cfg = asdict(config)
    prefix = f"{config.model}_"
    hp = {k: v for k, v in cfg.items() if k.startswith(prefix)}
    hp["stencil_k"] = config.stencil_k
    hp["random_state"] = config.random_state
    return hp


def new_experiment_id(model: str) -> str:
    """Mint a unique, time-sortable experiment identifier."""
    return f"{model}_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"


def record_experiment(config: SWOTConfig, metrics: dict, extra: dict | None = None) -> dict:
    """Append a full record of a completed run to the model's registry.

    Uses the ``experiment_id`` already minted for the run (which also names
    unnamed animation outputs), minting one only if the config lacks it.

    Returns the record (including its ``experiment_id``).
    """
    now = datetime.now()
    experiment_id = config.experiment_id or new_experiment_id(config.model)

    record = {
        "experiment_id": experiment_id,
        "recorded_at":   now.isoformat(timespec="seconds"),
        "model":         config.model,
        "run_id":        config.run_id,
        "inputs": {
            "swot_path":     config.swot_path,
            "swot_pkl_path": config.swot_pkl_path,
            "hfr_path":      config.hfr_path,
            "hfr_pkl_path":  config.hfr_pkl_path,
            "era5_path":     config.era5_path,
            "era5_pkl_path": config.era5_pkl_path,
            "goes_nc_path":  config.goes_nc_path,
            "sw_corner":     config.sw_corner,
            "ne_corner":     config.ne_corner,
            "mission":       config.mission,
            "region":        config.region,
            "cycles_start":  config.cycles_start,
            "cycles_end":    config.cycles_end,
        },
        "features":        list(config.features),
        "stencil_k":       config.stencil_k,
        "hyperparameters": _hyperparameters(config),
        "metrics":         metrics,
        "config":          asdict(config),   # complete snapshot, nothing omitted
        "environment":     _library_versions(config.model),
    }
    if extra:
        record.update(extra)

    reg_dir = REGISTRY_ROOT / config.model
    reg_dir.mkdir(parents=True, exist_ok=True)

    with open(reg_dir / "experiments.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, default=str) + "\n")

    summary = {
        "experiment_id":  experiment_id,
        "recorded_at":    record["recorded_at"],
        "model":          config.model,
        "run_id":         config.run_id,
        "region":         config.region or "",
        "mission":        config.mission,
        "cycles_start":   config.cycles_start,
        "cycles_end":     config.cycles_end,
        "stencil_k":      config.stencil_k,
        "n_features":     len(config.features),
        "features":       ",".join(config.features),
        "hyperparameters": json.dumps(_hyperparameters(config), default=str),
        "rmse_u":         metrics.get("rmse_u"),
        "rmse_v":         metrics.get("rmse_v"),
        "r2_u":           metrics.get("r2_u"),
        "r2_v":           metrics.get("r2_v"),
    }
    csv_path = reg_dir / "experiments_summary.csv"
    write_header = not csv_path.exists()
    with open(csv_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_SUMMARY_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(summary)

    return record


def scored_experiments(records: list[dict]) -> list[dict]:
    """Experiments that recorded both R² components, ranked best-first by mean R²."""
    scored = [
        r for r in records
        if isinstance(r.get("metrics"), dict)
        and r["metrics"].get("r2_u") is not None
        and r["metrics"].get("r2_v") is not None
    ]
    scored.sort(
        key=lambda r: (r["metrics"]["r2_u"] + r["metrics"]["r2_v"]) / 2,
        reverse=True,
    )
    return scored


def _experiment_label(record: dict) -> str:
    if record.get("run_id"):
        return str(record["run_id"])[:16]
    eid = record.get("experiment_id", "")
    parts = eid.split("_")
    return parts[-1] if len(parts) >= 2 else eid[:8]


# Plot theme defaults suit a white background; the app passes its dark theme.
_LIGHT_THEME = {
    "color_u":     "#0891b2",
    "color_v":     "#7c6ee4",
    "ink":         "#1a2733",
    "ink_muted":   "#5b6b7b",
    "grid":        "#d7dee6",
    "surface":     "#ffffff",
    "surface_alt": "#eef2f6",
    "accent":      "#0891b2",
}

_FONT = "Space Grotesk, Segoe UI, system-ui, sans-serif"


def _theme(overrides: dict | None) -> dict:
    t = dict(_LIGHT_THEME)
    if overrides:
        t.update(overrides)
    return t


def _apply_base_layout(fig, t: dict, height: int) -> None:
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family=_FONT, color=t["ink_muted"], size=12),
        hoverlabel=dict(
            bgcolor=t["surface_alt"], bordercolor=t["grid"],
            font=dict(family=_FONT, color=t["ink"], size=12), align="left",
        ),
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="closest",
    )


def _hover_metadata(record: dict) -> str:
    """Multi-line hover card for one experiment (plotly hovertemplate HTML)."""
    import html as _html

    m  = record.get("metrics") or {}
    hp = record.get("hyperparameters") or {}
    inputs = record.get("inputs") or {}
    esc = lambda s: _html.escape(str(s))

    lines = [f"<b>R²  u {m.get('r2_u', float('nan')):.3f} · v {m.get('r2_v', float('nan')):.3f}</b>"]
    if m.get("rmse_u") is not None and m.get("rmse_v") is not None:
        lines.append(f"RMSE  u {m['rmse_u']:.4f} · v {m['rmse_v']:.4f} m/s")
    lines.append(esc(record.get("experiment_id", "")))
    ctx = " · ".join(x for x in (
        record.get("model", ""),
        f"stencil k={record.get('stencil_k', '—')}",
        inputs.get("region") or "",
        inputs.get("mission") or "",
        (record.get("recorded_at") or "")[:16].replace("T", " "),
    ) if x)
    lines.append(esc(ctx))
    feats = record.get("features") or []
    for i in range(0, len(feats), 4):
        prefix = "features: " if i == 0 else "&nbsp;&nbsp;&nbsp;&nbsp;"
        lines.append(prefix + esc(", ".join(feats[i:i + 4])))
    if record.get("model") == "rf":
        lines.append(esc(f"trees={hp.get('rf_n_estimators')} · depth={hp.get('rf_max_depth')}"))
    else:
        hidden = hp.get("ann_hidden_layers") or []
        lines.append(esc(
            f"hidden={'-'.join(str(h) for h in hidden)}"
            f" · lr={hp.get('ann_lr')} · dropout={hp.get('ann_dropout')}"
        ))
    return "<br>".join(lines)


def leaderboard_figure(
    records: list[dict],
    max_experiments: int = 25,
    theme: dict | None = None,
):
    """Interactive ablation leaderboard: R² ranking over a feature-membership matrix.

    Top panel — paired R²(u)/R²(v) dots per experiment, best on the left;
    hovering any dot shows the full experiment metadata. Bottom panel —
    aligned matrix of which features each run used, plus its stencil size and
    model backend (and region/mission when they vary), so shared ingredients
    of the top runs line up visually.

    ``records`` are registry entries from :func:`load_experiments`. Returns a
    plotly Figure, or ``None`` if no record has both R² components. ``theme``
    overrides plot colors (keys: color_u, color_v, ink, ink_muted, grid,
    surface, surface_alt, accent); defaults suit a white background.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    from swotxai.config import AVAILABLE_FEATURES

    t = _theme(theme)
    scored = scored_experiments(records)[:max_experiments]
    if not scored:
        return None

    n = len(scored)
    xs = list(range(n))
    labels = [_experiment_label(r) for r in scored]
    r2_u = [r["metrics"]["r2_u"] for r in scored]
    r2_v = [r["metrics"]["r2_v"] for r in scored]
    meta = [_hover_metadata(r) for r in scored]

    feat_rows = [
        f for f in AVAILABLE_FEATURES
        if any(f in (r.get("features") or []) for r in scored)
    ]

    # Config facets rendered as text rows; region/mission only when they vary.
    def _region(r):  return (r.get("inputs") or {}).get("region") or "—"
    def _mission(r): return (r.get("inputs") or {}).get("mission") or "—"
    text_rows = [
        ("stencil k", lambda r: str(r.get("stencil_k", "—"))),
        ("model",     lambda r: r.get("model", "—")),
    ]
    if len({_region(r) for r in scored}) > 1:
        text_rows.append(("region", _region))
    if len({_mission(r) for r in scored}) > 1:
        text_rows.append(("mission", _mission))
    n_rows = len(feat_rows) + len(text_rows)

    top_px, matrix_px = 300, 26 * n_rows
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        row_heights=[top_px, matrix_px], vertical_spacing=0.05,
    )

    # --- Top panel: R² dumbbells ---
    cx, cy = [], []
    for x, (u, v) in enumerate(zip(r2_u, r2_v)):
        cx += [x, x, None]
        cy += [u, v, None]
    fig.add_trace(go.Scatter(
        x=cx, y=cy, mode="lines", line=dict(color=t["ink_muted"], width=1),
        opacity=0.55, hoverinfo="skip", showlegend=False,
    ), row=1, col=1)
    for name, vals, color in (("R² u", r2_u, t["color_u"]),
                              ("R² v", r2_v, t["color_v"])):
        fig.add_trace(go.Scatter(
            x=xs, y=vals, mode="markers", name=name,
            marker=dict(size=11, color=color, line=dict(color=t["surface"], width=1.5)),
            customdata=meta,
            hovertemplate="%{customdata}<extra></extra>",
        ), row=1, col=1)

    # --- Bottom panel: membership matrix ---
    used_x, used_y, used_meta = [], [], []
    off_x, off_y, off_meta = [], [], []
    for i, feat in enumerate(feat_rows):
        for x, r in enumerate(scored):
            if feat in (r.get("features") or []):
                used_x.append(x); used_y.append(i)
                used_meta.append(f"<b>{feat}</b> · used<br>{labels[x]}")
            else:
                off_x.append(x); off_y.append(i)
                off_meta.append(f"<b>{feat}</b> · not used<br>{labels[x]}")
    fig.add_trace(go.Scatter(
        x=used_x, y=used_y, mode="markers", marker=dict(size=9, color=t["ink"]),
        customdata=used_meta, hovertemplate="%{customdata}<extra></extra>",
        showlegend=False,
    ), row=2, col=1)
    if off_x:
        fig.add_trace(go.Scatter(
            x=off_x, y=off_y, mode="markers", marker=dict(size=4, color=t["grid"]),
            customdata=off_meta, hovertemplate="%{customdata}<extra></extra>",
            showlegend=False,
        ), row=2, col=1)
    for j, (_, value_fn) in enumerate(text_rows):
        i = len(feat_rows) + j
        fig.add_trace(go.Scatter(
            x=xs, y=[i] * n, mode="text", text=[value_fn(r) for r in scored],
            textfont=dict(color=t["ink"], size=11, family=_FONT),
            hoverinfo="skip", showlegend=False,
        ), row=2, col=1)
    for i in range(0, n_rows, 2):
        fig.add_shape(
            type="rect", xref="x2", yref="y2",
            x0=-0.6, x1=n - 0.4, y0=i - 0.5, y1=i + 0.5,
            fillcolor=t["surface_alt"], line_width=0, layer="below",
        )

    fig.update_xaxes(range=[-0.6, n - 0.4], showgrid=False, zeroline=False)
    fig.update_xaxes(row=1, col=1, showticklabels=False, ticks="")
    fig.update_xaxes(row=2, col=1, tickvals=xs, ticktext=labels,
                     tickangle=-40, tickfont=dict(size=10))
    fig.update_yaxes(row=1, col=1, title_text="R²", gridcolor=t["grid"],
                     zeroline=min(min(r2_u), min(r2_v)) < 0,
                     zerolinecolor=t["ink_muted"], zerolinewidth=1,
                     tickfont=dict(size=11))
    fig.update_yaxes(row=2, col=1, tickvals=list(range(n_rows)),
                     ticktext=feat_rows + [name for name, _ in text_rows],
                     autorange="reversed", showgrid=False, zeroline=False,
                     tickfont=dict(size=11))
    fig.update_layout(legend=dict(
        orientation="h", x=1, xanchor="right", y=1.0, yanchor="bottom",
        font=dict(color=t["ink"], size=12),
    ))
    _apply_base_layout(fig, t, top_px + matrix_px + 130)
    return fig


def importance_heatmap_figure(
    records: list[dict],
    component: str = "mean",
    max_experiments: int = 25,
    theme: dict | None = None,
):
    """Feature-importance heatmap across experiments, aligned with the leaderboard.

    Each column is one run; each cell is that feature's share of the run's
    total importance, so RF (impurity) and ANN (permutation) runs are
    comparable in pattern even though their raw units differ. ``component``
    selects the u model, the v model, or their mean. Hover shows both raw
    shares. Returns ``None`` when no shown run recorded importances.
    """
    import plotly.graph_objects as go

    from swotxai.config import AVAILABLE_FEATURES

    t = _theme(theme)
    scored = scored_experiments(records)[:max_experiments]
    if not scored:
        return None

    def _shares(r, comp):
        fi = (r.get("metrics") or {}).get(f"feature_importance_{comp}") or {}
        vals = {k: v for k, v in fi.items() if v is not None}
        total = sum(max(v, 0.0) for v in vals.values())
        if total <= 0:
            return {}
        return {k: max(v, 0.0) / total for k, v in vals.items()}

    shares_u = [_shares(r, "u") for r in scored]
    shares_v = [_shares(r, "v") for r in scored]
    if not any(shares_u) and not any(shares_v):
        return None

    feats_seen = set()
    for s in shares_u + shares_v:
        feats_seen.update(s)
    feats = [f for f in AVAILABLE_FEATURES if f in feats_seen]
    feats += sorted(feats_seen - set(feats))

    n = len(scored)
    xs = list(range(n))
    labels = [_experiment_label(r) for r in scored]

    z, custom = [], []
    for f in feats:
        z_row, c_row = [], []
        for i in range(n):
            su, sv = shares_u[i].get(f), shares_v[i].get(f)
            if component == "u":
                val = su
            elif component == "v":
                val = sv
            else:
                both = [x for x in (su, sv) if x is not None]
                val = sum(both) / len(both) if both else None
            z_row.append(val)
            c_row.append([
                f, labels[i],
                "—" if su is None else f"{su:.1%}",
                "—" if sv is None else f"{sv:.1%}",
            ])
        z.append(z_row)
        custom.append(c_row)

    fig = go.Figure(go.Heatmap(
        z=z, x=xs, y=feats, customdata=custom,
        colorscale=[[0.0, t["surface_alt"]], [1.0, t["accent"]]],
        zmin=0, xgap=2, ygap=2, hoverongaps=False,
        hovertemplate=(
            "<b>u %{customdata[2]} · v %{customdata[3]}</b> of run importance<br>"
            "%{customdata[0]} · %{customdata[1]}<extra></extra>"
        ),
        colorbar=dict(
            title=dict(text="share", font=dict(color=t["ink_muted"], size=11)),
            tickformat=".0%", thickness=12, outlinewidth=0,
            tickfont=dict(color=t["ink_muted"], size=10),
        ),
    ))
    fig.update_xaxes(tickvals=xs, ticktext=labels, tickangle=-40,
                     tickfont=dict(size=10), showgrid=False, zeroline=False)
    fig.update_yaxes(autorange="reversed", tickfont=dict(size=11),
                     showgrid=False, zeroline=False)
    _apply_base_layout(fig, t, 30 * len(feats) + 150)
    return fig


def importance_bar_figure(record: dict, theme: dict | None = None):
    """Grouped horizontal bars of one experiment's raw u/v feature importances.

    Returns a plotly Figure, or ``None`` if the record has no importances.
    """
    import plotly.graph_objects as go

    t = _theme(theme)
    m = record.get("metrics") or {}
    fi_u = m.get("feature_importance_u") or {}
    fi_v = m.get("feature_importance_v") or {}
    if not fi_u and not fi_v:
        return None

    feats = sorted(
        set(fi_u) | set(fi_v),
        key=lambda f: -((fi_u.get(f) or 0) + (fi_v.get(f) or 0)),
    )
    fig = go.Figure()
    for name, fi, color in (("u", fi_u, t["color_u"]), ("v", fi_v, t["color_v"])):
        fig.add_trace(go.Bar(
            y=feats, x=[fi.get(f) for f in feats], name=name, orientation="h",
            marker=dict(color=color, line=dict(width=0)),
            hovertemplate="<b>%{x:.4f}</b> · %{y} (" + name + ")<extra></extra>",
        ))
    fig.update_layout(
        barmode="group", bargap=0.35, bargroupgap=0.2,
        legend=dict(orientation="h", x=1, xanchor="right", y=1.0,
                    yanchor="bottom", font=dict(color=t["ink"], size=12)),
    )
    fig.update_xaxes(gridcolor=t["grid"], zeroline=False,
                     title_text="importance", tickfont=dict(size=10))
    fig.update_yaxes(autorange="reversed", showgrid=False, zeroline=False,
                     tickfont=dict(size=11))
    _apply_base_layout(fig, t, 28 * len(feats) + 130)
    return fig


# Sweep parameters derived from non-scalar config fields (in preference order).
_DERIVED_PARAMS = ("stencil_k", "n_features", "ann_n_hidden_layers", "ann_hidden_width")

# One-line meanings for the sweep parameters offered as response-surface axes.
PARAM_DESCRIPTIONS = {
    "stencil_k":           "spatial context window — each sample includes its k×k neighborhood "
                           "of grid cells as extra features (1 = just the cell itself)",
    "n_features":          "how many input features the run used (length of its feature list)",
    "ann_n_hidden_layers": "MLP depth — number of hidden layers",
    "ann_hidden_width":    "MLP width — mean neurons per hidden layer (exact when all layers are equal)",
    "ann_batch_size":      "training mini-batch size",
    "ann_dropout":         "dropout probability between hidden layers (0 = off)",
    "ann_lr":              "Adam learning rate",
    "ann_weight_decay":    "L2 regularization strength (AdamW weight decay)",
    "ann_max_epochs":      "upper bound on training epochs",
    "ann_patience":        "early-stopping patience — epochs without validation improvement before training stops",
    "ann_val_fraction":    "fraction of training data held out as the validation split",
    "rf_n_estimators":     "number of trees in the random forest",
    "rf_max_depth":        "maximum tree depth",
    "rf_n_jobs":           "CPU threads used for training (speed only — doesn't affect accuracy)",
    "random_state":        "RNG seed (run-to-run noise, not a tuning knob)",
}


def describe_param(param: str) -> str:
    """One-line meaning of a sweep parameter, with a generic fallback."""
    return PARAM_DESCRIPTIONS.get(param, "model hyperparameter (see config.yaml)")


def _param_value(record: dict, param: str):
    """Numeric value of a sweep parameter for one record, or None."""
    if param == "stencil_k":
        return record.get("stencil_k")
    if param == "n_features":
        feats = record.get("features")
        return len(feats) if feats else None
    if param in ("ann_n_hidden_layers", "ann_hidden_width"):
        hidden = (record.get("hyperparameters") or {}).get("ann_hidden_layers")
        if not isinstance(hidden, (list, tuple)) or not hidden:
            return None
        if param == "ann_n_hidden_layers":
            return len(hidden)
        return sum(hidden) / len(hidden)  # mean units/layer; exact for uniform stacks
    v = (record.get("hyperparameters") or {}).get(param)
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return None
    return v


def available_response_params(records: list[dict]) -> list[str]:
    """Numeric knobs that take ≥ 2 distinct values across the scored runs."""
    scored = scored_experiments(records)
    candidates = set(_DERIVED_PARAMS)
    for r in scored:
        for k, v in (r.get("hyperparameters") or {}).items():
            if not isinstance(v, bool) and isinstance(v, (int, float)):
                candidates.add(k)
    varying = [
        p for p in candidates
        if len({_param_value(r, p) for r in scored} - {None}) >= 2
    ]
    front = [p for p in _DERIVED_PARAMS if p in varying]
    return front + sorted(set(varying) - set(front))


def response_grid(records: list[dict], x_param: str, y_param: str):
    """Mean-R² grid over two sweep parameters.

    Returns ``(xs, ys, z, counts)`` — sorted unique parameter values, the
    grid of mean R² (mean of u/v, averaged over all runs at that combination,
    NaN where no run exists), and how many runs each cell averages. Returns
    ``None`` unless both parameters take ≥ 2 values among usable runs.
    """
    import numpy as np

    cells: dict[tuple, list] = {}
    for r in scored_experiments(records):
        xv, yv = _param_value(r, x_param), _param_value(r, y_param)
        if xv is None or yv is None:
            continue
        m = r["metrics"]
        cells.setdefault((xv, yv), []).append((m["r2_u"] + m["r2_v"]) / 2)
    if not cells:
        return None

    xs = sorted({x for x, _ in cells})
    ys = sorted({y for _, y in cells})
    if len(xs) < 2 or len(ys) < 2:
        return None

    z = np.full((len(ys), len(xs)), np.nan)
    counts = np.zeros((len(ys), len(xs)), dtype=int)
    for (xv, yv), vals in cells.items():
        i, j = ys.index(yv), xs.index(xv)
        z[i, j] = float(np.mean(vals))
        counts[i, j] = len(vals)
    return xs, ys, z, counts


def _fmt_param(v) -> str:
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def _mix(hex_a: str, hex_b: str, frac: float) -> str:
    """Blend two hex colors; frac=0 → a, frac=1 → b."""
    a = [int(hex_a.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    b = [int(hex_b.lstrip("#")[i:i + 2], 16) for i in (0, 2, 4)]
    return "#" + "".join(f"{round(x + (y - x) * frac):02x}" for x, y in zip(a, b))


def _response_hover_text(xs, ys, z, counts, x_param, y_param):
    import numpy as np

    text = []
    for i, yv in enumerate(ys):
        row = []
        for j, xv in enumerate(xs):
            if np.isnan(z[i, j]):
                row.append(f"no runs<br>{x_param}={_fmt_param(xv)} · {y_param}={_fmt_param(yv)}")
            else:
                n = counts[i, j]
                row.append(
                    f"<b>mean R² {z[i, j]:.3f}</b> ({n} run{'s' if n != 1 else ''})<br>"
                    f"{x_param}={_fmt_param(xv)} · {y_param}={_fmt_param(yv)}"
                )
        text.append(row)
    return text


def response_heatmap_figure(
    records: list[dict],
    x_param: str,
    y_param: str,
    theme: dict | None = None,
):
    """Mean-R² heatmap over two sweep parameters (the working view).

    Values are printed in the cells; empty cells are parameter combinations
    no run has covered. Returns ``None`` when :func:`response_grid` does.
    """
    import plotly.graph_objects as go

    t = _theme(theme)
    grid = response_grid(records, x_param, y_param)
    if grid is None:
        return None
    xs, ys, z, counts = grid

    fig = go.Figure(go.Heatmap(
        z=z,
        x=list(range(len(xs))), y=list(range(len(ys))),
        text=_response_hover_text(xs, ys, z, counts, x_param, y_param),
        hovertemplate="%{text}<extra></extra>",
        texttemplate="%{z:.3f}",
        textfont=dict(color=t["ink"], size=11, family=_FONT),
        colorscale=[[0.0, t["surface_alt"]], [1.0, t["accent"]]],
        xgap=2, ygap=2, hoverongaps=False,
        colorbar=dict(
            title=dict(text="mean R²", font=dict(color=t["ink_muted"], size=11)),
            thickness=12, outlinewidth=0,
            tickfont=dict(color=t["ink_muted"], size=10),
        ),
    ))
    fig.update_xaxes(tickvals=list(range(len(xs))), ticktext=[_fmt_param(v) for v in xs],
                     title_text=x_param, showgrid=False, zeroline=False,
                     tickfont=dict(size=10))
    fig.update_yaxes(tickvals=list(range(len(ys))), ticktext=[_fmt_param(v) for v in ys],
                     title_text=y_param, showgrid=False, zeroline=False,
                     tickfont=dict(size=10))
    _apply_base_layout(fig, t, 60 * len(ys) + 160)
    return fig


def response_surface_figure(
    records: list[dict],
    x_param: str,
    y_param: str,
    theme: dict | None = None,
):
    """3D scatter of runs over two sweep parameters (the presentation view).

    One point per run at (x_param, y_param, mean R²), colored by mean R²;
    runs repeating the same configuration stack vertically, so run-to-run
    spread is visible. Hovering a point shows that run's full metadata.
    Parameter values are treated as ordered categories (so log-spaced knobs
    like learning rate stay evenly spaced). Returns ``None`` unless both
    parameters take ≥ 2 values among usable runs.
    """
    import plotly.graph_objects as go

    t = _theme(theme)
    runs = []
    for r in scored_experiments(records):
        xv, yv = _param_value(r, x_param), _param_value(r, y_param)
        if xv is None or yv is None:
            continue
        m = r["metrics"]
        runs.append((xv, yv, (m["r2_u"] + m["r2_v"]) / 2, r))
    if not runs:
        return None
    xs = sorted({xv for xv, _, _, _ in runs})
    ys = sorted({yv for _, yv, _, _ in runs})
    if len(xs) < 2 or len(ys) < 2:
        return None

    pt_x = [xs.index(xv) for xv, _, _, _ in runs]
    pt_y = [ys.index(yv) for _, yv, _, _ in runs]
    pt_z = [zv for _, _, zv, _ in runs]
    pt_meta = [_hover_metadata(r) for _, _, _, r in runs]

    axis_common = dict(
        showbackground=False,
        gridcolor=t["grid"],
        zerolinecolor=t["grid"],
        tickfont=dict(color=t["ink_muted"], size=10),
    )
    fig = go.Figure(go.Scatter3d(
        x=pt_x, y=pt_y, z=pt_z, mode="markers",
        marker=dict(
            size=7, color=pt_z,
            # Low end lifted off the surface color so weak runs stay visible
            # as small marks, unlike heatmap cells which have area.
            colorscale=[[0.0, _mix(t["surface_alt"], t["accent"], 0.35)],
                        [1.0, t["accent"]]],
            cmin=min(pt_z), cmax=max(pt_z),
            line=dict(color=t["ink_muted"], width=1.5),
            colorbar=dict(
                title=dict(text="mean R²", font=dict(color=t["ink_muted"], size=11)),
                thickness=12, outlinewidth=0,
                tickfont=dict(color=t["ink_muted"], size=10),
            ),
        ),
        customdata=pt_meta, hovertemplate="%{customdata}<extra></extra>",
        showlegend=False,
    ))
    fig.update_layout(scene=dict(
        xaxis=dict(title=dict(text=x_param, font=dict(color=t["ink_muted"])),
                   tickvals=list(range(len(xs))), ticktext=[_fmt_param(v) for v in xs],
                   **axis_common),
        yaxis=dict(title=dict(text=y_param, font=dict(color=t["ink_muted"])),
                   tickvals=list(range(len(ys))), ticktext=[_fmt_param(v) for v in ys],
                   **axis_common),
        zaxis=dict(title=dict(text="mean R²", font=dict(color=t["ink_muted"])),
                   **axis_common),
        camera=dict(eye=dict(x=1.5, y=-1.5, z=0.9)),
    ))
    _apply_base_layout(fig, t, 560)
    fig.update_layout(margin=dict(l=0, r=0, t=10, b=0))
    return fig


def load_experiments(model: str | None = None) -> list[dict]:
    """Load all recorded experiments (newest first), optionally for one model."""
    records: list[dict] = []
    for m in ([model] if model else ["rf", "ann"]):
        path = REGISTRY_ROOT / m / "experiments.jsonl"
        if not path.exists():
            continue
        with open(path, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    records.append(json.loads(line))
    return sorted(records, key=lambda r: r.get("recorded_at", ""), reverse=True)
