"""Publication/poster figures for the AlphaEvolve results (white background).

1. fitness_evolution — every candidate of a run, chronological, colored by
   generation (viridis), with the best-so-far envelope; seed and champion
   annotated. Data: the run's database.jsonl.
2. champion_architecture — hand-drawn (accurate, not the app's regex
   heuristic) flow diagram of gen031_00: engineered features -> ridge
   baseline -> conv+MLP residual ensemble -> (u, v).

Writes PNG (300 dpi) + SVG to docs/figures/poster/.

Usage:
    python scripts/poster_figures.py --run evolve_all_science_v1
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

INK = "#1a1a2e"       # primary text
MUTED = "#5a6472"     # secondary text / arrows
GRID = "#d9dee5"
ACCENT = "#0f766e"    # teal — champion / output
BOX_FILL = "#f4f6f8"
BOX_EDGE = "#b8c0ca"

# LaTeX look without a LaTeX dependency: Computer Modern via mathtext
# (usetex would choke on the unicode arrows/symbols and needs a TeX run per label).
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white",
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "STIXGeneral", "Times New Roman"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,  # cmr10 has no U+2212 glyph
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
})


def _tt(name: str) -> str:
    """Identifier in typewriter mathtext — cmr10 has no plain '_' glyph."""
    return r"$\mathtt{" + name.replace("_", r"\_") + r"}$"


def _save(fig, out_dir: Path, stem: str):
    for ext in ("png", "svg"):
        p = out_dir / f"{stem}.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


def fitness_figure(run_dir: Path, out_dir: Path, run_name: str):
    recs = [json.loads(l) for l in open(run_dir / "database.jsonl", encoding="utf-8")]
    xs, ys, gens = [], [], []
    env_x, env_y, best = [], [], None
    champ = (None, -1e9, None)  # (x, fitness, id)
    for i, r in enumerate(recs):
        fit = (r.get("metrics") or {}).get("fitness")
        if r.get("status") != "ok" or fit is None:
            continue
        xs.append(i); ys.append(fit); gens.append(r.get("generation", 0))
        best = fit if best is None else max(best, fit)
        env_x.append(i); env_y.append(best)
        if fit > champ[1]:
            champ = (i, fit, r.get("candidate_id"))

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    ax.plot(env_x, env_y, ls=":", lw=2, color=INK, label="best so far", zorder=2)
    sc = ax.scatter(xs, ys, c=gens, cmap="viridis", s=26, linewidths=0, zorder=3,
                    label="candidates")
    cb = fig.colorbar(sc, ax=ax, pad=0.015)
    cb.set_label("generation", color=INK)
    cb.outline.set_edgecolor(GRID)
    cb.ax.tick_params(color=MUTED, labelcolor=MUTED)

    # Direct labels: seed and champion only.
    ax.annotate(f"seed  {ys[0]:.3f}", (xs[0], ys[0]), xytext=(10, 4),
                textcoords="offset points", fontsize=9, color=INK)
    ax.annotate(f"champion {_tt(champ[2])}  {champ[1]:.4f}", (champ[0], champ[1]),
                xytext=(-10, 10), textcoords="offset points", fontsize=9,
                color=ACCENT, ha="right", fontweight="bold")
    ax.scatter([champ[0]], [champ[1]], s=90, facecolors="none",
               edgecolors=ACCENT, linewidths=1.6, zorder=4)

    ax.set_xlabel("candidate (chronological)")
    ax.set_ylabel(r"fitness  (mean held-out $R^2$ of $u$, $v$)")
    ax.set_title(f"LLM-guided code evolution: {_tt(run_name)}", fontsize=12, pad=10)
    ax.grid(color=GRID, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    _save(fig, out_dir, f"fitness_{run_name}")


def _box(ax, cx, cy, w, h, text, fill=BOX_FILL, edge=BOX_EDGE, fs=8.6,
         title=None, tcolor=INK):
    ax.add_patch(FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                boxstyle="round,pad=0.012",
                                fc=fill, ec=edge, lw=1.1))
    if title:
        # Title pinned inside the top edge; body centered in the space below it.
        ax.text(cx, cy + h / 2 - 0.006, title, ha="center", va="top",
                fontsize=fs + 1, fontweight="bold", color=tcolor)
        ax.text(cx, cy - h * 0.16, text, ha="center", va="center", fontsize=fs,
                color=INK, linespacing=1.45)
    else:
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fs,
                color=INK, linespacing=1.45)


def _arrow(ax, x0, y0, x1, y1, label=None, ls="-"):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                 mutation_scale=13, color=MUTED, lw=1.3,
                                 linestyle=ls, shrinkA=2, shrinkB=2))
    if label:
        ax.text((x0 + x1) / 2 + 0.012, (y0 + y1) / 2, label, fontsize=7.8,
                color=MUTED, ha="left", va="center", style="italic")


def architecture_figure(out_dir: Path):
    """gen031_00, drawn from the candidate source (not the regex heuristic)."""
    fig, ax = plt.subplots(figsize=(7.0, 10.0))
    ax.set_xlim(0, 1); ax.set_ylim(-0.09, 1)
    ax.axis("off")

    ax.text(0.5, 0.995, "Champion architecture: " + _tt("gen031_00")
            + " (pooled science, fitness 0.4458)",
            ha="center", va="top", fontsize=11.5, fontweight="bold", color=INK)

    _box(ax, 0.5, 0.900, 0.60, 0.075,
         r"8 SWOT/ERA5 features $\times$ 3$\times$3 stencil = 72 columns",
         title="Input")
    _arrow(ax, 0.5, 0.849, 0.5, 0.841)

    _box(ax, 0.5, 0.752, 0.82, 0.15,
         "standardize; gradient-preserving mirror imputation of edge NaNs\n"
         "+ per-cell missingness (9); per-feature stencil std (8)\n"
         r"+ central differences $\partial_x,\ \partial_y$; Laplacian; diagonals; $|\nabla|$ (40)" "\n"
         "+ flow regime: speeds, vorticity, strains, Okubo-Weiss, advection (15)",
         title=r"Feature engineering $\rightarrow$ 152 columns")
    _arrow(ax, 0.5, 0.663, 0.5, 0.653)

    _box(ax, 0.5, 0.594, 0.64, 0.09,
         r"exact linear fit on engineered block, clip $\pm$3 m/s",
         title=r"Ridge baseline ($\alpha = 1$)")
    _arrow(ax, 0.30, 0.535, 0.30, 0.514, label="residuals")
    _arrow(ax, 0.70, 0.535, 0.70, 0.514, label="residuals")

    _box(ax, 0.285, 0.393, 0.40, 0.23,
         "stencil image (8 ch + mask)\n"
         r"3$\times$3 conv $\times$2:  9$\rightarrow$32$\rightarrow$48" "\n"
         r"conv proj $\rightarrow$ 128" "\n"
         r"dense path 256$\rightarrow$128" "\n"
         r"fused trunk $\rightarrow$ 128" "\n"
         r"heads: $\mu$, $\log\sigma^2$",
         title="Member 1: ConvWideDeep")
    _box(ax, 0.715, 0.393, 0.40, 0.23,
         "full 152-column vector\n"
         r"MLP 256$\rightarrow$256$\rightarrow$128" "\n"
         "LayerNorm; SiLU; dropout 0.1\n\n\n"
         r"heads: $\mu$, $\log\sigma^2$",
         title="Member 2: HeteroMLP")

    _arrow(ax, 0.285, 0.266, 0.43, 0.243)
    _arrow(ax, 0.715, 0.266, 0.57, 0.243)
    _box(ax, 0.5, 0.190, 0.56, 0.075,
         r"average residual $\mu$  (drop members $>$15% off best val)",
         title="Ensemble")
    _arrow(ax, 0.5, 0.138, 0.5, 0.123, label="+ ridge baseline")

    _box(ax, 0.5, 0.085, 0.34, 0.048, r"$(u, v)$ surface velocity", fill="#e0f2f0",
         edge=ACCENT, title=None)

    _box(ax, 0.5, -0.018, 0.94, 0.105,
         r"anchored heteroscedastic NLL; recency weights (latest 3$\times$); inverse-variance $u$/$v$ weights" "\n"
         "temporal last-10% validation; EMA 0.999 with raw-vs-EMA selection; recency fine-tune\n"
         "fp16 GPU-staged training matrix (the gen031 innovation); 60-epoch cap, 54-min budget",
         title="Training", fs=7.8, fill="white", edge=GRID, tcolor=MUTED)

    _save(fig, out_dir, "champion_architecture_gen031_00")


def architecture_figure_simple(out_dir: Path):
    """Poster version: one idea per box, left-to-right, no fine print."""
    fig, ax = plt.subplots(figsize=(12.0, 3.4))
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.axis("off")

    ax.text(0.5, 1.00, "Champion model " + _tt("gen031_00"),
            ha="center", va="top", fontsize=13, fontweight="bold", color=INK)

    _box(ax, 0.075, 0.5, 0.12, 0.34,
         "SWOT + ERA5\ninputs\n" r"8 feat $\times$ 3$\times$3", fs=10)
    _arrow(ax, 0.149, 0.5, 0.166, 0.5)
    _box(ax, 0.245, 0.5, 0.13, 0.34,
         "physics features\ngradients, vorticity,\nOkubo-Weiss", fs=10)
    _arrow(ax, 0.324, 0.5, 0.346, 0.5)
    _box(ax, 0.425, 0.5, 0.13, 0.34,
         "linear\n(geostrophic)\nridge baseline", fs=10)

    _arrow(ax, 0.504, 0.56, 0.536, 0.66)
    _arrow(ax, 0.504, 0.44, 0.536, 0.34)
    ax.text(0.497, 0.27, "residuals", fontsize=8.5, color=MUTED,
            ha="center", style="italic")
    _box(ax, 0.61, 0.72, 0.12, 0.30, "CNN\nspatial patterns", fs=10)
    _box(ax, 0.61, 0.28, 0.12, 0.30, "MLP\nfeature\ncombinations", fs=10)

    _arrow(ax, 0.684, 0.66, 0.713, 0.545)
    _arrow(ax, 0.684, 0.34, 0.713, 0.455)
    _box(ax, 0.78, 0.5, 0.105, 0.34, "average\n+ baseline\n(errors cancel)", fs=10)
    _arrow(ax, 0.845, 0.5, 0.861, 0.5)
    _box(ax, 0.925, 0.5, 0.10, 0.34, r"$(u, v)$" "\nsurface\nvelocity",
         fs=10, fill="#e0f2f0", edge=ACCENT)

    ax.text(0.5, 0.02,
            "architecture discovered by LLM-guided code evolution "
            "(141 candidates, 35 generations, temporal-holdout selection)",
            ha="center", va="top", fontsize=9.5, color=MUTED)
    _save(fig, out_dir, "champion_architecture_simple")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="evolve_all_science_v1")
    args = ap.parse_args()
    out_dir = REPO_ROOT / "docs" / "figures" / "poster"
    out_dir.mkdir(parents=True, exist_ok=True)
    fitness_figure(REPO_ROOT / "experiments" / "evolve" / args.run, out_dir, args.run)
    architecture_figure(out_dir)
    architecture_figure_simple(out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
