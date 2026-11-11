"""Paper figures for the Phase-1 baselines (docs/figures/science/ or docs/figures/calval/, by the run's mission).

Reads experiments/evolve/<run>/baselines/results.json (from
scripts/baselines_eval.py), the frozen data.npz and the replayed row
metadata. Writes PNG (300 dpi) + SVG, white background, Computer Modern,
same tokens as scripts/poster_figures.py. See docs/figures/science/README.md
for what each figure shows.

    python scripts/baselines_figures.py --run evolve_all_science_v1   # -> docs/figures/science/
    python scripts/baselines_figures.py --run evolve_uswc_v1          # -> docs/figures/calval/
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures" / "science"      # set per run in main(); hfr_rotary_spectrum.py overrides it
# per-mission protocol and caption text; REGIONS is narrowed in main() to those present in results.json
MISSION = {
    "science": dict(protocol="buffer30d/swot_valid", buffer="30 days",
                    window="cycles 15-16, Apr-Jun 2024", train="training cycles 1-12"),
    "calval":  dict(protocol="buffer7d/swot_valid", buffer="7 days (the record ends 3 weeks after training)",
                    window="26 Jun-10 Jul 2023", train="the calval training period (Apr-19 Jun 2023)"),
}
CHAMP_LABEL = {
    "science": {"champion_pooled": "AlphaEvolve champion MLP, pooled\n72 stencil cols, nonlinear",
                "champion_solo": "AlphaEvolve champion MLP, region-only\n72 stencil cols, nonlinear"},
    "calval":  {"champion_pooled": "AlphaEvolve champion MLP, uswc campaign\n81 stencil cols incl. SST, nonlinear",
                "champion_solo": "AlphaEvolve champion MLP, retrained per region\n81 cols incl. SST, mean of 3 seeds"},
}
CHAMP_NOTES = {
    "science": ("AlphaEvolve champion = candidate $\\mathtt{gen031\\_00}$ of run $\\mathtt{evolve\\_all\\_science\\_v1}$ "
                "(ridge-residual MLP evolved by LLM-guided search; pooled row uses its campaign predictions, "
                "retrain row the same code trained on one region)."),
    "calval":  ("AlphaEvolve champion = candidate $\\mathtt{gen020\\_02}$ of run $\\mathtt{evolve\\_uswc\\_v1}$ "
                "(1-day repeat, Apr-Jul 2023; evolved on the US West Coast with GOES SST as an input). uswc row uses its "
                "campaign predictions; usegc and ushi rows are the same code retrained on each region, mean of seeds 42-44."),
}
MIS = MISSION["science"]

INK, MUTED, GRID = "#1a1a2e", "#5a6472", "#d9dee5"
C_U, C_V = "#0e7490", "#ea580c"          # u teal / v orange (CVD-safe blue-orange axis)
C_NEUTRAL = "#8a93a0"
plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white", "savefig.facecolor": "white",
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "STIXGeneral", "Times New Roman"],
    "mathtext.fontset": "cm", "axes.unicode_minus": False,
    "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.linewidth": 0.8,
    "xtick.labelsize": 8.5, "ytick.labelsize": 8.5, "axes.labelsize": 9.5, "axes.titlesize": 10,
    "legend.fontsize": 8.5, "legend.frameon": False,
})
REGIONS = ["usegc", "uswc", "prvi", "ushi"]          # ladder order; narrowed per run in main()
REGION_NAME = {"usegc": "US East & Gulf Coast", "uswc": "US West Coast",
               "prvi": "Puerto Rico / USVI", "ushi": "Hawaii"}
# Every model predicts the same target (HFR 25-h u, v at the pass point and time);
# the second label line is the right-hand side, fitted on training cycles 1-12.
LADDER = [  # (results key, label)
    ("train_mean", "training-period mean\n$\\bar{y}$ (1 par.)"),
    ("cell_mean", "per-grid-point climatology\n$\\bar{y}(x)$ (1 per cell)"),
    ("cell_mean+wind_lsq", "climatology + instantaneous wind\n$\\bar{y}(x) + (1, W_u, W_v)$"),
    ("duacs_calibrated", "DUACS L4 geostrophy, calibrated\n$1,\\ d$ (2 par.)"),
    ("duacs_cal+wind_lsq", "DUACS + instantaneous wind\n$1,\\ d_u, d_v, W_u, W_v$ (5 par.)"),
    ("geos_calibrated", "SWOT L3 geostrophy, calibrated\n$1,\\ g$ (2 par.)"),
    ("geos_cal+wind_lsq", "SWOT + instantaneous wind\n$1,\\ g_u, g_v, W_u, W_v$ (5 par.)"),
    ("cell_linear_geos_wind", "per-grid-point linear\n$1, g_u, g_v, W_u, W_v$ per cell (5 per cell)"),
    ("ridge72_solo", "ridge on 3x3 stencil\n72 stencil cols, $L_2$ (73 par.)"),
    ("champion_pooled", "AlphaEvolve champion MLP, pooled\n72 stencil cols, nonlinear"),
    ("champion_solo", "AlphaEvolve champion MLP, region-only\n72 stencil cols, nonlinear"),
]
RHS_NOTE = ("Target in every row: HFR 25-h detided $u$, $v$ at the pass point and time. Second label line = predictors "
            "($g$ SWOT L3 geostrophy, $d$ DUACS L4, $W$ ERA5 10-m wind), fitted on training cycles 1-12.")
CHAMP_NOTE = ("AlphaEvolve champion = candidate $\\mathtt{gen031\\_00}$ of run $\\mathtt{evolve\\_all\\_science\\_v1}$ "
              "(ridge-residual MLP evolved by LLM-guided search; pooled row uses its campaign predictions, "
              "retrain row the same code trained on one region).")
TIER_ORDER = [("raw", "raw hourly"), ("bw18", "18-h low-pass"), ("target_25h", "25-h mean (target)"), ("bw48", "48-h low-pass")]


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{name}.{ext}", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  wrote", OUT / f"{name}.png")


def style(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)


# --------------------------------------------------------------------------
def fig_ladder(res, protocol=None):
    protocol = protocol or MIS["protocol"]
    """Skill ladder: R² with 95 % block-bootstrap CI per model, u and v, per region."""
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(3.0 + 2.85 * len(REGIONS), 5.0), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, rid in zip(axes, REGIONS):
        prot = res["regions"][rid]["protocols"][protocol]
        rows = [(k, lab) for k, lab in LADDER if k in prot]
        y = np.arange(len(rows))[::-1]
        for comp, color, dy in (("u", C_U, 0.17), ("v", C_V, -0.17)):
            vals = np.array([prot[k][comp]["r2"] for k, _ in rows])
            lo = np.array([prot[k][comp]["r2_ci"][0] for k, _ in rows])
            hi = np.array([prot[k][comp]["r2_ci"][1] for k, _ in rows])
            ax.hlines(y + dy, np.maximum(lo, -0.6), hi, color=color, lw=1.6, alpha=0.55, zorder=2)
            ax.scatter(vals, y + dy, s=34, color=color, edgecolor="white", lw=1.2, zorder=3,
                       label=f"${comp}$" if rid == REGIONS[0] else None)
        ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
        ax.set_xlim(-0.6, 1.0)
        ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75, 1.0])
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels([lab for _, lab in rows])
        else:
            ax.tick_params(labelleft=False)
        n = prot["train_mean"]["u"]["n"]; nb = prot["train_mean"]["u"]["n_blocks"]
        ax.set_title(f"{REGION_NAME[rid]}\n{n:,} rows, {nb} passes", fontsize=9.5)
        ax.set_xlabel("$R^2$ (held-out, 25-h detided HFR)")
        style(ax)
    axes[0].legend(loc="upper left", title=None, handletextpad=0.2)
    fig.text(0.5, -0.07, RHS_NOTE + "\nMarkers: point estimate; bars: 95 % CI from a block bootstrap over (cycle, pass) snapshots. "
             f"Rows with finite centre-cell SWOT geostrophy, at least {MIS['buffer']} after the last training pass ({MIS['window']}).\n"
             + CHAMP_NOTE,
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(w_pad=1.0)
    save(fig, "skill_ladder")


def fig_rmse_pct(res, protocol=None):
    protocol = protocol or MIS["protocol"]
    """% RMSE change relative to calibrated SWOT geostrophy."""
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(3.0 + 2.85 * len(REGIONS), 4.6), sharey=True, squeeze=False)
    axes = axes[0]
    for ax, rid in zip(axes, REGIONS):
        prot = res["regions"][rid]["protocols"][protocol]
        rows = [(k, lab) for k, lab in LADDER if k in prot and k != "geos_calibrated"]
        y = np.arange(len(rows))[::-1]
        for comp, color, dy in (("u", C_U, 0.19), ("v", C_V, -0.19)):
            vals = np.array([prot[k][comp]["pct_rmse_vs_ref"] for k, _ in rows])
            ax.barh(y + dy, vals, height=0.34, color=color, zorder=3,
                    label=f"${comp}$" if rid == REGIONS[0] else None)
        ax.axvline(0, color=MUTED, lw=0.8, zorder=4)
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels([lab for _, lab in rows])
        else:
            ax.tick_params(labelleft=False)
        ax.set_title(REGION_NAME[rid], fontsize=9.5)
        ax.set_xlabel("RMSE change (%)")
        lim = max(25, np.nanmax(np.abs([prot[k][c]["pct_rmse_vs_ref"] for k, _ in rows for c in "uv"])) * 1.1)
        ax.set_xlim(-lim, lim)
        style(ax)
    axes[0].legend(loc="lower left", handletextpad=0.2)
    fig.text(0.5, -0.07, RHS_NOTE + "\nRMSE change relative to the calibrated SWOT L3 geostrophic velocity on the same rows; negative = better.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(w_pad=1.0)
    save(fig, "rmse_change")


def fig_tiers(res, protocol=None):
    protocol = protocol or MIS["protocol"]
    """Champion and linear baseline skill vs the timescale of the HFR truth."""
    models = [("champion_pooled", CHAMP_LABEL[mission]["champion_pooled"].replace("\n", ", "), "-", "o"),
              ("geos_cal+wind_lsq", "SWOT + wind, linear ($1, g_u, g_v, W_u, W_v$)", "--", "s")]
    # regions where the common subset (rows finite in every tier) is big enough to score
    regs = [r for r in REGIONS if res["regions"][r].get("tiers", {}).get("target_25h", {}).get(protocol, {})
            .get("train_mean", {}).get("u", {}).get("n", 0) >= 1000]
    dropped = [r for r in REGIONS if r not in regs]
    fig, axes = plt.subplots(1, len(regs), figsize=(2.5 + 2.8 * len(regs), 3.6), sharey=True, squeeze=False)
    axes = axes[0]
    x = np.arange(len(TIER_ORDER))
    for ax, rid in zip(axes, regs):
        tiers = res["regions"][rid].get("tiers", {})
        for comp, color in (("u", C_U), ("v", C_V)):
            for key, lab, ls, mk in models:
                vals = [tiers.get(t, {}).get(protocol, {}).get(key, {}).get(comp, {}).get("r2", np.nan) for t, _ in TIER_ORDER]
                ax.plot(x, vals, ls=ls, marker=mk, ms=5.5, lw=1.8, color=color, mec="white", mew=1.0,
                        label=f"${comp}$, {lab}" if rid == regs[0] else None, zorder=3)
        n = tiers.get("target_25h", {}).get(protocol, {}).get("train_mean", {}).get("u", {}).get("n", 0)
        frac = res["regions"][rid].get("tiers_common_frac", 0)
        ax.set_title(f"{REGION_NAME[rid]}\n{n:,} common rows ({frac:.0%})", fontsize=9.5)
        ax.set_xticks(x); ax.set_xticklabels([lab for _, lab in TIER_ORDER], rotation=25, ha="right")
        ax.axhline(0, color=MUTED, lw=0.8, zorder=1)
        ax.set_ylim(-0.3, 0.8)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    axes[0].set_ylabel("$R^2$ (held-out)")
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=2, handlelength=2.2, bbox_to_anchor=(0.5, -0.06), columnspacing=2.0)
    fig.text(0.5, -0.2, "Same predictions scored against four versions of the radar truth, on rows where every tier is defined. "
             f"Both models predict HFR $u$, $v$ at the pass point and time from predictors fitted on {MIS['train']}.\n"
             "Low-pass tiers: order-2 zero-phase Butterworth on the hourly record.\n"
             + CHAMP_NOTE
             + "".join(f"\n{REGION_NAME[r]} omitted: under 1,000 rows common to every tier." for r in dropped),
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(w_pad=1.0)
    save(fig, "truth_tiers")


def fig_overshoot(run_dir, meta_all):
    """Joint density of SWOT L3 geostrophic velocity vs HFR: slope < 1, amplitude > 1."""
    F = meta_all["features"]; col = lambda f: F.index(f) * 9 + 4
    d = np.load(run_dir / "data.npz"); X, Y = d["X_test"], d["Y_test"]
    regs = ["usegc", "uswc"]
    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.0))
    for i, rid in enumerate(regs):
        idx = np.concatenate([np.arange(s, e) for s, e in meta_all["region_test_slices"][rid]])
        for j, (f, comp) in enumerate([("ugos_filtered", 0), ("vgos_filtered", 1)]):
            ax = axes[i, j]
            g, y = X[idx, col(f)], Y[idx, comp]
            ok = np.isfinite(g) & np.isfinite(y); g, y = g[ok], y[ok]
            lim = 1.0
            hb = ax.hexbin(g, y, gridsize=55, extent=(-lim, lim, -lim, lim), bins="log",
                           cmap="Blues", mincnt=1, linewidths=0.1)
            b = np.cov(g, y)[0, 1] / g.var(); a = y.mean() - b * g.mean()
            xx = np.array([-lim, lim])
            ax.plot(xx, xx, color=MUTED, lw=1.0, ls=":", label="1:1")
            ax.plot(xx, a + b * xx, color=C_U if comp == 0 else C_V, lw=2.0,
                    label=f"least squares, slope {b:.2f}")
            r = np.corrcoef(g, y)[0, 1]
            ax.text(0.03, 0.97, f"$r$ = {r:.2f}\n$\\sigma$ ratio {g.std()/y.std():.2f}\nn = {ok.sum():,}",
                    transform=ax.transAxes, va="top", fontsize=8.3, color=INK)
            ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim); ax.set_aspect("equal")
            comp_s = "u" if comp == 0 else "v"
            ax.set_xlabel(f"SWOT L3 ${comp_s}_{{gos}}$ (m s$^{{-1}}$)")
            ax.set_ylabel(f"HFR ${comp_s}$, 25-h mean (m s$^{{-1}}$)")
            ax.set_title(f"{REGION_NAME[rid]}, ${comp_s}$", fontsize=9.5)
            ax.legend(loc="lower right", fontsize=7.8)
            for s in ("top", "right"):
                ax.spines[s].set_visible(False)
    fig.text(0.5, -0.01, "Held-out rows with finite SWOT geostrophy; colour = log count. "
             "The regression slope is the calibration applied in the baselines.", ha="center", fontsize=8, color=MUTED)
    fig.tight_layout()
    save(fig, "geostrophy_overshoot")


def fig_coverage(run_dir, meta_all):
    """Where the comparison happens: held-out radar cells with / without SWOT geostrophy."""
    F = meta_all["features"]; col = lambda f: F.index(f) * 9 + 4
    d = np.load(run_dir / "data.npz"); X = d["X_test"]
    fig, axes = plt.subplots(1, len(REGIONS), figsize=(11.5, 3.8), layout="constrained")
    for ax, rid in zip(axes, REGIONS):
        meta = np.load(run_dir / "baselines" / f"meta_{rid}.npz", allow_pickle=True)
        idx = np.concatenate([np.arange(s, e) for s, e in meta_all["region_test_slices"][rid]])
        ok = np.isfinite(X[idx, col("ugos_filtered")]).astype(float)
        lat, lon = meta["lat"], meta["lon"]
        # per radar cell (rows repeat across passes): fraction of held-out passes with SWOT geostrophy
        key = np.round(np.column_stack([lon, lat]), 3)
        cells, inv = np.unique(key, axis=0, return_inverse=True)
        frac = np.bincount(inv, ok) / np.bincount(inv)
        sc = ax.scatter(cells[:, 0], cells[:, 1], c=frac, s=1.4, cmap="Blues", vmin=0, vmax=1, lw=0,
                        rasterized=True)
        ax.set_title(f"{REGION_NAME[rid]}\nSWOT-valid rows {ok.mean():.0%}, {len(cells):,} grid points", fontsize=9.5)
        ax.set_aspect(1 / np.cos(np.deg2rad(np.nanmean(lat))))
        ax.set_xlabel("longitude ($^\\circ$E)"); ax.set_ylabel("latitude ($^\\circ$N)" if ax is axes[0] else "")
        ax.grid(color=GRID, lw=0.6); ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    cb = fig.colorbar(sc, ax=axes, fraction=0.025, pad=0.02)
    cb.set_label("fraction of held-out passes with SWOT L3 geostrophy at the cell", fontsize=8.5)
    cb.outline.set_visible(False)
    fig.text(0.5, -0.04, "Held-out radar grid points on the swath-aligned SWOT grids (test cycles 13-16). Light points sit under the nadir gap, swath edges or the coastal mask of the L3 filtered product.",
             ha="center", fontsize=8, color=MUTED)
    save(fig, "coverage_map")


# --------------------------------------------------------------------------
GRID_MODELS = [  # (results key, "label:  predictors") - every model baselines_eval.py scores, ladder order
    ("train_mean", "training-period mean:  $\\bar{y}$"),
    ("cell_mean", "per-grid-point climatology:  $\\bar{y}(x)$"),
    ("cell_mean+wind_lsq", "climatology + wind, linear:  $\\bar{y}(x) + (1, W_u, W_v)$"),
    ("wind_only_lsq", "wind only, linear:  $1, W_u, W_v$"),
    ("swot_geos", "SWOT L3 geostrophy, raw:  $g$ (slope 1)"),
    ("geos+ekman_3pct45deg", "SWOT + Ekman:  $g + 0.03|W|$ at 45 deg"),
    ("geos+wind_lsq", "SWOT + wind, linear:  $g + (1, W_u, W_v)$"),
    ("geos_calibrated", "SWOT, calibrated:  $1, g$"),
    ("geos_cal+wind_lsq", "SWOT calibrated + wind:  $1, g_u, g_v, W_u, W_v$"),
    ("cell_linear_geos_wind", "per-cell linear:  $1, g_u, g_v, W_u, W_v$ per cell"),
    ("duacs_geos", "DUACS L4 geostrophy, raw:  $d$ (slope 1)"),
    ("duacs_calibrated", "DUACS, calibrated:  $1, d$"),
    ("duacs_cal+wind_lsq", "DUACS calibrated + wind:  $1, d_u, d_v, W_u, W_v$"),
    ("ridge72_solo", "ridge, 72 stencil cols, $L_2$"),
    ("champion_pooled", "AlphaEvolve MLP, pooled:  72 cols, nonlinear"),
    ("champion_solo", "AlphaEvolve MLP, region-only:  72 cols, nonlinear"),
]
GRID_TIERS = [("MAIN", "25-h target" + chr(10) + "(all rows)"), ("raw", "raw hourly" + chr(10) + "(common rows)"),
              ("bw18", "18-h low-pass" + chr(10) + "(common rows)"), ("target_25h", "25-h mean" + chr(10) + "(common rows)"),
              ("bw48", "48-h low-pass" + chr(10) + "(common rows)")]
GRID_PROTOCOLS = [("all_test/all_rows", "all / all"), ("all_test/swot_valid", "all / SWOT"),
                  ("buffer30d/all_rows", "30 d / all"), ("buffer30d/swot_valid", "30 d / SWOT")]   # science; swapped in main() for calval
GRID_REGIONS = ["usegc", "uswc", "prvi", "ushi", "gak", "glna"]                                   # narrowed in main()
GRID_REGION_NAME = dict(REGION_NAME, gak="Gulf of Alaska", glna="Great Lakes")


def fig_benchmark_grid(res, comp="u"):
    """Every scored (region, truth tier, protocol, model) cell as one heatmap tile."""
    from matplotlib.colors import TwoSlopeNorm, ListedColormap
    from matplotlib.patches import Rectangle
    cols = [(t, pk) for t, _ in GRID_TIERS for pk, _ in GRID_PROTOCOLS]
    nM, nC = len(GRID_MODELS), len(cols)
    cmap = plt.get_cmap("RdBu").copy(); cmap.set_bad("#eef0f3")
    norm = TwoSlopeNorm(vmin=-0.5, vcenter=0.0, vmax=0.7)
    fig, axes = plt.subplots(len(GRID_REGIONS), 1, figsize=(10.5, 2.6 * len(GRID_REGIONS)), sharex=True, squeeze=False)
    axes = axes[:, 0]
    total = 0
    for ax, rid in zip(axes, GRID_REGIONS):
        R = res["regions"][rid]
        M = np.full((nM, nC), np.nan)
        for j, (tier, pk) in enumerate(cols):
            block = R["protocols"].get(pk) if tier == "MAIN" else R.get("tiers", {}).get(tier, {}).get(pk)
            if not block:
                continue
            if tier != "MAIN" and block.get("train_mean", {}).get("u", {}).get("n", 0) < 1000:
                continue        # tier subset too small to score (ushi calval: 127 common rows)
            for i, (mk, _) in enumerate(GRID_MODELS):
                if mk in block:
                    M[i, j] = block[mk][comp]["r2"]
        n_cells = int(np.isfinite(M).sum()); total += n_cells
        ax.imshow(np.ma.masked_invalid(np.clip(M, -0.5, 0.7)), cmap=cmap, norm=norm, aspect="auto",
                  interpolation="nearest")
        for i in range(nM):
            for j in range(nC):
                if np.isfinite(M[i, j]):
                    v = M[i, j]
                    ax.text(j, i, f"{v:.2f}" if v > -1 else "$<$-1", ha="center", va="center", fontsize=5.6,
                            color="white" if (v > 0.45 or v < -0.3) else INK)
        # vertical separators between tiers, box around the headline column
        for t in range(1, len(GRID_TIERS)):
            ax.axvline(t * len(GRID_PROTOCOLS) - 0.5, color="white", lw=2.5)
        ax.add_patch(Rectangle((3 - 0.5, -0.5), 1, nM, fill=False, ec=INK, lw=1.4, zorder=5))
        ax.set_yticks(range(nM)); ax.set_yticklabels([lab for _, lab in GRID_MODELS], fontsize=7.5)
        ax.set_xticks(range(nC)); ax.set_xticklabels([pl for _ in GRID_TIERS for _, pl in GRID_PROTOCOLS],
                                                     rotation=60, ha="right", fontsize=7)
        ax.tick_params(length=0)
        for s in ax.spines.values():
            s.set_visible(False)
        ax.set_title(f"{GRID_REGION_NAME[rid]}:  {n_cells} scored cells, "
                     f"{R['n_test_rows']:,} test rows, SWOT-valid {100*R['swot_valid_frac']:.0f} %",
                     loc="left", fontsize=9, pad=26 if ax is axes[0] else 6)
    # tier headers above the top panel
    top = axes[0]
    for t, (_, tl) in enumerate(GRID_TIERS):
        top.text(t * len(GRID_PROTOCOLS) + 1.5, -0.9, tl, ha="center", va="bottom", fontsize=8, color=INK,
                 transform=top.transData, clip_on=False)
    fig.subplots_adjust(left=0.27, right=0.9, top=0.945 if len(GRID_REGIONS) > 3 else 0.9,
                        bottom=0.07 if len(GRID_REGIONS) > 3 else 0.17, hspace=0.32)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cb = fig.colorbar(sm, cax=fig.add_axes([0.915, 0.35, 0.012, 0.3]))
    cb.set_label(f"$R^2$ ({comp}), clipped to [-0.5, 0.7]", fontsize=9); cb.outline.set_visible(False)
    fig.suptitle(f"The full benchmark grid, {comp} component: {total} scored cells "
                 f"(region x truth tier x protocol x model)", y=0.985, fontsize=11)
    fig.text(0.5, 0.005,
             f"Columns: truth tier (header) x row protocol (test cycles: all or buffered by {MIS['buffer'].split(' (')[0]} / rows: all or "
             "centre-cell SWOT geostrophy finite). Black box: the paper headline protocol. "
             "Grey: not scored (no SWOT rows, DUACS not finite, no hourly HFR for tiers, or under 1,000 rows common to every tier). "
             "Rows: model label, then its predictors (g SWOT L3 geostrophy, d DUACS L4, W ERA5 wind; target always HFR u, v). "
             "Each cell also carries a bootstrap CI, RMSE, r, amplitude ratio and N_eff in results.json.",
             ha="center", fontsize=7.5, color=MUTED, wrap=True)
    save(fig, f"benchmark_grid_{comp}")


# --------------------------------------------------------------------------
def fig_two_orbits(res_sci, res_cal):
    """Same radar (US West Coast), two SWOT orbits: 21-day science repeat vs 1-day calval repeat."""
    panels = [("science (21-day repeat), test Apr-Jun 2024", res_sci["regions"]["uswc"]["protocols"]["buffer30d/swot_valid"],
               "AlphaEvolve champion MLP\n$\\mathtt{gen031\\_00}$, pooled, 72 cols"),
              ("calval (1-day repeat), test 26 Jun-10 Jul 2023", res_cal["regions"]["uswc"]["protocols"]["buffer7d/swot_valid"],
               "AlphaEvolve champion MLP\n$\\mathtt{gen020\\_02}$, uswc, 81 cols incl. SST")]
    rows_common = [(k, lab) for k, lab in LADDER if k not in ("champion_pooled", "champion_solo")]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 5.2), sharey=True)
    for ax, (title, prot, champ_lab) in zip(axes, panels):
        rows = [(k, lab) for k, lab in rows_common if k in prot] + [("champion_pooled", champ_lab)]
        y = np.arange(len(rows))[::-1]
        for comp, color, dy in (("u", C_U, 0.17), ("v", C_V, -0.17)):
            vals = np.array([prot[k][comp]["r2"] for k, _ in rows])
            lo = np.array([prot[k][comp]["r2_ci"][0] for k, _ in rows])
            hi = np.array([prot[k][comp]["r2_ci"][1] for k, _ in rows])
            ax.hlines(y + dy, np.maximum(lo, -0.6), hi, color=color, lw=1.6, alpha=0.55, zorder=2)
            ax.scatter(vals, y + dy, s=34, color=color, edgecolor="white", lw=1.2, zorder=3,
                       label=f"${comp}$" if ax is axes[0] else None)
        ax.axvline(0, color=MUTED, lw=0.8, zorder=1)
        ax.set_xlim(-0.6, 0.8); ax.set_xticks([-0.5, -0.25, 0, 0.25, 0.5, 0.75])
        ax.set_yticks(y)
        if ax is axes[0]:
            ax.set_yticklabels([lab for _, lab in rows])
        else:
            ax.tick_params(labelleft=False)
            # the champion label differs between panels: write it beside the last row
            ax.text(-0.58, y[-1], champ_lab, ha="left", va="center", fontsize=7.5, color=MUTED)
        n = prot["train_mean"]["u"]["n"]; nb = prot["train_mean"]["u"]["n_blocks"]
        ax.set_title(f"US West Coast, {title}\n{n:,} rows, {nb} passes", fontsize=9.5)
        ax.set_xlabel("$R^2$ (held-out, 25-h detided HFR)")
        style(ax)
    axes[0].legend(loc="lower right", handletextpad=0.2)
    fig.text(0.5, -0.06, RHS_NOTE.replace("fitted on training cycles 1-12", "fitted on each orbit's training period")
             + "\nSame radar, two orbits. Buffer after the last training pass: 30 days (science), 7 days "
             "(calval; the record ends 3 weeks after training). Bars: 95 % block-bootstrap CI over passes.",
             ha="center", fontsize=8, color=MUTED)
    fig.tight_layout(w_pad=1.0)
    save(fig, "uswc_two_orbits")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="evolve_all_science_v1")
    ap.add_argument("--calval-run", default="evolve_uswc_v1", help="single-region calval run for uswc_two_orbits")
    a = ap.parse_args()
    run_dir = ROOT / "experiments" / "evolve" / a.run
    res = json.load(open(run_dir / "baselines" / "results.json"))
    meta_all = json.load(open(run_dir / "data_meta.json"))
    mission = meta_all.get("mission", "science")
    MIS = MISSION[mission]
    OUT = ROOT / "docs" / "figures" / mission
    CHAMP_NOTE = CHAMP_NOTES[mission]
    RHS_NOTE = RHS_NOTE.replace("fitted on training cycles 1-12", "fitted on " + MIS["train"])
    if mission == "calval":
        # one champion row everywhere: the campaign predictions where they exist, else the retrained twin
        for rid, R in res["regions"].items():
            for blocks in [R["protocols"]] + [t for t in R.get("tiers", {}).values()]:
                for prot in blocks.values():
                    if "champion_pooled" not in prot and "champion_solo" in prot:
                        prot["champion_pooled"] = prot.pop("champion_solo")
                    else:
                        prot.pop("champion_solo", None)
        CHAMP_LABEL["calval"] = {"champion_pooled": "AlphaEvolve champion MLP\n81 cols incl. SST (uswc: campaign; usegc, ushi: retrained, 3 seeds)",
                                 "champion_solo": "unused"}
    LADDER = [(k, CHAMP_LABEL[mission].get(k, lab)) for k, lab in LADDER]
    GRID_MODELS = [(k, CHAMP_LABEL[mission][k].replace("\n", ":  ") if k in CHAMP_LABEL[mission] else lab) for k, lab in GRID_MODELS]
    GRID_MODELS = [m for m in GRID_MODELS if m[1] != "unused"]
    present = [r for r in GRID_REGIONS if r in res["regions"]]
    REGIONS = [r for r in REGIONS if r in present]
    GRID_REGIONS = present
    if mission == "calval":
        GRID_PROTOCOLS = [(k.replace("buffer30d", "buffer7d"), l.replace("30 d", "7 d")) for k, l in GRID_PROTOCOLS]
    # only tier columns that hold data somewhere (calval has no hourly-archive tiers yet)
    GRID_TIERS = [t for t in GRID_TIERS if t[0] == "MAIN" or any(res["regions"][r].get("tiers", {}).get(t[0]) for r in present)]
    print(f"run {a.run}: mission {mission}, regions {REGIONS}, protocol {MIS['protocol']} -> {OUT}")
    fig_ladder(res)
    fig_rmse_pct(res)
    if any(res["regions"][r].get("tiers") for r in REGIONS):
        fig_tiers(res)
    if "region_test_slices" in meta_all:          # pooled science run only
        fig_overshoot(run_dir, meta_all)
        fig_coverage(run_dir, meta_all)
    fig_benchmark_grid(res, "u")
    fig_benchmark_grid(res, "v")
    cal = run_dir.parent / a.calval_run / "baselines" / "results.json"
    if mission == "science" and cal.exists():
        fig_two_orbits(res, json.load(open(cal)))
