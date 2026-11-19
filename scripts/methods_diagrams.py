"""Schematic diagrams of the SWOTxAI methods, one figure each, for talks and the paper.

    python scripts/methods_diagrams.py            # writes docs/figures/methods/*.png and *.svg

Figures (docs/figures/methods/README.md explains each in prose):
    method_rows            one training row: the 3x3 SWOT stencil, the wind, the extras, the network, the target
    method_daily_target    Phase 1: one row per cell per pass against the 25-hour-mean radar current
    method_wind_history    Phase 2: lagged winds, f and season added as columns, same daily target (null result)
    method_hourly_target   Experiment 3: 25 rows per pass against the detided hourly current, wind at the target hour
    method_hourly_to_daily the hourly route scored as a daily product, and the pass-hour check
    method_holdout         the evaluation protocol: train on cycles 001-016, score 017-058; continents and leave-one-out
Nothing is read from data; every figure is drawn from the constants of the methods.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import holdout_eval as he  # noqa: E402

FIG = he.REPO / "docs" / "figures" / "methods"
BLUE, PINK, ORANGE, GREEN, GREY = "#0072B2", "#CC79A7", "#D55E00", "#009E73", "0.45"
plt = he._mpl()
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle  # noqa: E402


def save(fig, name: str) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.png", dpi=250, bbox_inches="tight")
    fig.savefig(FIG / f"{name}.svg", bbox_inches="tight")
    he.log(f"wrote {FIG / name}.png/.svg")


def box(ax, x, y, w, h, text, color="0.3", fc="white", fs=8, lw=1.0, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                ec=color, fc=fc, lw=lw, ls=ls))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color="k")


def arrow(ax, p, q, color="0.3", lw=1.0, style="-|>"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle=style, mutation_scale=10, color=color, lw=lw))


def timeline(ax, y, lo=-12, hi=12, label="hours from the SWOT pass", ticks=(-12, -6, 0, 6, 12)):
    ax.plot([lo, hi], [y, y], color="k", lw=0.9)
    for t in ticks:
        ax.plot([t, t], [y - 0.12, y + 0.12], color="k", lw=0.9)
        ax.text(t, y - 0.3, f"{t:+d}" if t else "0", ha="center", va="top", fontsize=7.5)
    ax.text((lo + hi) / 2, y - 0.75, label, ha="center", va="top", fontsize=8, color=GREY)


def stencil(ax, x, y, s=0.42, color=BLUE):
    for i in range(3):
        for j in range(3):
            ax.add_patch(Rectangle((x + i * s, y + j * s), s, s, ec=color, fc=color if (i, j) == (1, 1) else "white",
                                   lw=0.9, alpha=0.9 if (i, j) == (1, 1) else 1))
    ax.text(x + 1.5 * s, y - 0.18, "SWOT stencil\n$3\\times3$ cells", ha="center", va="top", fontsize=7.5, color=color)


# --------------------------------------------------------------------------
def fig_rows() -> None:
    """One row: what the network sees and what it is asked for."""
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.set_xlim(0, 10); ax.set_ylim(0, 4.2); ax.axis("off")
    box(ax, 0.3, 2.6, 2.3, 1.3, "SWOT Level-3 at the pass\n$3\\times3$ cells $\\times$ 8 fields = 72 columns\n"
        "mdt, ssha, $u_g$, $v_g$, $u_{ga}$, $v_{ga}$, era5$_u$, era5$_v$", color=BLUE, fs=7.5)
    stencil(ax, 0.55, 0.7)
    ax.text(2.0, 1.3, "centre cell is the\ntarget cell; the 8\nneighbours give\nspatial context", fontsize=7, color=GREY, va="center")
    box(ax, 3.0, 2.6, 1.7, 1.3, "extra columns\n(by experiment)\nwind lags, slab, $f$,\nseason, offset $k$", color=ORANGE, fs=7.5)
    box(ax, 3.0, 0.9, 1.7, 1.1, "engineered block\nspeed, angle,\nvorticity of the\nstencil", color=GREY, fs=7.5)
    box(ax, 5.3, 1.4, 2.0, 2.0, "learned model\n(the evolved champion:\nconvolutional path over\nthe stencil + MLP;\nmasked squared error;\nvalidation on the last\n10 % of passes)", color="0.2", fc="0.96", fs=7.3)
    arrow(ax, (2.6, 3.25), (3.0, 3.25)); arrow(ax, (4.7, 3.25), (5.3, 3.0)); arrow(ax, (4.7, 1.45), (5.3, 1.8))
    arrow(ax, (1.85, 2.6), (1.85, 2.0), color=BLUE)
    box(ax, 7.9, 2.5, 1.8, 0.9, "prediction\n$(\\hat u, \\hat v)$ at the target cell", color=GREEN, fs=8)
    box(ax, 7.9, 0.9, 1.8, 0.9, "truth: HF radar\ncurrent at the target\ncell and hour(s)", color=PINK, fs=7.5)
    arrow(ax, (7.3, 2.6), (7.9, 2.9), color=GREEN)
    ax.annotate("", (8.8, 1.8), (8.8, 2.5), arrowprops=dict(arrowstyle="<->", color="0.3", lw=1))
    ax.text(9.0, 2.15, "loss, $R^2$", fontsize=8, va="center")
    ax.text(5.0, 0.3, "one row = one radar cell at one SWOT pass (and, in the hourly experiment, one target hour)",
            ha="center", fontsize=8, color=GREY)
    fig.suptitle("What one training row contains", fontsize=10)
    save(fig, "method_rows")


# --------------------------------------------------------------------------
def fig_daily() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.6))
    ax.set_xlim(-16, 26); ax.set_ylim(-1.6, 3.4); ax.axis("off")
    timeline(ax, 0)
    ax.axvline(0, color=BLUE, lw=1.2, ymin=0.32, ymax=0.7)
    ax.text(-0.5, 2.0, "SWOT pass at $t_0$:\nheight stencil + wind at $t_0$", ha="right", va="bottom", fontsize=8, color=BLUE)
    ax.add_patch(Rectangle((-12.5, 0.35), 25, 0.5, fc=PINK, ec="none", alpha=0.35))
    ax.text(0, 1.0, "target: radar current averaged over the 25 hours $t_0 - 12 \\ldots t_0 + 12$\n(one number per cell per pass)",
            ha="center", va="bottom", fontsize=8, color=PINK)
    t = np.linspace(-12, 12, 200)
    ax.plot(t, 0.6 + 0.18 * np.sin(2 * np.pi * t / 19) + 0.06 * np.sin(2 * np.pi * t / 12.42), color=PINK, lw=0.8, alpha=0.8)
    ax.text(12.8, 0.6, "hourly radar current\n(inertial + tidal motion)\naverages to the box mean", fontsize=7, color=PINK, va="center")
    box(ax, 17.5, 1.9, 8.0, 1.2, "learned model\n72 stencil columns $\\rightarrow (\\hat u, \\hat v)$", color="0.2", fc="0.96", fs=8)
    arrow(ax, (0.4, 2.5), (17.4, 2.5), color=BLUE)
    arrow(ax, (19.5, 2.0), (12.6, 1.0), color=GREEN)
    ax.text(21.5, 1.5, "prediction scored\nagainst the box mean", fontsize=7, color=GREEN, va="center", ha="center")
    ax.text(-15.5, 3.3, "Phase 1 (the daily-target model): one row per cell per pass", fontsize=9, va="top")
    ax.text(-15.5, -1.2, "The 25-hour boxcar has its first null at 0.96 cycles per day, on the inertial band at 35 to 40$^\\circ$N:\n"
            "the target keeps the slow flow and discards most of the wind-driven oscillation.", fontsize=7.5, color=GREY, va="top")
    save(fig, "method_daily_target")


# --------------------------------------------------------------------------
def fig_wind_history() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.set_xlim(-15.5, 5.5); ax.set_ylim(-1.9, 3.4); ax.axis("off")
    lags = [192, 96, 48, 24, 18, 12, 6, 3, 0]
    x = lambda L: -np.sqrt(L)                      # square-root axis so the short lags do not pile up
    ax.plot([x(230), 2.5], [0, 0], color="k", lw=0.9)
    for i, L in enumerate(lags):
        ax.plot([x(L), x(L)], [-0.12, 0.12], color="k", lw=0.9)
        ax.text(x(L), -0.3 - 0.35 * (i % 2 if L < 30 else 0), f"$-{L}$ h" if L else "$t_0$", ha="center", va="top", fontsize=7)
        ax.plot(x(L), 0.55, marker="v", color=ORANGE, ms=6)
    ax.text(x(60), -1.1, "ERA5 wind read at the pass time minus each lag (8 lags back to 8 days), two columns each\n"
            "(axis: square root of the lag)", ha="center", fontsize=8, color=ORANGE, va="top")
    ax.axvline(0, color=BLUE, lw=1.2, ymin=0.36, ymax=0.72)
    ax.text(0.15, 1.9, "SWOT stencil at $t_0$", fontsize=8, color=BLUE, va="bottom")
    ax.add_patch(Rectangle((-1.2, 0.9), 2.4, 0.4, fc=PINK, ec="none", alpha=0.35))
    ax.text(1.4, 1.1, "same 25-hour-\nmean target", fontsize=7.5, color=PINK, va="center")
    box(ax, -15.3, 1.7, 5.2, 1.2, "+ Coriolis parameter $f$\n+ season ($\\sin, \\cos$ of day of year)", color=ORANGE, fs=7.5)
    box(ax, -9.3, 1.7, 5.2, 1.2, "feature sets: $o$ (none), $w$ (lags),\n$fs$ ($f$, season), $fsw$ (all)", color=GREY, fs=7.5)
    ax.text(-15.3, 3.3, "Phase 2 (wind history on the daily target): extra columns on the same rows", fontsize=9, va="top")
    ax.text(-15.3, -1.55, "Result: no change in $R^2$ inside the bootstrap intervals, three seeds, both coasts. The response function\n"
            "(finite-difference Jacobian, +2 m/s on one lag at a time) is physical, 1.2 % of wind speed at 30 to 40$^\\circ$ right,\n"
            "70 to 80 % of it at lag 0: the target has little of the lagged motion left to explain.", fontsize=7.5, color=GREY, va="top")
    save(fig, "method_wind_history")


# --------------------------------------------------------------------------
def fig_hourly() -> None:
    fig = plt.figure(figsize=(10, 6.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.7, 1.3], width_ratios=[1.1, 1], hspace=0.08, wspace=0.15)
    ax = fig.add_subplot(gs[0, :]); ins = fig.add_subplot(gs[1, 1]); note = fig.add_subplot(gs[1, 0])
    ax.set_xlim(-66, 46); ax.set_ylim(-1.3, 7.4); ax.axis("off"); note.axis("off")
    ax.text(-66, 7.3, "Experiment 3 (the hourly target): 25 rows per cell per pass, each asked for a different hour", fontsize=9, va="top")
    ax.plot([-62, 14], [0, 0], color="k", lw=0.9)
    for t in range(-60, 15, 12):
        ax.plot([t, t], [-0.12, 0.12], color="k", lw=0.9); ax.text(t, -0.3, f"{t:+d}" if t else "$t_0$", ha="center", va="top", fontsize=7.5)
    ax.text(-24, -0.95, "hours from the SWOT pass", ha="center", fontsize=8, color=GREY)
    ax.axvline(0, color=BLUE, lw=1.2, ymin=0.2, ymax=0.78)
    ax.text(0.8, 6.9, "SWOT stencil at $t_0$ (the only satellite input, shared by all 25 rows)", fontsize=7.5, color=BLUE, va="top")
    rows = [(-12, 4.9, "row $k=-12$"), (0, 3.4, "row $k=0$"), (9, 1.9, "row $k=+9$")]
    for k, y, lab in rows:
        ax.add_patch(Rectangle((k - 48, y - 0.22), 48, 0.44, fc=ORANGE, ec="none", alpha=0.25))
        for L in [0, 3, 6, 9, 12, 15, 18, 21, 24, 36, 48]:
            ax.plot(k - L, y, marker="v", color=ORANGE, ms=4)
            if k == 0:
                ax.text(k - L, y + 0.3, str(L), fontsize=6, color=ORANGE, ha="center", va="bottom")
        if k == 0:
            ax.text(k - 54, y + 0.3, "lag (h):", fontsize=6, color=ORANGE, ha="right", va="bottom")
        ax.plot(k, y, marker="s", color=PINK, ms=7)
        ax.plot([k, k], [0.15, y - 0.3], color=PINK, lw=0.6, ls=":")
        ax.text(16, y, f"{lab}: target = detided radar current at $t_0{k:+d}$ h\ninputs = offset $k$; ERA5 wind at 11 lags before $t_0{k:+d}$ h\n(0, 3, $\\ldots$, 24, 36, 48 h; two columns each); slab current at $t_0{k:+d}$ h",
                fontsize=7, va="center")
    ax.text(-62, 5.75, "wind window of each row: 11 lags ending at the target hour, nothing later enters. Every 3 h to 24 h\n"
            "(about six samples per inertial period of 18 to 21 h), then 36 and 48 h (slab damping time 4 d).", fontsize=7.3, color=ORANGE, va="bottom")
    for k in range(-12, 13, 3):
        ax.plot(k, 0.55, marker="s", color=PINK, ms=4, alpha=0.7)
    ax.text(16, 0.55, "nine target hours per pass ($-12, -9, \\ldots, +12$ h;\noffsets $\\neq 0$ on 1/8 of the base rows)",
            fontsize=7.5, color=PINK, va="center", ha="left")
    # detiding inset
    t = np.linspace(0, 72, 400)
    tide = 0.15 * np.sin(2 * np.pi * t / 12.42) + 0.08 * np.sin(2 * np.pi * t / 23.93 + 1)
    raw = tide + 0.12 * np.exp(-t / 60) * np.sin(2 * np.pi * t / 19) + 0.05
    ins.plot(t, raw, color="0.6", lw=0.9, label="raw hourly radar")
    ins.plot(t, tide, color=GREY, lw=0.8, ls="--", label="harmonic fit (M$_2$ S$_2$ N$_2$ K$_1$ O$_1$, training year)")
    ins.plot(t, raw - tide, color=PINK, lw=1.1, label="detided target")
    ins.set_xlabel("hours", fontsize=7); ins.set_yticks([]); ins.tick_params(labelsize=6.5)
    ins.legend(fontsize=6, frameon=False, loc="upper right"); ins.set_title("detiding, per radar cell", fontsize=7.5, loc="left")
    note.text(0, 0.95, "Slab-layer current (two input columns):\n"
              "$dZ/dt + (r + \\mathrm{i}f)\\,Z = \\tau/(\\rho H)$, $Z = u + \\mathrm{i}v$,\n"
              "$\\tau = \\rho_a C_D |\\mathbf{W}|\\mathbf{W}$ from the ERA5 wind, integrated\n"
              "exactly over each hour; $H = 25$ m, $r^{-1} = 4$ d.\n"
              "It supplies the phase of the inertial oscillation at the\ntarget hour, which the wind lags alone only imply.",
              fontsize=7.5, color="k", va="top", transform=note.transAxes)
    save(fig, "method_hourly_target")


# --------------------------------------------------------------------------
def fig_hourly_to_daily() -> None:
    fig, ax = plt.subplots(figsize=(10, 3.9))
    ax.set_xlim(-16, 34); ax.set_ylim(-1.9, 3.6); ax.axis("off")
    ax.text(-16, 3.5, "Scoring the hourly route as a daily product", fontsize=9, va="top")
    timeline(ax, 0)
    ax.axvline(0, color=BLUE, lw=1.0, ymin=0.3, ymax=0.62)
    ax.text(0, 2.25, "one pass", ha="center", fontsize=7.5, color=BLUE, va="bottom")
    ks = np.arange(-12, 13, 3)
    preds = 0.9 + 0.35 * np.sin(2 * np.pi * ks / 19 + 0.6)
    ax.plot(ks, preds, marker="o", color=GREEN, ms=5, lw=0.8)
    ax.text(-12.5, 1.6, "nine hourly predictions of the hourly model", fontsize=7.5, color=GREEN, va="bottom")
    ax.plot([-12, 12], [preds.mean()] * 2, color=GREEN, lw=1.4, ls="--")
    ax.text(12.6, preds.mean(), "their mean", fontsize=7.5, color=GREEN, va="center")
    ax.add_patch(Rectangle((-12.5, 0.32), 25, 0.22, fc=PINK, ec="none", alpha=0.35))
    ax.text(12.6, 0.43, "25-hour-mean radar target", fontsize=7.5, color=PINK, va="center")
    box(ax, 19.0, 1.3, 14.5, 2.0, "compare on the same rows:\nmean of 9 hourly predictions\nvs. the daily champion\n(trained on the daily target),\nboth against the 25 h target", color="0.2", fc="0.96", fs=7.3)
    ax.plot([0], [preds[4]], marker="o", color=ORANGE, ms=8, mfc="none", mew=1.5)
    ax.text(0.8, preds[4] + 0.25, "the check: the $k=0$ prediction alone\nagainst the daily mean", fontsize=7, color=ORANGE, va="bottom")
    ax.text(-16, -1.3, "Result (three seeds each side): the averaged hourly route beats the daily champion on the West Coast ($u$ 0.39 vs 0.35, intervals apart),\n"
            "ties or gains elsewhere, never loses. The $k=0$ prediction alone explains 14 % of the daily mean on uswc and 64 % on usegc:\n"
            "the West Coast current has far more sub-daily motion that cancels over the day.", fontsize=7.3, color=GREY, va="top")
    save(fig, "method_hourly_to_daily")


# --------------------------------------------------------------------------
def fig_holdout() -> None:
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(10, 3.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
    ax.set_xlim(0, 60); ax.set_ylim(0, 3.2); ax.axis("off")
    ax.add_patch(Rectangle((1, 1.2), 16, 0.9, fc=BLUE, alpha=0.35, ec="none"))
    ax.add_patch(Rectangle((17, 1.2), 42, 0.9, fc=PINK, alpha=0.35, ec="none"))
    ax.text(9, 1.65, "train\ncycles 001-016\nJul 2023 - Jun 2024", ha="center", va="center", fontsize=7.5)
    ax.text(38, 1.65, "score (held out)\ncycles 017-058, Jun 2024 - Aug 2026", ha="center", va="center", fontsize=7.5)
    for c in (1, 17, 59):
        ax.plot([c, c], [1.05, 2.25], color="k", lw=0.8)
    ax.text(30, 0.7, "SWOT 21-day science cycles", ha="center", fontsize=8, color=GREY)
    ax.text(30, 2.75, "Transfer in time: every model is fitted on year one and scored on the next two years;\n"
            "skill is tracked by quarter and a drift slope fitted (13 of 16 regions: slope $\\approx 0$).",
            ha="center", va="center", fontsize=7.5)
    ax.text(30, 0.2, "Intervals: block bootstrap over passes (rows of one pass are not independent).", ha="center", fontsize=7, color=GREY)
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 3.2); ax2.axis("off")
    conts = [("US (6)", 0.5, BLUE), ("Europe (6)", 3.6, GREEN), ("Australia (7)", 6.7, ORANGE)]
    for name, x, col in conts:
        box(ax2, x, 1.7, 2.6, 0.8, name, color=col, fs=8)
    ax2.text(5, 2.95, "Transfer in space: train on one continent, score the other two;\nwithin the US, leave one region out and score it.",
             ha="center", va="center", fontsize=7.5)
    ax2.annotate("", (3.6, 2.1), (3.1, 2.1), arrowprops=dict(arrowstyle="<->", color="0.3", lw=1))
    ax2.annotate("", (6.7, 2.1), (6.2, 2.1), arrowprops=dict(arrowstyle="<->", color="0.3", lw=1))
    ax2.text(5, 1.3, "Result: $u$ collapses, $v$ survives, worse than calibrated geostrophy;\n"
             "the all-19 model recovers every region it has seen.\nRule: train on every network you intend to predict.",
             ha="center", va="center", fontsize=7.3, color=GREY)
    ax2.text(5, 0.35, "Phase 1b and 2 used 7 training sets + 4 leave-one-out sets, 3 seeds each in-region.", ha="center", fontsize=7, color=GREY)
    fig.suptitle("The evaluation protocol", fontsize=10)
    save(fig, "method_holdout")


if __name__ == "__main__":
    for f in (fig_rows, fig_daily, fig_wind_history, fig_hourly, fig_hourly_to_daily, fig_holdout):
        f()
