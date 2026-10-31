"""Grouped permutation feature importance for an evolve champion (poster figure).

The champion is free-form candidate code with no importance API, so: retrain
it once on the run's frozen data (exact run params), then predict the held-out
test set plus one copy per base feature in which that feature's stencil-column
group is row-shuffled (all k*k columns together, so the spatial pattern breaks
too). Importance = drop in held-out R² vs the unpermuted baseline. Engineered
features inside the candidate (gradients, vorticity, ...) are derived from the
raw columns, so permuting raw columns propagates through them correctly.

Checkpointed: importances land in docs/figures/poster/champion_importance.json; if
that exists the (expensive) training is skipped and only the figure is drawn.

Usage:
    python scripts/champion_importance.py --run evolve_all_science_v1
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

import swotxai  # noqa: F401  (torch-before-numpy DLL ordering)
import numpy as np

EVOLVE_ROOT = REPO_ROOT / "experiments" / "evolve"
OUT_DIR = REPO_ROOT / "docs" / "figures" / "poster"

INK = "#1a1a2e"
MUTED = "#5a6472"
GRID = "#d9dee5"
C_U = "#0e7490"   # deep ocean teal / warm orange: CVD-safe pair (blue-orange axis)
C_V = "#ea580c"

PRETTY = {
    "mdt": "MDT", "ssha_filtered": "SSHA",
    "ugos_filtered": r"$u_{gos}$", "vgos_filtered": r"$v_{gos}$",
    "ugosa_filtered": r"$u_{gosa}$", "vgosa_filtered": r"$v_{gosa}$",
    "era5_u": r"ERA5 $u_{10}$", "era5_v": r"ERA5 $v_{10}$", "SST": "SST",
}


def compute(run_dir: Path, cid: str, json_path: Path):
    from swotxai.evolve.harness import score_predictions

    meta = json.loads((run_dir / "data_meta.json").read_text(encoding="utf-8"))
    params = json.loads((run_dir / "candidates" / cid / "params.json").read_text(encoding="utf-8"))
    feats = meta["features"]
    k2 = meta["n_inputs"] // len(feats)  # stencil cells per feature

    data = np.load(run_dir / "data.npz")
    X_train, Y_train, X_test, Y_test = (data[k] for k in
                                        ("X_train", "Y_train", "X_test", "Y_test"))
    n = len(X_test)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)  # one shared row-shuffle for every feature

    variants = [X_test]
    for i in range(len(feats)):
        Xp = X_test.copy()
        cols = slice(i * k2, (i + 1) * k2)
        Xp[:, cols] = X_test[perm][:, cols]
        variants.append(Xp)
    X_big = np.concatenate(variants)
    del variants

    spec = importlib.util.spec_from_file_location(
        "evolve_candidate", run_dir / "candidates" / cid / "candidate.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    t0 = time.time()
    print(f"Training {cid} + predicting {len(X_big):,} rows "
          f"(baseline + {len(feats)} permuted copies)...", flush=True)
    pred = np.asarray(module.train_and_predict(X_train, Y_train, X_big, params),
                      dtype=np.float32)
    print(f"done in {time.time() - t0:.0f}s", flush=True)

    base = score_predictions(pred[:n], Y_test)
    imp = {}
    for i, f in enumerate(feats):
        m = score_predictions(pred[(i + 1) * n:(i + 2) * n], Y_test)
        imp[f] = {"delta_r2_u": base["r2_u"] - m["r2_u"],
                  "delta_r2_v": base["r2_v"] - m["r2_v"]}
        print(f"{f:16s} d_r2_u {imp[f]['delta_r2_u']:+.4f}  "
              f"d_r2_v {imp[f]['delta_r2_v']:+.4f}", flush=True)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps({
        "run": run_dir.name, "candidate": cid, "n_test": int(n),
        "method": "grouped permutation (all stencil cells of a feature "
                  "shuffled together, shared permutation), scored on the "
                  "temporal holdout",
        "baseline": {k: base[k] for k in ("r2_u", "r2_v", "fitness")},
        "importance": imp,
    }, indent=2), encoding="utf-8")
    print(f"wrote {json_path}")


def plot(json_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "font.family": "serif",
        "font.serif": ["cmr10", "Computer Modern Roman", "STIXGeneral", "Times New Roman"],
        "mathtext.fontset": "cm", "axes.unicode_minus": False,
        "text.color": INK, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
        "xtick.color": MUTED, "ytick.color": MUTED,
    })

    d = json.loads(json_path.read_text(encoding="utf-8"))
    imp = d["importance"]
    feats = sorted(imp, key=lambda f: imp[f]["delta_r2_u"] + imp[f]["delta_r2_v"])
    du = [imp[f]["delta_r2_u"] for f in feats]
    dv = [imp[f]["delta_r2_v"] for f in feats]
    labels = [PRETTY.get(f, f) for f in feats]

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    y = np.arange(len(feats))
    xmax = max(max(du), max(dv))
    ax.set_xlim(0, xmax * 1.12)
    ax.set_ylim(-0.65, len(feats) - 0.35)

    bh = 0.38
    ax.barh(y + bh / 2 + 0.01, du, height=bh, color=C_U, zorder=3,
            label=r"$u$ (east)")
    ax.barh(y - bh / 2 - 0.01, dv, height=bh, color=C_V, zorder=3,
            label=r"$v$ (north)")
    for yy, w in [*zip(y + bh / 2 + 0.01, du), *zip(y - bh / 2 - 0.01, dv)]:
        ax.text(w + 0.012, yy, f"{w:.2f}", va="center", ha="left",
                fontsize=7.6, color=MUTED, zorder=4)

    ax.set_yticks(y, labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel(r"permutation importance  ($\Delta R^2$ on temporal holdout)")
    fig.text(0.5, -0.035,
             r"$\Delta R^2 = R^2(\mathrm{intact}) - R^2(\mathrm{feature\ shuffled})$"
             ": the skill lost when that feature is decoupled from the target",
             ha="center", va="top", fontsize=8.2, color=MUTED)
    ax.set_title("Champion " + r"$\mathtt{gen031\_00}$" + ": what the model relies on",
                 fontsize=12.5, pad=14, x=0.5)
    ax.plot([0.33, 0.67], [1.035, 1.035], transform=ax.transAxes, lw=2.2,
            color=C_U, clip_on=False, solid_capstyle="round")
    ax.grid(axis="x", color=GRID, lw=0.8, ls=(0, (1, 3)))
    ax.set_axisbelow(True)
    ax.axvline(0, color=MUTED, lw=0.8)
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=9)

    for ext in ("png", "svg"):
        p = json_path.parent / f"champion_importance.{ext}"
        fig.savefig(p, dpi=300, bbox_inches="tight")
        print(f"wrote {p}")
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--run", default="evolve_all_science_v1")
    ap.add_argument("--candidate", default="gen031_00")
    ap.add_argument("--force", action="store_true", help="recompute even if JSON exists")
    args = ap.parse_args()
    json_path = OUT_DIR / "champion_importance.json"
    if args.force or not json_path.exists():
        compute(EVOLVE_ROOT / args.run, args.candidate, json_path)
    else:
        print("Reusing checkpointed importances.")
    plot(json_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
