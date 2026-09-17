"""
Figure 1 (paper centerpiece): the adaptivity gap closes from below and never crosses zero.

Plots the pre-registered contrast `ours - best-non-adaptive` (PCC-top50) vs budget, one line per
imputer, with 95% bootstrap-CI bands, from results/ci_multislide_*.json. A dashed zero line marks
the win threshold; the whole story is that every band sits at or below it. Regenerates from the JSONs
so the figure can never drift from the numbers. Also emits the mechanism panel (ours - image-only
ablation) showing the measured signal only becomes useful under the data-dependent imputer.

Run:  python -u scripts/06_plot_figure1.py   ->  docs/figures/figure1_adaptivity_gap.png (+ .pdf)
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from stare import config

RESULTS = Path(config.RESULTS_ROOT)
OUTDIR = Path("docs/figures"); OUTDIR.mkdir(parents=True, exist_ok=True)
IMPUTERS = config.SWEEP_IMPUTERS
LABEL = {"knn": "kNN (stationary)", "kriging": "kriging (learned, linear)",
         "graphflow": "graphflow (STFlow-class)"}
COLOR = {"knn": "#4C72B0", "kriging": "#DD8452", "graphflow": "#55A868"}


def load_ci(imp):
    p = RESULTS / f"ci_multislide_{imp}.json"
    return json.loads(p.read_text()) if p.exists() else None


def series(cid, contrast):
    xs, mean, lo, hi = [], [], [], []
    for b in cid["budgets"]:
        m = cid["margins"][b][contrast]
        xs.append(float(b) * 100); mean.append(m["mean"]); lo.append(m["ci95"][0]); hi.append(m["ci95"][1])
    return np.array(xs), np.array(mean), np.array(lo), np.array(hi)


def panel(ax, contrast, title, ylabel):
    for imp in IMPUTERS:
        cid = load_ci(imp)
        if not cid:
            continue
        x, m, lo, hi = series(cid, contrast)
        ax.plot(x, m, "-o", ms=4, color=COLOR[imp], label=LABEL[imp], zorder=3)
        ax.fill_between(x, lo, hi, color=COLOR[imp], alpha=0.15, zorder=1)
    ax.axhline(0, ls="--", lw=1, color="0.4", zorder=2)
    ax.set_xscale("log")
    ax.set_xticks([0.5, 1, 2, 5, 10, 20]); ax.set_xticklabels(["0.5", "1", "2", "5", "10", "20"])
    ax.set_xlabel("budget (% of regions profiled)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(True, which="both", ls=":", lw=0.5, alpha=0.5)


def main():
    if not any(load_ci(i) for i in IMPUTERS):
        print("[skip] no ci_multislide_*.json present — run scripts/05_multislide_ci.py first.")
        return
    n_sl = len(next(load_ci(i) for i in IMPUTERS if load_ci(i))["slides"])
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))
    panel(axes[0], "ours_vs_best_nonadapt",
          "(a) Adaptivity gap: closed-loop − best non-adaptive",
          r"$\Delta$ PCC-top50  (ours − best non-adaptive)")
    panel(axes[1], "closed_vs_ablate",
          "(b) Mechanism: measured signal's value",
          r"$\Delta$ PCC-top50  (ours − image-only ablation)")
    axes[0].legend(fontsize=8, loc="lower right", framealpha=0.9)
    fig.suptitle(f"Adaptivity provides no reconstruction gain ({n_sl} slides, mean ± 95% CI)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    png = OUTDIR / "figure1_adaptivity_gap.png"
    pdf = OUTDIR / "figure1_adaptivity_gap.pdf"
    fig.savefig(png, dpi=200, bbox_inches="tight")
    fig.savefig(pdf, bbox_inches="tight")
    print(f"[write] {png}\n[write] {pdf}")


if __name__ == "__main__":
    main()
