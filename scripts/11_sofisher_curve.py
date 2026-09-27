"""
SOFisher native reproduction — ROI-capture learning curve (revision).

Reads the committed TensorBoard logs from the authors' native 5M-step DQN run (artifacts/sofish_seed222_5m,
seed 222, mouse-brain cortex) and produces the evaluation ROI-capture reward curve + a summary. This is
the native-reproduction evidence that SOFisher optimizes a DETECTION objective (regions of interest
captured), complementing our reconstruction-vs-detection split. Requires: pip install tbparse.

Run:  python -u scripts/11_sofisher_curve.py
"""
import os, sys, json
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tbparse import SummaryReader

LOGDIR = "artifacts/sofish_seed222_5m"


def rolling_median(x, w):
    if len(x) < w:
        return x
    return np.array([np.median(x[i:i + w]) for i in range(len(x) - w + 1)])


def main():
    df = SummaryReader(LOGDIR).scalars
    ev = df[df["tag"] == "evaluate_rewards"].sort_values("step")
    s, v = ev["step"].values.astype(float), ev["value"].values.astype(float)
    early = v[s < s.max() * 0.05]
    late = v[s > s.max() * 0.90]
    summary = {
        "run": "SOFisher native, cortex, seed 222, max_steps=5e6",
        "metric": "evaluate_rewards (ROI-capture episode reward)",
        "n_eval_points": int(len(v)), "step_max": int(s.max()),
        "early_median_<5pct": round(float(np.median(early)), 1),
        "late_median_>90pct": round(float(np.median(late)), 1),
        "early_mean_<5pct": round(float(np.mean(early)), 1),
        "late_mean_>90pct": round(float(np.mean(late)), 1),
        "best": round(float(v.max()), 1), "final": round(float(v[-1]), 1),
        "note": "Reward is noisy with rare catastrophic episodes (down to ~-1.7e5); we report medians "
                "as the robust trend and clip the figure y-axis accordingly.",
        "interpretation": "Eval ROI-capture reward rises from strongly negative (untrained) to positive; "
                          "SOFisher learns to capture regions of interest (a detection objective), not "
                          "whole-slide reconstruction.",
    }
    Path("results").mkdir(exist_ok=True)
    Path("results/sofisher_native.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))

    w = 100
    sm = rolling_median(v, w)
    ssm = s[w - 1:] if len(v) >= w else s
    lo, hi = np.percentile(v, [2, 98])
    pad = 0.15 * (hi - lo)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(s / 1e6, np.clip(v, lo - pad, hi + pad), lw=0.4, color="#CCCCCC", label="eval reward (raw, clipped)")
    ax.plot(ssm / 1e6, sm, lw=2.0, color="#55A868", label=f"rolling median ({w})")
    ax.axhline(0, ls="--", lw=1, color="0.4")
    ax.set_ylim(lo - pad, hi + pad)
    ax.set_xlabel("training steps (millions)")
    ax.set_ylabel("ROI-capture eval reward")
    ax.set_title("SOFisher native run: learns to capture ROIs (detection)", fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, ls=":", lw=0.5, alpha=0.5)
    fig.tight_layout()
    fp = Path("docs/figures/figure4_sofisher_native.png")
    fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fp, dpi=200, bbox_inches="tight")
    fig.savefig(fp.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[write] {fp}\n[write] results/sofisher_native.json")


if __name__ == "__main__":
    main()
