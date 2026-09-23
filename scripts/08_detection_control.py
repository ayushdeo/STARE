"""
Detection positive-control (revision, reviewer point 1: prove the simulator CAN express adaptivity).

Companion to 07_positive_control.py. Same synthetic slides with hidden, image-unpredictable, spatially
clustered ROI structure, but scored on a DETECTION objective instead of reconstruction: did the policy's
selected regions land on the hidden ROIs? This is the objective that adaptive spatial-omics samplers
(SOFisher, S2-omics) actually optimize -- "capture more regions of interest".

Metric at a fixed budget B:
  recall    = |selected ∩ ROI| / |ROI|        (fraction of ROI regions captured)
  precision = |selected ∩ ROI| / |selected|   (fraction of budget spent on ROI)
ROI = regions whose hidden-structure mask exceeds a threshold. The signal that REVEALS an ROI (measured
expression magnitude/variance) scales with alpha, so at alpha=0 no policy can detect ROIs (recall ~ area
fraction) and as alpha grows, value-conditioned policies should detect them (recall rises), while
coverage stays at the area fraction. If adaptive policies beat coverage here, the simulator is NOT
degenerate: it can express an adaptivity advantage -- just for detection, not reconstruction (07).

Run:  python -u scripts/08_detection_control.py
"""
import os, sys, json
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from stare import config, stats
from stare.simulator import Slide, AcquisitionSimulator
from stare.imputer import build_imputer
from stare.policies import build_policy

R_SIDE = 20
G = 200
N_IMG_GENES = 40
N_HOTSPOT_GENES = 40
BUDGET = 0.10
ALPHAS = [0.0, 1.0, 2.0, 4.0, 8.0]
ROUGH_ELL = 2.0
ROI_THRESH = 0.4
N_SLIDES = 8
IMPUTER = "knn"
BOOT = 5000


def _smoother(coords, ell_r):
    ell = ell_r * config.REGION_UM
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(2)
    A = np.exp(-d2 / (2 * ell ** 2))
    return A / A.sum(1, keepdims=True)


def make_slide(alpha, seed):
    """Returns (slide, roi_mask_bool)."""
    rng = np.random.default_rng(seed)
    R = R_SIDE * R_SIDE
    gx, gy = np.meshgrid(np.arange(R_SIDE), np.arange(R_SIDE))
    coords = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64) * config.REGION_UM
    A_long, A_short = _smoother(coords, 2.0), _smoother(coords, ROUGH_ELL)
    m = A_long @ rng.normal(size=(R, 8))
    P = rng.normal(size=(8, config.IMAGE_EMB_DIM))
    emb = m @ P + 0.5 * rng.normal(size=(R, config.IMAGE_EMB_DIM))
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)
    expr = np.zeros((R, G))
    expr[:, :N_IMG_GENES] = m @ rng.normal(size=(8, N_IMG_GENES)) + 0.3 * rng.normal(size=(R, N_IMG_GENES))
    n_spa = G - N_IMG_GENES - N_HOTSPOT_GENES
    expr[:, N_IMG_GENES:N_IMG_GENES + n_spa] = A_long @ rng.normal(size=(R, n_spa)) + 0.3 * rng.normal(size=(R, n_spa))
    centers = coords[rng.choice(R, 4, replace=False)]
    mask = np.zeros(R)
    for c in centers:
        mask += np.exp(-((coords - c) ** 2).sum(1) / (2 * (2.0 * config.REGION_UM) ** 2))
    mask = np.clip(mask, 0, 1)
    rough = A_short @ rng.normal(size=(R, N_HOTSPOT_GENES))
    expr[:, N_IMG_GENES + n_spa:] = alpha * mask[:, None] * rough + 0.3 * rng.normal(size=(R, N_HOTSPOT_GENES))
    expr = np.log1p(np.clip(expr - expr.min(0, keepdims=True), 0, None)).astype(np.float32)
    hvg = np.argsort(expr.var(0))[::-1][:config.N_HVG].astype(int)
    roi = mask > ROI_THRESH
    return Slide("synth", coords, emb.astype(np.float32), expr, hvg), roi


def selected_set(slide, imputer, policy_or_order):
    sim = AcquisitionSimulator(slide, imputer, BUDGET)
    st = sim.reset()
    if isinstance(policy_or_order, (list, np.ndarray)):
        for i in policy_or_order:
            if sim.done:
                break
            if not st.mask[int(i)]:
                st = sim.reveal(int(i))
    else:
        while not sim.done:
            st = sim.reveal(policy_or_order.select(st))
    return np.array(sim.profiled, dtype=int)


def oracle_order(slide, k=8):
    co = slide.region_coords
    Y = slide.expression[:, slide.hvg_idx]
    d2 = ((co[:, None, :] - co[None, :, :]) ** 2).sum(2)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, 1)[:, :k]
    diff = Y[nn].var(1).mean(1)
    return list(np.argsort(-diff))     # detection oracle: just go to highest true difficulty first


def recall(sel, roi):
    if roi.sum() == 0:
        return float("nan")
    return float(roi[sel].sum() / roi.sum())


def main():
    imputer = build_imputer(IMPUTER)
    POLICIES = ["random", "grid", "kmeans", "value_greedy", "greedy_ig", "oracle"]
    rows = {}
    roi_fracs = []
    for alpha in ALPHAS:
        rec = {p: [] for p in POLICIES}
        for s in range(N_SLIDES):
            slide, roi = make_slide(alpha, 3000 + s)
            roi_fracs.append(roi.mean())
            rec["random"].append(np.mean([recall(selected_set(slide, imputer, build_policy("random", seed=r)), roi) for r in range(3)]))
            rec["grid"].append(recall(selected_set(slide, imputer, build_policy("grid")), roi))
            rec["kmeans"].append(recall(selected_set(slide, imputer, build_policy("kmeans")), roi))
            rec["value_greedy"].append(recall(selected_set(slide, imputer, build_policy("value_greedy", beta=0.5)), roi))
            rec["greedy_ig"].append(recall(selected_set(slide, imputer, build_policy("greedy_ig", imputer=imputer)), roi))
            rec["oracle"].append(recall(selected_set(slide, imputer, oracle_order(slide)), roi))
        best_cover = np.maximum.reduce([rec["random"], rec["grid"], rec["kmeans"]])
        m_vg = stats.paired_margin_summary(np.array(rec["value_greedy"]), best_cover, BOOT, config.CI_ALPHA, config.CI_SEED)
        m_ig = stats.paired_margin_summary(np.array(rec["greedy_ig"]), best_cover, BOOT, config.CI_ALPHA, config.CI_SEED)
        rows[f"{alpha}"] = {"recall_means": {p: float(np.mean(v)) for p, v in rec.items()},
                            "value_greedy_vs_bestcover": m_vg, "greedy_ig_vs_bestcover": m_ig}
        print(f"alpha={alpha}: cover~{best_cover.mean():.3f} value_greedy={np.mean(rec['value_greedy']):.3f} "
              f"greedy_ig={np.mean(rec['greedy_ig']):.3f} oracle={np.mean(rec['oracle']):.3f} | "
              f"vg-cover={m_vg['mean']:+.3f}{'*' if m_vg['ci_excludes_zero'] else ''} "
              f"ig-cover={m_ig['mean']:+.3f}{'*' if m_ig['ci_excludes_zero'] else ''}")

    out = Path(config.RESULTS_ROOT) / "detection_control.json"
    out.write_text(json.dumps({"budget": BUDGET, "rough_ell": ROUGH_ELL, "roi_thresh": ROI_THRESH,
                               "roi_fraction_mean": float(np.mean(roi_fracs)), "n_slides": N_SLIDES,
                               "alphas": ALPHAS, "metric": "recall@budget", "rows": rows}, indent=2),
                   encoding="utf-8")
    print(f"[write] {out}  (mean ROI fraction={np.mean(roi_fracs):.3f}, budget={BUDGET})")

    xs = np.array(ALPHAS)
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    colors = {"random": "#999999", "grid": "#8172B3", "kmeans": "#CCB974",
              "value_greedy": "#4C72B0", "greedy_ig": "#55A868", "oracle": "#000000"}
    for p in POLICIES:
        ys = np.array([rows[f"{a}"]["recall_means"][p] for a in ALPHAS])
        ax.plot(xs, ys, "-o", color=colors[p], label=p)
    ax.axhline(np.mean(roi_fracs) and BUDGET, ls=":", lw=1, color="0.5")  # ~coverage floor
    ax.set_xlabel(r"ROI signal strength $\alpha$")
    ax.set_ylabel("ROI recall @ budget")
    ax.set_title(f"Detection control: adaptivity captures ROIs ({N_SLIDES} slides)", fontsize=10)
    ax.legend(fontsize=8, loc="best", ncol=2)
    ax.grid(True, ls=":", lw=0.5, alpha=0.5)
    fig.tight_layout()
    fp = Path("docs/figures/figure3_detection_control.png")
    fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fp, dpi=200, bbox_inches="tight")
    fig.savefig(fp.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[write] {fp}")


if __name__ == "__main__":
    main()
