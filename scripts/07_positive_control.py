"""
Positive control + adaptivity boundary (revision points 1, 2, 10).

Question: can our simulator/metric register an adaptivity advantage when one exists, and where is the
boundary? We build synthetic slides with image-unpredictable, spatially clustered structure whose
RECONSTRUCTABILITY is controlled by a correlation-length knob `rough_ell` (in region units):
  * small rough_ell  -> near-irreducible roughness (dense sampling cannot recover it);
  * large rough_ell  -> smooth, recoverable-from-few-samples structure.
Adaptivity should help only in the middle: structure that needs MORE than uniform sampling but is still
recoverable. We include an ORACLE policy that knows the true local difficulty (an upper bound on any
value-conditioned policy) plus our real policies, and score on PCC (top-50) and MSE.

Finding this quantifies: for whole-slide reconstruction, even the oracle's advantage over strong
coverage is small and appears only for reconstructable structure; our deployable policies capture less.
This is the boundary the revised paper reports, and it is why a null result on real ST data (which is
dominated by image-predictable + smoothly-varying signal) is expected rather than a policy failure.

Run:  python -u scripts/07_positive_control.py
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
ALPHA = 8.0                                  # strong hotspot amplitude (dominates the HVGs)
ROUGH_ELLS = [1.0, 1.5, 2.0, 2.5, 3.0]       # reconstructability knob (region units)
N_SLIDES = 8
IMPUTER = "knn"
BOOT = 5000


def _smoother(coords, ell_r):
    ell = ell_r * config.REGION_UM
    d2 = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(2)
    A = np.exp(-d2 / (2 * ell ** 2))
    return A / A.sum(1, keepdims=True)


def make_slide(rough_ell, seed, alpha=ALPHA):
    rng = np.random.default_rng(seed)
    R = R_SIDE * R_SIDE
    gx, gy = np.meshgrid(np.arange(R_SIDE), np.arange(R_SIDE))
    coords = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64) * config.REGION_UM
    A_long, A_short = _smoother(coords, 2.0), _smoother(coords, rough_ell)
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
    return Slide("synth", coords, emb.astype(np.float32), expr, hvg)


def _play(sim, order_or_policy, imputer):
    st = sim.reset()
    if isinstance(order_or_policy, (list, np.ndarray)):
        for i in order_or_policy:
            if sim.done:
                break
            if not st.mask[int(i)]:
                st = sim.reveal(int(i))
    else:
        while not sim.done:
            st = sim.reveal(order_or_policy.select(st))
    return sim.score()


def oracle_order(slide, k=8):
    """Upper-bound value-conditioned order: greedily maximize true local difficulty x coverage."""
    co = slide.region_coords
    Y = slide.expression[:, slide.hvg_idx]
    d2 = ((co[:, None, :] - co[None, :, :]) ** 2).sum(2)
    np.fill_diagonal(d2, np.inf)
    nn = np.argsort(d2, 1)[:, :k]
    diff = Y[nn].var(1).mean(1)
    R = co.shape[0]
    order = [int(np.argmax(diff))]
    mind = np.linalg.norm(co - co[order[0]], axis=1)
    for _ in range(R - 1):
        sc = diff * np.sqrt(np.clip(mind, 1e-9, None))
        sc[order] = -np.inf
        j = int(np.argmax(sc))
        order.append(j)
        mind = np.minimum(mind, np.linalg.norm(co - co[j], axis=1))
    return order


def score_all(slide, imputer):
    def sim():
        return AcquisitionSimulator(slide, imputer, BUDGET)
    out = {}
    out["kmeans"] = _play(sim(), build_policy("kmeans"), imputer)
    out["grid"] = _play(sim(), build_policy("grid"), imputer)
    out["oracle"] = _play(sim(), oracle_order(slide), imputer)
    out["greedy_ig"] = _play(sim(), build_policy("greedy_ig", imputer=imputer), imputer)
    out["value_greedy"] = _play(sim(), build_policy("value_greedy", beta=0.5), imputer)
    return out


def main():
    imputer = build_imputer(IMPUTER)
    rows = {}
    for ell in ROUGH_ELLS:
        acc = {m: {p: [] for p in ["kmeans", "grid", "oracle", "greedy_ig", "value_greedy"]}
               for m in ["pcc_top50", "mse"]}
        for s in range(N_SLIDES):
            sl = make_slide(ell, 2000 + s)
            r = score_all(sl, imputer)
            for p, sc in r.items():
                acc["pcc_top50"][p].append(sc["pcc_top50"])
                acc["mse"][p].append(sc["mse"])
        best_cover = np.maximum(acc["pcc_top50"]["kmeans"], acc["pcc_top50"]["grid"])
        m_oracle = stats.paired_margin_summary(acc["pcc_top50"]["oracle"], best_cover, BOOT, config.CI_ALPHA, config.CI_SEED)
        m_ig = stats.paired_margin_summary(acc["pcc_top50"]["greedy_ig"], best_cover, BOOT, config.CI_ALPHA, config.CI_SEED)
        rows[f"{ell}"] = {
            "pcc_means": {p: float(np.mean(v)) for p, v in acc["pcc_top50"].items()},
            "mse_means": {p: float(np.mean(v)) for p, v in acc["mse"].items()},
            "oracle_vs_bestcover": m_oracle, "greedy_ig_vs_bestcover": m_ig,
        }
        print(f"rough_ell={ell}: oracle_pcc={np.mean(acc['pcc_top50']['oracle']):.3f} "
              f"bestcover={best_cover.mean():.3f} | oracle-cover={m_oracle['mean']:+.3f} "
              f"CI={m_oracle['ci95']} excl0={m_oracle['ci_excludes_zero']} | "
              f"ig-cover={m_ig['mean']:+.3f}{'*' if m_ig['ci_excludes_zero'] else ''}")

    out = Path(config.RESULTS_ROOT) / "positive_control.json"
    out.write_text(json.dumps({"budget": BUDGET, "alpha": ALPHA, "imputer": IMPUTER,
                               "n_slides": N_SLIDES, "rough_ells": ROUGH_ELLS, "rows": rows}, indent=2),
                   encoding="utf-8")
    print(f"[write] {out}")

    xs = np.array(ROUGH_ELLS)
    def series(key):
        return (np.array([rows[f"{e}"][key]["mean"] for e in ROUGH_ELLS]),
                np.array([rows[f"{e}"][key]["ci95"][0] for e in ROUGH_ELLS]),
                np.array([rows[f"{e}"][key]["ci95"][1] for e in ROUGH_ELLS]))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.axhline(0, ls="--", lw=1, color="0.4")
    for key, color, lab in [("oracle_vs_bestcover", "#000000", "oracle (upper bound) − best coverage"),
                            ("greedy_ig_vs_bestcover", "#55A868", "error-reduction − best coverage")]:
        m, lo, hi = series(key)
        ax.plot(xs, m, "-o", color=color, label=lab)
        ax.fill_between(xs, lo, hi, color=color, alpha=0.15)
    ax.set_xlabel(r"reconstructability $\ell$ (correlation length of hidden structure, region units)")
    ax.set_ylabel(r"$\Delta$ PCC-top50")
    ax.set_title(f"When does adaptivity help reconstruction? ({N_SLIDES} slides, mean $\\pm$ 95% CI)", fontsize=10)
    ax.legend(fontsize=8, loc="best")
    ax.grid(True, ls=":", lw=0.5, alpha=0.5)
    fig.tight_layout()
    fp = Path("docs/figures/figure2_positive_control.png")
    fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fp, dpi=200, bbox_inches="tight")
    fig.savefig(fp.with_suffix(".pdf"), bbox_inches="tight")
    print(f"[write] {fp}")


if __name__ == "__main__":
    main()
