"""
Real-data reconstruction-vs-detection split (revision; RUN ON 4070).

Confirms on real HEST slides the split established on synthetic data (scripts/07, 08):
  * RECONSTRUCTION: does any policy beat strong coverage at reconstructing unprofiled expression (PCC)?
  * DETECTION:      does a value-conditioned policy beat coverage at capturing molecularly heterogeneous
                    regions of interest (recall @ budget) -- the S2-omics / SOFisher objective?

For each pinned slide (all CI_TISSUES) we define ROIs from ground-truth local molecular heterogeneity
(top ROI_QUANTILE of regions; used for EVALUATION only, like HVG selection). At each budget we run every
policy under a fixed imputer and record BOTH: reconstruction PCC-top50 and detection recall@budget. We
then report, across slides, the paired margin of the value-conditioned adaptive policy vs the best
coverage baseline, for each objective, with a bootstrap CI.

Expected (confirming the synthetic result): reconstruction margin <= 0 (adaptivity does not help), and
detection margin > 0 at useful budgets (adaptivity helps). Writes results/realdata_detection.json.

Run unbuffered:  python -u scripts/09_realdata_detection.py
"""
import os, sys, json, pickle
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from pathlib import Path
import numpy as np

from stare import config, metrics, stats
from stare.simulator import AcquisitionSimulator
from stare.imputer import build_imputer
from stare.slides import load_ci_slides
from stare.policies import build_policy

IMPUTER = "graphflow"                      # strongest imputer (matches the headline reconstruction result)
BUDGETS = [0.02, 0.05, 0.10, 0.20]
ROI_QUANTILE = 0.80                        # ROI = top 20% most heterogeneous regions
COVERAGE = ["kmeans", "grid", "mi_greedy"]  # non-adaptive comparators (best-of = coverage envelope)
BOOT = 5000


def log(m): print(m, flush=True)


# study clusters + tissue strata of the loaded slides (set in main; used by every margin summary)
GROUPING = {}




def roi_mask(slide):
    het = metrics.local_heterogeneity(slide.expression, slide.region_coords, slide.hvg_idx)
    return het >= np.quantile(het, ROI_QUANTILE), het


def profiled_and_score(slide, imputer, policy_or_order, bf):
    sim = AcquisitionSimulator(slide, imputer, bf)
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
    return np.array(sim.profiled, dtype=int), sim.score()["pcc_top50"]


def main():
    imputer = build_imputer(IMPUTER)
    slides, labels, group = load_ci_slides(log=log)
    GROUPING.update(clusters=group["clusters"], strata=group["strata"])
    log(f"=== real-data detection | slides={labels} imputer={IMPUTER} budgets={BUDGETS} ===")

    POLICIES = COVERAGE + ["value_greedy"]
    rows = {}
    for bf in BUDGETS:
        rec_pcc = {p: [] for p in POLICIES + ["oracle"]}
        rec_det = {p: [] for p in POLICIES + ["oracle"]}
        for si, slide in enumerate(slides):
            roi, het = roi_mask(slide)
            oracle_order = list(np.argsort(-het))          # detection oracle: most heterogeneous first
            for p in POLICIES:
                pk = {"imputer": imputer} if p == "greedy_ig" else ({"beta": 0.5} if p == "value_greedy" else {})
                prof, pcc = profiled_and_score(slide, imputer, build_policy(p, **pk), bf)
                rec_pcc[p].append(pcc); rec_det[p].append(metrics.detection_recall(prof, roi))
            prof, pcc = profiled_and_score(slide, imputer, oracle_order, bf)
            rec_pcc["oracle"].append(pcc); rec_det["oracle"].append(metrics.detection_recall(prof, roi))
            log(f"  [{bf:.2f}] slide {si+1}/{len(slides)} {labels[si]} done")
        best_cov_pcc = np.maximum.reduce([rec_pcc[p] for p in COVERAGE])
        best_cov_det = np.maximum.reduce([rec_det[p] for p in COVERAGE])
        m_recon = stats.paired_margin_summary(np.array(rec_pcc["value_greedy"]), best_cov_pcc, BOOT, config.CI_ALPHA, config.CI_SEED, **GROUPING)
        m_detect = stats.paired_margin_summary(np.array(rec_det["value_greedy"]), best_cov_det, BOOT, config.CI_ALPHA, config.CI_SEED, **GROUPING)
        rows[f"{bf:.2f}"] = {
            "reconstruction_pcc_means": {p: float(np.mean(v)) for p, v in rec_pcc.items()},
            "detection_recall_means": {p: float(np.mean(v)) for p, v in rec_det.items()},
            "value_greedy_recon_vs_bestcover": m_recon,
            "value_greedy_detect_vs_bestcover": m_detect,
        }
        log(f"  == budget {bf:.2f}: RECON value-cover={m_recon['mean']:+.3f} CI={m_recon['ci95']} "
            f"excl0={m_recon['ci_excludes_zero']} | DETECT value-cover={m_detect['mean']:+.3f} "
            f"CI={m_detect['ci95']} excl0={m_detect['ci_excludes_zero']} "
            f"(oracle_recall={np.mean(rec_det['oracle']):.3f} cover_recall={best_cov_det.mean():.3f})")

    out = Path(config.RESULTS_ROOT) / "realdata_detection.json"
    out.write_text(json.dumps({"imputer": IMPUTER, "slides": labels, "grouping": GROUPING, "region_mode": config.REGION_MODE, "budgets": BUDGETS,
                               "roi_quantile": ROI_QUANTILE, "rows": rows}, indent=2), encoding="utf-8")
    log(f"[write] {out}")
    log("\nExpected: reconstruction margin <= 0 (adaptivity no help) AND detection margin > 0 at useful "
        "budgets (adaptivity helps) -> the split holds on real data.")


if __name__ == "__main__":
    main()
