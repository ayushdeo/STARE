"""
Competitor head-to-head under matched objectives (revision; RUN ON 4070).

Runs the harmonized SCR2-ST and SOFisher acquisition policies (src/stare/policies/competitors.py)
alongside our coverage/placement baselines and value_greedy, on real HEST slides, scoring BOTH
objectives per policy:
  * RECONSTRUCTION: PCC-top50 on unprofiled regions.
  * DETECTION:      recall of top-20% molecularly heterogeneous ROIs (S2-omics objective).

Hypothesis (the reconciliation): the adaptive competitors help DETECTION but not whole-slide
RECONSTRUCTION vs strong placement -- their published gains are on a detection/diversity objective, not
reconstruction. We report, across slides, each adaptive policy's margin vs the best coverage baseline for
each objective, with a bootstrap CI. Companion (native reproduction of their code): see
docs/COMPETITOR_REPRODUCTION.md. Writes results/competitors.json.

Run unbuffered:  python -u scripts/10_competitors.py
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

from stare import config, data, metrics, stats
from stare.simulator import AcquisitionSimulator
from stare.imputer import build_imputer
from stare.policies import build_policy

IMPUTER = "graphflow"
BUDGETS = [0.02, 0.05, 0.10, 0.20]
ROI_QUANTILE = 0.80
COVERAGE = ["kmeans", "grid", "mi_greedy"]         # non-adaptive envelope
ADAPTIVE = ["scr2st", "sofisher", "value_greedy"]  # scr2st is non-adaptive but grouped as a competitor
BOOT = 5000


def log(m): print(m, flush=True)


def load_or_cache_slides(tissue):
    p = Path(config.CACHE_ROOT) / f"slides_{tissue}_{int(config.REGION_UM)}um_{config.IMAGE_ENCODER}.pkl"
    if p.exists():
        log(f"[cache] {p}"); return pickle.load(open(p, "rb"))
    log(f"[build] {tissue} (minutes)..."); Path(config.CACHE_ROOT).mkdir(parents=True, exist_ok=True)
    s = data.load_tissue(tissue); pickle.dump(s, open(p, "wb")); return s


def roi_mask(slide):
    het = metrics.local_heterogeneity(slide.expression, slide.region_coords, slide.hvg_idx)
    return het >= np.quantile(het, ROI_QUANTILE), het


def profiled_and_pcc(slide, imputer, pol_or_order, bf):
    sim = AcquisitionSimulator(slide, imputer, bf); st = sim.reset()
    if isinstance(pol_or_order, (list, np.ndarray)):
        for i in pol_or_order:
            if sim.done: break
            if not st.mask[int(i)]: st = sim.reveal(int(i))
    else:
        while not sim.done: st = sim.reveal(pol_or_order.select(st))
    return np.array(sim.profiled, dtype=int), sim.score()["pcc_top50"]


def main():
    imputer = build_imputer(IMPUTER)
    slides, labels = [], []
    for tissue in config.CI_TISSUES:
        for s in load_or_cache_slides(tissue):
            slides.append(s); labels.append(f"{tissue}:{getattr(s,'patient_id','?')}")
    log(f"=== competitors | slides={labels} imputer={IMPUTER} budgets={BUDGETS} ===")
    ALL = COVERAGE + ADAPTIVE
    rows = {}
    for bf in BUDGETS:
        pcc = {p: [] for p in ALL + ["oracle"]}
        det = {p: [] for p in ALL + ["oracle"]}
        for si, slide in enumerate(slides):
            roi, het = roi_mask(slide)
            for p in ALL:
                prof, v = profiled_and_pcc(slide, imputer, build_policy(p), bf)
                pcc[p].append(v); det[p].append(metrics.detection_recall(prof, roi))
            prof, v = profiled_and_pcc(slide, imputer, list(np.argsort(-het)), bf)
            pcc["oracle"].append(v); det["oracle"].append(metrics.detection_recall(prof, roi))
            log(f"  [{bf:.2f}] slide {si+1}/{len(slides)} {labels[si]}")
        bc_pcc = np.maximum.reduce([pcc[p] for p in COVERAGE])
        bc_det = np.maximum.reduce([det[p] for p in COVERAGE])
        rows[f"{bf:.2f}"] = {
            "reconstruction_pcc_means": {p: float(np.mean(v)) for p, v in pcc.items()},
            "detection_recall_means": {p: float(np.mean(v)) for p, v in det.items()},
            "recon_margin_vs_bestcover": {p: stats.paired_margin_summary(np.array(pcc[p]), bc_pcc, BOOT, config.CI_ALPHA, config.CI_SEED) for p in ADAPTIVE},
            "detect_margin_vs_bestcover": {p: stats.paired_margin_summary(np.array(det[p]), bc_det, BOOT, config.CI_ALPHA, config.CI_SEED) for p in ADAPTIVE},
        }
        log(f"  == budget {bf:.2f} ==")
        for p in ADAPTIVE:
            rm = rows[f"{bf:.2f}"]["recon_margin_vs_bestcover"][p]; dm = rows[f"{bf:.2f}"]["detect_margin_vs_bestcover"][p]
            log(f"     {p:14s} RECON {rm['mean']:+.3f}{'*' if rm['ci_excludes_zero'] else ' '} | "
                f"DETECT {dm['mean']:+.3f}{'*' if dm['ci_excludes_zero'] else ' '}")

    out = Path(config.RESULTS_ROOT) / "competitors.json"
    out.write_text(json.dumps({"imputer": IMPUTER, "slides": labels, "budgets": BUDGETS,
                               "roi_quantile": ROI_QUANTILE, "rows": rows}, indent=2), encoding="utf-8")
    log(f"[write] {out}")


if __name__ == "__main__":
    main()
