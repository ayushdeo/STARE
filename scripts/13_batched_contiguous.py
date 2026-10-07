"""
Batched + contiguous acquisition (revision, reviewer point 7: physical realism; RUN ON 4070).

Real ROI/FOV platforms do not reveal one scattered region at a time: acquisition happens in a few lab
ROUNDS, and each ROI is a spatially CONTIGUOUS block. We test whether the findings survive these
constraints, scoring BOTH objectives (reconstruction PCC-top50 and detection recall of top-20%
heterogeneity ROIs), vs the best coverage baseline, across slides with bootstrap CIs.

Part A -- batched rounds (scattered): reveal `round_size` regions per round, adapting only between rounds.
  round_size in {1 (fully sequential), 5, 25, one-shot}. Non-adaptive placement is order-invariant to
  batching, so best-coverage is computed once. If adaptivity helped detection, coarser batching should
  erode that advantage toward one-shot.
Part B -- contiguous ROIs: each round a policy picks a SEED and we reveal a contiguous block of `block_k`
  nearest regions (a realistic ROI). Both the adaptive policy and the coverage baselines take contiguous
  blocks, for a fair comparison.

Writes results/batched_contiguous.json.  Run:  python -u scripts/13_batched_contiguous.py
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
from stare.simulator import AcquisitionSimulator, AcquisitionState
from stare.imputer import build_imputer
from stare.slides import load_ci_slides
from stare.policies import build_policy

IMPUTER = "graphflow"
BUDGET = 0.10
ROUND_SIZES = [1, 5, 25, 10 ** 9]          # 1e9 => one-shot (single round)
BLOCK_K = 7                                # contiguous ROI size (regions)
COVERAGE = ["kmeans", "grid", "mi_greedy"]
ADAPTIVE = "value_greedy"
BOOT = 5000


def log(m): print(m, flush=True)


# study clusters + tissue strata of the loaded slides (set in main; used by every margin summary)
GROUPING = {}




def roi_mask(slide):
    het = metrics.local_heterogeneity(slide.expression, slide.region_coords, slide.hvg_idx)
    return het >= np.quantile(het, 0.80)


def _new_policy(name):
    return build_policy(name, **({"beta": 0.5} if name == ADAPTIVE else {}))


def play_scattered(slide, imputer, name, bf, round_size):
    """Reveal round_size regions per round; the policy sees measured values only between rounds."""
    sim = AcquisitionSimulator(slide, imputer, bf); sim.reset()
    pol = _new_policy(name)
    while not sim.done:
        k = min(round_size, sim.budget - len(sim.profiled))
        cur_mask = sim._mask.copy()
        prof_start = np.array(sim.profiled, dtype=int)
        pending = []
        for _ in range(k):
            temp = AcquisitionState(slide=slide, profiled=prof_start, mask=cur_mask, budget=sim.budget)
            pick = int(pol.select(temp))
            pending.append(pick); cur_mask[pick] = True
        sim.reveal(pending)
    return np.array(sim.profiled, dtype=int), sim.score()["pcc_top50"]


def play_contiguous(slide, imputer, name, bf, block_k):
    """Each round: policy picks a seed; reveal a contiguous block of the block_k nearest regions."""
    sim = AcquisitionSimulator(slide, imputer, bf); sim.reset()
    pol = _new_policy(name)
    co = slide.region_coords
    while not sim.done:
        cur_mask = sim._mask.copy()
        prof_start = np.array(sim.profiled, dtype=int)
        temp = AcquisitionState(slide=slide, profiled=prof_start, mask=cur_mask, budget=sim.budget)
        seed = int(pol.select(temp))
        bk = min(block_k, sim.budget - len(sim.profiled))
        unpro = np.where(~cur_mask)[0]
        d = np.linalg.norm(co[unpro] - co[seed], axis=1)
        block = unpro[np.argsort(d)[:bk]]
        sim.reveal([int(x) for x in block])
    return np.array(sim.profiled, dtype=int), sim.score()["pcc_top50"]


def summarize(adaptive_vals, cover_matrix):
    best = np.maximum.reduce(cover_matrix)
    return stats.paired_margin_summary(np.array(adaptive_vals), best, BOOT, config.CI_ALPHA, config.CI_SEED, **GROUPING)


def main():
    imputer = build_imputer(IMPUTER)
    slides, labels, group = load_ci_slides(log=log)
    GROUPING.update(clusters=group["clusters"], strata=group["strata"])
    rois = [roi_mask(s) for s in slides]
    log(f"=== batched/contiguous | slides={labels} imputer={IMPUTER} budget={BUDGET} ===")

    out = {"imputer": IMPUTER, "budget": BUDGET, "slides": labels, "grouping": GROUPING, "region_mode": config.REGION_MODE, "block_k": BLOCK_K,
           "round_sizes": [("oneshot" if r > 10 ** 8 else r) for r in ROUND_SIZES], "partA_batched": {}, "partB_contiguous": {}}

    # Part A: coverage is order-invariant to batching -> compute once (round_size=1)
    cov_recon = {}; cov_detect = {}
    for c in COVERAGE:
        rc, dt = [], []
        for s, roi in zip(slides, rois):
            prof, pcc = play_scattered(s, imputer, c, BUDGET, 1)
            rc.append(pcc); dt.append(metrics.detection_recall(prof, roi))
        cov_recon[c] = rc; cov_detect[c] = dt
    cov_recon_mat = [cov_recon[c] for c in COVERAGE]
    cov_detect_mat = [cov_detect[c] for c in COVERAGE]

    log("\n-- Part A: batched rounds (adaptive = value_greedy) --")
    for rs in ROUND_SIZES:
        rc, dt = [], []
        for s, roi in zip(slides, rois):
            prof, pcc = play_scattered(s, imputer, ADAPTIVE, BUDGET, rs)
            rc.append(pcc); dt.append(metrics.detection_recall(prof, roi))
        m_rec = summarize(rc, cov_recon_mat); m_det = summarize(dt, cov_detect_mat)
        key = "oneshot" if rs > 10 ** 8 else str(rs)
        out["partA_batched"][key] = {"recon_margin": m_rec, "detect_margin": m_det,
                                     "adaptive_detect_mean": float(np.mean(dt))}
        log(f"  round_size={key:7s}  RECON {m_rec['mean']:+.3f}{'*' if m_rec['ci_excludes_zero'] else ' '}"
            f"  DETECT {m_det['mean']:+.3f}{'*' if m_det['ci_excludes_zero'] else ' '} "
            f"(adaptive recall={np.mean(dt):.3f})")

    # Part B: contiguous blocks -- both adaptive and coverage take contiguous ROIs
    log("\n-- Part B: contiguous ROIs (block_k={}) --".format(BLOCK_K))
    cc_recon = {}; cc_detect = {}
    for name in COVERAGE + [ADAPTIVE]:
        rc, dt = [], []
        for s, roi in zip(slides, rois):
            prof, pcc = play_contiguous(s, imputer, name, BUDGET, BLOCK_K)
            rc.append(pcc); dt.append(metrics.detection_recall(prof, roi))
        cc_recon[name] = rc; cc_detect[name] = dt
    m_rec = summarize(cc_recon[ADAPTIVE], [cc_recon[c] for c in COVERAGE])
    m_det = summarize(cc_detect[ADAPTIVE], [cc_detect[c] for c in COVERAGE])
    out["partB_contiguous"] = {"recon_margin": m_rec, "detect_margin": m_det,
                               "adaptive_detect_mean": float(np.mean(cc_detect[ADAPTIVE])),
                               "cover_detect_mean": float(np.mean(np.maximum.reduce([cc_detect[c] for c in COVERAGE])))}
    log(f"  contiguous  RECON {m_rec['mean']:+.3f}{'*' if m_rec['ci_excludes_zero'] else ' '}"
        f"  DETECT {m_det['mean']:+.3f}{'*' if m_det['ci_excludes_zero'] else ' '} "
        f"(adaptive recall={np.mean(cc_detect[ADAPTIVE]):.3f} vs cover {np.mean(np.maximum.reduce([cc_detect[c] for c in COVERAGE])):.3f})")

    p = Path(config.RESULTS_ROOT) / "batched_contiguous.json"
    p.write_text(json.dumps(out, indent=2), encoding="utf-8")
    log(f"\n[write] {p}")
    log("Expected: RECON margin stays <= 0 in every setting; DETECT margin positive when adaptation is "
        "frequent (round_size small / contiguous seeded by signal) and shrinks toward 0 as batching coarsens "
        "(one-shot) -- quantifying how physical batching limits the detection advantage.")


if __name__ == "__main__":
    main()
