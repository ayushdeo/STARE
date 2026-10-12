"""
Stage-3 experiments (reviewer round 3; RUN ON 4070). Registered in docs/ITERATION_LOG.md (iteration 11) before
this script was run; analysed by scripts/22_stage3_analysis.py.

Part H (hybrids, point 1): the first ceil(f x budget) regions come from a FIXED coverage rule, the rest from a
  feedback policy; f in {0.5, 0.8}. Each hybrid also runs with permuted feedback (seeds 1, 2).
    reconstruction: coverage = k-means (0.5, 1%) / MI-greedy (2-20%); feedback = residual policy
    detection     : coverage = variance-greedy (2-10%) / one-shot ranker (20%); feedback = local-variance policy
  The fixed rules are those chosen for most held-out studies in stage 2 (results/revision_summary.json).
Part S (extra permutation seeds, point 3): local-variance with permuted feedback, seeds 1-4 (detection budgets);
  residual policy with permuted feedback, seeds 1-2 (all budgets). Stage 2 used seed 2021.
Part R (batched acquisition with stage-2 controls, point 5): rounds of 5 regions at 10% and 20%; local-variance
  with real feedback and with permuted feedback (seeds 1, 2). The fixed detection rules are non-adaptive and
  unaffected by batching, so their stage-2 values serve as comparators.

Every (slide, budget) cell is checkpointed to results/stage3_runs.json; rerunning resumes.
Run:  python -u scripts/21_stage3_experiments.py --part H     (then S, then R; or --part HSR)
"""
import os, sys, json, time, argparse
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from pathlib import Path
import numpy as np

from stare import config, metrics
from stare.simulator import AcquisitionSimulator, AcquisitionState
from stare.imputer import build_imputer
from stare.policies import build_policy, FeedbackAblation, HybridPolicy
from stare.slides import load_ci_slides

IMPUTER = "graphflow"
REC_BUDGETS = ["0.005", "0.010", "0.020", "0.050", "0.100", "0.200"]
DET_BUDGETS = ["0.020", "0.050", "0.100", "0.200"]
REC_RULE = {"0.005": "kmeans", "0.010": "kmeans", "0.020": "mi_greedy", "0.050": "mi_greedy", "0.100": "mi_greedy", "0.200": "mi_greedy"}
DET_RULE = {"0.020": "var_greedy", "0.050": "var_greedy", "0.100": "var_greedy", "0.200": "oneshot"}
FRACS = [0.5, 0.8]
HYB_SEEDS = [1, 2]
S_SEEDS_DET = [1, 2, 3, 4]
S_SEEDS_REC = [1, 2]
R_BUDGETS = ["0.100", "0.200"]
R_ROUND = 5
R_SEEDS = [1, 2]
ROI_QUANTILE = 0.80
OUT = Path(config.RESULTS_ROOT) / "stage3_runs.json"


def log(m): print(m, flush=True)


def build(name, b, imp):
    """Names: hyb_{rec|det}_f{0.5|0.8}[_perm{s}] ; vg_perm{s} ; rg_perm{s} ; batch5_vg[_perm{s}]"""
    seed = None
    if "_perm" in name:
        name, s = name.split("_perm"); seed = int(s)
    if name.startswith("hyb_"):
        _, obj, f = name.split("_")
        frac = float(f[1:])
        if obj == "rec":
            pol = HybridPolicy(build_policy(REC_RULE[b]), build_policy("residual_guided", imputer=imp), frac)
        else:
            pol = HybridPolicy(build_policy(DET_RULE[b]), build_policy("value_greedy", beta=0.5), frac)
    elif name in ("vg", "batch5_vg"):
        pol = build_policy("value_greedy", beta=0.5)
    elif name == "rg":
        pol = build_policy("residual_guided", imputer=imp)
    else:
        raise ValueError(name)
    return FeedbackAblation(pol, "permute", seed=seed) if seed is not None else pol


def play(slide, imp, name, b, roi, round_size=1):
    bf = float(b)
    sim = AcquisitionSimulator(slide, imp, bf); sim.reset()
    pol = build(name, b, imp)
    while not sim.done:
        k = min(round_size, sim.budget - len(sim.profiled))
        mask, prof = sim._mask.copy(), np.array(sim.profiled, dtype=int)
        pending = []
        for _ in range(k):                      # policy sees only measurements available at the start of the round
            pick = int(pol.select(AcquisitionState(slide=slide, profiled=prof, mask=mask, budget=sim.budget)))
            pending.append(pick); mask[pick] = True
        sim.reveal(pending)
    prof = np.array(sim.profiled, dtype=int)
    return {"pcc": float(sim.score()["pcc_top50"]), "recall": metrics.detection_recall(prof, roi)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="HSR", help="any of H, S, R")
    args = ap.parse_args()
    imp = build_imputer(IMPUTER)
    slides, labels, group = load_ci_slides(log=log)
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    res.update({"slides": labels, "grouping": {k: group[k] for k in ("clusters", "strata")}, "region_mode": config.REGION_MODE,
                "rec_rule": REC_RULE, "det_rule": DET_RULE, "fracs": FRACS, "r_round": R_ROUND})
    save = lambda: OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")

    plan = {}
    if "H" in args.part:
        plan["H"] = {b: [f"hyb_rec_f{f}" + s for f in FRACS for s in [""] + [f"_perm{k}" for k in HYB_SEEDS]]
                        + ([f"hyb_det_f{f}" + s for f in FRACS for s in [""] + [f"_perm{k}" for k in HYB_SEEDS]] if b in DET_BUDGETS else [])
                     for b in REC_BUDGETS}
    if "S" in args.part:
        plan["S"] = {b: [f"rg_perm{k}" for k in S_SEEDS_REC] + ([f"vg_perm{k}" for k in S_SEEDS_DET] if b in DET_BUDGETS else [])
                     for b in REC_BUDGETS}
    if "R" in args.part:
        plan["R"] = {b: ["batch5_vg"] + [f"batch5_vg_perm{k}" for k in R_SEEDS] for b in R_BUDGETS}

    for slide, lab in zip(slides, labels):
        het = metrics.local_heterogeneity(slide.expression, slide.region_coords, slide.hvg_idx)
        roi = het >= np.quantile(het, ROI_QUANTILE)
        for part, by_b in plan.items():
            for b, names in by_b.items():
                cell = res.setdefault(part, {}).setdefault(lab, {}).setdefault(b, {})
                todo = [n for n in names if n not in cell]
                if not todo:
                    continue
                t0 = time.time()
                for n in todo:
                    cell[n] = play(slide, imp, n, b, roi, R_ROUND if part == "R" else 1)
                save()
                log(f"[{part}] {lab} budget {b}: {len(todo)} runs in {time.time() - t0:.0f}s")
    save()
    log(f"[write] {OUT}")


if __name__ == "__main__":
    main()
