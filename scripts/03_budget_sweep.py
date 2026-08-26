"""
Stage-2 DIAGNOSTIC: imputation-error-vs-budget sweep (RUN ON 4070).

Philosophy: NOT a single-point pass/fail. This produces the accuracy-vs-budget Pareto curve
across LOW-to-high budgets, so we can SEE where (if anywhere) the closed loop adds value and
ITERATE — the field-standard presentation (PERSIST panel-size curves; AFA-RL reward vs #features).

For every budget in config.SWEEP_BUDGET_FRACTIONS, run every policy through the SAME frozen
imputer + simulator. The random baseline is stochastic, so it's averaged over config.RANDOM_SEEDS
(mean +/- std) to be a fair floor. Deterministic policies run once.

The two questions that actually decide the thesis, evaluated AT EACH BUDGET:
    Q1  does ours (closed-loop) beat RANDOM (the real floor)?
    Q2  does the CLOSED LOOP help, i.e. ours(measured on) > ours(no_measured)?
Beating one-shot is reported but is NOT sufficient (one-shot seeks outliers, bad for reconstruction).

Writes results/sweep_<tissue>.json (committed) with per-budget/per-policy metrics + verdicts.
Run unbuffered:  python -u scripts/03_budget_sweep.py
"""
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import json
import statistics
import time
from pathlib import Path

from stare import config, data
from stare.simulator import AcquisitionSimulator
from stare.imputer import build_imputer
from stare.policies import build_policy

NEEDS_IMPUTER = {"greedy_ig"}
METRICS = ["pcc_top50", "pcc_mean"]   # primary + full-transcriptome (honest, both reported)


def log(m): print(m, flush=True)


def load_or_cache_slides(tissue):
    cache_dir = Path(config.CACHE_ROOT); cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"slides_{tissue}_{int(config.REGION_UM)}um_{config.IMAGE_ENCODER}.pkl"
    if p.exists():
        import pickle
        log(f"[cache] {p}")
        return pickle.load(open(p, "rb"))
    log("[build] no cache; building (minutes)...")
    slides = data.load_tissue(tissue)
    import pickle
    pickle.dump(slides, open(p, "wb"))
    return slides


def score_policy(slides, imputer, policy_name, bf, pkw, seed=None):
    """Mean over slides of both metrics for one (policy, budget)."""
    pk = dict(pkw)
    if policy_name in NEEDS_IMPUTER:
        pk.setdefault("imputer", imputer)
    if seed is not None:
        pk["seed"] = seed
    per = []
    for s in slides:
        sim = AcquisitionSimulator(s, imputer, bf)
        pol = build_policy(policy_name, **pk)
        st = sim.reset()
        while not sim.done:
            st = sim.reveal(pol.select(st))
        per.append(sim.score())
    return {m: float(sum(r[m] for r in per) / len(per)) for m in METRICS}


def main():
    tissue = config.FEASIBILITY_TISSUE
    slides = load_or_cache_slides(tissue)
    imputer = build_imputer(config.IMPUTER)
    R = slides[0].n_regions
    budgets = config.SWEEP_BUDGET_FRACTIONS
    log(f"=== budget sweep | tissue={tissue} regions={R} encoder={config.IMAGE_ENCODER} "
        f"imputer={config.IMPUTER} budgets={budgets} ===")

    # policy_name, kwargs, stochastic?  (random averaged over seeds)
    policies = {
        "random":              ("random",    {}, True),
        "grid":                ("grid",      {}, False),
        "variance":            ("variance",  {}, False),
        "kmeans":              ("kmeans",    {}, False),
        "oneshot_spatialfinder": ("oneshot", {}, False),
        "ours_closed_loop":    ("greedy_ig", {}, False),
        "ablate_no_measured":  ("greedy_ig", {"use_measured": False}, False),
        "ablate_no_image":     ("greedy_ig", {"use_image": False}, False),
        "ablate_no_spatial":   ("greedy_ig", {"use_spatial": False}, False),
    }

    # results[policy][budget] = {"pcc_top50": mean, "pcc_top50_std": s, "pcc_mean": ...}
    results = {name: {} for name in policies}
    out = Path(config.RESULTS_ROOT) / f"sweep_{tissue}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    def flush(extra=None):
        payload = {"tissue": tissue, "regions": R, "encoder": config.IMAGE_ENCODER,
                   "imputer": config.IMPUTER, "budgets": budgets, "metrics": METRICS,
                   "results": results}
        if extra: payload.update(extra)
        out.write_text(json.dumps(payload, indent=2))

    for bf in budgets:
        k = int(round(bf * R))
        log(f"\n--- budget {bf:.3f}  (~{k} regions) ---")
        for name, (pn, pkw, stochastic) in policies.items():
            t0 = time.time()
            try:
                if stochastic:
                    runs = [score_policy(slides, imputer, pn, bf, pkw, seed=s)
                            for s in config.RANDOM_SEEDS]
                    cell = {}
                    for m in METRICS:
                        vals = [r[m] for r in runs]
                        cell[m] = float(statistics.mean(vals))
                        cell[m + "_std"] = float(statistics.pstdev(vals))
                else:
                    cell = score_policy(slides, imputer, pn, bf, pkw)
                results[name][f"{bf:.3f}"] = cell
                log(f"    {name:24s} pcc_top50={cell['pcc_top50']:.4f} "
                    f"pcc_mean={cell['pcc_mean']:.4f} ({time.time()-t0:.1f}s)")
            except Exception:
                import traceback
                results[name][f"{bf:.3f}"] = {"error": traceback.format_exc().splitlines()[-1]}
                log(f"    {name:24s} FAILED: {results[name][f'{bf:.3f}']['error']}")
            flush()

    # ---- diagnostic verdicts per budget (the real read) ----
    log("\n=== DIAGNOSTIC (primary metric = pcc_top50; higher is better) ===")
    verdicts = {}
    for bf in budgets:
        b = f"{bf:.3f}"
        def val(n): return results.get(n, {}).get(b, {}).get("pcc_top50")
        ours, rand = val("ours_closed_loop"), val("random")
        nomeas, oneshot = val("ablate_no_measured"), val("oneshot_spatialfinder")
        v = {}
        if ours is not None and rand is not None:
            v["beats_random"] = bool(ours > rand); v["ours_minus_random"] = round(ours - rand, 4)
        if ours is not None and nomeas is not None:
            v["closed_loop_helps"] = bool(ours > nomeas); v["closed_minus_nomeasured"] = round(ours - nomeas, 4)
        if ours is not None and oneshot is not None:
            v["beats_oneshot"] = bool(ours > oneshot)
        verdicts[b] = v
        log(f"  budget {b}: beats_random={v.get('beats_random')} (Δ{v.get('ours_minus_random')}) | "
            f"closed_loop_helps={v.get('closed_loop_helps')} (Δ{v.get('closed_minus_nomeasured')}) | "
            f"beats_oneshot={v.get('beats_oneshot')}")

    # overall read: at how many budgets do BOTH core tests hold?
    core_pass = [b for b, v in verdicts.items()
                 if v.get("beats_random") and v.get("closed_loop_helps")]
    log(f"\nCORE-PASS budgets (beat random AND closed loop helps): {core_pass if core_pass else 'NONE'}")
    if core_pass:
        log("  => the closed loop adds value in this budget regime; iterate/scale there.")
    else:
        log("  => closed loop adds no value at any tested budget with the CURRENT imputer/criterion.")
        log("     ITERATE before deciding STOP: (a) stronger imputer (STFlow/SpaIM instead of kNN),")
        log("     (b) better acquisition score (expected error-reduction, not raw uncertainty),")
        log("     (c) harder target (full transcriptome / lower-signal genes). Then re-sweep.")
    flush(extra={"verdicts": verdicts, "core_pass_budgets": core_pass})
    log(f"\n[write] {out}")


if __name__ == "__main__":
    main()
