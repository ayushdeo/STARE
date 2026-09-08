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


# policy_name, kwargs, stochastic?  (random averaged over seeds)
POLICIES = {
    "random":                ("random",    {}, True),
    "grid":                  ("grid",      {}, False),
    "kmeans":                ("kmeans",    {}, False),
    "oneshot_spatialfinder": ("oneshot",   {}, False),   # SpatialFinder-style competitor
    "var_greedy":            ("var_greedy", {}, False),   # Krause non-adaptive (variance)
    "mi_greedy":             ("mi_greedy",  {}, False),   # Krause non-adaptive (MI) — principled opponent
    "ours_closed_loop":      ("greedy_ig", {}, False),    # ours (closed-loop, adaptive)
    "ablate_no_measured":    ("greedy_ig", {"use_measured": False}, False),
}


def run_one_imputer(imputer_name: str, tissue: str, slides, budgets):
    R = slides[0].n_regions
    imputer = build_imputer(imputer_name)
    log(f"\n########## IMPUTER = {imputer_name} (regions={R}) ##########")
    results = {name: {} for name in POLICIES}
    out = Path(config.RESULTS_ROOT) / f"sweep_{tissue}_{imputer_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    def flush(extra=None):
        payload = {"tissue": tissue, "regions": R, "encoder": config.IMAGE_ENCODER,
                   "imputer": imputer_name, "budgets": budgets, "metrics": METRICS,
                   "results": results}
        if extra: payload.update(extra)
        out.write_text(json.dumps(payload, indent=2))

    for bf in budgets:
        log(f"\n--- {imputer_name} | budget {bf:.3f} (~{int(round(bf*R))} regions) ---")
        for name, (pn, pkw, stochastic) in POLICIES.items():
            t0 = time.time()
            try:
                if stochastic:
                    runs = [score_policy(slides, imputer, pn, bf, pkw, seed=s) for s in config.RANDOM_SEEDS]
                    cell = {}
                    for m in METRICS:
                        vals = [r[m] for r in runs]
                        cell[m] = float(statistics.mean(vals)); cell[m + "_std"] = float(statistics.pstdev(vals))
                else:
                    cell = score_policy(slides, imputer, pn, bf, pkw)
                results[name][f"{bf:.3f}"] = cell
                log(f"    {name:24s} pcc_top50={cell['pcc_top50']:.4f} pcc_mean={cell['pcc_mean']:.4f} ({time.time()-t0:.1f}s)")
            except Exception:
                import traceback
                results[name][f"{bf:.3f}"] = {"error": traceback.format_exc().splitlines()[-1]}
                log(f"    {name:24s} FAILED: {results[name][f'{bf:.3f}']['error']}")
            flush()

    # Honest gate: ours (adaptive/closed-loop) vs mi_greedy (principled NON-ADAPTIVE opponent) [Krause],
    # AND closed_loop_helps (ours vs ablate_no_measured). Beating random/kmeans is table stakes.
    log(f"\n=== DIAGNOSTIC (imputer={imputer_name}; pcc_top50) ===")
    verdicts = {}
    for bf in budgets:
        b = f"{bf:.3f}"
        def val(n): return results.get(n, {}).get(b, {}).get("pcc_top50")
        ours, mi, rand, nomeas = val("ours_closed_loop"), val("mi_greedy"), val("random"), val("ablate_no_measured")
        v = {}
        if ours is not None and mi is not None:
            v["beats_mi_greedy"] = bool(ours > mi); v["ours_minus_mi"] = round(ours - mi, 4)
        if ours is not None and rand is not None:
            v["beats_random"] = bool(ours > rand)
        if ours is not None and nomeas is not None:
            v["closed_loop_helps"] = bool(ours > nomeas); v["closed_minus_nomeasured"] = round(ours - nomeas, 4)
        verdicts[b] = v
        log(f"  budget {b}: beats_mi_greedy={v.get('beats_mi_greedy')} (d={v.get('ours_minus_mi')}) | "
            f"closed_loop_helps={v.get('closed_loop_helps')} (d={v.get('closed_minus_nomeasured')}) | "
            f"beats_random={v.get('beats_random')}")
    core_pass = [b for b, v in verdicts.items() if v.get("beats_mi_greedy") and v.get("closed_loop_helps")]
    log(f"\n[{imputer_name}] CORE-PASS budgets (beat MI-greedy AND closed loop helps): {core_pass or 'NONE'}")
    flush(extra={"verdicts": verdicts, "core_pass_budgets": core_pass})
    log(f"[write] {out}")
    return imputer_name, core_pass


def main():
    tissue = config.FEASIBILITY_TISSUE
    slides = load_or_cache_slides(tissue)
    budgets = config.SWEEP_BUDGET_FRACTIONS
    log(f"=== budget sweep | tissue={tissue} encoder={config.IMAGE_ENCODER} "
        f"imputers={config.SWEEP_IMPUTERS} budgets={budgets} ===")
    summary = [run_one_imputer(name, tissue, slides, budgets) for name in config.SWEEP_IMPUTERS]
    log("\n===== HEADLINE =====")
    for name, cp in summary:
        log(f"  imputer={name:8s} closed-loop earns its keep at budgets: {cp or 'NONE'}")
    log("Expectation from theory [Krause 2008]: stationary (knn) -> NONE; learned (kriging) -> maybe some.\n"
        "If kriging is ALSO NONE, the agentic framing does not survive -> pivot to the reconstruction-"
        "placement benchmark paper (see docs/PROGRESS.md sec.6).")


if __name__ == "__main__":
    main()
