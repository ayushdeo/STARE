"""
Stage-2 GO/NO-GO (RUN ON 4070): the feasibility experiment.

On ONE tissue at the pinned feasibility budget, run every policy through the SAME frozen
imputer + simulator and compare imputation error. The decisive result:
    does GreedyInfoGain (closed-loop) beat OneShotSpatialFinder at equal budget?
Plus the ablations: closed-loop on/off, image on/off, spatial on/off.

Hardened runner:
  * caches built Slide objects to config.CACHE_ROOT (reruns are instant; no re-embedding)
  * streams per-policy progress with flush (survives buffered terminals)
  * writes results/feasibility_<tissue>.json even if a run raises (partial + traceback)
  * fixes the imputer-argument handling (imputer injected only for policies that need it)

Run it unbuffered so you see progress live:
    python -u scripts/02_run_feasibility.py

See docs/REPRODUCTION_PLAN.md §4 Stage 2 and docs/SOURCE_OF_TRUTH.md §6.
"""
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")  # silence sklearn/joblib wmic warning on Windows

import json
import pickle
import time
import traceback
from pathlib import Path

from stare import config, data
from stare.simulator import AcquisitionSimulator
from stare.imputer import build_imputer
from stare.policies import build_policy


# Policies whose constructor requires the imputer. Everything else must NOT receive it.
NEEDS_IMPUTER = {"greedy_ig"}


def log(msg: str):
    print(msg, flush=True)


def load_or_cache_slides(tissue: str):
    """Load built slides from cache if present, else build via data.load_tissue and cache.
    Cache key includes tissue + region size + encoder so a config change invalidates it."""
    cache_dir = Path(config.CACHE_ROOT)
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = f"slides_{tissue}_{int(config.REGION_UM)}um_{config.IMAGE_ENCODER}.pkl"
    cache_path = cache_dir / key
    if cache_path.exists():
        log(f"[cache] loading slides from {cache_path}")
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    log(f"[build] no cache; building slides for '{tissue}' (this can take minutes: UNI embedding)")
    t0 = time.time()
    slides = data.load_tissue(tissue)
    with open(cache_path, "wb") as f:
        pickle.dump(slides, f)
    log(f"[build] built + cached {len(slides)} slide(s) in {time.time()-t0:.1f}s -> {cache_path}")
    return slides


def run_policy(slide, imputer, policy_name, budget_fraction, policy_kwargs=None):
    """Instantiate simulator + policy, run acquisition rounds to budget, return score dict.
    The imputer is passed to build_policy ONLY for policies that require it (no duplicate arg)."""
    pkw = dict(policy_kwargs or {})
    if policy_name in NEEDS_IMPUTER:
        pkw.setdefault("imputer", imputer)
    sim = AcquisitionSimulator(slide, imputer, budget_fraction)
    policy = build_policy(policy_name, **pkw)
    state = sim.reset()
    while not sim.done:
        state = sim.reveal(policy.select(state))
    return sim.score()


def main():
    tissue = config.FEASIBILITY_TISSUE
    bf = config.FEASIBILITY_BUDGET_FRACTION
    log(f"=== Stage-2 feasibility | tissue={tissue} budget_fraction={bf} "
        f"encoder={config.IMAGE_ENCODER} imputer={config.IMPUTER} metric={config.PRIMARY_METRIC} ===")

    slides = load_or_cache_slides(tissue)
    log(f"[data] {len(slides)} slide(s); "
        f"regions={[s.n_regions for s in slides]}; budget≈{int(round(bf*slides[0].n_regions))}")
    imputer = build_imputer(config.IMPUTER)   # frozen, shared across all policies

    # run-name -> (policy_name, policy_kwargs). imputer is NOT put here; run_policy injects it.
    runs = {
        "random":                 ("random",    {}),
        "oneshot_spatialfinder":  ("oneshot",   {}),           # PRIMARY competitor
        "ours_closed_loop":       ("greedy_ig", {}),
        # --- ablations (WS-E decisive) ---
        "ablate_no_measured":     ("greedy_ig", {"use_measured": False}),
        "ablate_no_image":        ("greedy_ig", {"use_image": False}),
        "ablate_no_spatial":      ("greedy_ig", {"use_spatial": False}),
    }

    out = Path(config.RESULTS_ROOT) / f"feasibility_{tissue}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    results = {}

    def flush_json(extra=None):
        payload = {"tissue": tissue, "budget_fraction": bf,
                   "encoder": config.IMAGE_ENCODER, "imputer": config.IMPUTER,
                   "primary_metric": config.PRIMARY_METRIC, "results": results}
        if extra:
            payload.update(extra)
        out.write_text(json.dumps(payload, indent=2))

    for name, (policy_name, pkw) in runs.items():
        t0 = time.time()
        try:
            per_slide = [run_policy(s, imputer, policy_name, bf, pkw) for s in slides]
            vals = [r[config.PRIMARY_METRIC] for r in per_slide]
            results[name] = {"mean": float(sum(vals) / len(vals)), "per_slide": per_slide}
            log(f"[run] {name:24s} {config.PRIMARY_METRIC}={results[name]['mean']:.4f} "
                f"({time.time()-t0:.1f}s)")
        except Exception:
            tb = traceback.format_exc()
            results[name] = {"error": tb.splitlines()[-1], "traceback": tb}
            log(f"[run] {name:24s} FAILED:\n{tb}")
        flush_json()   # write after every run so we never lose progress

    log(f"\n[write] {out}")

    # GO/NO-GO (only if both key runs produced a number)
    ours = results.get("ours_closed_loop", {}).get("mean")
    base = results.get("oneshot_spatialfinder", {}).get("mean")
    if ours is not None and base is not None:
        gate = ours > base
        verdict = "PROCEED to scale" if gate else "STOP; reassess with TA"
        log(f"\nGO/NO-GO: closed-loop ({ours:.4f}) {'>' if gate else '<='} one-shot ({base:.4f}) "
            f"=> {verdict}")
        flush_json(extra={"go_no_go": {"closed_loop": ours, "one_shot": base,
                                       "passed": bool(gate), "verdict": verdict}})
    else:
        log("\nGO/NO-GO: could not evaluate — one of the key runs errored (see JSON).")


if __name__ == "__main__":
    main()
