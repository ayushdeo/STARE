"""
Stage-2 GO/NO-GO (RUN ON 4070 after Stage-0/1 complete): the feasibility experiment.

On ONE tissue at the pinned feasibility budget, run every policy through the SAME frozen
imputer + simulator and compare imputation error. The decisive result:
    does GreedyInfoGain (closed-loop) beat OneShotSpatialFinder at equal budget?
Plus the ablations: closed-loop on/off, image on/off, spatial on/off.

Emits results/feasibility_<tissue>.json (this summary IS committed; raw arrays are gitignored).
See docs/REPRODUCTION_PLAN.md §4 Stage 2 and docs/SOURCE_OF_TRUTH.md §6.
"""
import json
from pathlib import Path

from stare import config, data
from stare.simulator import AcquisitionSimulator
from stare.imputer import build_imputer
from stare.policies import build_policy


def run_policy(slide, imputer, policy_name, budget_fraction, **pkw):
    sim = AcquisitionSimulator(slide, imputer, budget_fraction)
    policy = build_policy(policy_name, **pkw)
    state = sim.reset()
    while not sim.done:
        state = sim.reveal(policy.select(state))
    return sim.score()


def main():
    tissue = config.FEASIBILITY_TISSUE
    bf = config.FEASIBILITY_BUDGET_FRACTION
    slides = data.load_tissue(tissue)
    imputer = build_imputer(config.IMPUTER)   # frozen, shared across all policies

    runs = {
        "random": dict(policy_name="random"),
        "oneshot_spatialfinder": dict(policy_name="oneshot"),          # PRIMARY competitor
        "ours_closed_loop": dict(policy_name="greedy_ig", imputer=imputer),
        # --- ablations (WS-E decisive) ---
        "ablate_no_measured": dict(policy_name="greedy_ig", imputer=imputer, use_measured=False),
        "ablate_no_image":    dict(policy_name="greedy_ig", imputer=imputer, use_image=False),
        "ablate_no_spatial":  dict(policy_name="greedy_ig", imputer=imputer, use_spatial=False),
    }

    results = {}
    for name, kw in runs.items():
        per_slide = [run_policy(s, imputer, budget_fraction=bf, **kw) for s in slides]
        # average PRIMARY_METRIC across slides
        vals = [r[config.PRIMARY_METRIC] for r in per_slide]
        results[name] = {"mean": float(sum(vals) / len(vals)),
                         "per_slide": per_slide}

    out = Path(config.RESULTS_ROOT) / f"feasibility_{tissue}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"tissue": tissue, "budget_fraction": bf,
                               "primary_metric": config.PRIMARY_METRIC,
                               "results": results}, indent=2))
    print(f"wrote {out}")

    gate = results["ours_closed_loop"]["mean"] > results["oneshot_spatialfinder"]["mean"]
    print(f"\nGO/NO-GO: closed-loop ({results['ours_closed_loop']['mean']:.4f}) "
          f"{'>' if gate else '<='} one-shot ({results['oneshot_spatialfinder']['mean']:.4f}) "
          f"=> {'PROCEED to scale' if gate else 'STOP; reassess with TA'}")


if __name__ == "__main__":
    main()
