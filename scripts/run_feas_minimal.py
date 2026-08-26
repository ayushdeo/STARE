import json
from pathlib import Path

from stare import config, data
from stare.imputer import build_imputer
from stare.policies import build_policy
from stare.simulator import AcquisitionSimulator


def run_policy(slide, imputer, policy_name, **pkw):
    sim = AcquisitionSimulator(slide, imputer, config.FEASIBILITY_BUDGET_FRACTION)
    policy = build_policy(policy_name, **pkw)
    state = sim.reset()
    while not sim.done:
        state = sim.reveal(policy.select(state))
    return sim.score()


def main():
    tissue = config.FEASIBILITY_TISSUE
    slides = data.load_tissue(tissue)
    imputer = build_imputer(config.IMPUTER)

    runs = {
        'random': dict(policy_name='random'),
        'oneshot_spatialfinder': dict(policy_name='oneshot'),
        'ours_closed_loop': dict(policy_name='greedy_ig', imputer=imputer),
        'ablate_no_measured': dict(policy_name='greedy_ig', imputer=imputer, use_measured=False),
        'ablate_no_image': dict(policy_name='greedy_ig', imputer=imputer, use_image=False),
        'ablate_no_spatial': dict(policy_name='greedy_ig', imputer=imputer, use_spatial=False),
    }

    results = {}
    for name, kw in runs.items():
        per_slide = [run_policy(s, imputer, **kw) for s in slides]
        vals = [r[config.PRIMARY_METRIC] for r in per_slide]
        results[name] = {'mean': float(sum(vals) / len(vals)), 'per_slide': per_slide}

    out = Path(config.RESULTS_ROOT) / f"feasibility_{tissue}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'tissue': tissue, 'budget_fraction': config.FEASIBILITY_BUDGET_FRACTION, 'primary_metric': config.PRIMARY_METRIC, 'results': results}, indent=2))
    print('wrote', out)


if __name__ == '__main__':
    main()
