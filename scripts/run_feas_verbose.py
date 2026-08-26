import json
import traceback
from pathlib import Path

from stare import config, data
from stare.simulator import AcquisitionSimulator
from stare.imputer import build_imputer
from stare.policies import build_policy


def run_policy(slide, imputer, policy_name, budget_fraction, **pkw):
    print(f'  running policy {policy_name} on slide {slide.patient_id}')
    sim = AcquisitionSimulator(slide, imputer, budget_fraction)
    policy = build_policy(policy_name, **pkw)
    state = sim.reset()
    step = 0
    while not sim.done:
        sel = policy.select(state)
        state = sim.reveal(sel)
        step += 1
        if step % 50 == 0:
            print(f'    progressed {step} rounds')
    print(f'  finished policy {policy_name}, score keys: {list(sim.score().keys())}')
    return sim.score()


def main():
    try:
        print('loading slides')
        tissue = config.FEASIBILITY_TISSUE
        slides = data.load_tissue(tissue)
        print('loaded slides:', [s.patient_id for s in slides])

        print('building imputer')
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
            print('=== RUN', name)
            per_slide = [run_policy(s, imputer, budget_fraction=config.FEASIBILITY_BUDGET_FRACTION, **kw) for s in slides]
            vals = [r[config.PRIMARY_METRIC] for r in per_slide]
            results[name] = {'mean': float(sum(vals) / len(vals)), 'per_slide': per_slide}

        out = Path(config.RESULTS_ROOT) / f"feasibility_{tissue}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({'tissue': tissue, 'budget_fraction': config.FEASIBILITY_BUDGET_FRACTION, 'primary_metric': config.PRIMARY_METRIC, 'results': results}, indent=2))
        print('wrote', out)

        gate = results['ours_closed_loop']['mean'] > results['oneshot_spatialfinder']['mean']
        print('GO/NO-GO:', gate)

    except Exception:
        traceback.print_exc()


if __name__ == '__main__':
    main()
