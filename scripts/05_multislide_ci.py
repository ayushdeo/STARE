"""
Iteration-3 CI run (RUN ON 4070): cross-slide significance for the closed-loop margin.

Pools every pinned slide across config.CI_TISSUES (prostate+kidney+lung+colon = 7 slides) and, for
each imputer and budget, records PER-SLIDE pcc_top50 for every policy. It then summarizes the paired
margins that decide the thesis, with a bootstrap CI + exact sign test (src/stare/stats.py):

    ours_vs_mi            : ours_closed_loop  -  mi_greedy            (adaptive vs strong non-adaptive)
    ours_vs_best_nonadapt : ours_closed_loop  -  max(non-adaptive)    (the FAIREST, hardest test)
    adaptive_vs_var       : adaptive_uncert.  -  var_greedy           (clean adaptive vs its twin)
    closed_vs_ablate      : ours_closed_loop  -  ablate_no_measured   (does using measured help)

DECISION RULE (honest): a closed-loop WIN requires ours_vs_best_nonadapt CI to EXCLUDE 0 (positive)
at some budget. If every CI straddles 0, the verdict is "tie / no gain" and the negative result
stands with cross-slide evidence. Writes results/ci_multislide_<imputer>.json (committed).

Run unbuffered:  python -u scripts/05_multislide_ci.py
"""
import os
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
import sys
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
import json, pickle, time
from pathlib import Path

from stare import config, data, stats
from stare.simulator import AcquisitionSimulator
from stare.imputer import build_imputer
from stare.policies import build_policy

NEEDS_IMPUTER = {"greedy_ig"}
# Keep this policy set in sync with scripts/03_budget_sweep.py POLICIES.
POLICIES = {
    "random":                ("random", {}, True),
    "grid":                  ("grid", {}, False),
    "kmeans":                ("kmeans", {}, False),
    "oneshot_spatialfinder": ("oneshot", {}, False),
    "var_greedy":            ("var_greedy", {}, False),
    "mi_greedy":             ("mi_greedy", {}, False),
    "ours_closed_loop":      ("greedy_ig", {}, False),
    "adaptive_uncertainty":  ("greedy_ig", {"criterion": "uncertainty",
                                            "use_image": False, "use_spatial": False}, False),
    "ablate_no_measured":    ("greedy_ig", {"use_measured": False}, False),
}
NON_ADAPTIVE = ["grid", "kmeans", "oneshot_spatialfinder", "var_greedy", "mi_greedy"]


def log(m): print(m, flush=True)


def load_or_cache_slides(tissue):
    cache_dir = Path(config.CACHE_ROOT); cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"slides_{tissue}_{int(config.REGION_UM)}um_{config.IMAGE_ENCODER}.pkl"
    if p.exists():
        log(f"[cache] {p}")
        return pickle.load(open(p, "rb"))
    log(f"[build] {tissue}: no cache; building (minutes)...")
    slides = data.load_tissue(tissue)
    pickle.dump(slides, open(p, "wb"))
    return slides


def play_slide(slide, imputer, pn, bf, pkw, seed=None):
    """pcc_top50 for one (slide, policy, budget). Random is averaged over RANDOM_SEEDS by the caller."""
    pk = dict(pkw)
    if pn in NEEDS_IMPUTER:
        pk.setdefault("imputer", imputer)
    if seed is not None:
        pk["seed"] = seed
    sim = AcquisitionSimulator(slide, imputer, bf)
    pol = build_policy(pn, **pk)
    st = sim.reset()
    while not sim.done:
        st = sim.reveal(pol.select(st))
    return float(sim.score()["pcc_top50"])


def run_imputer(imputer_name, slides, slide_labels, budgets):
    imputer = build_imputer(imputer_name)
    log(f"\n########## IMPUTER = {imputer_name} | {len(slides)} slides ##########")
    # per_slide[policy][budget] = [value per slide]
    per_slide = {name: {f"{b:.3f}": [] for b in budgets} for name in POLICIES}
    for si, slide in enumerate(slides):
        for b in budgets:
            bkey = f"{b:.3f}"
            for name, (pn, pkw, stoch) in POLICIES.items():
                t0 = time.time()
                if stoch:
                    vals = [play_slide(slide, imputer, pn, b, pkw, seed=s) for s in config.RANDOM_SEEDS]
                    v = float(sum(vals) / len(vals))
                else:
                    v = play_slide(slide, imputer, pn, b, pkw)
                per_slide[name][bkey].append(v)
            log(f"  slide {si+1}/{len(slides)} [{slide_labels[si]}] budget {bkey} done ({time.time()-t0:.1f}s/policy-last)")

    # paired margins with CIs, per budget
    import numpy as np
    margins = {}
    for b in budgets:
        bkey = f"{b:.3f}"
        ours = np.array(per_slide["ours_closed_loop"][bkey])
        mi = np.array(per_slide["mi_greedy"][bkey])
        adapt = np.array(per_slide["adaptive_uncertainty"][bkey])
        varg = np.array(per_slide["var_greedy"][bkey])
        ablate = np.array(per_slide["ablate_no_measured"][bkey])
        best_na = np.max(np.vstack([per_slide[n][bkey] for n in NON_ADAPTIVE]), axis=0)
        margins[bkey] = {
            "ours_vs_mi": stats.paired_margin_summary(ours, mi, config.CI_BOOTSTRAP, config.CI_ALPHA, config.CI_SEED),
            "ours_vs_best_nonadapt": stats.paired_margin_summary(ours, best_na, config.CI_BOOTSTRAP, config.CI_ALPHA, config.CI_SEED),
            "adaptive_vs_var": stats.paired_margin_summary(adapt, varg, config.CI_BOOTSTRAP, config.CI_ALPHA, config.CI_SEED),
            "closed_vs_ablate": stats.paired_margin_summary(ours, ablate, config.CI_BOOTSTRAP, config.CI_ALPHA, config.CI_SEED),
        }
        m = margins[bkey]["ours_vs_best_nonadapt"]
        log(f"  [{imputer_name}] budget {bkey}: ours-best_nonadapt mean={m['mean']} "
            f"CI95={m['ci95']} excl0={m['ci_excludes_zero']} sign={m['n_pos']}/{m['n_pos']+m['n_neg']} p={m['sign_p']}")

    # win = any budget where ours beats the BEST non-adaptive with CI excluding 0
    win_budgets = [b for b in margins if margins[b]["ours_vs_best_nonadapt"]["ci_excludes_zero"]
                   and margins[b]["ours_vs_best_nonadapt"]["mean"] > 0]
    out = Path(config.RESULTS_ROOT) / f"ci_multislide_{imputer_name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "imputer": imputer_name, "tissues": config.CI_TISSUES, "slides": slide_labels,
        "budgets": [f"{b:.3f}" for b in budgets], "metric": "pcc_top50",
        "per_slide": per_slide, "margins": margins,
        "win_budgets_ours_vs_best_nonadapt": win_budgets,
    }, indent=2), encoding="utf-8")
    log(f"[write] {out}  | WIN budgets (ours>best non-adaptive, CI excl 0): {win_budgets or 'NONE'}")
    return imputer_name, win_budgets


def main():
    budgets = config.SWEEP_BUDGET_FRACTIONS
    slides, labels = [], []
    for tissue in config.CI_TISSUES:
        ts = load_or_cache_slides(tissue)
        for s in ts:
            slides.append(s); labels.append(f"{tissue}:{getattr(s, 'patient_id', '?')}")
    log(f"=== multislide CI | tissues={config.CI_TISSUES} slides={labels} "
        f"imputers={config.SWEEP_IMPUTERS} budgets={budgets} ===")
    summary = [run_imputer(name, slides, labels, budgets) for name in config.SWEEP_IMPUTERS]
    log("\n===== CI HEADLINE (ours vs BEST non-adaptive, CI excludes 0) =====")
    for name, wins in summary:
        log(f"  imputer={name:9s} closed-loop WIN budgets: {wins or 'NONE'}")
    log("Interpretation: NONE everywhere -> negative result holds with cross-slide evidence.\n"
        "Any budget listed -> a real (if small) closed-loop gain there; report it with its CI.")


if __name__ == "__main__":
    main()
