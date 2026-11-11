"""
Post hoc run for the GPU machine: the tuned local-variance settings with constant feedback (stage-2 protocol).

Why: under constant feedback the local-variance term should vanish, which would make the constant version identical
for every setting. It does not at kernel width 0.05: the policy clips the kernel sum at 1e-12, so a candidate more
than about 7.4 kernel widths from every profiled region gets a small artificial variance, which changes the first
few picks (scripts/30_posthoc_stage4.py checks this on synthetic slides; widths 0.1 and 0.2 are unaffected). The
stage-2 constant run (width 0.1) is therefore only an approximate control for the tuned settings. This script runs
the exact controls. Status: post hoc.

Runs, per slide and detection budget: local_variance:<beta>:0.05 with constant feedback for beta in {0.25, 0.5}
(every setting the tuning chose), and local_variance:0.5:0.1 with constant feedback as a consistency check against
the stage-2 run.
Output: results/stage4_posthoc_runs.json (checkpointed per slide). Then rerun scripts/30_posthoc_stage4.py.
Run:    python -u scripts/31_posthoc_tuned_constant.py
"""
import os, sys, json, time, importlib.util
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pathlib import Path
import numpy as np
from stare import config, metrics
from stare.imputer import build_imputer
from stare.slides import load_ci_slides

spec = importlib.util.spec_from_file_location("s27", Path(__file__).with_name("27_stage4_experiments.py"))
s27 = importlib.util.module_from_spec(spec); spec.loader.exec_module(s27)
RUNS = ["local_variance:0.25:0.05_const", "local_variance:0.5:0.05_const", "local_variance:0.5:0.1_const"]
OUT = Path(config.RESULTS_ROOT) / "stage4_posthoc_runs.json"


def main():
    slides, labels, group = load_ci_slides(log=s27.log)
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    res.update({"slides": labels, "grouping": {k: group[k] for k in ("clusters", "strata")}, "runs": RUNS})
    save = lambda: OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for i, (slide, lab) in enumerate(zip(slides, labels)):
        het = metrics.local_heterogeneity(slide.expression, slide.region_coords, slide.hvg_idx)
        roi = het >= np.quantile(het, s27.ROI_QUANTILE)
        imp = build_imputer(s27.IMPUTER)
        s27.log(f"[{i + 1}/{len(slides)}] {lab} ({slide.n_regions} regions)")
        for b in s27.DET_BUDGETS:
            cell = res.setdefault("P_const", {}).setdefault(lab, {}).setdefault(b, {})
            n = int(round(float(b) * slide.n_regions))
            t0 = time.time()
            for x in [r for r in RUNS if r not in cell]:
                prof = s27.select(slide, imp, s27.make(x, b, imp), n)
                cell[x] = {"recall": float(roi[prof].sum() / roi.sum())}
            save(); s27.log(f"    {b}: {time.time() - t0:.0f}s")
    save(); s27.log(f"[write] {OUT}")


if __name__ == "__main__":
    main()
