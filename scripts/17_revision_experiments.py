"""
Revision experiments, round 2 (reviewer major revision; RUN ON 4070). Pre-registered in docs/ITERATION_LOG.md
(iteration 9) before this script was run.

Part A (real slides, all budgets): every policy is run once per (slide, budget) and BOTH metrics are stored per
policy per slide, so feasible comparators can be formed afterwards (scripts/18_revision_analysis.py):
  non-adaptive pool : grid, kmeans, var_greedy, mi_greedy, oneshot     (LOSO comparator chosen from these)
  diversity         : scr2st                                           (non-adaptive)
  value-conditioned : value_greedy, + feedback ablations _perm (permuted values) and _const (constant values)
  reconstruction-aligned value-conditioned : residual_guided, + _perm
Part B (semi-synthetic reconstruction positive control on the same real slides): append 20 genes carrying an
image-unpredictable spatial hotspot field (3 hotspots, radius 5% of the slide diagonal) whose SD is alpha times
the median SD of the slide's top-50 genes, recompute the top-50 genes, and run the non-adaptive pool and residual_guided (+ _perm) with the primary imputer.

Results are checkpointed after every (slide, budget[, alpha]) so the run resumes after interruption.
Writes results/revision_runs.json.

Run:  python -u scripts/17_revision_experiments.py            # both parts
      python -u scripts/17_revision_experiments.py --part A   # or B
"""
import os, sys, json, time, argparse, zlib
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass
from dataclasses import replace
from pathlib import Path
import numpy as np

from stare import config, metrics
from stare.simulator import AcquisitionSimulator
from stare.imputer import build_imputer
from stare.policies import build_policy, FeedbackAblation
from stare.slides import load_ci_slides

IMPUTER = "graphflow"
BUDGETS_A = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
BUDGETS_B = [0.05, 0.10]
ALPHAS_B = [0.0, 1.0, 2.0]
ROI_QUANTILE = 0.80
NA_POOL = ["grid", "kmeans", "var_greedy", "mi_greedy", "oneshot"]
N_SPIKE, K_HOT, HOT_RADIUS = 20, 3, 0.05          # spiked genes, hotspots, radius as fraction of slide diagonal
OUT = Path(config.RESULTS_ROOT) / "revision_runs.json"


def log(m): print(m, flush=True)


def make_policy(name, imp):
    """'<policy>', '<policy>_perm' (permuted feedback) or '<policy>_const' (constant feedback)."""
    abl_mode, base = None, name
    for suffix, mode in (("_perm", "permute"), ("_const", "constant")):
        if name.endswith(suffix):
            abl_mode, base = mode, name[: -len(suffix)]
    kw = {"beta": 0.5} if base == "value_greedy" else ({"imputer": imp} if base == "residual_guided" else {})
    pol = build_policy(base, **kw)
    return FeedbackAblation(pol, abl_mode) if abl_mode else pol


def play(slide, imp, name, bf, roi, spike_idx=None):
    sim = AcquisitionSimulator(slide, imp, bf); st = sim.reset()
    pol = make_policy(name, imp)
    while not sim.done:
        st = sim.reveal(pol.select(st))
    prof = np.array(sim.profiled, dtype=int)
    out = {"pcc": float(sim.score()["pcc_top50"]), "recall": metrics.detection_recall(prof, roi)}
    if spike_idx is not None:                       # Part B: PCC over the spiked genes alone (secondary)
        state = sim._state()
        r = metrics.per_gene_pcc(slide.expression[state.unprofiled], imp.impute(state))
        out["pcc_spike"] = float(np.nanmean(r[spike_idx]))
    return out


def spike(slide, alpha):
    """Append N_SPIKE genes with an image-unpredictable hotspot field; recompute top-50 genes."""
    rng = np.random.default_rng(zlib.crc32(str(slide.patient_id).encode()) + config.SEED)
    co = slide.region_coords.astype(np.float64)
    rho = HOT_RADIUS * float(np.linalg.norm(co.max(0) - co.min(0)))
    centers = co[rng.choice(len(co), size=K_HOT, replace=False)]
    field = sum(np.exp(-((co - c) ** 2).sum(1) / (2 * rho ** 2)) for c in centers)
    field = field / max(field.max(), 1e-12)
    X = slide.expression.astype(np.float32)
    # scale: each spiked gene's SD = alpha x the median SD of the slide's own top-50 genes, so at alpha >= 1
    # the spiked genes compete for the evaluation set; plus small noise (0.1 x that SD)
    s_ref = float(np.median(X[:, np.asarray(slide.hvg_idx, dtype=int)].std(0)))
    amp = rng.uniform(0.5, 1.5, size=N_SPIKE)
    base = field[:, None] * amp[None, :]
    base = base / np.clip(base.std(0, keepdims=True), 1e-12, None)
    new = alpha * s_ref * base + rng.normal(0, 0.1 * s_ref, size=(len(co), N_SPIKE))
    Xa = np.concatenate([X, new.astype(np.float32)], axis=1)
    hvg = np.argsort(Xa.var(0))[::-1][: config.N_HVG].astype(int)
    meta = dict(slide.meta or {}); meta["spike"] = {"alpha": alpha, "n_genes": N_SPIKE, "k_hot": K_HOT, "radius": HOT_RADIUS}
    return replace(slide, expression=Xa, hvg_idx=hvg,
                   gene_names=(list(slide.gene_names) if slide.gene_names else []) + [f"SPIKE{i}" for i in range(N_SPIKE)],
                   meta=meta), int(np.sum(hvg >= X.shape[1]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", choices=["A", "B", "AB"], default="AB")
    args = ap.parse_args()
    imp = build_imputer(IMPUTER)
    slides, labels, group = load_ci_slides(log=log)
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    res.update({"imputer": IMPUTER, "slides": labels, "grouping": {k: group[k] for k in ("clusters", "strata")},
                "region_mode": config.REGION_MODE, "roi_quantile": ROI_QUANTILE, "na_pool": NA_POOL})
    res.setdefault("A", {}); res.setdefault("B", {})

    def save():
        OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")

    POL_A = NA_POOL + ["scr2st", "value_greedy", "value_greedy_perm", "value_greedy_const",
                       "residual_guided", "residual_guided_perm"]
    POL_B = NA_POOL + ["residual_guided", "residual_guided_perm"]

    if "A" in args.part:
        for slide, lab in zip(slides, labels):
            het = metrics.local_heterogeneity(slide.expression, slide.region_coords, slide.hvg_idx)
            roi = het >= np.quantile(het, ROI_QUANTILE)
            for bf in BUDGETS_A:
                cell = res["A"].setdefault(lab, {}).setdefault(f"{bf:.3f}", {})
                todo = [p for p in POL_A if p not in cell]
                if not todo:
                    continue
                t0 = time.time()
                for p in todo:
                    cell[p] = play(slide, imp, p, bf, roi)
                save()
                log(f"[A] {lab} budget {bf:.3f}: {len(todo)} policies in {time.time() - t0:.0f}s")

    if "B" in args.part:
        for slide, lab in zip(slides, labels):
            for alpha in ALPHAS_B:
                sl, n_in = spike(slide, alpha)
                het = metrics.local_heterogeneity(sl.expression, sl.region_coords, sl.hvg_idx)
                roi = het >= np.quantile(het, ROI_QUANTILE)
                for bf in BUDGETS_B:
                    cell = res["B"].setdefault(lab, {}).setdefault(f"{alpha:g}", {}).setdefault(f"{bf:.3f}", {})
                    cell["_spiked_in_top50"] = n_in
                    todo = [p for p in POL_B if p not in cell]
                    if not todo:
                        continue
                    t0 = time.time()
                    spike_idx = np.arange(slide.expression.shape[1], sl.expression.shape[1])
                    for p in todo:
                        cell[p] = play(sl, imp, p, bf, roi, spike_idx)
                    save()
                    log(f"[B] {lab} alpha {alpha:g} budget {bf:.3f}: {n_in}/50 spiked genes in top-50, "
                        f"{len(todo)} policies in {time.time() - t0:.0f}s")
    save()
    log(f"[write] {OUT}")


if __name__ == "__main__":
    main()
