"""
Stage-4 experiments (RUN ON THE GPU MACHINE). Registered in docs/STAGE4_PLAN.md before this script is run; analysed
by scripts/28_stage4_analysis.py. Each part answers one critique of stages 1-3.

Part E  common evaluation set + constant-feedback controls (spot level, 22 slides, all six budgets).
        Per slide a fixed random 20% of regions is held out: no policy may select it, and reconstruction is scored on
        it for every policy (PCC over the top-50 genes and MSE), so policies are compared on the same regions.
        Policies run on the remaining 80%; the budget is still round(b x N) of all N regions.
        Runs: non-adaptive pool; residual (+ permuted, + constant); local-variance (+ permuted, + constant);
        half-coverage hybrids for reconstruction and detection (+ permuted, + constant).
Part U  ground-truth greedy upper bound for reconstruction (4 slides, budgets 0.5, 1, 2%). Each step adds, among
        K randomly drawn selectable candidates, the region that most increases held-out PCC computed with the true
        expression. Not deployable; it shows how much a better selection could gain under this imputer.
Part T  ROI-scale acquisition (22 slides). Spots are pooled into 300 um square tiles (>= 3 spots); a tile's
        expression is the mean of its spots, its embedding the mean of their embeddings. Budgets are 12, 24, and 48
        tiles. ROIs are the 20% of tiles with the highest within-tile heterogeneity (mean variance of the top-50 genes
        across the tile's spots). A 20% tile hold-out is used for reconstruction, as in Part E. Value-conditioned
        policies run sequentially, in 4 rounds, and in 2 rounds.
Part P  local-variance settings sweep (spot level, standard protocol of stage 2, detection budgets): beta in
        {0.25, 0.5, 1, 2} x kernel width in {0.05, 0.1, 0.2}. The analysis picks a setting per held-out study on the
        other 17 (nested), so the tuned policy is compared with the feasible design fairly.

Every (part, slide, budget) cell is checkpointed to results/stage4_runs.json; rerunning resumes.
Run:  python -u scripts/27_stage4_experiments.py --part E      (then T, P, U; or --part ETPU)
"""
import os, sys, json, time, math, argparse
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
from stare.simulator import AcquisitionSimulator, AcquisitionState
from stare.imputer import build_imputer
from stare.policies import build_policy, FeedbackAblation, HybridPolicy
from stare.slides import load_ci_slides

IMPUTER = "graphflow"
BUDGETS = ["0.005", "0.010", "0.020", "0.050", "0.100", "0.200"]
DET_BUDGETS = ["0.020", "0.050", "0.100", "0.200"]
POOL = ["grid", "kmeans", "var_greedy", "mi_greedy", "oneshot"]
REC_RULE = {"0.005": "kmeans", "0.010": "kmeans", "0.020": "mi_greedy", "0.050": "mi_greedy", "0.100": "mi_greedy", "0.200": "mi_greedy"}
DET_RULE = {"0.020": "var_greedy", "0.050": "var_greedy", "0.100": "var_greedy", "0.200": "oneshot"}
HOLDOUT = 0.20
ROI_QUANTILE = 0.80
PERM_SEED = 1
U_SLIDES = ["lung:TENX171", "colon:TENX155", "prostate:TENX174", "kidney:TENX173"]
U_BUDGETS = ["0.005", "0.010", "0.020"]
U_CANDIDATES = 48
TILE_UM = 300.0
TILE_MIN_SPOTS = 3
T_BUDGETS = [12, 24, 48]
T_ROUNDS = [0, 4, 2]                # 0 = sequential
P_BETA = [0.25, 0.5, 1.0, 2.0]
P_WIDTH = [0.05, 0.1, 0.2]
OUT = Path(config.RESULTS_ROOT) / "stage4_runs.json"


def log(m): print(m, flush=True)


# ------------------------------------------------------------------ slides, hold-outs, tiles
def subslide(slide, idx):
    """The slide restricted to regions idx (policies only ever see the selectable regions)."""
    meta = dict(slide.meta or {}, subset_of=slide.n_regions)
    meta["region_mode"] = f"{meta.get('region_mode', 'spot')}_holdout"      # own key in the placement memo
    return replace(slide, region_coords=slide.region_coords[idx], image_embeddings=slide.image_embeddings[idx],
                   expression=slide.expression[idx], meta=meta)


def holdout(n, slide_index):
    """Fixed random 20% of regions per slide (seeded by slide position), never selectable."""
    rng = np.random.default_rng([config.SEED, slide_index])
    h = np.sort(rng.choice(n, size=int(round(HOLDOUT * n)), replace=False))
    sel = np.setdiff1d(np.arange(n), h)
    return h, sel


def tile_slide(slide):
    """Pool spots into TILE_UM squares; returns the tile slide and each tile's within-tile heterogeneity."""
    xy = slide.region_coords - slide.region_coords.min(0)
    key = np.floor(xy / TILE_UM).astype(int)
    ids, inv = np.unique(key[:, 0] * 100000 + key[:, 1], return_inverse=True)
    counts = np.bincount(inv)
    keep = np.flatnonzero(counts >= TILE_MIN_SPOTS)
    expr, emb, xy_t, het = [], [], [], []
    hv = slide.hvg_idx
    for t in keep:
        m = inv == t
        e = slide.expression[m]
        expr.append(e.mean(0)); emb.append(slide.image_embeddings[m].mean(0)); xy_t.append(slide.region_coords[m].mean(0))
        het.append(float(e[:, hv].var(0).mean()))
    ts = replace(slide, region_coords=np.array(xy_t), image_embeddings=np.array(emb, dtype=slide.image_embeddings.dtype),
                 expression=np.array(expr, dtype=slide.expression.dtype),
                 meta=dict(slide.meta or {}, region_mode=f"tile{TILE_UM:g}", n_spots=slide.n_regions))
    return ts, np.array(het)


# ------------------------------------------------------------------ policies
def make(name, b, imp):
    """'<base>[_perm|_const]'; bases: pool names, residual, local_variance, local_variance:<beta>:<width>,
    hyb_rec, hyb_det (half coverage)."""
    mode = None
    for suf, m in (("_perm", "permute"), ("_const", "constant")):
        if name.endswith(suf):
            name, mode = name[: -len(suf)], m
    if name in POOL:
        pol = build_policy(name)
    elif name == "residual":
        pol = build_policy("residual_guided", imputer=imp)
    elif name.startswith("local_variance"):
        parts = name.split(":")
        beta, width = (float(parts[1]), float(parts[2])) if len(parts) == 3 else (0.5, 0.1)
        pol = build_policy("value_greedy", beta=beta, lengthscale_frac=width)
    elif name == "hyb_rec":
        pol = HybridPolicy(build_policy(REC_RULE[b]), build_policy("residual_guided", imputer=imp), 0.5)
    elif name == "hyb_det":
        pol = HybridPolicy(build_policy(DET_RULE[b]), build_policy("value_greedy", beta=0.5), 0.5)
    else:
        raise ValueError(name)
    return FeedbackAblation(pol, mode, seed=PERM_SEED) if mode else pol


def select(slide, imp, pol, n, round_size=1):
    """Run a policy for n picks on `slide`; within a round it sees only measurements from earlier rounds."""
    sim = AcquisitionSimulator(slide, imp, n / slide.n_regions)
    assert sim.budget == n, (sim.budget, n)
    sim.reset()
    while not sim.done:
        k = min(round_size, sim.budget - len(sim.profiled))
        mask, prof = sim._mask.copy(), np.array(sim.profiled, dtype=int)
        pending = []
        for _ in range(k):
            pick = int(pol.select(AcquisitionState(slide=slide, profiled=prof, mask=mask, budget=sim.budget)))
            pending.append(pick); mask[pick] = True
        sim.reveal(pending)
    return np.array(sim.profiled, dtype=int)


def score_holdout(slide, imp, profiled, h):
    """Impute the full slide from `profiled` and score the held-out regions h (top-50 genes)."""
    sim = AcquisitionSimulator(slide, imp, len(profiled) / slide.n_regions)
    sim.reset(); sim.reveal([int(i) for i in profiled])
    st = sim._state()
    pred = imp.impute(st)
    pos = np.searchsorted(st.unprofiled, h)
    assert np.array_equal(st.unprofiled[pos], h)
    hv = slide.hvg_idx
    yt, yp = slide.expression[h][:, hv], pred[pos][:, hv]
    return {"pcc": float(np.nanmean(metrics.per_gene_pcc(yt, yp))), "mse": float(np.mean((yt - yp) ** 2))}


def run_holdout(slide, sub, h, sel, roi, name, b_key, n, imp_pol, imp_score, round_size=1):
    picks = sel[select(sub, imp_pol, make(name, b_key, imp_pol), n, round_size)]
    out = score_holdout(slide, imp_score, picks, h)
    out["recall"] = float(roi[picks].sum() / max(roi[sel].sum(), 1))
    return out


# ------------------------------------------------------------------ parts
def part_E(slide, i, roi, cell_for, save):
    h, sel = holdout(slide.n_regions, i)
    sub = subslide(slide, sel)
    imp_pol, imp_score = build_imputer(IMPUTER), build_imputer(IMPUTER)
    for b in BUDGETS:
        n = int(round(float(b) * slide.n_regions))
        names = POOL + ["residual", "residual_perm", "residual_const", "hyb_rec", "hyb_rec_perm", "hyb_rec_const"]
        if b in DET_BUDGETS:
            names += ["local_variance", "local_variance_perm", "local_variance_const", "hyb_det", "hyb_det_perm", "hyb_det_const"]
        cell = cell_for(b)
        todo = [x for x in names if x not in cell]
        t0 = time.time()
        for x in todo:
            cell[x] = run_holdout(slide, sub, h, sel, roi, x, b, n, imp_pol, imp_score)
        if todo:
            cell["_n_select"], cell["_n_holdout"] = int(sel.size), int(h.size)
            save(); log(f"    E {b}: {len(todo)} runs in {time.time() - t0:.0f}s")


def part_U(slide, i, cell_for, save, rng_seed):
    h, sel = holdout(slide.n_regions, i)
    imp = build_imputer(IMPUTER)
    for b in U_BUDGETS:
        cell = cell_for(b)
        if "greedy_truth" in cell:
            continue
        n = int(round(float(b) * slide.n_regions))
        rng = np.random.default_rng([rng_seed, i, int(float(b) * 1000)])
        prof, avail, path, t0 = [], list(sel), [], time.time()
        for step in range(n):
            cand = rng.choice(avail, size=min(U_CANDIDATES, len(avail)), replace=False)
            scores = [score_holdout(slide, imp, prof + [int(c)], h)["pcc"] for c in cand]
            best = int(cand[int(np.nanargmax(scores))])
            prof.append(best); avail.remove(best); path.append(float(np.nanmax(scores)))
            if step == 0:
                log(f"    U {b}: first step {time.time() - t0:.1f}s for {len(cand)} candidates; {n} steps")
        cell["greedy_truth"] = {**score_holdout(slide, imp, prof, h), "path_pcc": path}
        save(); log(f"    U {b}: done in {time.time() - t0:.0f}s")


def part_T(slide, i, cell_for, save):
    ts, het = tile_slide(slide)
    roi = het >= np.quantile(het, ROI_QUANTILE)
    h, sel = holdout(ts.n_regions, i)
    sub = subslide(ts, sel)
    imp_pol, imp_score = build_imputer(IMPUTER), build_imputer(IMPUTER)
    for n in T_BUDGETS:
        if n > sel.size // 2:
            continue
        cell = cell_for(str(n))
        cell.setdefault("_n_tiles", int(ts.n_regions))
        todo = []
        for x in POOL:
            if x not in cell:
                todo.append((x, 1))
        for base in ("residual", "local_variance"):
            for suf in ("", "_perm", "_const"):
                for r in T_ROUNDS:
                    key = f"{base}{suf}" + (f"@r{r}" if r else "")
                    if key not in cell and (r == 0 or suf == ""):     # ablations sequential only
                        todo.append((key, r))
        t0 = time.time()
        for key, r in todo:
            name = key.split("@")[0]
            rs = 1 if r in (0, 1) else math.ceil(n / r)
            cell[key] = run_holdout(ts, sub, h, sel, roi, name, "0.020", n, imp_pol, imp_score, rs)
        if todo:
            save(); log(f"    T {n} tiles: {len(todo)} runs in {time.time() - t0:.0f}s")


def part_P(slide, roi, cell_for, save):
    imp = build_imputer(IMPUTER)
    for b in DET_BUDGETS:
        cell = cell_for(b)
        n = int(round(float(b) * slide.n_regions))
        todo = [f"local_variance:{be:g}:{w:g}" for be in P_BETA for w in P_WIDTH]
        todo = [x for x in todo if x not in cell]
        t0 = time.time()
        for x in todo:
            prof = select(slide, imp, make(x, b, imp), n)
            cell[x] = {"recall": float(roi[prof].sum() / roi.sum())}
        if todo:
            save(); log(f"    P {b}: {len(todo)} runs in {time.time() - t0:.0f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--part", default="ETPU", help="any of E, T, P, U")
    args = ap.parse_args()
    slides, labels, group = load_ci_slides(log=log)
    res = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    res.update({"slides": labels, "grouping": {k: group[k] for k in ("clusters", "strata")}, "imputer": IMPUTER,
                "holdout": HOLDOUT, "rec_rule": REC_RULE, "det_rule": DET_RULE, "tile_um": TILE_UM,
                "t_rounds": T_ROUNDS, "u_candidates": U_CANDIDATES, "perm_seed": PERM_SEED})
    save = lambda: OUT.write_text(json.dumps(res, indent=1), encoding="utf-8")
    for i, (slide, lab) in enumerate(zip(slides, labels)):
        het = metrics.local_heterogeneity(slide.expression, slide.region_coords, slide.hvg_idx)
        roi = het >= np.quantile(het, ROI_QUANTILE)
        cf = lambda part: (lambda b: res.setdefault(part, {}).setdefault(lab, {}).setdefault(b, {}))
        log(f"[{i + 1}/{len(slides)}] {lab} ({slide.n_regions} regions)")
        if "E" in args.part:
            part_E(slide, i, roi, cf("E"), save)
        if "T" in args.part:
            part_T(slide, i, cf("T"), save)
        if "P" in args.part:
            part_P(slide, roi, cf("P"), save)
        if "U" in args.part and lab in U_SLIDES:
            part_U(slide, i, cf("U"), save, config.SEED)
    save()
    log(f"[write] {OUT}")


if __name__ == "__main__":
    main()
