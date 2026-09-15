"""
Offline smoke test — validates the full acquisition/imputer/policy pipeline on a SYNTHETIC slide,
so bugs are caught WITHOUT the HEST download or a GPU (real runs happen on the 4070).

Builds a fake Slide with (a) an image-predictable gene component and (b) a spatially-smooth
component + noise — the structure real ST has — then exercises every imputer (knn, kriging,
graphflow) and a few policies (random, mi_greedy, ours) at two budgets. Asserts shapes/finiteness
and prints a mini per-imputer verdict so we can eyeball that graphflow behaves sanely before the
decisive run.  Run:  python -u tests/smoke_synth.py
"""
import os, sys
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "4")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass

import numpy as np
from stare import config
from stare.simulator import Slide, AcquisitionSimulator
from stare.imputer import build_imputer
from stare.policies import build_policy


def make_synth_slide(R_side=20, emb_dim=None, G=200, n_img_genes=40, seed=0) -> Slide:
    """Synthetic slide: image partially predicts a marker-gene subset; the rest is a smooth spatial
    field + noise. Mirrors the real regime (morphology predicts some genes, spatial predicts more)."""
    emb_dim = emb_dim or config.IMAGE_EMB_DIM
    rng = np.random.default_rng(seed)
    R = R_side * R_side
    gx, gy = np.meshgrid(np.arange(R_side), np.arange(R_side))
    coords = np.stack([gx.ravel(), gy.ravel()], 1).astype(np.float64) * config.REGION_UM

    # latent morphology (8-dim), smoothed spatially so nearby regions look alike
    m = rng.normal(size=(R, 8))
    d = ((coords[:, None, :] - coords[None, :, :]) ** 2).sum(2)
    A = np.exp(-d / (2 * (2 * config.REGION_UM) ** 2)); A /= A.sum(1, keepdims=True)
    m = A @ m                                             # spatial smoothing

    # image embeddings = random projection of morphology + noise, then L2-normalized (as in data.py)
    P = rng.normal(size=(8, emb_dim))
    emb = m @ P + 0.5 * rng.normal(size=(R, emb_dim))
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-8, None)

    # expression: image-predictable marker genes + smooth spatial genes + noise, made nonneg (log1p-like)
    expr = np.zeros((R, G))
    Wimg = rng.normal(size=(8, n_img_genes))
    expr[:, :n_img_genes] = m @ Wimg + 0.3 * rng.normal(size=(R, n_img_genes))
    spatial_field = A @ rng.normal(size=(R, G - n_img_genes))
    expr[:, n_img_genes:] = spatial_field + 0.3 * rng.normal(size=(R, G - n_img_genes))
    expr = np.log1p(np.clip(expr - expr.min(0, keepdims=True), 0, None))

    hvg_idx = np.argsort(expr.var(0))[::-1][:config.N_HVG].astype(int)
    return Slide(tissue="synth", region_coords=coords, image_embeddings=emb.astype(np.float32),
                 expression=expr.astype(np.float32), hvg_idx=hvg_idx)


def play(slide, imputer, policy_name, bf, pkw):
    sim = AcquisitionSimulator(slide, imputer, bf)
    pol = build_policy(policy_name, **pkw)
    st = sim.reset()
    while not sim.done:
        st = sim.reveal(pol.select(st))
    return sim.score()


def main():
    slide = make_synth_slide()
    print(f"[synth] R={slide.n_regions} G={slide.expression.shape[1]} hvg={len(slide.hvg_idx)}")
    budgets = [0.05, 0.20]
    policies = {
        "random":    ("random", {}, True),
        "var_greedy": ("var_greedy", {}, False),
        "mi_greedy": ("mi_greedy", {}, False),
        "ours":      ("greedy_ig", {}, False),          # imputer injected below
        "adaptive":  ("greedy_ig", {"criterion": "uncertainty", "use_image": False,
                                    "use_spatial": False}, False),   # clean adaptive placement
        "ours_nomeas": ("greedy_ig", {"use_measured": False}, False),
    }
    ok = True
    for imp_name in config.SWEEP_IMPUTERS:
        print(f"\n===== imputer={imp_name} =====")
        imputer = build_imputer(imp_name)
        for bf in budgets:
            row = {}
            for label, (pn, pkw, stoch) in policies.items():
                pk = dict(pkw)
                if pn == "greedy_ig":
                    pk["imputer"] = imputer
                if stoch:
                    vals = [play(slide, imputer, pn, bf, {**pk, "seed": s})["pcc_top50"] for s in range(3)]
                    v = float(np.mean(vals))
                else:
                    v = play(slide, imputer, pn, bf, pk)["pcc_top50"]
                assert np.isfinite(v), f"non-finite pcc_top50 for {imp_name}/{label}@{bf}"
                row[label] = v
            gap = row["ours"] - row["ours_nomeas"]
            adapt_gap = row["adaptive"] - row["var_greedy"]
            print(f"  bf={bf:.2f}  random={row['random']:.3f}  var_greedy={row['var_greedy']:.3f}  "
                  f"mi_greedy={row['mi_greedy']:.3f}  ours={row['ours']:.3f}  adaptive={row['adaptive']:.3f}")
            print(f"           closed_loop_gap(ours-nomeas)={gap:+.3f}  "
                  f"adaptivity_gap(adaptive-var_greedy)={adapt_gap:+.3f}  "
                  f"beats_mi={row['ours'] > row['mi_greedy']}")
    print(f"\n[smoke] all imputers ran, all metrics finite. OK={ok}")


if __name__ == "__main__":
    main()
