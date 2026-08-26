"""Integration test: every policy runs end-to-end through the simulator on a synthetic slide,
and the closed-loop metric is well-formed. Data-independent (no HEST/UNI needed)."""
import numpy as np
import pytest

from stare.simulator import AcquisitionSimulator, Slide
from stare.imputer import KNNImputer, build_imputer
from stare.policies import build_policy, REGISTRY
from stare.metrics import morans_i


def make_slide(R=200, G=60, seed=0):
    """Synthetic slide with spatially-structured expression correlated to image embeddings,
    so a sensible policy should beat random (sanity, not a real result)."""
    rng = np.random.default_rng(seed)
    coords = rng.uniform(0, 2000, size=(R, 2))
    latent = rng.normal(size=(R, 8))
    emb = np.concatenate([latent, rng.normal(scale=0.1, size=(R, 1536 - 8))], axis=1)
    W = rng.normal(size=(8, G))
    expr = np.maximum(latent @ W + rng.normal(scale=0.3, size=(R, G)), 0.0)
    hvg = np.argsort(-expr.var(axis=0))[:50]
    return Slide(tissue="synthetic", region_coords=coords, image_embeddings=emb,
                 expression=expr, hvg_idx=hvg, patient_id="synthetic")


@pytest.mark.parametrize("name", list(REGISTRY))
def test_policy_runs_end_to_end(name):
    slide = make_slide()
    imp = KNNImputer(k=10)
    sim = AcquisitionSimulator(slide, imp, budget_fraction=0.10)
    kw = {"imputer": imp} if name in ("greedy_ig",) else {}
    policy = build_policy(name, **kw)
    state = sim.reset()
    steps = 0
    while not sim.done:
        state = sim.reveal(policy.select(state))
        steps += 1
    assert steps == sim.budget == 20
    score = sim.score()
    assert np.isfinite(score["pcc_top50"]) or np.isnan(score["pcc_top50"])
    assert score["mse"] >= 0


def test_closed_loop_flag_toggles_with_ablation():
    imp = build_imputer("knn")
    full = build_policy("greedy_ig", imputer=imp)                       # closed loop on
    ablated = build_policy("greedy_ig", imputer=imp, use_measured=False)  # closed loop off
    assert full.closed_loop is True
    assert ablated.closed_loop is False


def test_greedy_beats_random_on_structured_synthetic():
    """Sanity (NOT a paper result): on a slide where expression correlates with image, the
    greedy closed-loop policy should impute at least as well as random at equal budget."""
    slide = make_slide(seed=1)
    imp = KNNImputer(k=10)

    def run(policy_name, **kw):
        sim = AcquisitionSimulator(slide, imp, budget_fraction=0.15)
        pol = build_policy(policy_name, **kw)
        s = sim.reset()
        while not sim.done:
            s = sim.reveal(pol.select(s))
        return sim.score()["pcc_top50"]

    greedy = run("greedy_ig", imputer=imp)
    rand = run("random")
    assert greedy >= rand - 0.02   # allow tiny noise; expect greedy >= random


def test_morans_i_high_for_smooth_field():
    coords = np.stack(np.meshgrid(np.arange(10), np.arange(10)), -1).reshape(-1, 2).astype(float)
    smooth = (coords[:, [0]] + coords[:, [1]])            # spatially smooth -> high Moran's I
    noise = np.random.default_rng(0).normal(size=(100, 1))
    assert morans_i(smooth, coords, k=4) > morans_i(noise, coords, k=4)
