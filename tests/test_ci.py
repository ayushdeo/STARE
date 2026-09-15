"""
Offline verification of the CI math (src/stare/stats.py) and the multi-slide margin pipeline,
on synthetic slides — no HEST/GPU. Run:  python -u tests/test_ci.py
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.path.insert(0, os.path.dirname(__file__))
import numpy as np
from stare import stats
from stare.imputer import build_imputer
from smoke_synth import make_synth_slide, play as play_policy


def approx(a, b, tol=1e-9): return abs(a - b) <= tol


def test_bootstrap_ci_constant():
    lo, hi = stats.bootstrap_ci(np.full(6, 0.3), B=200, seed=1)
    assert approx(lo, 0.3) and approx(hi, 0.3), (lo, hi)


def test_bootstrap_ci_singleton():
    assert stats.bootstrap_ci(np.array([0.7])) == (0.7, 0.7)


def test_paired_all_positive():
    s = stats.paired_margin_summary([1, 1, 1, 1, 1, 1, 1], [0]*7, B=500, seed=0)
    assert s["mean"] == 1.0 and s["n_pos"] == 7 and s["n_neg"] == 0
    assert s["ci_excludes_zero"] is True
    # exact two-sided sign test for 7/7: 2/128 = 0.0156
    assert approx(s["sign_p"], 0.0156, 1e-3), s["sign_p"]


def test_paired_split_straddles_zero():
    s = stats.paired_margin_summary([1, -1, 1, -1], [0, 0, 0, 0], B=500, seed=0)
    assert s["n_pos"] == 2 and s["n_neg"] == 2
    assert s["ci_excludes_zero"] is False
    assert approx(s["sign_p"], 1.0), s["sign_p"]   # 2/4 -> perfectly balanced


def test_multislide_margin_pipeline():
    """Build several synthetic slides, compute per-slide ours vs mi margins, summarize. Just needs to
    run and produce a well-formed summary (values finite, n matches)."""
    slides = [make_synth_slide(seed=k) for k in range(4)]
    imp = build_imputer("graphflow")
    bf = 0.10
    ours = np.array([play_policy(s, imp, "greedy_ig", bf, {"imputer": imp})["pcc_top50"] for s in slides])
    mi = np.array([play_policy(s, imp, "mi_greedy", bf, {})["pcc_top50"] for s in slides])
    summ = stats.paired_margin_summary(ours, mi, B=1000, seed=0)
    assert summ["n_slides"] == 4
    assert len(summ["per_slide"]) == 4
    assert np.isfinite(summ["mean"])
    assert summ["ci95"][0] <= summ["ci95"][1]
    print(f"  [pipeline] ours-mi per slide={summ['per_slide']} mean={summ['mean']} "
          f"CI={summ['ci95']} excl0={summ['ci_excludes_zero']}")


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    for t in tests:
        t(); print(f"  ok: {t.__name__}")
    print(f"\n[test_ci] {len(tests)} tests passed.")


if __name__ == "__main__":
    main()
