# Iteration Log — honest record of what we tried and what it showed

Each entry: what we changed, the result file, the verdict, and what it implies. Tagged
**[OURS]** (our run) / **[LIT]** (published) so nothing rests on assumption. Companion to
`PROGRESS.md`. Prostate, 2249 regions, UNI2-h, top-50 HVG PCC (primary), over unprofiled regions.

---

## Iteration 0 — image-kNN imputer + uncertainty-greedy policy  [OURS: results/sweep_prostate.json @ 9d78e4a-era]
- Imputer: kNN in image-embedding space. Policy: greedy pick of the single most-uncertain region.
- Result: `kmeans` (one-shot image coverage) wins at every budget; `ours` ~ `random`; loop-on ≈ loop-off.
- Verdict: **closed loop adds nothing.** Diagnosed to the imputer (coverage interpolator).

## Iteration 1 — joint image+spatial kNN imputer + expected-error-reduction greedy  [OURS: results/sweep_prostate.json @ f2c0c9c]
- Imputer: kNN in joint (image (+) physical-space) features. Policy: greedy pick of region that
  reduces error over its neighbourhood (AFA / Bayesian-experimental-design objective [AFA-RL]).
- Result: **WORSE.** `ours_closed_loop` now loses to `random`, `kmeans`, and `ablate_no_measured`
  at every budget (e.g. 20%: ours 0.781 vs kmeans 0.885). Error-reduction greedy over-clusters picks
  in high-variance regions -> poor coverage -> worse reconstruction. `kmeans` still wins everywhere.
- Bug found: under the `error_reduction` criterion the policy ignores `use_image`/`use_spatial`, so
  `ablate_no_image` == `ablate_no_spatial` == `ours` (byte-identical). Those ablations are currently
  meaningless and must be redefined (see "Fixes owed").

## The crystallized finding (robust across 2 heuristics x 2 imputers)  [OURS + LIT]
**One-shot spatial+image COVERAGE (k-means) is near-optimal for reconstruction; conditioning the
acquisition on measured expression does not help.**

This is **predicted by established theory**, not a failure of engineering:
- Krause, Guestrin & Singh, "Near-Optimal Sensor Placements in Gaussian Processes," JMLR 2008
  [https://jmlr.org/papers/v9/krause08a.html]. For a GP with a FIXED/known kernel the posterior
  variance is **independent of the observed values** -> the best set of locations can be chosen
  offline; **adaptive (closed-loop) placement gives no gain over one-shot placement.** A kNN imputer
  is exactly this fixed-kernel regime. Our World-2 result is the theorem's predicted behaviour.
- The same literature says variance/mutual-information greedy placement is submodular -> greedy is
  (1-1/e)-optimal, and mutual-information placement beats pure-variance/coverage. We have NOT yet run
  the principled MI/variance placement baselines — we ran ad-hoc heuristics instead.

**Implication (this is the pivot):** the closed loop can only earn its complexity when the imputer's
uncertainty DEPENDS on measured values — i.e. a **non-stationary or learned** model. Our kNN is
stationary by construction, so it can never show a closed-loop benefit. Continuing to hand-tune
greedy heuristics on a kNN imputer is futile; the theory says so.

## Iteration 2 — principled placement baselines + learned kriging imputer  [OURS: results/sweep_prostate_{knn,kriging}.json @ 2dcb20d]
- Did the "fixes owed": added Krause variance-greedy + MI-greedy (the correct NON-adaptive opponents),
  and a learned non-stationary imputer (regression-kriging: ridge image->expr refit on profiled +
  spatial residual kriging). Gate = beat MI-greedy AND closed_loop_helps.
- Result: **CORE-PASS = NONE under both imputers.** `closed_loop_helps=False` at every budget.
  Under kriging, MI-greedy is the single best method (0.856 @20% vs ours 0.676 vs kmeans 0.811).
  ours-MI @20% = -0.10 (knn), -0.18 (kriging).
- Verdict: **closed loop still earns nothing**, now against the *principled* opponent, under a
  stationary AND a linear-learned imputer. Caveat logged: ours (error-reduction greedy) over-clusters
  and is a weak policy; kriging is a weak imputer at low budget. One door left open: a STRONG learned
  imputer (STFlow-class) untested.

## Iteration 3 — STFlow-class imputer + adaptive policy, single slide  [OURS: results/sweep_prostate_graphflow.json @ 513c7a2]
- Added GraphFlowImputer (STFlow-inspired: nonlinear RFF image->expr backbone + iterative spatial
  message passing that clamps measured regions; data-dependent uncertainty), and a clean ADAPTIVE
  max-uncertainty policy (the adaptive twin of var_greedy). Closes the weak-imputer AND weak-policy gaps.
- Result on prostate (n=1 slide): the literal gate **passes at 5/10/20%** (ours-MI = +0.002, +0.003,
  +0.006) and `closed_loop_helps=True`, `adaptive_beats_nonadaptive=True` at all budgets.
- **BUT the wins are not real yet:** (a) margins (+0.002..+0.006) are ≤ the random baseline's own
  seed std (±0.002..±0.005); (b) they occur only at HIGH budget where the task saturates (all methods
  0.84-0.88); (c) at LOW budget (0.5-2%, where acquisition matters) a one-shot image **k-means wins
  outright** (0.569 vs ours 0.368 @0.5%); (d) n=1 slide, deterministic policies carry no error bars.
- The genuinely informative signal: **the adaptivity gap closes monotonically as the imputer becomes
  more data-dependent** — ours-MI @20% = -0.10 (kNN) -> -0.18 (kriging) -> ~0 (graphflow). This is
  qualitatively what Krause 2008 predicts, but it approaches 0 *from below* and never becomes a robust
  positive on one slide.
- Verdict: **NOT a win, NOT a clean null — a statistical tie at saturation, pending significance.**

## Iteration 3-CI — cross-slide significance run  [OURS: results/ci_multislide_*.json — launched 2026-09-15]
- Change: pool all pinned slides across prostate+kidney+lung+colon (7 slides), report the paired
  margin **ours - best-non-adaptive** per budget with a percentile bootstrap CI + exact sign test
  (src/stare/stats.py, verified in tests/test_ci.py). Runner: scripts/05_multislide_ci.py.
- **Pre-registered decision rule:** a closed-loop WIN requires `ours - best_non_adaptive` mean > 0
  with 95% CI excluding 0 at ≥1 budget. If every CI straddles 0 -> negative result stands with
  cross-slide evidence; the paper is the theory-backed characterization.
- **RESULT [OURS: results/ci_multislide_*.json @ 8f26d06, 7 slides prostate+kidney+lung+colon]:**
  **WIN budgets = NONE, for all three imputers.** `ours - best_non_adaptive` is negative at EVERY
  budget under every imputer; the CI excludes 0 (on the negative side) in most cells and merely
  straddles 0 (mean still negative) at graphflow's high budgets (10%: -0.005 [-0.017,+0.003];
  20%: -0.004 [-0.009,+0.001]). The single-slide graphflow "wins" at 5/10/20% were NOISE — they
  vanish and go slightly negative across 7 slides. **The negative result holds with cross-slide
  significance.**
- **Mechanism confirmed (the interesting part):** `closed_vs_ablate` (ours vs its image-only greedy
  twin) is significantly POSITIVE only under graphflow (+0.08..+0.12, CI excl 0, up to 7/7 slides at
  1-5%) and null/negative under kNN & kriging. So conditioning the POLICY on measured expression
  extracts real, robust signal — but ONLY when the imputer is data-dependent (exactly Krause's
  precondition), and even then only enough to TIE one-shot placement at high budget, never to beat it,
  while still LOSING at the low budgets where acquisition matters. The adaptivity gap closes toward 0
  from below as the imputer strengthens (kNN -0.13 -> kriging -0.13 -> graphflow -0.004 @20%) but
  never crosses 0.
- **Verdict: the closed-loop / agentic framing does not earn its complexity.** One-shot placement
  (k-means / MI-greedy) is near-optimal for reconstruction; adaptivity provides no significant gain
  at any budget under any imputer we tested, up to the STFlow-class limit. -> Paper = the theory-backed
  negative characterization (docs/PAPER_OUTLINE.md), now with cross-slide CIs as the key evidence.

## Iteration 4 — reviewer response: reconstruction-vs-detection split  [OURS, 2026-09-24]
Reviewer (weak reject) correctly flagged: (a) missing adaptive prior work (SOFisher Nat Commun 2026,
SCR2-ST MIDL 2026, S2-omics Nat Cell Biol 2025 — all verified real); (b) our "closed-loop" selection
was not actually value-conditioned (graphflow uncertainty = 1-influence is location-only) and the
ablation swapped the whole rule; (c) no positive control; (d) oracle comparator; (e) n=7; (f) one metric.
- Fix (b): added `value_greedy` — selection score s(i|A_t,y_{A_t},X) = local measured VARIANCE + coverage,
  explicitly using measured values.
- Positive control (07): synthetic reconstructability sweep with an ORACLE upper bound. **Even the oracle
  never beats coverage on reconstruction (PCC or MSE), at any structure length.** Whole-slide
  reconstruction is intrinsically near-insensitive to acquisition adaptivity.
- Detection control (08): same slides, ROI-recall objective. **value_greedy and oracle DO beat coverage**
  once ROI signal is strong (value_greedy +0.10 recall @alpha=8, oracle 0.449 vs cover 0.121). Proves the
  simulator is not degenerate — it expresses adaptivity for DETECTION.
- **Real HEST data (09, graphflow, 7 slides, ROI = top-20% molecular heterogeneity):** the split HOLDS.
  * RECONSTRUCTION: value_greedy - best coverage = **-0.081/-0.067/-0.065/-0.074** at 2/5/10/20% (all CI
    excl 0). Adaptivity significantly WORSE (value_greedy 0.61 vs coverage 0.68 @20%).
  * DETECTION: value_greedy - best coverage = **+0.006/+0.014/+0.046/+0.088** at 2/5/10/20% (CI excl 0 at
    2,10,20%). Oracle recall 0.100->0.999 vs coverage 0.018->0.228. Adaptivity significantly BETTER.
- **New claim (replaces the flat negative):** adaptive spatial-omics acquisition helps ROI DETECTION
  (SOFisher/S2-omics objective) but not whole-slide RECONSTRUCTION; the two objectives were conflated,
  and prior positive results and our negative result are both correct under their own objective. Evidence:
  results/{positive_control,detection_control,realdata_detection}.json + figures 2/3.

## Fixes owed / experimental-design corrections
1. Redefine the ablations so they are meaningful under whichever criterion is active (e.g. ablate the
   spatial vs image component of the imputer's feature space, and ablate measured-vs-not at the policy).
2. Add the **principled sensor-placement baselines** (variance-greedy, mutual-information-greedy
   [Krause 2008]) as the correct non-adaptive references — these, not k-means, are the literature's
   named methods for this exact problem.
3. Faithful **Tier-1 SpatialFinder reproduction** is still owed: our current one-shot port produces
   near-zero / negative PCC, which means it is an unfaithful port (SpatialFinder reports high rho on
   its own task) — do not trust the head-to-head until the native repro matches.

## Iteration 5 — HVG-leakage sensitivity (reviewer point 8)  [OURS: results/hvg_sensitivity.json, 7 slides]
Reviewer: top-50 HVG selected from the full slide leaks eval-target info. Re-scored value_greedy vs best
coverage under three gene sets. The reconstruction margin stays negative with CI excluding 0 under ALL:
slide_hvg (leaky) -0.081/-0.067/-0.065/-0.074; ref_hvg (prospective pilot, no leak) -0.080/-0.067/-0.066/
-0.075 (~identical to slide_hvg); all_genes (no selection, zero leak) -0.027/-0.025/-0.029/-0.043 @2/5/10/20%.
Conclusion: full-slide HVG selection does NOT drive the result; the leak is immaterial. Imputer fits on
profiled regions and the policy conditions on measured values, so the HVG target was the only leak -> closed.

## Iteration 6 — batched/contiguous acquisition (reviewer point 7)  [OURS: results/batched_contiguous.json, 7 slides]
Physical-realism constraints. value_greedy vs best coverage, reconstruction PCC + detection recall, CIs.
- RECONSTRUCTION null is ROBUST: margin significantly negative at every round_size (-0.065/-0.062/-0.058
  @ rs=1/5/25; -0.119 one-shot) and under contiguous ROIs (-0.017*). Adaptivity never helps reconstruction.
- DETECTION advantage is FRAGILE to realism: significant ONLY in the idealized fully-sequential scattered
  setting (rs=1: +0.046*, CI excl 0). Under batched rounds (rs>=5) it drops to +0.013/+0.014 (CI includes
  0, NOT significant); one-shot +0.033 (n.s.); contiguous ROIs +0.021 (CI [-0.002,+0.047], n.s.).
- Honest reading: the detection benefit depends on the idealized one-region-at-a-time, scattered-selection
  model; under physically realistic batched and/or contiguous acquisition it is not statistically
  distinguishable from strong coverage at n=7 (point estimates remain small-positive; low power possible).
  This tempers the "adaptivity helps detection" half and reinforces the overall limited practical value.
