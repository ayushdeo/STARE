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
  cross-slide evidence; the paper is the theory-backed characterization. Result: _pending 4070 run._

## Fixes owed / experimental-design corrections
1. Redefine the ablations so they are meaningful under whichever criterion is active (e.g. ablate the
   spatial vs image component of the imputer's feature space, and ablate measured-vs-not at the policy).
2. Add the **principled sensor-placement baselines** (variance-greedy, mutual-information-greedy
   [Krause 2008]) as the correct non-adaptive references — these, not k-means, are the literature's
   named methods for this exact problem.
3. Faithful **Tier-1 SpatialFinder reproduction** is still owed: our current one-shot port produces
   near-zero / negative PCC, which means it is an unfaithful port (SpatialFinder reports high rho on
   its own task) — do not trust the head-to-head until the native repro matches.
