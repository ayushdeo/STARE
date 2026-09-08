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

## Fixes owed / experimental-design corrections
1. Redefine the ablations so they are meaningful under whichever criterion is active (e.g. ablate the
   spatial vs image component of the imputer's feature space, and ablate measured-vs-not at the policy).
2. Add the **principled sensor-placement baselines** (variance-greedy, mutual-information-greedy
   [Krause 2008]) as the correct non-adaptive references — these, not k-means, are the literature's
   named methods for this exact problem.
3. Faithful **Tier-1 SpatialFinder reproduction** is still owed: our current one-shot port produces
   near-zero / negative PCC, which means it is an unfaithful port (SpatialFinder reports high rho on
   its own task) — do not trust the head-to-head until the native repro matches.
