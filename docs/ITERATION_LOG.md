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

## Iteration 7 — expanded slide set: PRE-REGISTERED plan (reviewer point 5)  [written 2026-10-06, BEFORE any expanded result]
Audit of the HEST metadata (public v1.1 CSV; gated v1.3 on the 4070) found problems that had to be fixed
before expanding:
- **Unit scale.** The loader assumed 0.5 um/px; HEST WSIs are 0.17-3.2 um/px (HD slides ~0.274). The legacy
  "112 um" grid was therefore ~61 um on the HD slides (expected: one pseudo-spot per region) and would
  have varied 59-224 um across standard Visium slides. Fix: `config.REGION_MODE = "spot"` — one region =
  one native (pseudo-)spot, coordinates in true microns from HEST's spot-spacing pixel-size estimate
  (`pixel_size_um_embedded` is missing for ~half the samples and off by 30-2000x for some). Policies and
  imputers use relative scales (unit-box coords, extent fractions, median-heuristic / kNN), so only the
  pooling unit changes. `00b_scale_audit.py` checks legacy == spot on the original 7; the paper's "112 um
  region" wording must be corrected whatever the audit shows.
- **Cache.** Per-tissue pickles ignored sample IDs (an expanded config would have silently re-run n=7).
  Replaced by per-slide caches in `stare/slides.py`.
- **Selection.** v1 suggestions included a mouse kidney (TENX146), a duplicate specimen (TENX128 ==
  TENX155, identical WSI dims), a targeted panel, degraded-RNA studies, and serial sections; HEST's
  `patient` field is unreliable (four CRC patients all "Patient 1"). v2 curates: human, whole
  transcriptome, exact organ, >=1000 spots, WSI <= 0.6 um/px, one slide per specimen, <= 2 per study.
- **Statistics.** At n=7 the smallest attainable exact two-sided p is 2/2^7 = 0.0156; with 24 primary
  margins Holm needs p < 0.0021. So NO n=7 margin survives multiplicity correction (reconstruction
  headline: raw p 0.0156 -> Holm 0.37), although the percentile-bootstrap CIs excluded 0. The n=7 "*"
  claims are therefore not robust to an exact, corrected test; this is a power limit, not evidence
  of an effect in either direction.

**Pre-specified analysis for the expanded set (fixed now):**
1. Unit = native spot (`REGION_MODE="spot"`), all slides including the original 7 rebuilt in this mode.
2. Independence unit = HEST study (cluster). Primary test = two-sided cluster-level sign-flip
   randomization test of the mean margin; CI = cluster bootstrap (whole studies resampled). Tissue-
   stratified (equal-weight) estimate reported alongside. Slide-level fields kept for comparison.
3. Primary family (Holm, alpha = 0.05), 24 margins: 05 graphflow `ours_vs_best_nonadapt` (6 budgets);
   09 `value_greedy_{recon,detect}_vs_bestcover` (4 budgets x 2); 13 all batched/contiguous margins (10).
   Everything else (10, 12, other 05/09 margins) is secondary, Holm within experiment.
4. Decision rules. Reconstruction null confirmed if the primary reconstruction margins are <= 0 and none
   is significantly > 0 after Holm (a significantly negative margin strengthens it). Detection advantage
   claimed only where the detect margin is > 0 with Holm p < 0.05; if it holds at rs=1 but not at rs>=5 /
   contiguous, the claim stays "fragile to physical realism"; if rs>=5 becomes significant, we report the
   n=7 non-significance as a power artifact and revise that claim upward.
5. No slide is dropped after results are seen; any build failure is reported with the reason.
- **Compute (added before the expanded run).** At 4-9k spots the old code was infeasible (graphflow built an
  R x R x d distance tensor, ~42 GB at 9k; greedy_ig refit a kNN index every step; value_greedy recomputed a
  U x t kernel times all genes every step; Moran's I used a dense R x R product in every score()). Replaced
  by sparse kNN diffusion, an uncertainty-only path, cached neighbour lists, running kernel sums, memoized
  placements (k-means persisted to cache/placements), and a sparse Moran's I. Verified on a 132-run
  equivalence harness (11 policies x 3 imputers x 2 slide sizes x 2 budgets): identical selections and
  scores to the old code. Timing at R=6000, 20% budget: greedy_ig 8.6 s (was 319 s at R=2200).

## Iteration 8 — expanded slide set RESULTS (n=22 slides, 18 study clusters)  [4070 run 3bd63a6; analysed 2026-10-09]
**Slide set (gated v1.3 metadata).** Two of the original 7 are not human and were excluded by the pre-set filter:
TENX186 (kidney) = *Mus musculus*; TENX175 (colon) = "Homo sapiens/Mus musculus". The n=7 results in the
current manuscript therefore include a mouse and a mixed-species slide -> must be corrected. Final set: prostate 6,
kidney 4, lung 6, colon 6 (7 Visium HD + 15 Visium; 18 HEST studies). TENX171/170 are two arms of one 10x lung
experiment (same study cluster).
**Scale audit.** All 5 retained original HD slides had exactly one pseudo-spot per legacy region (legacy grid
= 61 um true; HD pseudo-spot pitch 128 um), so the old HD results were on native units. 5 of the new Visium slides
would have pooled up to 3 spots under the legacy loader. Units still differ by platform (HD 128 um pitch vs
Visium 55 um spots at 100 um pitch) -> platform-split reported.
**Reproduction.** The 5 overlapping slides give per-slide margins close to n=7 (same sign; |diff| ~0.01-0.06,
from true spot coordinates vs snapped grid centres). Pooled shifts come from the new slides.
**Stats fix.** The 4070 JSONs stored a 4-decimal Monte Carlo sign-flip p (small p rounded to 0). The test is now
exact up to 20 clusters (2^18 patterns; floor 7.6e-6) and script 14 recomputes it unrounded from per-slide margins.

Pre-registered primary family (24 margins, Holm):
- **Reconstruction null: CONFIRMED and strengthened.** greedy_ig ("ours") - best non-adaptive: -0.128/-0.083/
  -0.040 at 0.5/1/2% (all 18 clusters negative, Holm p=1.8e-4), -0.011 at 5% (Holm 0.057), -0.002/-0.002 at
  10/20% (90% cluster CI inside +-0.01 -> equivalent within 0.01 PCC). value_greedy - best coverage:
  -0.136/-0.148/-0.152/-0.166 at 2/5/10/20% (Holm 1.8e-4), larger than at n=7 (-0.065 to -0.081); larger on
  Visium (-0.15 to -0.19) than HD (-0.10). Every batched/contiguous recon margin is negative (Holm 1.8e-4).
- **Detection advantage: NOT SUPPORTED after correction.** value_greedy - best coverage recall: +0.001/+0.013/
  +0.021/+0.042 at 2/5/10/20% (raw p 0.72/0.042/0.040/0.044; Holm 0.36). Batched rs=5 +0.034 (raw 0.0073, Holm
  0.073), rs=25 +0.011 (Holm 0.36), contiguous +0.024 (Holm 0.36); one-shot -0.051 (Holm 0.008, significantly
  WORSE). Effects shrank vs n=7 (+0.046 -> +0.021 at 10%; +0.088 -> +0.042 at 20%). Absolute: at 20% value_greedy
  recall 0.278 vs coverage ~0.20-0.23 vs heterogeneity oracle 0.999.
- Per pre-registered rule 4 the detection claim is withdrawn: small positive point estimates, nominal p~0.04
  uncorrected, not significant under Holm; not a power artifact of n=7 (power is now ample: the recon margins
  reach the exact floor).

Secondary:
- **Platform split.** Detection gain sits in HD (+0.048 / +0.099 at 10/20%, 5 clusters -> min p 0.0625) and is
  ~0 on Visium (+0.008 / +0.015, p 0.43/0.47).
- **Competitors (10).** The NON-adaptive SCR2-ST-style diversity sampler beats the coverage envelope on detection
  (+0.020 at 10%, p=0.0014; +0.029 at 20%, p=0.0002) about as much as adaptive value_greedy (+0.021/+0.042) ->
  the detection gain over kmeans/grid/MI is not specific to adaptivity. Harmonized SOFisher port: recon
  -0.19 to -0.28, detection ~0 (port remains weaker than value_greedy; native SOFisher result unchanged).
- **HVG sensitivity (12).** Recon margin unchanged under ref_hvg (prospective) and negative under all_genes
  (-0.03 to -0.08, CIs exclude 0): no leakage artifact.

**Revised claim.** On 22 human slides from 18 studies, adaptive acquisition does not improve whole-slide
reconstruction over strong non-adaptive placement (worse at low budgets, equivalent within 0.01 PCC at
10-20%). For ROI detection, value-conditioned adaptivity gives at most a small recall gain (<=0.04 at 20%) that
does not survive multiplicity correction, is concentrated in Visium HD slides, and is matched by a non-adaptive
diversity design.

## Iteration 9 — reviewer round 2 (major revision): PRE-REGISTERED plan  [written 2026-10-10, BEFORE scripts/17 is run]
Reviewer concerns: (1) "best non-adaptive" is a per-slide hindsight maximum, not a deployable rule; (2) under
graphflow, error-reduction / max-uncertainty are sequential FIXED designs (value-independent), and the
value-conditioned policy seeks local variance, not reconstruction error; (3) controls do not establish sensitivity
for reconstruction or power for small detection effects; (4) detection needs a direct value-conditioned vs
diversity comparison and a feedback ablation; (5) scope (computational ROI, spot unit) must be explicit.

**Post-review re-analyses of EXISTING results (post hoc; scripts/16_revision_local.py, results/revision_local.json):**
- Value-independence verified directly: under graphflow, error-reduction selects identical regions when the
  measured values are permuted (unit test, synthetic slide).
- Feasible comparator (leave-one-study-out choice among grid/k-means/var-greedy/MI-greedy/one-shot; picks k-means at
  0.5-1%, MI-greedy at 2-20% for most held-out studies): error-reduction - LOSO = -0.107/-0.067/-0.024/-0.010 at
  0.5/1/2/5% (unadjusted p <= 0.021), -0.001 at 10% (p 0.76), +0.001 at 20% (p 0.54). Max-uncertainty - LOSO at 20% =
  +0.003 [+0.002, +0.004], p = 0.0007 (a small positive margin for a sequential FIXED design that the hindsight
  comparator had hidden). Hindsight gap (per-slide best - LOSO): +0.016 to +0.021 at <=2%, <=0.002 at >=5%.
- Direct value-conditioned - diversity recall: +0.001/+0.003/+0.001/+0.012 at 2/5/10/20%, all CIs include 0
  (p >= 0.62), positive on 8-11 of 22 slides.
- Power (detection, value-conditioned - best coverage): MDE at 80% power = 0.020/0.030/0.055 recall at 5/10/20%
  unadjusted, 0.033/0.046/0.083 at alpha = 0.05/24; power at the observed means ~0.49-0.69 unadjusted,
  ~0.08-0.11 corrected. Q2 is therefore UNRESOLVED (underpowered), not negative.

**New experiments (scripts/17_revision_experiments.py), analysed by scripts/18_revision_analysis.py:**
- New policy `residual_guided` (value-conditioned, aligned with reconstruction error): error-reduction form with the
  graphflow uncertainty replaced by (kNN-averaged leave-one-out ridge residual of measured values) x (1 - graph
  influence). Feedback ablation wrapper `FeedbackAblation` (permute = policy sees another region's values;
  constant = policy sees the slide mean). Selection rule unchanged by the wrapper.
- Part A: 22 slides x 6 budgets; per-policy per-slide PCC and recall for grid, k-means, var-greedy, MI-greedy,
  one-shot, scr2st, value_greedy (+_perm, +_const), residual_guided (+_perm).
- Part B (reconstruction positive control): 20 appended genes with a 3-hotspot field (radius 5% of diagonal), SD =
  alpha x median SD of the slide's top-50 genes, alpha in {0,1,2}; budgets 5,10%; NA pool + residual_guided (+_perm).

**Primary family (18 margins, Holm, exact study-cluster sign-flip, 95% cluster CIs reported as unadjusted):**
- Q1: residual_guided - LOSO_recon (PCC), 6 budgets.
- Q2: value_greedy - value_greedy_perm (recall); value_greedy - LOSO_detect (recall); value_greedy - scr2st (recall);
  each at 2/5/10/20%.
LOSO comparators are chosen per budget and objective from the five non-adaptive policies by mean performance on the
other studies. The per-slide maximum is kept only as a hindsight benchmark.
**Secondary (Holm within family):** residual_guided - residual_guided_perm (PCC, 6); value_greedy - value_greedy_const
(recall, 4); residual_guided - best_NA (PCC, 6); value_greedy - LOSO_recon (PCC, 4).
**Positive control B (descriptive):** sensitivity for reconstruction is considered shown if residual_guided - LOSO > 0
with a 95% cluster CI excluding 0 at alpha = 2 for either budget; if not, the paper states that the protocol's
sensitivity to reconstruction gains is not established.
**Decision rules:** Q1 benefit claimed only if a residual_guided - LOSO margin is > 0 with p_H < 0.05. Feedback benefit
for detection claimed only if value_greedy - value_greedy_perm > 0 with p_H < 0.05. Non-significant margins are
reported as "not established", with the MDE at 80% power, never as evidence of no effect.
**Presentation (no new data):** error-reduction / max-uncertainty relabelled "sequential fixed designs"; CIs labelled
unadjusted; the post hoc equivalence statement removed from the main narrative; single-round loss attributed to the
centroid start rule; scope (computational heterogeneity ROIs, spot unit, retrospective simulator) in abstract and
conclusion.
- **Bug in the first 4070 run of script 17 (fixed 2026-10-10, before any result was analysed).** Crash at MEND144:
  "index 4231 is out of bounds for axis 0 with size 4074". Cause: imputer caches (features, RFF, graph) and the
  k-means / placement memos were keyed by id(slide); the feedback-ablation wrapper creates a temporary copy of the
  slide per run, and a discarded copy's id was reused by a later copy of a different (smaller) slide, which then
  received the stale cached graph. Fix: caches keyed by the identity of the image-embedding and coordinate arrays
  (their only inputs), with references held so ids cannot be reused; copies share those arrays and so share
  entries correctly. Reproduced on the pushed code (same IndexError) and verified fixed (48 shared-imputer runs
  identical to fresh-imputer runs; 132-run equivalence check unchanged).
  Validity of results saved before the crash: a stale entry from a different slide has a different region count
  and makes the graph product fail on the first non-cold-start step, so any cross-slide reuse would have crashed
  immediately; same-slide reuse returns identical values (shared arrays). The only pair of slides with equal
  region counts (TENX174, ZEN48; 2,385) had not been reached. Saved results are therefore unaffected and the run
  is resumed, not restarted.

## Iteration 10 — reviewer round 2 RESULTS  [4070 run c4d3326; analysed 2026-10-12 with scripts/18 as pre-registered]
Data check: all 22 slides x 6 budgets (Part A) and 22 x 3 alpha x 2 budgets (Part B) complete; the five non-adaptive
policies reproduce script 05 per-slide PCC exactly (max |diff| 0), value-conditioned and diversity reproduce 09/10.
All 20 spiked genes entered the top-50 on every slide at alpha = 1 and 2.
LOSO choices: recon k-means (0.5-1%), MI-greedy (2-20%, k-means for 4 studies at 2%); detection var-greedy (2-10%,
one-shot for some studies), one-shot ranker at 20%.

**Primary family (18, Holm):**
- Q1 residual_guided - LOSO (PCC): -0.092/-0.057/-0.025/-0.016/-0.007/-0.002 at 0.5/1/2/5/10/20%; p_H = 0.003/0.010/
  0.099/0.003/0.010/0.13. No positive margin -> no Q1 benefit (rule); significantly WORSE at 0.5, 1, 5, 10%.
- Q2 value_greedy - value_greedy_perm (recall): +0.005/+0.017/+0.030/+0.063 at 2/5/10/20%; unadjusted p 0.13/0.057/
  0.053/0.027; p_H >= 0.29 -> feedback benefit NOT established. MDE80 (unadj.) 0.010/0.026/0.041/0.084: observed
  effects are below the MDE at 5-20% (underpowered).
- Q2 value_greedy - LOSO_detect: +0.008/+0.006/+0.017/+0.026, p_H >= 0.25.
- Q2 value_greedy - scr2st: +0.001/+0.003/+0.001/+0.012, p >= 0.62, p_H = 1.
**Secondary:**
- residual_guided - residual_guided_perm (PCC): +0.031/+0.039/+0.031 at 0.5/1/2% (p_H 0.056/0.002/0.014), +0.009 at
  5%, ~0 at 10-20%: informative feedback improves this policy at low budgets relative to the same rule with
  uninformative values, but not to the level of the feasible non-adaptive comparator.
- value_greedy - value_greedy_const (recall): +0.004/+0.019/+0.034/+0.074 (p_H 0.52/0.083/0.076/0.038).
- value_greedy - LOSO_recon (PCC): -0.119 to -0.163 (p_H <= 0.0002). residual_guided - best_NA: -0.005 to -0.113.
**Positive control B (pre-set criterion: residual_guided - LOSO > 0, CI excl. 0, at alpha = 2): NOT MET.**
residual_guided - LOSO = -0.021 (5%) and -0.011 (10%) at alpha = 2 (CIs below 0); -0.015/-0.007 at alpha = 0.
residual_guided - perm = +0.003 to +0.013 in all settings. Per the plan, the paper states that the protocol's
sensitivity to reconstruction gains is not established.

**Revised conclusions (to carry into the manuscript):**
- Q1: none of the tested value-conditioned policies, including one aligned with reconstruction error, beat a
  feasible non-adaptive rule at any budget; feedback improved the error-aligned policy over its permuted-feedback
  version at 0.5-2%, so measured values carried usable information, but less than the coverage advantage of
  k-means/MI-greedy. Because the positive control did not show a gain, Q1 is a statement about these policies, not
  a demonstration that useful value-conditioned reconstruction is impossible.
- Q2: unresolved. Feedback-ablation estimates are positive and grow with budget but are not significant after
  correction and lie below the MDE; no advantage over the non-adaptive diversity design was found.

## Iteration 11 — reviewer round 3: stage-3 PRE-REGISTERED plan  [written 2026-10-12, BEFORE scripts/21 is run]
Reviewer points: (1) coverage-plus-feedback hybrid untested; (2) LOSO comparator couples the study margins through
rule selection; (3) one permutation seed; clarify what the residual policy fits; (4) MDE quoted at unadjusted alpha;
(5) Q3 uses stage-1 hindsight comparator without feedback control; (6) scope. Fixes: Table 1 caption (local-variance
WAS run at 0.5-1% in stage 2), appendix float placement, precise wording for the feedback received.

**Post hoc sensitivity of stage-2 data (scripts/20_stage2_sensitivity.py, results/stage2_sensitivity.json):**
- Selection-aware bootstrap (LOSO re-selected inside each resample): Q1 residual - comparator CIs all below 0
  (0.5%: [-0.132, -0.020] ... 20%: [-0.0043, -0.0005]); Q2 local-variance - comparator CIs all include 0.
- Fixed MI-greedy comparator (valid sign-flip; Holm over the 18 analogous margins): Q1 residual - MI = -0.055/-0.045/
  -0.033/-0.016/-0.007/-0.002 (p_H 0.42/0.09/0.014/0.003/0.011/0.11). Q2 local-variance - MI = +0.009/+0.024/+0.043/
  +0.078 recall (p_H 0.17/0.063/0.043/0.043): SIGNIFICANT at 10-20% against MI-greedy, which is a weak detection
  rule (recall 0.200 at 20% vs 0.252 for the cross-validated choice). The detection conclusion is comparator-
  dependent: an advantage over a reconstruction-oriented coverage design, none over the best non-adaptive design
  for detection. Fixed k-means: residual - k-means is +0.009 at 20% (p 0.03; k-means is weak at high budget).
- Holm-level MDE (alpha = 0.05/18 bound) for stage-2 primary margins: Q2 feedback 0.015/0.035/0.066/0.112 recall at
  2/5/10/20%; the unadjusted MDEs are descriptive only.
- Clarification (point 3): with permuted feedback the residual policy's leave-one-out ridge model is fitted to the
  permuted values it is shown; the scoring imputer is always fitted to the true values.

**Stage-3 experiments (scripts/21_stage3_experiments.py; analysis scripts/22_stage3_analysis.py):**
- Part H hybrids: first ceil(f x budget) regions from a FIXED coverage rule, the rest from feedback; f in {0.5, 0.8}.
  Reconstruction: k-means (0.5, 1%), MI-greedy (2-20%), then the residual policy. Detection: variance-greedy (2-10%),
  one-shot ranker (20%), then the local-variance policy. These rules are the stage-2 LOSO choices for most held-out
  studies, i.e. chosen on stage-2 data from the same slides (this favours the baseline). Verified: the hybrid's
  coverage phase is exactly the prefix of the rule's own full-budget selection (all 4 rules, both f, 2 budgets).
  Each hybrid also with permuted feedback, seeds 1, 2.
- Part S: extra permutation seeds for stage-2 ablations (local-variance seeds 1-4; residual seeds 1-2).
- Part R: rounds of 5 regions at 10% and 20%, local-variance with real and permuted (seeds 1, 2) feedback.

**Primary family (20, Holm; fixed-rule comparators, so the study-level sign-flip test is valid as specified):**
hybrid(f) - the same fixed rule for the whole budget: reconstruction (PCC) f in {0.5, 0.8} x 6 budgets, detection
(recall) f in {0.5, 0.8} x 4 budgets.
**Secondary (Holm within, 24):** hybrid - permuted-feedback hybrid (seed mean), both objectives (20); rounds of 5:
local-variance - fixed detection rule and - permuted (seed mean), 10% and 20% (4).
**Descriptive:** per-seed stage-2 ablation margins and their across-seed SD; seed-averaged ablation margins.
**Decision rule:** a hybrid benefit is claimed only for a positive margin with p_H < 0.05; otherwise "not
established", reported with CIs and the Holm-level MDE bound (alpha/20).
**Q3 framing:** batched results (stage 1 and Part R) are reported as an exploratory stress test, not a coequal
confirmatory answer.

**Stage-3 results (analysed 2026-10-15; parts H and R complete on 22 slides, part S stopped at slide 5 of 22):**
- Primary (20): no hybrid margin positive at p_H < 0.05. One significant negative: reconstruction, f = 0.5, 0.5%:
  hybrid - k-means = -0.036 PCC (p_H 0.006). From 2% upward all margins within 0.007 PCC / 0.006 recall; Holm-level
  MDE at 2-10% is 0.004-0.017 PCC and 0.004-0.026 recall (detection 20%: 0.035-0.068).
- Secondary (24): hybrid - permuted significant at 2% only (reconstruction f 0.5 +0.016, f 0.8 +0.007; detection
  f 0.5 +0.007). Permuted feedback drops the hybrid below the fixed rule; real feedback restores it (e.g. detection
  10%: fixed 0.127, permuted 0.110, observed 0.133). Rounds of 5: local-variance - fixed +0.023 / +0.035, - permuted
  +0.050 / +0.084 at 10 / 20% (p_H >= 0.11). Rounds of 5 recall 0.150 / 0.288 vs one-at-a-time 0.137 / 0.278.
- Seed variation (descriptive) not yet computed: rerun part S to completion, then scripts 22 and 23.
- Manuscript revised: stage 3 as section "Feedback on top of a coverage design"; Q3 exploratory; Holm-level MDEs;
  post hoc feasible-comparator inference in an appendix; scope of computational ROIs in Limitations.
