# STARE — Progress & Positioning (status regroup)

Last updated: 2026-09-08. Companion to `SOURCE_OF_TRUTH.md` (spec), `KNOWLEDGE_BASE.md` (paper notes),
`lit_review/REALITY_CHECK.md` (problem-exists evidence). This file is the honest "where we are"
snapshot, written so every claim is traceable to either **[LIT]** (published, cited) or **[OURS]**
(our own experiment, with the result file) — and everything else is explicitly a **[HYP]** hypothesis
with the experiment that would confirm/refute it. No claim rests on our own assumption alone.

---

## 1. The claim (one sentence)
We study **closed-loop, budget-constrained region acquisition** for spatial transcriptomics: given a
fixed measurement budget on a tissue slide, sequentially decide *which regions to profile next* —
conditioned on H&E morphology **and on the expression already measured** — to minimize
whole-transcriptome reconstruction error.

## 2. Why this matters (motivation — all [LIT])
- **Targeted ST is budget-limited by physics, not choice.** Imaging platforms cap panels (~1,000
  genes; Xenium ≤5,000) [scGIST, Genome Biol 2024; HEST-1k, NeurIPS 2024]. Region-based platforms
  (GeoMx DSP) require a pathologist to *manually* select a budget-limited set of ROIs to profile
  [Bruker/NanoString GeoMx docs; `REALITY_CHECK.md`]. "Which regions to profile under budget" is a
  real, currently-manual decision.
- **Reducing this cost is a published goal.** SpatialFinder (Frontiers Bioinf. 2026) frames exactly
  "which regions to sequence from H&E to cut per-slide cost" [SpatialFinder]. → demand is real.
- **Sequential information-gain acquisition is an established, working paradigm** in adjacent
  experimental sciences — self-driving labs and adaptive microscopy run closed loops that measure →
  update a surrogate → choose the next measurement by expected information gain [Royal Society Open
  Science 2025 review; FAST, PMC10485018]. Formally this is active feature acquisition / Bayesian
  experimental design [AFA-RL, Yin et al. 2020].

## 3. The gap we occupy (positioning — all [LIT], from `KNOWLEDGE_BASE.md`)
| Prior work | What it does | What it does NOT do (= our differentiator) |
|---|---|---|
| PERSIST (Nat Commun'23), ReconST (NAR'26), scGIST/gpsFISH (Genome Biol'24) | static gene-panel selection from scRNA reference | one fixed panel; non-spatial; no image; no sequential conditioning (ReconST states this verbatim) |
| SpatialFinder (Frontiers'26) — nearest competitor | one-shot ROI ranking from H&E, diversity heuristic | one-shot (no feedback from measured expression); diversity objective, not reconstruction; not a learned policy |
| BLEEP/Stem/STFlow/GenAR (NeurIPS'23/ICLR'25/ICML'25/preprint) | predict expression from image (no acquisition) | assume everything measured or nothing; never *choose what to measure* |
| CellAgent (ICLR'25), SpatialAgent (bioRxiv'25) | LLM agents automating analysis *workflow* | agency over pipeline steps, not over physical measurement |

**Unoccupied cell:** closed-loop, image + measured-expression-conditioned, reconstruction-objective,
*learned* region acquisition. This is our contribution surface. (Confirmed unoccupied by deep reads +
the reality check.)

## 4. What we have built (methods — [OURS], reproducible)
- **Wet-lab-free acquisition simulator** on HEST-1k Visium HD [HEST-1k]: mask the full slide, a policy
  reveals regions under budget B, then a frozen imputer reconstructs the unprofiled transcriptome and
  we score it. *Masking-based acquisition simulation is the field norm* — PERSIST simulates FISH panels
  by masking scRNA genes [PERSIST]; we do the spatial analogue. **[LIT-justified design]**
- **Region unit = 112 µm = one 224px@20× UNI2-h patch** — aligns the acquisition unit with the image
  encoder patch (HEST convention) and sits mid-range of GeoMx ROI sizes [HEST; GeoMx docs].
- **Evaluation matches HEST-Benchmark**: top-50 HVG, log1p, mean per-gene Pearson, over held-out
  (unprofiled) regions [HEST-1k]. Full-transcriptome PCC reported as secondary.
- **Shared frozen imputer across all policies** (fair comparison) — the reproduction-plan §6 principle.
- **Baselines ported into the same simulator**: SpatialFinder-style one-shot (Tier-2 port; the native
  cell-classifier term is replaced by an image-embedding diversity proxy — logged in
  `experiments/deviations.md`), static-panel proxies, and naive coverage baselines (random averaged
  over seeds, grid, variance, k-means).
- **Our closed-loop policy + ablations** (`use_measured`/`use_image`/`use_spatial` toggles).

## 5. What we have found so far ([OURS] — `results/sweep_prostate.json`; see `ITERATION_LOG.md`)
**Robust finding across 2 acquisition heuristics × 2 imputers (prostate, 2249 regions, UNI2-h):**
**one-shot spatial+image COVERAGE (k-means) is near-optimal for reconstruction; conditioning
acquisition on measured expression does not help.** Iteration-0 (image-kNN + uncertainty greedy):
closed-loop ≈ no-loop, k-means wins. Iteration-1 (joint image+spatial kNN + error-reduction greedy):
*worse* — our policy over-clusters and loses to random/k-means/no-loop at every budget.

**Interpretation — this is a KNOWN THEOREM, not an engineering failure [LIT].** Krause, Guestrin &
Singh (JMLR 2008, "Near-Optimal Sensor Placements in Gaussian Processes") show that for spatial
reconstruction modelled as a GP with a **fixed/known kernel**, the posterior variance is **independent
of the observed values** → the optimal set of measurement locations can be chosen **offline**, and
**adaptive (closed-loop) placement gives no gain over one-shot placement.** A kNN imputer *is* that
fixed-kernel regime, so our World-2 result is the predicted behaviour. Corollary (the pivot): the
closed loop can only earn its complexity when the imputer's uncertainty **depends on measured values**
— i.e. a **non-stationary / learned** imputer (STFlow/SpaIM), the same precondition AFA-RL identifies
[AFA-RL]. Hand-tuning greedy heuristics on a stationary kNN imputer is provably futile.

## 6. The one existential question we must answer honestly (the decisive objection)
> **Does the closed loop earn its complexity, or is coverage-based one-shot selection already optimal
> for reconstruction?**

If, even with a *joint* imputer, a one-shot coverage policy matches the closed loop, then the honest
contribution is NOT "agentic acquisition" — it is "*a reconstruction-objective one-shot region selector
beats SpatialFinder's diversity objective and static panels*," which is still a real, publishable result
(cf. our earlier finding that decoder complexity plateaus — negative/scoping results are legitimate),
but it is a different, smaller paper. We do **not** get to assume the closed loop wins. Iterations 1–2
(§7) are designed to *test* this, not to force it.

## 7. Next steps — reframed around LITERATURE REPLICABILITY (stop inventing heuristics)
Iterations 0–1 hand-rolled greedy heuristics and both failed exactly as the sensor-placement theory
predicts. The disciplined move is to **replicate established frameworks/methods**, not invent more.

- **Step A — adopt the sensor-placement framing + its named baselines [LIT: Krause 2008].** Our problem
  IS "select measurement locations to reconstruct a spatial field." Implement the field's principled
  non-adaptive baselines: **variance-greedy** and **mutual-information-greedy** placement (submodular,
  (1−1/e)-optimal). These — not our ad-hoc greedy, not bare k-means — are the correct references. This
  gives our method a name, guarantees, and a fair comparison, and reproduces a known result.
- **Step B — replicate a REAL imputer so the verdict isn't a kNN artifact [LIT].** Two regimes:
  (i) a **GP/kriging** spatial regressor = the stationary-kernel reference (closed-form variance;
  theory predicts closed-loop gives ≈0 gain — we should reproduce that as a sanity check);
  (ii) a **learned / non-stationary** imputer the field uses — **SpaIM (Nat Commun'25)** or **STFlow
  (ICML'25)** — where the posterior CAN depend on measured values, so closed-loop *can* help. This is
  the decisive experiment; it replaces "tune our greedy" with "swap in a published imputer."
- **Step C — faithful Tier-1 SpatialFinder reproduction [LIT].** Current one-shot port gives near-zero/
  negative PCC → unfaithful. Reproduce SpatialFinder's native ρ before any head-to-head is trusted.
- **Step D — decide the paper's identity on evidence (§6):** if closed-loop beats non-adaptive
  placement ONLY under a learned imputer → the agentic/adaptive-acquisition paper, grounded in the
  Krause stationary-vs-nonstationary distinction. If it never beats non-adaptive placement → the honest
  paper is "reconstruction-objective sensor-placement selection beats SpatialFinder's diversity
  objective and static panels," a benchmarking/negative result. Either is publishable; evidence picks.
- **Scale [LIT].** prostate → kidney + SpatialFinder's 4 tissues + HEST Visium; report the full
  imputation-error-vs-budget Pareto (field-standard: PERSIST panel-size curves, AFA-RL reward-vs-budget).

**Fixes owed before more runs (see `ITERATION_LOG.md`):** redefine the now-degenerate ablations
(`no_image`==`no_spatial`==`ours` under error_reduction); add the variance/MI-greedy placement
baselines; do the Tier-1 SpatialFinder repro.

## 8. Caveats we must own up front (not hide)
1. **In-silico, not wet-lab.** We simulate acquisition by masking Visium HD; real GeoMx ROI acquisition
   differs (assay noise, ROI shape). Justified by field practice (PERSIST masks to simulate FISH; HEST
   is the standard benchmark), but we state it and, ideally, validate the top policy on a held-out
   platform/tissue.
2. **Results are conditional on the imputer.** The imputer is the environment; a single imputer's
   verdict is not the field's. We therefore report **across imputers** (kNN → SpaIM/STFlow), not one.
3. **Single tissue so far.** Prostate only; credibility requires the multi-tissue Pareto (§7 Scale).
4. **Competitor faithfulness.** Every non-1:1 port is logged in `experiments/deviations.md`; a Tier-1
   native SpatialFinder-ρ reproduction is still owed before the head-to-head is fully trustworthy.

## 9. Status vs the spec
- `SOURCE_OF_TRUTH.md` §4/§5/§6 (Option B objective, workstreams, corrected gate) — current.
- We are in **WS-D iteration loop** (not "Stage 0" — the README status line is stale and should read
  "Stage 2: closed-loop policy under diagnostic iteration on prostate").
- Go/no-go is a *diagnostic across budgets + iterations*, not a single pass/fail (`SOURCE_OF_TRUTH §6`).
