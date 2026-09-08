# STARE — Progress & Positioning (reviewer-facing regroup)

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

## 2. Why a reviewer should care (motivation — all [LIT])
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

## 5. What we have found so far ([OURS] — `results/sweep_prostate.json`, prostate, 2249 regions, UNI2-h)
**Result 1 (World-2 diagnosis).** With an **image-similarity kNN imputer**, acquisition degenerates to
image-space coverage: a *one-shot k-means* wins at every budget (0.5–20%), our closed-loop policy never
wins, and closed-loop ≈ no-loop (loop off is often marginally better). → *at this setting the closed
loop adds no value.*
**Interpretation (reviewer-facing, [LIT]-anchored, NOT an excuse):** this is precisely the AFA-RL
"zero-imputing NonSeq baseline" failure mode — the closed loop can only help if the imputer models the
*joint* distribution, so that a measurement updates beliefs about *neighbours* [AFA-RL]. An image-kNN
imputer has no such channel; the acquisition problem collapses to coverage, which one-shot solves. The
experiment is working; it localized the bottleneck to the **imputer**, not the idea.

## 6. The one existential question we must answer honestly (the reviewer's kill-shot)
> **Does the closed loop earn its complexity, or is coverage-based one-shot selection already optimal
> for reconstruction?**

If, even with a *joint* imputer, a one-shot coverage policy matches the closed loop, then the honest
contribution is NOT "agentic acquisition" — it is "*a reconstruction-objective one-shot region selector
beats SpatialFinder's diversity objective and static panels*," which is still a real, publishable result
(cf. our earlier finding that decoder complexity plateaus — negative/scoping results are legitimate),
but it is a different, smaller paper. We do **not** get to assume the closed loop wins. Iterations 1–2
(§7) are designed to *test* this, not to force it.

## 7. Next steps — each justified, none on assumption
- **Iteration 1 (DONE, re-sweep running) [OURS+LIT].** Joint image+spatial imputer (a measured region
  now has local consequence) + **expected-error-reduction** acquisition (measure the region that most
  reduces error over its neighbourhood — the standard AFA / Bayesian-experimental-design objective
  [AFA-RL]). Test: does closed-loop now beat random/k-means AND loop-on beat loop-off at 1–2% budget?
- **Iteration 2 (if 1 fails) [LIT].** Replace the kNN imputer with a genuine **joint spatial imputer**
  the field already uses — SpaIM (Nat Commun'25), STFlow (ICML'25 Spotlight), or a spatial GNN. These
  *do* model the joint/spatial distribution, the precondition AFA-RL identifies. This is the WS-B
  "scale imputer," pulled forward by the diagnosis rather than scheduled by assumption.
- **Iteration 3 (only after a working imputer) [LIT].** Learned/amortized policy (AFA-RL seq-VAE-over-
  partial-state template) and non-greedy lookahead (NOCTA) *if* greedy leaves headroom — escalate only
  on evidence.
- **Scale [LIT].** Extend prostate → kidney + SpatialFinder's 4 tissues + HEST Visium; report the full
  imputation-error-vs-budget Pareto (the field-standard presentation, cf. PERSIST panel-size curves,
  AFA-RL reward-vs-#features).

## 8. Reviewer caveats we must own up front (not hide)
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
