# SOURCE OF TRUTH — Master Reference

**This is the single document to return to for project status, direction, and rationale.** Everything else defers to it:
- `CHECKLIST.md` — the executable checklist; every item maps to a section here (§ references in brackets).
- `RESEARCH_PLAN.md` — earlier phased plan; superseded by this file for the committed direction, kept for history.
- `lit_review/` — the three decision documents that led here.

**Committed direction (locked):** Direction 1, refined to **Option B — Sequential, Closed-Loop, Budget-Constrained ROI/Spot ACQUISITION with a learned, image-conditioned policy that optimizes whole-transcriptome imputation error.** Decided after literature survey + deep reads (§3) + a dedicated reality check (`lit_review/REALITY_CHECK.md`).

**Why Option B (region/spot acquisition) and NOT the within-slide adaptive gene panel:** the reality check found that within-slide adaptive gene panels are **physically infeasible** on mainstream imaging platforms (Xenium/MERFISH/CosMx image the whole slide with one fixed panel per run). The *real* budget-limited, image-guided decision practitioners make is **which regions/spots to profile** (GeoMx ROI selection; cost-per-region Visium HD) — currently done manually by pathologists.

**Problem confirmed real (not forced), 3 independent signals:** (1) GeoMx ROI selection is manual + budget-limited (~6 ROIs/type min, ~$15k/slide); (2) SpatialFinder (Frontiers Bioinf. 2026) publishes the exact cost-reduction framing → community demand; (3) self-driving labs / adaptive microscopy prove sequential information-gain acquisition works in adjacent fields. See `lit_review/REALITY_CHECK.md`.

**Sharpened gap / differentiator (vs SpatialFinder 2026, the nearest entrant):** SpatialFinder is **one-shot, heuristic, diversity-objective, not a learned policy, no closed loop.** Ours must be **sequential + closed-loop (conditioned on already-measured regions) + reconstruction/imputation objective + learned policy.** The single decisive experiment: closed-loop acquisition beats one-shot selection at equal budget.

Last updated: 2026-07-05 (post reality check).

---

## 1. The one-paragraph problem statement

Targeted spatial transcriptomics platforms (Xenium, MERFISH, CosMx) can physically measure only a few hundred to ~1,000 genes per experiment, and every probe/spot costs money and imaging time. Today, biologists pick **one fixed gene panel from a reference scRNA-seq dataset before ever seeing the tissue**, and apply it uniformly across the whole slide. We propose to make that acquisition decision **adaptive and sequential, conditioned on the H&E histology image and on the spots/genes already measured** — spending a fixed measurement budget where it recovers the unmeasured transcriptome best, and allowing the panel to differ across regions of the *same* tissue (tumor core vs. stroma need different genes). The objective is minimum whole-transcriptome imputation error under a fixed budget.

## 2. Why this is the right problem (motivation, with citations)

1. **The constraint is physical and real, not invented.** Imaging-based ST is capped at ~1,000 genes [scGIST, Genome Biology 2024, https://genomebiology.biomedcentral.com/articles/10.1186/s13059-024-03185-y]. This is the honest cost/budget that makes agency meaningful — unlike single-shot prediction, where there is no decision to make.
2. **Our own prior oracle result is now evidence FOR the thesis.** We previously showed that conditioning on true values of some genes lifts prediction of others from ~0.20 → ~0.24–0.54 PCC [internal, `CSCI 566 (2).pdf`]. Under acquisition framing this is exactly the required claim: *measured genes inform unmeasured ones.* The learned-policy collapse that sank the old project is irrelevant here, because the decision is now "what to measure," not "what order to emit predictions."
3. **The field's SOTA panel selectors explicitly do NOT do this** — their own papers list it as a limitation (see §3.2). We are not inventing a gap; the incumbents named it.

## 3. Deep-read findings — does the gap exist? (YES, confirmed)

### 3.1 The 2×2 that defines the gap

| | Static panel (chosen once, uniform) | Sequential / spatially-adaptive acquisition |
|---|---|---|
| **Non-spatial, scRNA reference only** | PERSIST, ReconST, scGIST, gpsFISH, scGPD | RL-panel (2501.04718), AFA (2011.00825) |
| **Spatially-resolved + histology-image-conditioned** | *(essentially empty)* | **★ OUR CELL — confirmed empty by deep reads below** |

### 3.2 Evidence the target cell is empty (from deep reads, 2026-07-05)

- **ReconST** (Nucleic Acids Research 2026 / bioRxiv 2025) — SOTA reconstruction-objective panel selector. Deep read confirms: selects **one globally static panel**; uses **only scRNA-seq** for selection; spatial data used **post-hoc for validation only**; **no histology image**; **no sequential/active acquisition**. Its own stated limitations verbatim: *"Panel is globally static across tissue"* and *"No explicit incorporation of spatial structure ... during selection."* [https://academic.oup.com/nar/article/54/11/gkag621/8709438] **← The incumbent SOTA names our exact gap as its limitation.**
- **RL gene-panel selection** (arXiv:2501.04718, 2025) — deep read confirms: scRNA-seq only, **no spatial, no histology**, **static single panel per dataset**, reward = NMI clustering + parsimony. Not a top venue. [https://arxiv.org/html/2501.04718]
- **Active Feature Acquisition — RL** (arXiv:2011.00825) — the canonical AFA formulation we must position against and can borrow from: a **sequential VAE over partially-observed state** feeding an RL policy that trades task reward vs. acquisition cost (POMDP). Generic ML, not spatial/genomic. This is the *methodological template*, not a competitor on our task. [https://arxiv.org/abs/2011.00825]

**Conclusion:** the spatially-adaptive, image-conditioned, sequential-acquisition cell is unoccupied. The contribution is the *combination*; novelty must be earned by beating static PERSIST/ReconST and a non-spatial AFA policy on the same benchmark (see §6 baselines).

## 4. Objective (formal) — Option B: closed-loop region/spot acquisition

A slide has a set of candidate **regions/spots** $R$ (e.g., ROIs or Visium(-HD) spots), each with an H&E patch $I_r$ and spatial coordinate; the full transcriptome $X\in\mathbb{R}^{|R|\times|G|}$ is the (held-out) ground truth. Acquisition proceeds in **rounds** $t=1..T$: at each round the policy $\pi$ selects the next region(s) to *profile* given everything seen so far; profiling reveals that region's expression. Let $M_t$ = set of profiled regions after round $t$, budget $B$ = total regions profileable. A frozen imputer $\hat{X}(\cdot)$ reconstructs the full slide transcriptome from the profiled subset.

$$\min_{\text{policy }\pi}\ \mathbb{E}\Big[\,\mathcal{L}_{\text{impute}}\big(\hat{X}(M_T),\,X\big)\Big]\quad\text{s.t.}\quad |M_T|\le B,\qquad r_{t+1}\sim\pi\big(\cdot \mid \underbrace{\{I_r\}_{r\in R}}_{\text{whole-slide image}},\ \underbrace{M_t,\,X_{M_t}}_{\text{regions profiled so far + their measured expression}}\big)$$

This is a **budget-constrained POMDP / sequential Bayesian experimental design** problem. The three properties that define our contribution (and separate us from SpatialFinder, §3):
1. **Closed-loop / sequential** — $r_{t+1}$ conditions on $X_{M_t}$ (the expression *actually measured so far*), not just the image. SpatialFinder scores all regions once, ignoring measured expression.
2. **Reconstruction objective** — minimize whole-transcriptome imputation error $\mathcal{L}_{\text{impute}}$, not a diversity/relevance heuristic.
3. **Learned policy** — $\pi$ is learned (greedy expected-information-gain first, then amortized/RL), not a hand-tuned score.

**Research question:** *does a closed-loop, image-conditioned acquisition policy recover the unmeasured transcriptome better, at equal region budget, than (a) one-shot region selection (SpatialFinder-style), (b) a static reference gene panel (PERSIST/ReconST), and (c) naive region sampling?*

## 5. How we approach it — subtasks (map 1:1 to CHECKLIST.md)

Five workstreams; each is a checklist block. WS-A/B/C parallel; D needs A+B+C; E needs D.

- **WS-A — Benchmark & region-acquisition simulator.** Data = HEST-1k (Visium whole-transcriptome slides = ground truth; Xenium/Visium-HD for realistic region granularity) + STImage-1K4M. Build a **closed-loop region simulator**: define candidate regions (ROIs/spots), start with none profiled, expose a `reveal(region) → measured expression` API, enforce budget $B$, and after each round/at end score imputation of all *unprofiled* regions' transcriptomes. Region granularity chosen to match GeoMx-ROI (~100–600µm) and Visium-HD spot scales. Wet-lab-free. [HEST-1k, NeurIPS 2024 D&B Spotlight]
- **WS-B — Imputer (frozen environment / reward model).** A fixed imputer reconstructing full-slide transcriptome from profiled regions + image. Candidates: STFlow/Stem (image-conditioned generative) or SpaIM/TISSUE-wrapped. Must expose **per-region/per-gene predictive uncertainty** (Stem sample variance / GenAR heteroscedastic head / TISSUE conformal intervals) — this is the information-gain signal WS-D consumes. Frozen so all policy comparisons are fair. [TISSUE; SpaIM; STFlow; Stem]
- **WS-C — Baselines (the "their runs" to beat).** (1) **One-shot region selection — SpatialFinder-style** diversity/relevance scoring, no closed loop (THE key competitor). (2) **Static gene panel** — PERSIST/ReconST (the orthogonal "measure fixed genes everywhere" strategy) mapped into the same budget. (3) **Naive region sampling** — random, grid/uniform, greedy-variance, k-means-on-image-embeddings. All scored through the identical simulator + frozen imputer. [SpatialFinder 2026; PERSIST; ReconST]
- **WS-D — Our closed-loop policy (the contribution).** Over the simulator: (i) H&E conditioning via frozen pathology encoder (UNI2-h, reuse existing pipeline); (ii) spatial structure (k-NN/graph over regions, cf. STFlow E(2)-invariant attention); (iii) **conditioning on already-measured expression** $X_{M_t}$ (the closed loop). Progression: **greedy expected-information-gain** policy first (cheap, no RL), then **amortized/RL policy** (AFA-RL seq-VAE-over-partial-state template) only if greedy leaves headroom. Non-greedy lookahead (NOCTA) as a later escalation. [AFA-RL; NOCTA; STFlow]
- **WS-E — Evaluation, ablations, writeup.** Imputation-error-vs-budget Pareto curves across tissues; the **decisive ablation: closed-loop (condition on $X_{M_t}$) vs. one-shot (image-only)** at equal budget — this single result proves our contribution over SpatialFinder; plus ablate image-conditioning and spatial structure; significance across slides/tissues; secondary metric Moran's I (spatial-pattern preservation, cf. ReconST).

## 6. What "beating the baselines" must show (the paper's core claim)

At equal region budget $B$, our closed-loop image-conditioned policy must achieve lower whole-transcriptome imputation error than the baselines, AND an ablation must show the gain comes specifically from conditioning on measured expression + image + spatial structure, not merely from being sequential.

**Corrected gate criterion (do not single-point false-pass).** A first prostate run at 5% budget exposed that a naive "ours > one-shot" gate FALSE-PASSES: one-shot (SpatialFinder-style) seeks *diverse/outlier* regions, which are poor anchors for reconstruction, so it underperforms even random — beating it proves nothing. The tests that actually matter, evaluated across a **budget sweep** (not one point), are:
1. **Beat random** — ours(closed-loop) > random (the real floor; random averaged over seeds).
2. **The closed loop helps** — ours(measured-on) > ours(no_measured ablation).
Beating one-shot and static panels (PERSIST/ReconST) is reported but is *not sufficient* on its own.

**Iterate, don't binary-fail.** The go/no-go is a *diagnostic across budgets*, not a single pass/fail. At high budget the task saturates (every policy ~equal); the signal lives at LOW budgets. If the two core tests fail at all tested budgets with the current setup, the response is to ITERATE — stronger imputer (STFlow/SpaIM vs kNN), a better acquisition score (expected error-reduction vs raw uncertainty), or a harder target (full transcriptome) — and re-sweep, BEFORE concluding STOP. Only a persistent failure across iterations is a real STOP (then reassess with the TA, §8). Tooling: `scripts/03_budget_sweep.py` produces the imputation-error-vs-budget Pareto + per-budget verdicts (the field-standard presentation, cf. PERSIST panel-size curves, AFA-RL reward-vs-#features).

## 7. Target venues & timing

Primary targets (last-2-year acceptances in this exact space prove the venue fit): **NeurIPS** (BLEEP'23, HEST-1k'24), **ICLR** (Stem'25, CellAgent'25), **ICML** (STFlow'25), **CVPR** (MERGE'25). Datasets & Benchmarks / main track both viable. See `RESEARCH_PLAN.md` §1 for the full venue-mapped survey.

## 8. Risk register

| Risk | Severity | Mitigation |
|---|---|---|
| **Closed-loop does not beat one-shot (SpatialFinder) at equal budget** | **Critical (go/no-go)** | This IS the contribution. Front-load the WS-D early-signal test (greedy closed-loop vs one-shot on one tissue) before any full build |
| SpatialFinder / a fast-follower publishes the closed-loop version first | High (area is active, 2026) | Move fast on the early-signal prototype; keep the reconstruction-objective + learned-policy differentiators crisp; the closed-loop ablation is the moat |
| "Just AFA with a CNN prior" reviewer verdict | High | Center the *spatial-structure + measured-expression conditioning* novelty; ablate to prove each is necessary (WS-E) |
| Imputer quality caps the whole result | Medium | Use best available imputer, expose its uncertainty; treat imputer as fixed/frozen so comparisons are fair (WS-B) |
| Within-slide adaptive gene panel resurfaces as a goal | Medium | REJECTED — physically infeasible (see `lit_review/REALITY_CHECK.md`); keep scope on region/spot acquisition |
| Compute for RL policy training | Medium | Greedy expected-information-gain first (no RL); escalate to amortized/RL only if greedy leaves headroom |
| HEST-1k / region-simulator integration effort | Medium | Budget real time in WS-A; hard dependency for everything |

## 9. Full reference list (Direction-1 relevant; extends RESEARCH_PLAN §6)

**Benchmark / data**
- HEST-1k — NeurIPS 2024 D&B Spotlight. https://arxiv.org/abs/2406.16192
- STImage-1K4M. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11213178/

**Region acquisition (primary competitor + reality-check sources)**
- SpatialFinder — Frontiers in Bioinformatics 2026 (one-shot ROI prioritization; nearest competitor). https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2026.1746714/full
- GeoMx DSP ROI workflow (manual, budget-limited). https://brukerspatialbiology.com/blog/geomx-dsp-whats-in-an-roi/
- Self-driving labs review (sequential active acquisition paradigm). https://royalsocietypublishing.org/rsos/article/12/7/250646/235354
- Autonomous scanning microscopy / FAST (adaptive sparse measurement). https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10485018/
- See `lit_review/REALITY_CHECK.md` for the full analysis.

**Panel selection (baselines / incumbents)**
- PERSIST — Nature Communications 2023. https://www.nature.com/articles/s41467-023-37392-1
- ReconST — Nucleic Acids Research 2026 / bioRxiv 2025. https://academic.oup.com/nar/article/54/11/gkag621/8709438
- scGIST — Genome Biology 2024. https://genomebiology.biomedcentral.com/articles/10.1186/s13059-024-03185-y
- gpsFISH — Genome Biology 2024. https://genomebiology.biomedcentral.com/articles/10.1186/s13059-024-03174-1
- RL gene-panel (2501.04718) — arXiv 2025. https://arxiv.org/html/2501.04718

**Active feature acquisition / experimental design (method template)**
- RL with Efficient Active Feature Acquisition — arXiv 2011.00825. https://arxiv.org/abs/2011.00825
- Stochastic Encodings for AFA — 2025. https://arxiv.org/html/2508.01957v3
- NOCTA (non-greedy AFA) — 2025. https://arxiv.org/pdf/2507.12412

**Imputation + uncertainty (environment / reward)**
- TISSUE — 2023. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10168375/
- SpaIM — Nature Communications 2025. https://www.nature.com/articles/s41467-025-63185-9
- PASTA — Nature Communications 2025. https://www.nature.com/articles/s41467-025-67421-0
- Reliable ST imputation w/ uncertainty — Patterns 2024. https://www.cell.com/patterns/fulltext/S2666-3899(24)00154-5

**Prediction-from-histology context (related work, non-acquisition)**
- BLEEP — NeurIPS 2023. https://arxiv.org/abs/2306.01859
- Stem — ICLR 2025. https://arxiv.org/abs/2501.15598
- STFlow — ICML 2025 Spotlight. https://arxiv.org/abs/2506.05361
- MERGE — CVPR 2025. https://arxiv.org/abs/2412.02601
- GenAR — preprint 2025. https://arxiv.org/abs/2510.04315
- PEaRL — preprint 2025. https://arxiv.org/abs/2510.03455

**Agentic-AI-in-biology context**
- CellAgent — ICLR 2025. https://arxiv.org/abs/2407.09811
- SpatialAgent — bioRxiv 2025. https://www.biorxiv.org/content/10.1101/2025.04.03.646459v1
