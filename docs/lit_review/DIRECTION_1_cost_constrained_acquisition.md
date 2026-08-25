# Direction 1 — Cost-Constrained Acquisition / Active Profiling

**One-line thesis:** *Given a finite measurement budget (probes, spots, or dollars), decide which measurements to take so the unmeasured transcriptome is best recovered — and make that decision sequentially/spatially, conditioned on what's already been measured and on the histology image.*

This document reviews the literature, isolates what is genuinely unsolved, and states honestly how crowded the space is.

---

## 1. What motivates the problem (the real-world constraint)

Targeted spatial platforms are **panel-limited by physics, not by choice**:
- FISH-based / imaging platforms (MERFISH, Xenium, CosMx) are typically limited to ~100–1,000 genes per experiment. [scGIST, Genome Biology 2024, states the ~1,000-gene ceiling explicitly — https://genomebiology.biomedcentral.com/articles/10.1186/s13059-024-03185-y]
- Every extra gene/probe costs money, imaging time, and optical-crowding risk.
- Whole-transcriptome sequencing-based ST (Visium) is cheaper per-gene but coarse in resolution and still costs per-slide/per-spot.

So there is a genuine, quantifiable trade-off: **which K genes (or which spots) do I physically measure so that everything I didn't measure can be computationally imputed with least error?** This is a real experimental-design decision biologists make before every targeted-ST run. That is the honest motivation — and it is much stronger than the "predict-all-genes-from-image" motivation, because here the decision has real cost and real consequence.

Your oracle result becomes *supporting evidence* under this framing: it demonstrated that conditioning on true values of some genes improves prediction of others (~0.24–0.54 PCC vs ~0.20 static). Under acquisition framing, that's exactly the claim you want — measured genes carry information about unmeasured ones — rather than an embarrassment about a policy that wouldn't train.

---

## 2. Prior work — and this is a crowded, mature field (read this honestly)

### 2.1 Static gene-panel selection (the dominant paradigm)

All of these pick a **single fixed panel** from a reference scRNA-seq dataset, applied uniformly to the whole tissue:

| Method | Venue | Idea | Limitation w.r.t. our angle |
|---|---|---|---|
| PERSIST | Nature Communications 2023 | Deep autoencoder with a learned binary mask; sparsifies to K genes that reconstruct the genome-wide profile; binarizes expression to bridge scRNA→ST domain shift | Static panel; no spatial context; no image; no sequential conditioning. [https://www.nature.com/articles/s41467-023-37392-1] |
| scGIST | Genome Biology 2024 | Constrained feature selection; lets users prioritize genes while preserving cell-type detection | Static; classification-oriented, not full-transcriptome reconstruction. [https://genomebiology.biomedcentral.com/articles/10.1186/s13059-024-03185-y] |
| gpsFISH | Genome Biology 2024 | Optimizes cell-type detection, models & corrects platform effects | Static; cell-type objective, not imputation error. [https://genomebiology.biomedcentral.com/articles/10.1186/s13059-024-03174-1] |
| ReconST | bioRxiv/NAR 2025 | Gated autoencoder learns importance weights; selects panel that optimally reconstructs whole transcriptome | **Closest objective to ours** (reconstruction), but still static & non-spatial, no image. [https://academic.oup.com/nar/article/54/11/gkag621/8709438] |
| scGPD | bioRxiv 2025 | Single-cell-informed panel design as feature selection | Static; scRNA-reference-driven. [https://www.biorxiv.org/content/10.1101/2025.10.02.680117] |

**Takeaway:** the *static panel selection* problem is thoroughly solved and competitive. You cannot win by proposing "yet another panel selector."

### 2.2 RL / sequential formulations (the ones that threaten our novelty most)

| Method | Venue | Idea | Gap it leaves |
|---|---|---|---|
| Knowledge-Guided Gene Panel Selection: An RL Perspective | arXiv 2025 (not a top venue) | RL with expert-shaped reward for dynamic gene-panel refinement | Single-cell, **not spatial**; no image conditioning; no per-tissue adaptive acquisition. [https://arxiv.org/html/2501.04718] |
| Enhanced Gene Selection: Pre-Filtering Synergy + Reinforced Optimization | arXiv 2024 | RL-based gene selection with pre-filtering | Single-cell classification; not spatial imputation. [https://arxiv.org/pdf/2406.07418] |

### 2.3 Active Feature Acquisition (AFA) — the general ML subfield you'd be importing from

This is a **mature, well-studied** ML formulation, which is good (established theory to cite) and bad (reviewers know the baselines):
- Reinforcement Learning with Efficient Active Feature Acquisition (arXiv:2011.00825) — instance-wise, sequential "which feature to measure next" as an RL/POMDP. [https://arxiv.org/abs/2011.00825]
- Stochastic Encodings for Active Feature Acquisition (2025) — latent-variable AFA. [https://arxiv.org/html/2508.01957v3]
- NOCTA: Non-Greedy Objective Cost-Tradeoff Acquisition (2025) — non-greedy longitudinal AFA. [https://arxiv.org/pdf/2507.12412]
- Adapt-RADIN — RL-guided variable-length cost-sensitive acquisition.

**This is the honest risk:** "sequential cost-constrained acquisition where each measurement updates your posterior" is *exactly* the AFA problem, already formalized with RL. Your contribution cannot be "apply AFA to genes" — that's a straightforward application.

### 2.4 Imputation + uncertainty (the "impute the rest" half of the loop)

Already strong; you'd use these as components/baselines, not reinvent them:
- TISSUE (2023) — uncertainty-calibrated spatial imputation wrapper. [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10168375/]
- SpaIM (Nat Commun 2025) — style-transfer imputation of unmeasured ST genes. [https://www.nature.com/articles/s41467-025-63185-9]
- TransImpute — imputation + uncertainty attributes + spatial regularization.
- PASTA (Nat Commun 2025) — pathway-specific imputation. [https://www.nature.com/articles/s41467-025-67421-0]

---

## 3. The actual unsolved gap (where a real contribution could live)

Cross the two axes and there is a genuinely empty cell:

| | Static panel | Sequential/adaptive acquisition |
|---|---|---|
| **Non-spatial, scRNA reference** | PERSIST, scGIST, gpsFISH, ReconST, scGPD | RL panel (2501.04718), AFA (2011.00825) |
| **Spatially-resolved, image-conditioned** | *sparse* | **← EMPTY: no method does per-tissue, image-conditioned, sequential acquisition that adapts the panel/locations based on already-measured spots** |

The defensible contribution is therefore **narrow and specific**:
> A policy that, *for a given tissue*, uses the H&E image + the spots/genes measured so far to decide the next measurement (which gene, or which spatial location), such that the fixed budget is spent to minimize whole-transcriptome imputation error — as opposed to a one-size-fits-all panel chosen from a reference before seeing the tissue.

Two concrete sub-variants:
1. **Spatially-adaptive gene selection**: the panel can differ across regions of the *same* slide (tumor core vs stroma need different genes), decided online from the image + early measurements. No static-panel method can do this.
2. **Active spot acquisition**: fix the gene panel, decide *which spots/regions to physically profile* next to best impute the rest of the slide — a spatial active-learning loop with the image as prior.

---

## 4. How novel is it, honestly?

- **Conceptually novel:** Medium. The *combination* (spatial + image-conditioned + sequential + imputation-objective) appears genuinely unoccupied. But every individual ingredient exists, so novelty rests entirely on the combination + a convincing demonstration that spatial/image conditioning beats the strong static baselines.
- **Risk of "incremental" verdict:** Real. A reviewer who knows AFA + PERSIST will ask "isn't this just AFA with a CNN prior?" You need a result showing the spatial/image structure yields a gain that non-spatial AFA/PERSIST *cannot* achieve. If it doesn't, there's no paper.
- **Data feasibility:** Good. HEST-1k / STImage-1K4M give paired image+full-transcriptome ST, so you can *simulate* the acquisition loop (mask genes/spots, reveal under budget, impute rest) without wet-lab experiments. This is a major practical advantage — the whole thing is doable computationally.

---

## 5. Is "agentic" the right word here?

Partly. This is genuinely a **sequential decision problem under a budget constraint** (a constrained MDP / POMDP), which is the honest technical home for "agentic." But note: framed rigorously it is **active feature acquisition / Bayesian experimental design**, and those framings are stronger and more precise than "agentic AI." Reviewers respond better to "constrained sequential experimental design" than to "agent." Use the precise term; keep "agentic" as accessible framing only.

---

## 6. Verdict for Direction 1

**Strengths:** genuine real-world motivation; your oracle result becomes evidence *for* the thesis; fully simulatable on existing benchmarks; precise technical home (AFA/OED).
**Weaknesses:** crowded neighborhood (PERSIST/ReconST/RL-panel/AFA all adjacent); novelty hinges entirely on the spatial+image+sequential combination delivering a measurable, ablatable gain over strong static baselines; risk of "incremental application of AFA" verdict.
**Verdict:** *Viable but must be scoped tightly.* Only pursue if you commit to the spatially-adaptive / image-conditioned angle (the empty cell in §3) and can show it beats a static PERSIST-style panel and a non-spatial AFA policy on HEST-1k. If the gain is marginal, it collapses back into "incremental."
