> **SUPERSEDED FOR DIRECTION:** As of 2026-07-05 the project has committed to Direction 1. The master reference is now [`SOURCE_OF_TRUTH.md`](SOURCE_OF_TRUTH.md), with the executable checklist in [`CHECKLIST.md`](CHECKLIST.md). This file is retained for the venue-mapped literature survey (§1) and historical phasing — read §1 for related-work, but take direction/status from `SOURCE_OF_TRUTH.md`.

# Research Plan: Spatial Transcriptomics + Agentic AI — Restart

**Status:** Restarting from motivation, per TA feedback that the prior architecture (ranking-policy + value-head Gene-Token Decoder) is too simple for a NeurIPS/ICLR-level submission. This document is the single source of truth for the new direction. Update it as a living document — do not let the checklist (`CHECKLIST.md`) drift out of sync with this file.

---

## 0. Why we are restarting

Prior attempt (see `MidTermReport.pdf`, `CSCI 566 (1/2/3).pdf`) established:
- Static decoders (MLP/DeepMLP/Transformer/SpatialTransformer) on frozen UNI2-h embeddings all converge to ~0.20 mean PCC on top-50 HVGs (Visium DLPFC, leave-one-slide-out). [Internal result]
- An oracle-guided sequential decoder (ground-truth gene values as prefix) reaches ~0.24–0.54 PCC depending on config, showing sequential conditioning *can* help.
- A **learned** policy (the part that would actually need to ship) collapsed to ~0.04–0.06 PCC despite ~86% policy "accuracy" — i.e., imitating the oracle ordering did not transfer to better regression. [Internal result, `CSCI 566 (2).pdf`]
- Our local BLEEP reproduction is broken at the gene level (gene PCC ≈ 0.0001, vs. paper's reported strong performance) — confirmed still broken in `BLEEP/bleep_eval_out/metrics.csv`.

TA's objection: this is a fixable-bug-and-tune-hyperparameters problem, not a publishable contribution. We need a problem statement that a top-tier AI conference would consider novel, not just "add a policy head to an existing regression pipeline."

---

## 1. Literature survey: what's actually at NeurIPS / ICML / ICLR / CVPR (last ~2 years)

### 1.1 Gene-expression-from-histology prediction — the established task, no agency involved

| Paper | Venue | Year | Method | Why it matters to us |
|---|---|---|---|---|
| BLEEP — Bi-modal Contrastive Learning for ST Prediction [Xie et al.] | NeurIPS (main conf.) | 2023 | Contrastive image↔gene joint embedding + kNN retrieval | Original baseline we already (mis-)reproduced. [arXiv:2306.01859](https://arxiv.org/abs/2306.01859), [NeurIPS proceedings](https://proceedings.neurips.cc/paper_files/paper/2023/hash/df656d6ed77b565e8dcdfbf568aead0a-Abstract-Conference.html) |
| HEST-1k [Jaume et al.] | NeurIPS Datasets & Benchmarks (Spotlight) | 2024 | 1,229 paired ST–WSI dataset + standardized HEST-Benchmark (10 tasks) for histology foundation models | **This is the field's standard benchmark now.** We were using an ad hoc 12-fold split on one dataset; if we want comparable numbers we should evaluate on HEST-Benchmark. [arXiv:2406.16192](https://arxiv.org/abs/2406.16192), [GitHub](https://github.com/mahmoodlab/hest) |
| Stem [Zhu, Zhu, Tao, Qiu — Georgia Tech] | ICLR (poster) | 2025 | Conditional diffusion model over gene expression vector | We approximated this badly (transformer + AdaLN, not real diffusion). [arXiv:2501.15598](https://arxiv.org/abs/2501.15598), [OpenReview](https://openreview.net/forum?id=FtjLUHyZAO), [code](https://github.com/SichenZhu/Stem) |
| STFlow [Huang et al.] | **ICML (Spotlight)** | 2025 | Flow matching over the **joint distribution of all spots on a slide** (explicit cell–cell interaction), memory-efficient slide-level encoder | Strongest current non-agentic baseline; beats foundation models by 18%+ on HEST-1k/STImage-1K4M. **Primary baseline to reproduce.** [arXiv:2506.05361](https://arxiv.org/abs/2506.05361), [ICML proceedings](https://proceedings.mlr.press/v267/huang25t.html), [code](https://github.com/Graph-and-Geometric-Learning/STFlow) |
| MERGE [—] | **CVPR** | 2025 | Multi-faceted hierarchical graph + GNN over spot clusters (intra/inter-cluster edges model distant tissue interaction) | Shows graph-structured spatial context is a productive direction (cf. our failed "Spatial Transformer"). [arXiv:2412.02601](https://arxiv.org/abs/2412.02601) |
| GenAR [—] | preprint, under review | 2025 | **Multi-scale autoregressive, codebook-free discrete gene-token generation**, hierarchical gene clustering for cross-gene dependency, coarse-to-fine refinement | **This is the closest published work to our original "Gene-Token Decoder" idea — and it is not agentic, it's a properly engineered autoregressive generative model.** Must be addressed directly: what would an agentic version add beyond this? [arXiv:2510.04315](https://arxiv.org/abs/2510.04315), [OpenReview](https://openreview.net/forum?id=KlnAP2F2Zx) |
| PEaRL [—] | preprint | 2025 | Pathway-activation-score supervision (ssGSEA) + contrastive alignment with histology | Shows biologically-grounded auxiliary supervision (pathways, not just raw HVGs) improves PCC up to +58.9%. Relevant to "which genes/structure to predict" question. [arXiv:2510.03455](https://arxiv.org/abs/2510.03455) |

**Conclusion of 1.1:** Zero top-venue papers on the *prediction task itself* use agent/policy/RL architectures. The field's progress has come from (a) better generative objectives (diffusion, flow matching), (b) better spatial structure modeling (graphs, joint whole-slide distributions), and (c) better gene-side structure (hierarchical token clustering, pathway supervision) — not from sequential decision-making per se.

### 1.2 "Agentic AI" in spatial biology / genomics — exists, but solves a different problem

| Paper | Venue | Year | What the agent actually does | Relevance |
|---|---|---|---|---|
| CellAgent [Xiao et al.] | **ICLR (poster)** | 2025 | LLM multi-agent (planner/executor/evaluator) automates the scRNA-seq/ST **analysis workflow** (preprocessing, clustering, annotation) via natural language | Agency = orchestrating *which tool/pipeline step to run next*, not predicting gene expression values. [bioRxiv](https://www.biorxiv.org/content/10.1101/2024.05.13.593861v4), [arXiv:2407.09811](https://arxiv.org/abs/2407.09811), [ICLR listing](https://iclr.cc/virtual/2026/poster/10010902) |
| SpatialAgent [Genentech] | bioRxiv preprint (not peer-reviewed top venue) | 2025 | LLM agent spanning experimental design → multimodal analysis → hypothesis generation | Same pattern: agency over the *scientific workflow*, not the regression task. [bioRxiv](https://www.biorxiv.org/content/10.1101/2025.04.03.646459v1), [code](https://github.com/Genentech/SpatialAgent) |
| CellForge | submitted ICLR 2026, **desk-rejected** | 2025 | Multi-agent design of virtual-cell *models* (task analysis → method design → experiment execution) | Cannot be cited as "agentic AI got into a top venue" — it didn't. Useful only as a cautionary data point. [OpenReview](https://openreview.net/forum?id=6v7GkwSwtp) |
| LLM4Cell (survey) | arXiv survey | 2025 | Surveys agentic/LLM approaches across single-cell biology | Good source for further citation mining, not itself evidence of a result. [arXiv:2510.07793](https://arxiv.org/html/2510.07793v2) |

**Conclusion of 1.2:** Every credible "agentic AI + spatial/single-cell" paper is automating *experimental/analytical workflow decisions* (what to run, what to test next, how to design a model), not the gene-expression-from-image inference task itself. This is the real gap: **agentic AI has not been shown necessary or beneficial for the core ST prediction problem** — and our original pitch implicitly assumed it would be, without evidence.

### 1.3 Where agency *could* be honestly motivated (candidate problem reframings)

These are not yet committed directions — they need discussion before we pick one:

1. **Test-time compute / inference-budget allocation across spots or genes.** A slide has thousands of spots and a model that can choose *where* to spend more compute (e.g., diffusion steps, ensemble size) under a fixed budget is a genuine sequential-decision problem, distinct from GenAR's fixed-schedule autoregression.
2. **Cost-constrained gene panel selection** (active panel design, not prediction) — this already has non-agentic deep learning solutions (PERSIST, ReconST, scGPD) and at least one explicit RL formulation outside top-tier venues ([arXiv:2501.04718](https://arxiv.org/html/2501.04718v2) — Knowledge-Guided Gene Panel Selection via RL). Worth reading closely: is there a gap an agent framing fills that this RL paper didn't?
3. **Multimodal tool-use agent that decides when to fall back to retrieval (BLEEP-style) vs. generation (STFlow-style) vs. abstain**, conditioned on per-spot uncertainty — i.e., agency as a *meta-controller over existing strong models*, not a replacement for them. This directly uses our existing baselines as components rather than discarding them.
4. **Active learning / sequential acquisition during real experiments** (which spot/region to profile next given a budget) — closer to the experimental-design framing of SpatialAgent but applied to the *prediction* pipeline rather than wet-lab workflow.

Action: pick one of these (or something else literature surfaces) only after Phase 1 replication below — we need calibrated baselines before any architecture decision.

---

## 2. Phase 1: Replicate the original architectures, not our flawed approximations

We previously approximated Stem, STFlow, and BLEEP rather than running them faithfully (see midterm report, section on "We cannot plug UNI embeddings directly into STFlow pipeline..."). This is the actual root cause the TA is flagging, beyond the agentic-architecture question. Before any new architecture: get real, faithful numbers.

1. **BLEEP** — debug the existing local reproduction (`BLEEP/`) until gene-level PCC matches the published range, OR run the [official repo](https://github.com/bowang-lab/BLEEP) unmodified on our data split. Current state is broken (gene PCC ≈ 0.0001; see `BLEEP/bleep_eval_out/metrics.csv`).
2. **Stem** — run the [official code](https://github.com/SichenZhu/Stem) (real diffusion training, not our AdaLN-transformer stand-in) on our split or on their reported dataset to get a faithful number to anchor against.
3. **STFlow** — run the [official code](https://github.com/Graph-and-Geometric-Learning/STFlow) on HEST-1k/STImage-1K4M (their actual benchmark) to get the real SOTA number we'd be competing against.
4. **HEST-Benchmark** — migrate our evaluation off the ad hoc Visium DLPFC 12-fold split and onto [HEST-Benchmark's 10 tasks](https://github.com/mahmoodlab/hest) so any future number we report is comparable to the field.
5. **GenAR** — run the [official code](https://github.com/oyjr/genar) since it is the nearest published idea to ours; if our agentic reframing can't beat or meaningfully differ from GenAR's ablations, that's a strong signal the framing needs to change.

Only after these 5 numbers are in hand (faithful reproductions, not approximations) do we have a legitimate "their runs" to compare "our runs" against.

---

## 3. Phase 2: Establish motivation for agency (not architecture)

Before writing any model code:
- Write a short (1-2 page) motivation memo answering: *what decision is being made, by whom/what, under what uncertainty or constraint, that a single forward pass cannot make?* If we can't answer this crisply, we don't have a paper.
- Survey the RL/active-learning-adjacent papers in section 1.3 in depth (full read, not abstract-only) before deciding.
- Explicitly write down why GenAR's autoregressive (but non-agentic) formulation is insufficient for our chosen motivation — this is the single most likely reviewer question.

---

## 4. Phase 3: New architecture, only after Phase 1 + 2 are done and reviewed

Out of scope to design until Phase 2 produces a committed motivation. Do not skip ahead.

---

## 5. Open questions / risks

- Is "agentic AI" even the right framing, or should we drop it and instead pursue a purely generative/structural contribution (e.g., a HEST-Benchmark-evaluated flow/diffusion model with a genuinely novel structural prior)? The literature survey suggests static/generative work is still where top-venue acceptances are happening; agentic framing is the riskier bet.
- Compute budget: STFlow/Stem are heavier than our prior MLP baselines — confirm we have the GPU budget to faithfully reproduce them before committing.
- Timeline: faithful reproduction (Phase 1) is itself nontrivial; budget real time for it rather than treating it as a formality.

---

## 6. Reference list (full)

1. Xie et al. "Spatially Resolved Gene Expression Prediction from H&E Histology Images via Bi-modal Contrastive Learning" (BLEEP). NeurIPS 2023. https://arxiv.org/abs/2306.01859
2. Jaume et al. "HEST-1k: A Dataset for Spatial Transcriptomics and Histology Image Analysis." NeurIPS 2024 (Datasets & Benchmarks, Spotlight). https://arxiv.org/abs/2406.16192
3. Zhu, Zhu, Tao, Qiu. "Diffusion Generative Modeling for Spatially Resolved Gene Expression Inference from Histology Images" (Stem). ICLR 2025. https://arxiv.org/abs/2501.15598
4. Huang et al. "Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching" (STFlow). ICML 2025 (Spotlight). https://arxiv.org/abs/2506.05361
5. "MERGE: Multi-faceted Hierarchical Graph-based GNN for Gene Expression Prediction from Whole Slide Histopathology Images." CVPR 2025. https://arxiv.org/abs/2412.02601
6. "GenAR: Next-Scale Autoregressive Generation for Spatial Gene Expression Prediction." Preprint/under review, 2025. https://arxiv.org/abs/2510.04315
7. "PEaRL: Pathway-Enhanced Representation Learning for Gene and Pathway Expression Prediction from Histology." Preprint, 2025. https://arxiv.org/abs/2510.03455
8. Xiao et al. "CellAgent: An LLM-driven Multi-Agent Framework for Automated Single-cell Data Analysis." ICLR 2025 (poster). https://arxiv.org/abs/2407.09811
9. "SpatialAgent: An autonomous AI agent for spatial biology." bioRxiv preprint, 2025. https://www.biorxiv.org/content/10.1101/2025.04.03.646459v1
10. "CellForge: Agentic Design of Virtual Cell Models." Submitted ICLR 2026, desk-rejected. https://openreview.net/forum?id=6v7GkwSwtp
11. "LLM4Cell: A Survey of Large Language and Agentic Models for Single-Cell Biology." arXiv survey, 2025. https://arxiv.org/html/2510.07793v2
12. "Knowledge-Guided Gene Panel Selection for Label-Free Single-Cell RNA-Seq Data: A Reinforcement Learning Perspective." arXiv, 2025. https://arxiv.org/html/2501.04718v2
