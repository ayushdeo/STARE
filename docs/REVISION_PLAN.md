# Revision Plan — stress-test reframe (response to critique)

The critique's verdict (low novelty as framed) is largely correct. We are taking the "stronger,
genuinely publishable" path: reframe from a flat negative result into a **stress test and boundary
study** of adaptive spatial-transcriptomics acquisition. New claim target:

> Adaptive sampling outperforms strong one-shot placement only beyond an identifiable level of
> value-dependent, spatially clustered structure. We quantify that boundary, reproduce published
> adaptive samplers, and show that under matched strong (MI-greedy) placement and a matched imputer,
> the reported advantage of adaptive ST sampling narrows or disappears in the regimes we test.

Verified prior work that must be engaged (all real, checked 2026-09):
- **SOFisher** — Nat. Commun. 2026 (bioRxiv 2024.07.05.602236), RL sequential FOV selection. github Heathcliff-Ng/SOFisher.
- **SCR2-ST** — arXiv 2512.13635 / MIDL 2026, RL active sampling + reconstruction, positive low-budget result. github hrlblab/SCR2ST.
- **S2-omics** — Nat. Cell Biol. 2025 (bioRxiv 2025.09.21.677634), H&E ROI selection + whole-slide recovery. github ddb-qiwang/S2Omics.

---

## Critique point -> action -> where it runs

| # | Critique point | Action | Runs |
|---|---|---|---|
| 1 | Missing adaptive prior work; "non-adaptive" claim false | Rewrite related work; cite SOFisher/SCR2-ST/S2-omics; position our angle (matched strong baseline + matched imputer) | local (writing) |
| 1b | Benchmark the adaptive competitors | Port SOFisher and SCR2-ST policies into our simulator; run vs MI-greedy under matched imputer | **4070** |
| 2 | Policy/uncertainty inconsistent; §3.3 not auditable | DONE: added `value_greedy` with explicit score s(i|A_t,y_{A_t},X); add pseudocode; fix/remove old §3.3 | local |
| 1/2 | Weak heuristics; no positive control | DONE (running): `07_positive_control.py` boundary + positive control on synthetic hotspot data | local |
| 3 | Theory overstated | Soften to "motivates the hypothesis"; drop "kNN is fixed-kernel regime" as fact; optional limited proposition | local |
| 4 | "Best non-adaptive" is an oracle | Rename "oracle non-adaptive envelope"; also report vs each baseline and vs a prospectively-fixed MI-greedy (LOSO) | local (re-analysis of existing JSONs) |
| 5 | n=7 too small; multiplicity | More slides; exact paired/sign inference; cluster by patient; max-T across budgets; tissue-stratified; report all per-slide values | **4070** (slides) + local (stats) |
| 6 | Null != equivalence | Preregister delta=0.01 PCC; TOST equivalence or hierarchical; remove residual "no gain" language | local |
| 7 | Not physically realistic | Add batched acquisition (1/2/4/8 rounds) and contiguous ROIs/FOVs; OR reposition as in-silico value-of-information | **4070** + local |
| 8 | HVG leakage | Separate genes for imputer-fit / policy / eval; sensitivity: global HVG, marker set, fixed panel, all-expressed | **4070** |
| 9 | "patient-stratified" misdescribed | State it is within-slide transductive reconstruction; remove "patient-stratified" | local |
| 10 | One metric | Add MSE/MAE, Spearman, gene-wise dist, Moran's I preservation, per-tissue/per-gene-class | local (metrics exist) + 4070 (rerun) |
| pres | Figure-thin | Add pipeline schematic; spatial maps of selected regions + reconstruction error; positive-control figure (Fig 2) | local |

---

## Execution order

**Stage A — local, no new real-data runs (in progress):**
1. `value_greedy` policy with auditable value-conditioned score. [DONE]
2. Positive control + boundary curve on synthetic hotspot data (`07_positive_control.py`, Fig 2). [RUNNING]
3. Three-way comparator re-analysis from existing `ci_multislide_*.json` (per-baseline, fixed-MI, oracle envelope). [TODO]
4. Manuscript: fix related work, §3.3, theory language, "patient-stratified", oracle naming, equivalence framing. [TODO]

**Stage B — 4070, new runs:**
5. Expand slide set (more HEST Visium HD samples per tissue; target >=20 slides). Re-run sweep + CI.
6. Port SOFisher and SCR2-ST acquisition policies into `policies/`; benchmark under matched imputer + MI-greedy.
7. Batched acquisition (rounds in {1,2,4,8}) and contiguous-ROI constraint in the simulator.
8. HVG-leakage sensitivity (global/marker/panel/all-expressed gene sets).
9. Extra metrics + spatial maps on real slides.

**Stage C — integrate:**
10. Rewrite results around the boundary claim; add equivalence tests; full per-slide appendix; multiplicity control.

---

## Ownership (per the machine split)
- Local (this repo, my machine): policies, simulator changes, stats, re-analysis, manuscript, figures, synthetic experiments.
- 4070: real-data runs (`data.py`, encoders), competitor reproductions that need HEST + GPU.
- Keep `config.py` the single source of pinned parameters; log every new run in `ITERATION_LOG.md`.
