# Decision Summary — Direction 1 vs Direction 2

Read `DIRECTION_1_cost_constrained_acquisition.md` and `DIRECTION_2_test_time_compute.md` first. This is the side-by-side to decide from.

## The unifying insight

Both directions are the **same underlying problem**: *budget-constrained sequential allocation guided by expected information gain.*
- **Direction 1** allocates a **physical measurement** budget (which genes/spots to profile). Information gain = uncertainty reduction in the *unmeasured transcriptome*.
- **Direction 2** allocates a **compute** budget (diffusion/flow steps per spot). Information gain = accuracy gained per unit FLOP.

Your framing in the prompt — "spending compute where needed, that's an optimization problem" — is exactly Direction 2's core, and it generalizes to Direction 1 by swapping "compute" for "measurement." So this is genuinely one research program with two application faces.

## Side-by-side

| Criterion | Direction 1: Acquisition | Direction 2: Test-time compute |
|---|---|---|
| Real-world motivation | **Strong** — targeted ST is physically panel-limited & costly | Medium — inference cost matters but is a "nicer-to-have" |
| Uses your oracle result as evidence | **Yes, directly** | No |
| Reuses reproduced baselines (Phase 2) | Partially (imputer) | **Yes, entirely** (STFlow/Stem = environment) |
| Novelty ceiling | Medium (crowded: PERSIST/ReconST/AFA/RL-panel) | Medium-high *for the domain* (empty cell) |
| "Agentic" framing fits? | **Yes** (constrained sequential decision / AFA / OED) | Weakly (it's adaptive computation — drop "agentic") |
| Biggest risk | "Incremental application of AFA/panel-selection" | "Just an efficiency paper" |
| What defuses the risk | Spatial + image-conditioned + adaptive panel (the empty cell) beating static PERSIST | Spatially-**structured** allocation (submodular over spot graph), not plain early-exit |
| Wet-lab needed? | No — simulate acquisition by masking on HEST-1k | No |
| Hard dependency | A working imputer | A **faithfully reproduced** STFlow/Stem (harder) |
| Fit to "agentic AI for spatial transcriptomics" thesis | **Best fit** — genuine agency under budget | Poor fit — really adaptive inference |

## Honest recommendation

**If the overarching thesis must stay "agentic AI for spatial transcriptomics": choose Direction 1**, scoped tightly to the empty cell (spatially-adaptive, image-conditioned, sequential acquisition minimizing whole-transcriptome imputation error). It's the only option where "agentic" is technically honest (it's a real constrained MDP / active feature acquisition problem), where your oracle result supports the thesis, and where the motivation (physical panel limits) is concrete and undeniable. The cost is a crowded neighborhood — you must beat static PERSIST/ReconST and non-spatial AFA to have a paper.

**If you're willing to drop the "agentic" label for a stronger technical bet: Direction 2** is more timely and reuses your baselines completely, but you should sell it as *structured adaptive test-time compute*, and the win depends on a working STFlow reproduction.

**My ranking for a NeurIPS/ICLR-level attempt:** Direction 1 (as the empty-cell spatial-adaptive-acquisition problem) is the better bet *given your existing assets* (oracle result, the agentic thesis, no reproduction dependency). Direction 2 is the better bet *only if* you can make the structured-spatial-allocation insight the centerpiece and you've already got STFlow running.

## What to verify before committing (do these reads, per CHECKLIST Phase 1)

1. Full read of **RL panel paper** (arXiv:2501.04718) and **ReconST** (NAR 2025) — confirm neither does spatially-adaptive / image-conditioned acquisition. If one does, Direction 1's empty cell shrinks.
2. Full read of **AFA RL** (arXiv:2011.00825) — understand exactly what the standard baseline policy is, so you can position against it.
3. Full read of **STFlow** (ICML 2025) — confirm it spends uniform per-spot compute (it does, but verify the integration schedule) to validate Direction 2's premise.
4. Full read of **PathCTM / Thinking in Scales** (arXiv:2605.19491) — the nearest adaptive-compute-in-pathology prior art to differentiate from.

## Open decision (for you / TA)

- Commit to Direction 1, Direction 2, or a deliberate "drop agentic, go structural" (the safer path noted in `RESEARCH_PLAN.md` §5)?
- If Direction 1: which sub-variant — adaptive **gene panel** per region, or active **spot** acquisition? (Gene-panel is closer to your oracle result; spot-acquisition is a cleaner spatial story.)
