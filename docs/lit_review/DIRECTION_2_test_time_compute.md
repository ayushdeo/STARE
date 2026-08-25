# Direction 2 — Test-Time Compute Allocation

**One-line thesis:** *Generative ST models (diffusion/flow) spend uniform inference compute on every spot. Under a fixed total compute budget, allocate more compute where the model is uncertain / where the marginal information gain is highest, and less where it's confident — recovering most of the accuracy at a fraction of the cost.*

Your own intuition in the prompt is exactly right: **this is a constrained resource-allocation / optimization problem** — maximize total prediction quality subject to a fixed compute budget, by distributing per-spot compute according to expected marginal gain. This document reviews the literature and stress-tests the idea.

---

## 1. What motivates the problem (the real cost)

The two current SOTA generative ST models are inference-heavy:
- **Stem** (ICLR 2025) — conditional diffusion; quality scales with number of denoising steps. [https://arxiv.org/abs/2501.15598]
- **STFlow** (ICML 2025 Spotlight) — whole-slide flow matching; integration steps × thousands of spots per slide. [https://arxiv.org/abs/2506.05361]

A single whole-slide inference is expensive precisely because compute is spent **uniformly** — the easy stroma spot gets the same number of diffusion/flow steps as the ambiguous tumor-boundary spot. If most spots are "easy," uniform allocation wastes the majority of the budget. This is the same observation that drives adaptive test-time compute in LLMs, applied to a domain where **nobody has done it yet.**

The optimization statement (your instinct, formalized):
> maximize Σ_spots quality(spot, c_spot) subject to Σ_spots c_spot ≤ C_total,
> where c_spot is per-spot compute (steps/samples) and quality is concave in c_spot (diminishing returns). Optimal allocation equalizes *marginal* gain per unit compute across spots → spend where marginal information gain is highest.

That's a clean, defensible problem with a known optimization structure (water-filling / greedy marginal allocation), and the research question is: *can you estimate per-spot marginal gain cheaply enough (from uncertainty proxies) that adaptive allocation beats uniform?*

---

## 2. Prior work

### 2.1 Adaptive test-time compute — exploding, but almost entirely LLM-land

- "Scaling LLM Test-Time Compute Optimally" — ICLR 2025. Foundational: adaptively allocating inference compute beats uniform/naive scaling. [https://proceedings.iclr.cc/paper_files/paper/2025/file/1b623663fd9b874366f3ce019fdfdd44-Paper-Conference.pdf]
- "A Survey of Adaptive and Controllable Test-Time Compute" (2025). [https://arxiv.org/pdf/2507.02076]
- Cross-query allocation: don't over-invest in easy instances, reserve compute for hard ones — *directly analogous to per-spot allocation.* [multiple 2025-2026 arXiv]

**Key point:** the entire field is LLM reasoning / verifier-based sampling. The core idea (allocate by difficulty/uncertainty) transfers, but **none of it is about dense spatial regression/generation.** That's your opening.

### 2.2 Test-time compute for diffusion models (closer)

- "Scaling Properties of Diffusion Models for Perceptual Tasks" (2024) — more denoising steps = more test-time compute = better perceptual output; establishes the "steps = compute knob" you'd allocate. [https://arxiv.org/pdf/2411.08034]
- General diffusion test-time scaling work treats step count / search over noise as the compute dimension.

**Gap:** these scale compute *uniformly or via global search*, not via *per-element (per-spot) adaptive allocation under a joint budget.*

### 2.3 Adaptive computation in pathology / imaging (nearest neighbor in the application domain)

- **PathCTM / "Thinking in Scales"** (2026) — adaptive continuous reasoning over gigapixel WSIs: dynamic scale-switching, attention-guided region pruning, **confidence-aware early stopping** that terminates once uncertainty is bounded. [https://arxiv.org/html/2605.19491v1]
- Early-exit networks (mature) — per-sample computation depth at runtime. [ACM EMDL 2021; ScienceDirect 2020]
- Window-based early-exit cascades for uncertainty (2023). [https://arxiv.org/pdf/2303.08010]

**Point:** adaptive/early-exit computation is proven to work in pathology *for classification/diagnosis*, but **not for spatial gene-expression generation.** PathCTM is the closest prior art and the one to differentiate against — it does adaptive compute for *diagnosis over scales*, not per-spot compute for *transcriptome generation under a budget.*

### 2.4 Uncertainty estimation in ST (the signal you'd allocate on)

You need a per-spot uncertainty/difficulty proxy; these already exist and are citable:
- TISSUE (2023) — calibrated per-gene/per-spot imputation uncertainty. [https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10168375/]
- Reliable ST imputation with uncertainty estimation (Patterns 2024). [https://www.cell.com/patterns/fulltext/S2666-3899(24)00154-5]

---

## 3. The actual unsolved gap

> **No existing method does per-spot adaptive test-time compute allocation for generative spatial transcriptomics.** STFlow and Stem spend uniform steps across all spots; adaptive-compute methods exist for LLMs and for pathology *classification*, but not for dense ST *generation* under a joint slide-level compute budget.

Concrete contribution shape:
1. Take a *frozen, pretrained* STFlow/Stem as the compute environment (you don't retrain the expensive model).
2. Learn/derive a lightweight **allocation policy** that reads a cheap early-signal (e.g., a few-step preview, or an uncertainty head) and assigns each spot a compute level (number of flow/diffusion steps or ensemble samples).
3. Show the accuracy–compute Pareto curve dominates uniform allocation: e.g., "match STFlow's full-budget PCC at 40% of the FLOPs," or "at equal FLOPs, +X PCC."

This is attractive because it **reuses your reproduced baselines as the environment** (Phase 2 of the research plan directly feeds this) rather than discarding them, and it lands in a *hot* area (test-time compute) with a clearly empty application cell.

---

## 4. How novel is it, honestly?

- **Conceptually novel:** Medium-high *for this domain*. The mechanism (adaptive test-time compute) is well-trodden in LLMs, so the novelty is domain-transfer + the spatial/joint-budget structure (allocation across spots that are *spatially correlated*, so measuring/computing one spot informs its neighbors — a twist LLM per-query allocation doesn't have).
- **The spatial twist is the intellectual core:** in LLM allocation, queries are independent. Here spots are spatially correlated → compute spent on one spot reduces uncertainty on neighbors → the allocation problem is a *structured* one (submodular over a spatial graph), not independent per-item. **That structure is what could make it a real ML contribution rather than "adaptive compute, now for genes."**
- **Risk of "just an efficiency paper" verdict:** This is the main danger. Pure efficiency/Pareto results are hard at NeurIPS/ICLR unless (a) the gains are large, or (b) there's a genuine methodological/theoretical insight (e.g., the submodular-over-spatial-graph allocation with guarantees). Frame around the *structured allocation* insight, not just "we saved FLOPs."

---

## 5. Is "agentic" the right word here?

**Weakly.** This is honestly **adaptive computation / constrained optimization / a control policy**, not an "agent" in any rich sense. It's a policy that allocates a budget — you *can* call it agentic, but reviewers will (correctly) read it as adaptive inference. If you go this route, drop "agentic" from the framing entirely and sell it as "structured adaptive test-time compute for spatial generative models." That's a cleaner, more honest, more current framing than "agentic AI."

---

## 6. Verdict for Direction 2

**Strengths:** rides a hot topic (test-time compute); reuses reproduced STFlow/Stem as environment (no expensive retraining); genuinely empty application cell; the spatial-correlation structure gives a real ML angle (structured/submodular allocation) beyond plain efficiency.
**Weaknesses:** "agentic" framing doesn't really fit (it's adaptive computation); risk of being dismissed as an efficiency paper unless the structured-allocation insight is the centerpiece; requires a *working, faithfully-reproduced* STFlow/Stem first (hard dependency on Phase 2).
**Verdict:** *Viable and timely, but reframe away from "agentic."* Strongest if the contribution is the **spatially-structured allocation** (submodular over the spot graph, exploiting spatial correlation), not just per-spot early-exit. Weakest if it stays "we made diffusion cheaper."

---

## 7. Note on the two directions sharing a backbone

Directions 1 and 2 are secretly the same math: both are **budget-constrained sequential allocation guided by expected information gain** — Direction 1 allocates *physical measurement* budget, Direction 2 allocates *compute* budget. The information-gain machinery (per-spot/per-gene expected uncertainty reduction) is identical. If one direction stalls, the other reuses ~70% of the formulation. See `DECISION_SUMMARY.md`.
