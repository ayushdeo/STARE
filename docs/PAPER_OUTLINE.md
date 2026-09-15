# Paper Outline — Paragraph-Level Plan

**Working title:** *Adaptivity Without Reward: A Negative Result for Closed-Loop
Region Acquisition in Spatial Transcriptomics*

**Venue target:** arXiv (cs.LG / q-bio.GN) → workshop (e.g., ICML/NeurIPS
computational-biology or ML4H). Negative-result-with-theory format.

**Audience:** CS researchers working in computational biology.

**Length:** ≤ 8 pages *including* related work, figures, and references.

**Honesty contract (governs every paragraph):**
- We report that closed-loop adaptivity provided **no gain** over principled
  non-adaptive placement, under both imputers we tested. We do **not** claim the
  agentic method wins.
- Every quantitative claim traces to `results/sweep_prostate_knn.json` or
  `results/sweep_prostate_kriging.json`, or to a cited paper. No invented numbers.
- Caveats (weak greedy policy; weak kriging imputer at low budget; strong learned
  imputer untested) are stated in the main text, not buried.
- The "why it matters" is the redirection of effort + the forward-looking
  condition, not an inflated result.

**Citation key (only these; all in `docs/KNOWLEDGE_BASE.md`):**
`[Krause08]` Krause, Guestrin, Singh, JMLR 2008 (sensor placement / submodular MI /
adaptivity gap) · `[AFA]` AFA-RL, arXiv 2011.00825 · `[SF]` SpatialFinder,
Frontiers Bioinf. 2026 · `[PERSIST]` Nat. Commun. 2023 · `[ReconST]` NAR 2026 ·
`[STFlow]` ICML 2025 · `[Stem]` ICLR 2025 · `[GenAR]` · `[BLEEP]` NeurIPS 2023 ·
`[HEST]` HEST-Benchmark · `[UNI]` UNI/UNI2-h · `[GeoMx]` GeoMx DSP ROI practice.
`[OURS-knn]` / `[OURS-kri]` = our two result files.

---

## Abstract  (~1 paragraph, ~180 words)
- **Purpose:** State the problem (budget-constrained region acquisition for ST
  reconstruction), what we built (a reproducible closed-loop acquisition simulator
  on HEST), the headline finding (closed-loop adaptivity gives no gain over
  non-adaptive MI-greedy placement at any budget, under a stationary *and* a
  learned imputer), the mechanism (sensor-placement theory), and the actionable
  condition under which adaptivity would help.
- **Cite:** `[OURS-knn]`, `[OURS-kri]`, `[Krause08]`.

---

## 1. Introduction  (~1 page, 4 paragraphs)

**P1 — Why the problem exists.**
- *Purpose:* Establish that ST profiling is budget-limited and that region choice
  is a real operational decision, not a manufactured one. Region-based platforms
  (GeoMx DSP) require a human to pick which ROIs to profile; the goal is to
  reconstruct whole-transcriptome / top-HVG expression over the *unprofiled* tissue
  from H&E morphology.
- *Cite:* `[GeoMx]` (manual budget-limited ROI selection), `[HEST]`, `[BLEEP]`
  (image→expression is feasible but bounded).

**P2 — The intuitive (agentic) hypothesis.**
- *Purpose:* Frame the natural idea we set out to test: acquire regions
  *sequentially*, conditioning each choice on the expression measured so far
  ("active feature acquisition" / closed-loop). State the prior belief that this
  should beat one-shot heuristics.
- *Cite:* `[AFA]` (adaptive acquisition RL), `[SF]` (the deployed one-shot /
  diversity ROI-ranking incumbent this would improve on).

**P3 — What we found, stated up front.**
- *Purpose:* No suspense: we built the simulator and the closed-loop policy, and
  across a 0.5–20% budget sweep the closed loop does **not** beat a principled
  non-adaptive placement rule (Krause MI-greedy), under either a stationary kNN
  imputer or a learned non-stationary kriging imputer.
- *Cite:* `[OURS-knn]`, `[OURS-kri]`.

**P4 — Contributions and why this matters anyway.**
- *Purpose:* List four contributions and their value: (i) a clean formalization of
  closed-loop ST acquisition; (ii) an open, reproducible HEST-based benchmark +
  simulator with strong non-adaptive baselines; (iii) a negative result paired with
  a *theoretical explanation* rather than an anecdote; (iv) a precise, testable
  condition — the imputer's posterior must be strongly data-dependent — that tells
  the field when to build an agent and when not to. Impact = saving the community
  from complex machinery that provably cannot win in the common regime.
- *Cite:* `[Krause08]`.

---

## 2. Related Work  (~0.75 page, 3 paragraphs)

**P1 — Image→expression prediction and its ceiling.**
- *Purpose:* Situate the imputer landscape and note the well-documented low PCC
  ceiling — morphology predicts a marker subset well, most genes weakly.
- *Cite:* `[BLEEP]`, `[STFlow]`, `[Stem]`, `[GenAR]`, `[HEST]` (encoder-scaling law).

**P2 — Gene / region selection is non-adaptive today.**
- *Purpose:* Show the incumbents choose *what to measure* without conditioning on
  measured values: static gene panels and one-shot ROI ranking. This is the gap our
  closed loop targeted.
- *Cite:* `[PERSIST]`, `[ReconST]` (static panels), `[SF]` (one-shot ROI).

**P3 — Active acquisition and sensor-placement theory.**
- *Purpose:* Name the two poles — adaptive AFA on one side, and the submodular
  sensor-placement theory on the other that *predicts when adaptivity is useless* —
  and position our study at their intersection.
- *Cite:* `[AFA]`, `[Krause08]`.

---

## 3. Problem Formulation  (~0.5 page, 2 paragraphs)

**P1 — Formal setup.**
- *Purpose:* Define a slide as regions with frozen image embeddings and latent
  expression; a budget B; a policy that reveals regions (sequentially for
  closed-loop); a frozen shared imputer that reconstructs unprofiled regions;
  objective = maximize PCC over top-50 HVG on unprofiled regions.
- *Cite:* `[HEST]` (metric protocol); reference `config.py` for pinned values.

**P2 — The quantity of interest: the adaptivity gap.**
- *Purpose:* Define adaptivity gap = (best closed-loop) − (best non-adaptive
  placement). State plainly that this single number is what the paper measures, and
  that a fair test requires a *strong* non-adaptive opponent, not just random.
- *Cite:* `[Krause08]` (open-loop vs adaptive placement).

---

## 4. Method / Experimental Substrate  (~0.75 page, 3 paragraphs)

**P1 — Data and encoder.**
- *Purpose:* HEST prostate slide (TENX174), 112 µm regions chosen so one region = one
  UNI2-h 224px patch, top-50 HVG, log1p; frozen UNI2-h embeddings. Emphasize the
  substrate is shared identically across all policies for fairness.
- *Cite:* `[HEST]`, `[UNI]`.

**P2 — Imputers = the environment.**
- *Purpose:* Describe the two frozen imputers and *why two*: a stationary kNN
  (theory predicts zero gap) and a learned regression-kriging whose predictor is
  refit on the profiled regions so its posterior *depends on measured values* — the
  precondition for any adaptivity gain. This is the fair steel-manning of the agent.
- *Cite:* geostatistics/kriging (standard ref); our `imputer.py`.

**P3 — Policies.**
- *Purpose:* Enumerate baselines and the agent: random / grid / k-means /
  one-shot(SpatialFinder-style) / Krause variance-greedy / Krause MI-greedy
  (the principled non-adaptive opponent) / our closed-loop error-reduction greedy.
- *Cite:* `[Krause08]`, `[SF]`.

---

## 5. Experiments  (~1.25 pages, 3 paragraphs + 1 table + 1 figure)

**P1 — Protocol.**
- *Purpose:* Budget sweep {0.5,1,2,5,10,20}%, random averaged over 5 seeds; report
  PCC-top50 and PCC-mean at every budget; two verdicts per budget — beats the random
  floor, and closed-loop-helps (agent vs its no-measured ablation). Present as a
  Pareto curve, the field-standard.
- *Cite:* `[PERSIST]` (panel-size curve convention); our `03_budget_sweep.py`.
- **Figure 1:** PCC-top50 vs budget, all policies, both imputers (2 panels).
- **Table 1:** PCC-top50 at 20% budget per policy per imputer + the two verdicts.

**P2 — Main result, reported honestly.**
- *Purpose:* Under kNN: closed loop shows no gain at any budget; k-means / MI-greedy
  dominate. Under kriging: still no gain; MI-greedy is the best method
  (0.856 @ 20% vs ours 0.676 vs k-means 0.811). `closed_loop_helps = False` at every
  budget under both imputers.
- *Cite:* `[OURS-knn]`, `[OURS-kri]` (all numbers verbatim from the JSONs — pull
  exact values when writing; do not paraphrase from memory).

**P3 — Ablations and caveats, in the open.**
- *Purpose:* Bound the claim so a reviewer cannot accuse us of overreach: (a) our
  error-reduction greedy over-clusters and is a *weak* agent; (b) kriging is a weak
  imputer at low budget (image barely predicts expression there); (c) a strong
  generative/GNN imputer (STFlow-class) is **not** tested and could in principle
  reopen the gap. State the claim precisely: *adaptivity provided no gain under a
  stationary kNN and a learned regression-kriging imputer.*
- *Cite:* `[OURS-*]`, `[STFlow]`.

---

## 6. Why Adaptivity Provides No Gain  (~0.75 page, 2 paragraphs)

**P1 — The theorem that explains it.**
- *Purpose:* For a fixed-kernel GP, posterior variance is data-*independent*, so the
  optimal set of locations can be chosen offline and observing values cannot improve
  the plan → adaptivity gap = 0; MI-greedy is (1−1/e)-optimal. Our stationary-kNN
  result is exactly this theory instantiated.
- *Cite:* `[Krause08]`.

**P2 — Why it persisted under the learned imputer.**
- *Purpose:* Explain empirically that although kriging is non-stationary in
  principle, at these budgets its data-dependence was too weak to concentrate error
  in a way the agent could exploit — so the gap stayed ~0. Connects the mechanism to
  the observation rather than asserting it.
- *Cite:* `[OURS-kri]`.

---

## 7. Discussion — When *Would* Adaptivity Help  (~0.5 page, 2 paragraphs)

**P1 — The condition we hand the field.**
- *Purpose:* Adaptivity pays only when the imputer's predictive posterior is
  *strongly* data-dependent — where a measurement changes *where* the remaining
  error lives. Candidates: generative/GNN imputers (STFlow), heteroscedastic models.
  Frame this as a concrete, testable design rule, the paper's forward value.
- *Cite:* `[STFlow]`, `[AFA]`.

**P2 — Practical takeaway for ST practitioners.**
- *Purpose:* Actionable guidance today: spend budget on principled one-shot
  placement (MI-greedy) plus a stronger imputer, not on closed-loop agents. Honest
  impact statement.
- *Cite:* `[OURS-kri]`, `[SF]`.

---

## 8. Conclusion  (~0.25 page, 1 paragraph)
- *Purpose:* Restate the arc: we built the agent, measured the adaptivity gap
  against a strong non-adaptive opponent, found it ~0, explained it with
  sensor-placement theory, and specified the condition to revisit. Value = redirected
  effort + a reusable benchmark. No new citations.

---

## References  (~0.75 page)
- ~12–15 entries, the citation key above. Keep tight; negative-result paper.

---

### Flow check (does it connect end to end?)
Problem is real (§1.P1) → the obvious fix is an agent (§1.P2) → we built and tested
it (§4–5) → it doesn't help (§5.P2) → here is exactly why (§6) → here is when it
*would* (§7) → so redirect effort (§7.P2, §8). Each section answers the question the
previous one raises; no idea is repeated.

### Page budget (single-column arXiv est.)
Abstract 0.15 · §1 1.0 · §2 0.75 · §3 0.5 · §4 0.75 · §5 1.25 · §6 0.75 · §7 0.5 ·
§8 0.25 · refs 0.75 · figure/table overhead already counted → ≈ 6.6–7.5 pp. Fits 8.
