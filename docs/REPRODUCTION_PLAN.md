# Reproduction & Experiment Plan — Option B

**Reference:** `SOURCE_OF_TRUTH.md` (objective §4, workstreams §5). This document specifies HOW we set up competitors, prove feasibility, then scale — the concrete execution order you asked for. Maps to `CHECKLIST.md` WS-A/B/C/D/E.

Execution philosophy (your instruction): **(1) set up competitors' runs and prove we can reproduce them → (2) build a small feasibility ablation that shows our method can beat them on ONE tissue → (3) only then scale.**

---

## 1. The comparability problem (read first — it dictates everything)

Our competitors were built for **different tasks on different datasets**:
- **SpatialFinder** — one-shot ROI *ranking* on 4 Visium HD tissues; metric = rank correlation vs cell-type-entropy ground truth.
- **PERSIST / ReconST** — static *gene-panel* selection from scRNA-seq; metric = reconstruction of genome-wide profile / clustering quality.
- **STFlow / Stem / GenAR** — *prediction* of expression from image (no acquisition); metric = PCC on top-HVG.

None of these is our task (closed-loop region acquisition minimizing whole-transcriptome imputation error under budget). Therefore **we cannot compare their published numbers to ours directly.** We use a **two-tier reproduction**:

- **Tier 1 — Native sanity reproduction.** Run each competitor's official code on ITS OWN dataset and confirm we reproduce its published headline number (±reasonable tolerance). Purpose: *prove we understand and implemented the method correctly.* This is a validation gate, NOT a comparison; its numbers never go in our main results table.
- **Tier 2 — Harmonized reproduction.** Port each competitor's *strategy* into OUR common simulator on OUR common dataset, under identical budget, identical frozen imputer, identical metric. Purpose: *apples-to-apples comparison.* These numbers ARE our results table.

Any deviation from a competitor's native setup is logged with justification (§6). The rule: **Tier 1 must be as close to 1:1 as their code allows; Tier 2 is deliberately harmonized and every harmonization is justified.**

---

## 2. Per-competitor reproduction spec

### 2.1 SpatialFinder — PRIMARY competitor (the one we must beat)
- Code: GitHub gist https://gist.github.com/jonrxu/63e9319016c257a54c823fbaab26bb27 (informal — expect gaps; may need reimplementation from paper). bioRxiv 2025.08.16.670684 / Frontiers 2026.
- Native data: 4 Visium HD tissues (lung, colon, prostate, kidney). Native metric: Spearman ρ / Overlap@K vs cell-type-entropy ranking (claims up to 0.89 ρ).
- Method to replicate: MUSK VLM embeddings + StarDist+PLIP+XGBoost cell classifier + hybrid diversity/targeted score, one-shot greedy top-K.
- **Tier 1 target:** reproduce their ρ on ≥1 of their 4 tissues (±0.05). If the gist is incomplete, reimplement the scoring from the paper and note it.
- **Tier 2 port:** run their one-shot scoring to pick K regions in OUR simulator, then score by OUR metric (whole-transcriptome imputation error via OUR frozen imputer). This is the head-to-head: their region choice vs our closed-loop region choice at equal K.
- Deviation risk: their objective is diversity, ours is imputation — that's the point, not a bug. Justify: we evaluate all methods on the same downstream (imputation), the objective every acquisition method ultimately serves.

### 2.2 PERSIST — static-panel baseline
- Code: https://github.com/iancovert/persist (mature PyTorch). Native data: SSv4/10X scRNA + MERFISH MOp. Native metric: explained variance / expressed-gene accuracy vs panel size.
- **Tier 1 target:** reproduce PERSIST explained-variance-vs-panel-size curve on its data.
- **Tier 2 port:** PERSIST selects a fixed gene panel from reference; in our region-acquisition simulator this = the "measure these fixed genes across all regions" strategy. Map its gene-budget to our region-budget on equal-cost terms, score imputation. Represents the orthogonal "fixed genes everywhere" incumbent.

### 2.3 ReconST — SOTA static-panel baseline
- Code: https://github.com/haoranlustat/ReconST (lightweight Python package). Native data: mouse-brain MERFISH, fetal-lung Xenium. Native metric: reconstruction, Moran's I, ARI.
- **Tier 1 target:** reproduce ReconST reconstruction advantage over PERSIST on its data.
- **Tier 2 port:** same as PERSIST — fixed panel mapped into our simulator; adopt its **Moran's I** as our secondary spatial-preservation metric.

### 2.4 Imputer (STFlow / Stem / SpaIM) — NOT a competitor, the ENVIRONMENT (WS-B)
- Code: STFlow https://github.com/Graph-and-Geometric-Learning/STFlow · Stem https://github.com/SichenZhu/Stem · GenAR https://github.com/oyjr/genar.
- **Tier 1 target:** reproduce the imputer's published PCC on HEST-1k/STImage to confirm correct setup (this is the number our whole simulator's reward depends on).
- Role: frozen reward/uncertainty model shared by ALL methods in Tier 2 (fair comparison). Pick ONE for feasibility (recommend the simplest to run reliably; STFlow if its k-NN attention runs cleanly, else SpaIM).

### 2.5 Naive baselines — trivial, implement directly
- Random / uniform-grid / greedy-variance / k-means-on-image-embedding region selection. No reproduction needed; implement in the simulator.

---

## 3. Common evaluation substrate (the shared dataset)

**Choice: Visium HD tissue(s) that overlap SpatialFinder AND HEST/GenAR.** Rationale:
- SpatialFinder is native to Visium HD → we can run it 1:1 (Tier 1) there.
- Visium HD = whole transcriptome + fine spatial resolution → gives BOTH ground-truth for imputation AND region definition (bin/pool HD spots into ROIs at GeoMx-like ~100–600µm).
- **Prostate and kidney appear in BOTH SpatialFinder (4 tissues) and GenAR/HEST** → convenient overlap for cross-checking imputers and competitors.
- **Feasibility: start with ONE tissue** (prostate or kidney). **Scale: all 4 SpatialFinder tissues + HEST Visium slides.**
- HEST-Library handles Visium HD loading/patching → reduces plumbing (WS-A).

---

## 4. Staged execution order (feasibility → gate → scale)

### Stage 0 — Environment + data (WS-A/B setup)
1. Set up HEST-1k loading for 1 Visium HD tissue; define region grid (~100–600µm bins).
2. Stand up the closed-loop simulator skeleton (`reveal(region)→expr`, budget B, impute-and-score).
3. Get ONE imputer running + Tier-1 sanity reproduction of its published PCC.

### Stage 1 — Competitor setup (Tier 1 sanity, then Tier 2 ports)
4. SpatialFinder: Tier 1 sanity on 1 native tissue → Tier 2 port into simulator.
5. PERSIST + ReconST: Tier 1 sanity → Tier 2 fixed-panel ports.
6. Naive baselines implemented in simulator.
7. Produce the **baseline imputation-error-vs-budget Pareto on 1 tissue** (all competitors, no us yet).

### Stage 2 — FEASIBILITY ABLATION (the go/no-go, small scale) `[CHECKLIST WS-D GATE]`
8. Implement greedy expected-information-gain closed-loop policy (image + spatial + condition-on-measured).
9. **Decisive feasibility test on 1 tissue at matched budget:**
   - (a) closed-loop (ours) vs one-shot (SpatialFinder port) — the primary claim.
   - (b) closed-loop vs static panel (PERSIST/ReconST port).
   - (c) closed-loop vs naive sampling.
10. Feasibility ablations (small): closed-loop ON vs OFF (condition on measured expr vs image-only); image-conditioning ON vs OFF; spatial-structure ON vs OFF.
11. **GATE:** if closed-loop does NOT beat one-shot at equal budget → STOP, reassess with TA (we'd have reinvented SpatialFinder). If YES → scale.

### Stage 3 — SCALE (only after gate passes)
12. Extend to all 4 Visium HD tissues + HEST Visium slides.
13. Add amortized/RL policy if greedy left headroom; non-greedy lookahead (NOCTA) later.
14. Full Pareto + all ablations + significance across tissues (WS-E).
15. Writeup.

---

## 5. Deliverable result tables (what we're building toward)
- **Table A (Tier 1, appendix):** each competitor, native dataset, published # vs our reproduced # → credibility that we implemented them correctly.
- **Table B (Tier 2, main result):** every method (ours + all competitors + naive) on the common substrate, imputation error at fixed budgets B ∈ {…}, identical frozen imputer + metric.
- **Figure 1 (main):** imputation-error-vs-budget Pareto curves, all methods, ours dominating.
- **Table C (ablation):** closed-loop vs one-shot; ±image; ±spatial — isolating each contribution.

---

## 6. Justification framework for non-1:1 reproduction
For every deviation from a competitor's native setup, log in an `experiments/deviations.md`: (1) what we changed, (2) why (technical necessity / harmonization for fair comparison), (3) why it does not advantage us unfairly. Standard justified deviations we already anticipate:
- **Different dataset than their native one** (Tier 2) — justified: all methods evaluated on the identical common substrate; native numbers preserved in Tier 1.
- **Different objective/metric than their native one** — justified: imputation error is the common downstream all acquisition strategies serve; we also report their native metric where meaningful.
- **Reimplementation when code is incomplete** (esp. SpatialFinder gist) — justified: reimplement from the paper, validate against their published number in Tier 1 before use.
- **Shared frozen imputer** — justified: removes imputer quality as a confound; every method gets the same reward model.
