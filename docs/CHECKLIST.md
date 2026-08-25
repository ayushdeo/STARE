# Progress Checklist — Direction 1 / Option B (Closed-Loop Region/Spot Acquisition)

**Master reference:** `SOURCE_OF_TRUTH.md`. Execution order & competitor-reproduction spec: `REPRODUCTION_PLAN.md` (two-tier reproduction, staged feasibility→gate→scale). Every block maps to a section there (`[§x]`). Update `SOURCE_OF_TRUTH.md` first if scope changes, then reflect it here. Workstreams WS-A…WS-E: A/B/C parallel; D needs A+B+C; E needs D.

**Committed problem (locked):** a learned, **closed-loop**, budget-constrained ROI/spot **acquisition** policy that optimizes **whole-transcriptome imputation error**, conditioned on the H&E image AND on already-measured regions. Differentiators vs SpatialFinder (2026): closed-loop + reconstruction-objective + learned-policy. `[SOURCE_OF_TRUTH §0, §4]`

Legend: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` gate

---

## Phase 0 — Decision, framing, reality check (DONE)
- [x] Survey top venues; write Direction 1 vs 2 lit reviews; commit Direction 1 `[lit_review/]`
- [x] Deep-read AFA-RL, PERSIST, ReconST, HEST-1k, STFlow, TISSUE, Stem, GenAR `[KNOWLEDGE_BASE.md]`
- [x] Reality check: confirm problem is real (GeoMx ROI, self-driving labs); find SpatialFinder; reject within-slide adaptive panel `[lit_review/REALITY_CHECK.md]`
- [x] Refine to Option B (region/spot acquisition); record sharpened gap `[SOURCE_OF_TRUTH §0, §4]`

## Phase 1 — Remaining deep reads (finish before building WS-D)
- [ ] Full read PERSIST mechanism for the static-panel baseline mapping `[§5 WS-C]`
- [ ] Full read HEST-1k library API + region/patch handling `[§5 WS-A]`
- [ ] Full read one imputer to be the environment (STFlow or SpaIM or TISSUE) `[§5 WS-B]`
- [ ] Re-read SpatialFinder method to replicate its one-shot scoring as a baseline `[§5 WS-C]`
- [ ] Skim NOCTA (non-greedy) + self-driving-lab BO/GP acquisition for the policy design `[§5 WS-D]`

## WS-A — Benchmark & closed-loop region-acquisition simulator `[§5 WS-A]`
- [ ] Obtain HEST-1k (Visium whole-transcriptome slides as ground truth) + optionally Xenium/Visium-HD; pick 1–2 tissues to start
- [ ] Define the acquisition unit: ROI (~100–600µm, GeoMx-like) vs Visium-HD spot vs spot-cluster — decide and document `[§4]`
- [ ] Define budget model: total regions profileable B; cost per region (uniform first)
- [ ] Build closed-loop simulator: state = profiled set + measured expression; `reveal(region)→expression` API; enforce B
- [ ] Scoring: whole-transcriptome imputation error over UNPROFILED regions (PCC/MSE), per-round + final
- [ ] Slide-level train/val/test split (no leakage); adopt HEST patient-stratified protocol where possible
- [ ] Secondary metric: Moran's I (spatial-pattern preservation)

## WS-B — Imputer (frozen environment / reward model) `[§5 WS-B]`
- [ ] Choose imputer that reconstructs full-slide transcriptome from profiled regions + image (STFlow / Stem / SpaIM / TISSUE-wrapped)
- [ ] Reproduce its published behavior on one dataset (sanity check)
- [ ] Wrap so it imputes unprofiled regions from an arbitrary profiled subset
- [ ] Expose per-region/per-gene predictive UNCERTAINTY (Stem sample var / GenAR heteroscedastic head / TISSUE intervals) = info-gain signal for WS-D
- [ ] Freeze + version the imputer (fair comparison across all policies)

## WS-C — Baselines / competitor reproduction (the "their runs" we must beat) `[§5 WS-C, §6; REPRODUCTION_PLAN §2,§4]`
Two-tier per `REPRODUCTION_PLAN §1`: Tier 1 = native sanity repro (prove correct impl); Tier 2 = harmonized port into our simulator (the actual comparison).
- [ ] SpatialFinder — Tier 1 sanity (reproduce ρ on ≥1 native Visium HD tissue; gist may need reimplement) `[repro §2.1]`
- [ ] SpatialFinder — Tier 2 port (one-shot region choice scored by our imputation metric) — PRIMARY competitor
- [ ] PERSIST — Tier 1 sanity (explained-variance curve) → Tier 2 fixed-panel port `[repro §2.2]`
- [ ] ReconST — Tier 1 sanity → Tier 2 fixed-panel port; adopt its Moran's I metric `[repro §2.3]`
- [ ] Naive region sampling: random, grid/uniform, greedy-variance, k-means-on-image-embeddings
- [ ] Baseline imputation-error-vs-budget Pareto (all competitors, 1 tissue) BEFORE building our policy → Table B seed
- [ ] Log every non-1:1 deviation in `experiments/deviations.md` `[repro §6]`

## WS-D — Our closed-loop policy (the contribution) `[§5 WS-D]`
- [ ] Greedy expected-information-gain policy (no RL): pick region maximizing expected reduction in imputation uncertainty
- [ ] Add H&E image conditioning via frozen UNI2-h
- [ ] Add spatial structure (k-NN/graph over regions; consider STFlow E(2)-invariant attention)
- [ ] Add closed-loop conditioning on already-measured expression X_{M_t} (the defining feature)
- [ ] **GATE — early-signal test:** does greedy CLOSED-LOOP beat ONE-SHOT (SpatialFinder-style) at matched budget on ONE tissue? `[§6, §8 critical risk]`
  - [ ] If NO → stop, we've reinvented SpatialFinder; re-evaluate with TA
  - [ ] If YES → proceed
- [ ] Amortized / RL policy (AFA-RL seq-VAE-over-partial-state template) if greedy leaves headroom
- [ ] Non-greedy lookahead (NOCTA-style) as later escalation

## WS-E — Evaluation, ablations, writeup `[§5 WS-E, §6]`
- [ ] Full imputation-error-vs-budget Pareto: ours vs all WS-C baselines, multiple tissues
- [ ] **Decisive ablation: closed-loop (condition on X_{M_t}) vs one-shot (image-only)** at equal budget — proves contribution over SpatialFinder
- [ ] Ablation: remove image-conditioning → measure drop
- [ ] Ablation: remove spatial structure → measure drop
- [ ] Statistical significance across slides/tissues; Moran's I secondary
- [ ] Draft: problem+motivation from `SOURCE_OF_TRUTH §1–2`+`REALITY_CHECK`; related work from `§3`+`KNOWLEDGE_BASE` (esp. SpatialFinder contrast)
- [ ] Every writeup claim traceable to a citation in `SOURCE_OF_TRUTH §9`

## Standing gates (do not skip)
- [!] WS-D closed-loop-vs-one-shot GATE must pass before WS-E full build `[§6, §8]`
- [!] TA/advisor sign-off on the Option-B problem statement (`SOURCE_OF_TRUTH §4`) before heavy implementation
