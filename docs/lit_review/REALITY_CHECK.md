# Reality Check — Does the acquisition problem actually exist? (2026-07-05)

Purpose: verify, before building anything, whether "budget-constrained, image-guided ROI/measurement acquisition for spatial transcriptomics" is a **real** decision practitioners make (not a forced problem), and whether agentic active-acquisition systems exist here and in adjacent fields. Companion to `../SOURCE_OF_TRUTH.md`. Verdict at bottom.

---

## Q1 — Is ROI selection in targeted ST a real, manual, budget-limited decision? → YES

**GeoMx DSP (NanoString/Bruker) — region-based platform:**
- User images the tissue (IF/H&E), then **manually chooses which ROIs to profile** based on staining/morphology; "a skilled pathologist is an essential part of the research team." ROIs are any shape, 10–600 µm, can be segmented into biological compartments (AOIs). [Bruker knowledge base; NanoString "getting started" guide; "What's in an ROI?"]
- **Budget is real and hard:** minimum ~6 ROIs per type recommended for statistical power; each ROI/segment consumes assay resources. Region-based profiling exists precisely because you *cannot* profile everything. [NanoString best-practices]
- Confirmed the decision is currently **manual / pathologist-heuristic**, image-guided. AI support is only "emerging" (Visiopharm digital-pathology software can *assist* ROI drawing; ROICellTrack integrates imaging *beyond* ROI selection). [ScienceDirect "Computational solutions for spatial transcriptomics"; ROICellTrack, Bioinformatics 2025]

**Imaging platforms (Xenium/MERFISH/CosMx) — panel-based:**
- Fixed probe panel per run, images whole slide in one pass. **Within-slide spatially-adaptive panels are NOT physically realizable** on these platforms → the "adaptive panel per region" sub-variant (Option A) is a forced problem. Confirmed and rejected.

**Conclusion Q1:** the *region-acquisition* decision (which regions/spots to profile under a budget, guided by the image) is a genuine, currently-manual, cost-limited workflow decision — on GeoMx-style and cost-per-region Visium HD workflows. The *within-slide adaptive gene panel* is not (physics). So the real problem is **region/spot acquisition**, not gene-panel adaptation.

---

## Q2 — Has anyone already built this for ST? → PARTIALLY (one direct 2026 entrant; gap survives, sharpened)

### ★ SpatialFinder (Frontiers in Bioinformatics, 2026) — the direct entrant. READ IN FULL.
- **What it is:** prioritizes high-value ROIs (~500µm) to **sequence** from H&E alone, to cut cost ("$15,000/slide → multiple subjects per slide"). This is exactly the Option-B problem framing → **proves the community treats this as a real, publishable problem.**
- **Method:** VLM (MUSK) patch embeddings + active-learning cell classifier (StarDist + PLIP + pathologist labels + XGBoost) + **hand-designed hybrid score** (diversity: UMAP distance + Shannon entropy; targeted: text-image similarity + target-cell abundance). Human-in-the-loop labeling (~27 min/tissue).
- **What it does NOT do (= our surviving gap, now precise):**
  1. **One-shot ranking, not sequential/closed-loop.** All ~10k ROIs scored simultaneously; greedy top-K. **No feedback from already-measured expression** → not adaptive acquisition, not an MDP.
  2. **Objective = diversity/relevance heuristic, NOT whole-transcriptome imputation/reconstruction error.** It never asks "which regions best let me recover the unmeasured transcriptome."
  3. **Budget is implicit**, no formal constrained-optimization objective; user manually picks K.
  4. **Not a learned policy** — hand-tuned scores (α grid-searched per tissue) + supervised XGBoost + human labels. No RL, no amortized policy.
  5. Per-tissue, no cross-tissue transfer; single annotator.
- Data: 4 Visium HD tissues; eval = Spearman ρ / Overlap@K vs cell-type-entropy ground truth (ρ 0.37–0.80).

### Other adjacent ST entrants
- **SpatialAgent** (Genentech, bioRxiv 2025) — autonomous LLM agent for experimental design/analysis at the *workflow* level; not a trainable acquisition policy with a reconstruction objective. [prior KB]
- General **region-based active learning** (arXiv 2507.17367) and **active timepoint selection** (arXiv 2605.30625) — active-learning machinery exists but not for ST region acquisition with an imputation objective.
- **PERSIST / ReconST / scGIST / gpsFISH** — static gene-panel design, not region acquisition. [KB]

**Conclusion Q2:** the exact cell — **sequential, closed-loop, budget-constrained ROI/spot acquisition with a LEARNED policy optimizing whole-transcriptome imputation error, conditioned on the image AND on already-measured regions** — is still unoccupied. SpatialFinder is the nearest, and its own limitations (one-shot, heuristic, diversity-objective, no closed loop, no learned policy) are exactly our differentiators. But the space is now demonstrably active (2026), so this is a *nascent competitive* area, not a wide-open one.

---

## Q3 — Do agentic active-acquisition systems work in adjacent fields? → YES, it's an established paradigm

This validates that the *approach* is real and not forced — it's a proven paradigm we'd be transferring, not inventing.
- **Self-driving labs / autonomous experimentation:** closed-loop systems that perform experiment → measure → update surrogate model → **choose next experiment by expected information gain**, most commonly **Bayesian optimization / GP surrogates**. Mature across chemistry, materials, optoelectronics. [Royal Society Open Science 2025 review; PMC9454899; Acceleration Consortium]
- **Adaptive microscopy — FAST toolkit:** NN + route optimization actively identifies and measures a **sparse but representative subset** instead of the full raster scan — i.e., budget-constrained adaptive spatial acquisition, exactly analogous to ROI acquisition. [PMC10485018, "AI-driven autonomous high-resolution scanning microscopy"]
- **Bayesian experimental design / active feature acquisition:** the formal theory (maximize expected information gain per measurement under cost) is well-established. [AFA-RL, KB #1]

**Conclusion Q3:** sequential, information-gain-driven, budget-constrained acquisition is a real, working paradigm in adjacent experimental sciences. Bringing it to ST region acquisition — with an imputation objective and image conditioning — is a legitimate, well-precedented transfer, not a forced construct.

---

## OVERALL VERDICT

**The problem exists.** Three independent confirmations: (1) GeoMx ROI selection is real, manual, budget-limited; (2) SpatialFinder (2026) publishes the exact cost-reduction framing, proving community demand; (3) self-driving labs prove sequential active acquisition works in adjacent fields. This is **not** a forced problem.

**A differentiated solution can be devised.** The surviving, sharpened gap: *a learned, closed-loop, budget-constrained ROI/spot acquisition policy that optimizes whole-transcriptome imputation error, conditioned on the H&E image and on already-measured regions.* SpatialFinder (one-shot, heuristic, diversity-objective) and self-driving-lab methods (not ST, not image-conditioned) bracket it without occupying it.

**Honest caveats (do not ignore):**
- The area is **now active** (SpatialFinder 2026, SpatialAgent 2025). The wide-open window has narrowed to a *nascent-competitive* one. Speed and a crisp differentiator matter.
- Novelty rests entirely on: **sequential/closed-loop + reconstruction-objective + learned-policy**. If we drift back toward one-shot heuristic scoring, we reinvent SpatialFinder. The go/no-go gate (beat one-shot selection AND a static panel at equal budget, via the closed loop) is unchanged and essential.
- Validation for the paper must show the **closed loop** (conditioning on measured regions) beats **one-shot** selection — that is the single experiment that proves our contribution over SpatialFinder.

## Sources
- GeoMx ROI workflow: https://brukerspatialbiology.com/support/knowledgebase/geomx-assays-applications/ · https://nanostring.com/blog/getting-started-with-your-first-geomx-dsp-experiment/ · https://brukerspatialbiology.com/blog/geomx-dsp-whats-in-an-roi/
- ROICellTrack (imaging beyond ROI selection): https://academic.oup.com/bioinformatics/article/41/5/btaf152/8108826
- SpatialFinder (2026): https://www.frontiersin.org/journals/bioinformatics/articles/10.3389/fbinf.2026.1746714/full
- Region-based active learning: https://arxiv.org/html/2507.17367 · Active timepoint selection: https://arxiv.org/html/2605.30625v1
- Self-driving labs review: https://royalsocietypublishing.org/rsos/article/12/7/250646/235354 · Autonomous chemistry: https://pmc.ncbi.nlm.nih.gov/articles/PMC9454899/
- Autonomous scanning microscopy (FAST): https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10485018/
