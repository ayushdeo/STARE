# STARE — Spatial Transcriptomics Acquisition via Reconstruction Evaluation

> **Closed-loop, budget-constrained ROI/spot acquisition that minimizes whole-transcriptome imputation error.**
> Given a fixed measurement budget on a tissue slide, learn a policy that decides *which regions to profile next* — conditioned on the H&E image **and on the expression measured so far** — so the unprofiled transcriptome is reconstructed as accurately as possible.

Target venues: NeurIPS / ICLR / ICML / CVPR. This repo is the working home for the project. **The provisional name STARE is a placeholder; rename freely.**

---

## 📍 READ THIS FIRST (orientation for whoever/whatever picks this up)

This README is written so a fresh agent (e.g. Claude Code on the 4070 laptop) can continue with zero prior context. Read these in order:

1. **`docs/SOURCE_OF_TRUTH.md`** — the master spec. Problem statement, formal objective (§4), the five workstreams WS-A…WS-E (§5), the go/no-go claim (§6), risks (§8), full citations (§9). **Everything defers to this file.**
2. **`docs/REPRODUCTION_PLAN.md`** — HOW we run: the two-tier competitor reproduction and the staged feasibility→gate→scale order. **This is the execution playbook.**
3. **`docs/CHECKLIST.md`** — the executable checklist; tick items as you go. Maps 1:1 to the source of truth.
4. **`docs/KNOWLEDGE_BASE.md`** — in-depth notes on all 23 papers (method, math, datasets, how each transfers to us). Query this before implementing anything.
5. **`docs/lit_review/REALITY_CHECK.md`** — why the problem is real (GeoMx ROI selection, self-driving labs) and how we differ from the nearest competitor, **SpatialFinder (2026)**.
6. **`docs/PROGRESS.md`** — reviewer-facing regroup: current findings, the World-2 diagnosis, next iterations, and the caveats we must own. Every claim tagged [LIT]/[OURS]/[HYP]. **Read for "where are we actually."**

**The one-sentence thesis to protect:** we beat one-shot region selection (SpatialFinder) by *closing the loop* — conditioning each next measurement on what's already been measured — and by optimizing *reconstruction error* rather than a diversity heuristic. If a change makes the method one-shot or heuristic, it reinvents SpatialFinder. Don't.

---

## 🎯 Current status: Stage 2 — closed-loop policy under diagnostic iteration (prostate). See `docs/PROGRESS.md`.

Real prostate data built (2249 regions, UNI2-h); baselines + our policy + ablations run across a budget
sweep. First finding = **World 2**: with an image-similarity kNN imputer, acquisition degenerates to
image coverage (one-shot k-means wins; closed-loop == no-loop). Diagnosed to the *imputer*, not the
idea (AFA-RL "NonSeq-ZI" failure mode). **Iterating** (joint image+spatial imputer + error-reduction
acquisition; next = a real joint imputer SpaIM/STFlow if needed). Not a pass/fail — a diagnostic loop.

Stages (from `docs/REPRODUCTION_PLAN.md §4`):
- **Stage 0** — env + data + simulator + imputer running. ✅ done
- **Stage 1** — baselines ported into the simulator (SpatialFinder one-shot, naive coverage). ✅ done
- **Stage 2** — closed-loop policy vs baselines across a budget sweep + ablations. ◀ **HERE (iterating)**
- **Stage 3** — scale to 4 tissues + HEST Visium, learned/RL policy, full Pareto, writeup.

---

## 🖥️ NEXT STEPS for Claude on the 4070 (do these in order)

> The 4070 (~8–12 GB VRAM) is enough for the feasibility slice: UNI embedding extraction (frozen, batched) and the kNN imputer are light. Heavy imputers (STFlow/Stem training) are deferred to Stage 3.

### Step 0 — Environment
```bash
git clone https://github.com/ayushdeo/STARE.git && cd STARE
python -m venv .venv && . .venv/Scripts/activate   # (Windows Git Bash)
# install the CUDA torch build matching your driver FIRST (see pytorch.org), e.g.:
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -e .
pip install -r requirements.txt
```
Also: get **UNI2-h** weights (HuggingFace `MahmoodLab/UNI2-h`, gated — request access) and system **openslide** binaries for WSI reading.

### Step 1 — Pin the exact HEST samples  (fills the one CONFIRM_ON_MACHINE gap)
```bash
python scripts/00_pick_samples.py
```
Implement the HEST-metadata filter in that script (Visium HD + prostate/kidney), choose 1–2 samples per tissue, and **paste the IDs into `src/stare/config.py → HEST_SAMPLE_IDS`**, then commit. If Visium HD is too sparse for a tissue, flip `config.PLATFORM='visium'` and log it in `experiments/deviations.md`.

### Step 2 — Build cached slides (WS-A)
Implement `stare.data.load_slide` (HEST → region grid → UNI embeddings + HVG + log1p), then:
```bash
python scripts/01_build_slides.py
```
Verify region counts / gene counts print sanely for the feasibility tissue (prostate).

### Step 3 — Finish the simulator + kNN imputer (WS-A / WS-B)
Fill the `NotImplementedError` bodies in:
- `src/stare/simulator.py` → `AcquisitionSimulator.score()`
- `src/stare/imputer.py` → `KNNImputer._neighbors/impute/uncertainty`
- `src/stare/metrics.py` → `morans_i` (optional for the gate)
Smoke-test with the `random` policy end-to-end (reset→reveal loop→score).

### Step 4 — Implement the two policies that decide the gate (WS-C / WS-D)
- `src/stare/policies/oneshot.py` → SpatialFinder-style scoring (do a Tier-1 sanity repro of their ρ first; log any simplification in `deviations.md`).
- `src/stare/policies/greedy_ig.py` → greedy expected-information-gain using `imputer.uncertainty(state)`, honoring the three ablation switches (`use_measured`, `use_image`, `use_spatial`).

### Step 5 — Run the feasibility go/no-go (Stage 2)
```bash
python scripts/02_run_feasibility.py
```
Reads `results/feasibility_prostate.json` and prints **PROCEED / STOP**. The gate: does `ours_closed_loop` beat `oneshot_spatialfinder` on `PRIMARY_METRIC` at the feasibility budget? Commit the JSON summary.

**If PROCEED** → move to Stage 3 (scale tissues, upgrade imputer to STFlow/SpaIM, add RL policy).
**If STOP** → the closed loop didn't help; do NOT keep tuning blindly — report back and reassess with the TA (see `docs/SOURCE_OF_TRUTH.md §8`).

---

## 🔧 Pinned parameters (Stage 0) — all in `src/stare/config.py`

| Parameter | Value | Why |
|---|---|---|
| Platform | Visium HD (fallback: Visium) | whole transcriptome + fine resolution = imputation GT + region grid; SpatialFinder's native platform |
| Feasibility tissue | **prostate** (validation: kidney) | in both SpatialFinder ∩ GenAR/HEST; large acquisition headroom |
| Region unit | **112 µm** (= 224 px @ 20×) | one region == one UNI patch (no resampling mismatch); mid-range GeoMx ROI |
| Budget grid | 1/2/5/10/20 % of regions (feasibility: 5 %) | Pareto x-axis; single point for the gate |
| Round size | 1 region/round | fully sequential closed loop |
| Gene target | top-50 HVG (primary) + full transcriptome | HEST-Benchmark protocol; comparable numbers |
| Image encoder | UNI2-h (frozen, 1536-d) | reuse existing pipeline; HEST-competitive |
| Imputer | kNN (feasibility) → STFlow/SpaIM (scale) | transparent + 4070-friendly first; SOTA later |
| Primary metric | mean per-gene PCC over top-50 HVG | field standard |
| Seed | 2021 | matches GenAR for cross-comparability |

Change a value → update `docs/SOURCE_OF_TRUTH.md` and this table in the same commit.

---

## 🗂️ Repo layout
```
STARE/
├── README.md                 <- you are here (operating manual + next steps)
├── docs/                     <- the brains: spec, plans, checklist, paper notes
│   ├── SOURCE_OF_TRUTH.md    <- master spec (start here)
│   ├── REPRODUCTION_PLAN.md  <- how to run competitors + staged order
│   ├── CHECKLIST.md          <- executable checklist
│   ├── KNOWLEDGE_BASE.md     <- deep notes on all 23 papers
│   ├── RESEARCH_PLAN.md      <- venue-mapped survey (history)
│   ├── papers_index.md       <- the 22 downloaded PDFs (kept locally, gitignored)
│   └── lit_review/           <- direction decisions + REALITY_CHECK
├── src/stare/                <- package
│   ├── config.py             <- ALL pinned parameters
│   ├── simulator.py          <- closed-loop acquisition environment
│   ├── data.py               <- HEST-1k -> Slide (region grid, UNI, HVG)
│   ├── imputer.py            <- frozen reward/uncertainty model (kNN feasibility)
│   ├── metrics.py            <- PCC / MSE / Moran's I over unprofiled regions
│   └── policies/             <- random, grid, variance, kmeans, oneshot(SpatialFinder), greedy_ig(ours)
├── scripts/                  <- 00_pick_samples · 01_build_slides · 02_run_feasibility
├── experiments/
│   └── deviations.md         <- log every non-1:1 reproduction choice
├── requirements.txt · pyproject.toml · .gitignore
```
Not committed (see `.gitignore`): `data/`, `cache/`, `papers/`, raw `results/*` arrays (JSON summaries ARE committed).

---

## ⚠️ Guardrails
- **Don't commit data or paper PDFs** (huge / copyrighted) — `.gitignore` handles it.
- **Keep the imputer frozen and shared** across all policies — it's the fair-comparison anchor.
- **Log every deviation** from a competitor's setup in `experiments/deviations.md`.
- **The gate is real**: if closed-loop doesn't beat one-shot at equal budget, stop and reassess — don't force it.
