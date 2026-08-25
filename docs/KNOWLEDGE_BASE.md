# KNOWLEDGE BASE — Deep notes on every paper

Purpose: an in-depth, queryable synthesis of all 22 papers in `papers/`, so any question about method, math, data, results, or relevance to our Direction-1 project can be answered. Companion to `SOURCE_OF_TRUTH.md`.

Reading status: `[x]` full-PDF read & noted · `[F]` deep full-text WebFetch · `[s]` summary-level (abstract + session searches). All 22 covered.

**Part I — deep reads (core to Direction 1):**
- [x] 1. AFA-RL (2011.00825) — method template
- [x] 2. PERSIST — primary static-panel baseline
- [F] 3. ReconST — SOTA static-panel baseline (its limitations = our gap)
- [x] 4. HEST-1k — benchmark + protocol
- [F] 5. STFlow — SOTA predictor / strongest related work
- [F] 6. TISSUE — imputer + uncertainty (info-gain signal)

**Part II — summary-level:**
- [F] 7. RL-gene-panel (2501.04718) · [s] 8. BLEEP · [x] 9. Stem (full PDF) · [F] 10. GenAR (full text) · [s] 11. MERGE · [s] 12. PEaRL
- [s] 13. scGIST · [s] 14. gpsFISH · [s] 15. SpaIM · [s] 16. PASTA · [s] 17. Stochastic-Encodings-AFA · [s] 18. NOCTA
- [s] 19. STImage-1K4M · [s] 20. CellAgent · [s] 21. SpatialAgent · [s] 22. LLM4Cell

See end for cross-paper synthesis.

---

## 1. AFA-RL — "Reinforcement Learning with Efficient Active Feature Acquisition" (Yin et al., 2020, under review; authors incl. Yingzhen Li, Cheng Zhang, Tschiatschek)

**Role for us:** THE methodological template for Direction 1. Formalizes "acquire features under cost, sequentially, conditioned on what's measured so far" as an MDP. Our task = this framework, with genes/spots as features and H&E image as extra conditioning.

### Problem — the AFA-POMDP (Definition 1)
A tuple `⟨S, A, T, O, R, C, γ⟩` where the action space is **joint**: `A = A^c × A^f` — control actions `A^c` AND feature-acquisition actions `A^f`. Key distinctions from a standard POMDP:
- Partial observability is **action-dependent**: at state `s`, the agent observes only the subset `x^p` of full features `x=(x^p, x^u)` that its acquisition action `a^f` selected. `x^u` = unobserved.
- `A^f = 2^[N_f]` — the **powerset** of features → acquisition action space is **exponential** in #features. (Directly relevant: our gene action space is huge; they flag hierarchical action representations as future work — an opening for us.)
- Reward `R: S×A^c→ℝ`; acquisition **cost** `C: S×A^f→ℝ≥0`, default uniform `C(a^f,s)=c·|a^f|` (each feature same cost, but generalizes to per-feature cost).

### Objective (Eq. 1)
Maximize expected discounted (task reward − acquisition cost):
`max_{π^f,π^c} E[ Σ_t γ^t ( R(s_t,a^c_t) − Σ_i c·I(a^{f(i)}_t) ) ]`.
Two coupled policies: **acquisition policy** `π^f(a^f|b_t)` and **task policy** `π^c(a^c|b_t)`, both conditioned on a **belief** `b_t` over state. (For us: there may be no "task/control" action — pure acquisition + imputation. Our "reward" = negative imputation error of the held-out transcriptome. So we use the `π^f` half.)

### The core technical contribution — Seq-PO-VAE (sequential VAE over partial observations)
The hard part is representing a partially-observed, evolving state. Their answer: a **sequential VAE** pre-trained to **impute unobserved features** from the history of partial observations + actions.
- Generative model: `p_θ(x^p_{1:T}, x^u_{1:T}, z_{1:T}) = Π_t p(x^p_t,x^u_t|z_t)p(z_t)`, standard Gaussian prior on `z` (deliberately static, not learned/autoregressive — they found static prior gives more normalized latents → **more stable RL policy training**; a practical lesson for us).
- Given `z_t`: observed & unobserved conditionally independent: `p(x^p_t,x^u_t|z_t)=p(x^p_t|z_t)p(x^u_t|z_t)`.
- Inference network `q_φ(z_{1:T}|x_{1:T},a_{<T}) = Π_t q_φ(z_t|x^p_{≤t}, a_{<t})` — autoregressive, RNN/LSTM backbone (forward filtering). Belief `b_t` = mean of `z_t`.
- **Training (Eq. 5):** maximize an ELBO that includes BOTH `log p_θ(x^p_t|z_t)` AND `log p_θ(x^u_t|z_t)` — the latter term turns unsupervised VAE into a **supervised imputation** task (needs fully-observed data at *training* time; deployment is partial). This is the key trick: *learn to impute the unobserved features → the belief carries information about them → policy conditions on a richer state.*

### Training loop (Algorithm 1)
1. Pre-train the Seq-PO-VAE offline on a dataset `D` (Eq. 5). 2. Then multi-stage RL: at each step compute belief `b_t=q_φ(x^p_{≤t},a_{<t})`, sample `a^f_t~π^f(b_t)` and `a^c_t~π^c(b_t)`, step env, get partial features + reward + cost, store, update policies with **LSTM-A3C**. VAE is frozen during policy training.

### Baselines (what they compare against — the canonical AFA baselines for us to reuse)
- **NonSeq-ZI (partial / full):** β-VAE on non-time-dependent instances, missing features filled by **zero-imputing**; "partial" loss = only over observed features, "full" loss = over all features (uses ground-truth full obs as target). These are the "static, non-sequential" baselines.
- **End-to-End:** no pretrained representation; policy learns representation jointly. (Fails under cost — representation learning is essential.)
- **Random acquisition** at fixed rates (25/50/75/90%).

### Datasets & results
- **Bouncing Ball+** (image control, 32×32; acquisition = choose subset of 4 image quadrants) and **Sepsis** medical simulator (8 features, acquisition = subset of 4 measurements: heart rate/sysBP/percoxyg/glucose; treatments = antibiotic/ventilation/vasopressor).
- Imputation loss (Table 1): Seq-PO-VAE beats NonSeq-ZI on both (Ball+ NLL 0.032 vs 0.072; Sepsis MSE 0.183 vs 0.484).
- Policy: Seq-PO-VAE learns faster, converges higher, and acquires **~half the features** for comparable task performance (Fig 5). On Ball+, solves task with **<10 observations** vs ~20 for baselines. Sepsis: >5% higher discharge rate than best baseline.
- **Cost knob works as expected:** higher `c` → fewer acquisitions, lower task reward — a clean cost-performance Pareto (Fig 8–10).
- Ablation (Appendix D): even when the VAE is trained on only 50/90% observed data (relaxing the full-observation assumption), it still beats non-sequential baselines — reassuring for real settings where full ground truth is scarce.

### Limitations / openings for us
- Action space exponential in #features → they explicitly name **hierarchical action representations** as future work. Our gene/spot acquisition has the same explosion; a spatial/hierarchical structure over the action space is a concrete novelty lever.
- No spatial structure, no images, generic features. We add: H&E-image conditioning + spatial graph over spots + the transcriptome-imputation objective.
- Assumes transition depends only on control action (measurements non-invasive). In our pure-acquisition setting there is no environment dynamics beyond "what's revealed" — arguably simpler (a static hidden state per slide), which means we may not even need the control policy `π^c` or A3C; a greedy expected-information-gain policy may suffice (cheaper first cut, per our CHECKLIST WS-D).

### Direct design transfer to our project
- Our imputer (WS-B) ≈ their Seq-PO-VAE decoder `p(x^u|z)` — reconstruct unmeasured genes from measured ones.
- Our acquisition policy (WS-D) ≈ their `π^f`, but conditioned additionally on the frozen UNI2-h image embedding.
- Our "reward" = −imputation error on held-out genes/spots; our "cost/budget" = number of measurements = their `Σ c·I(a^f)`.
- Their static-Gaussian-prior stability lesson and their pretraining-then-freeze recipe are directly adoptable.

---

## 2. PERSIST — "Predictive and robust gene selection for spatial transcriptomics" (Covert, Gala, Wang, Svoboda, Sümbül, Su-In Lee; Nature Communications 2023, 14:2091)

**Role for us:** THE primary static-panel baseline to beat (WS-C). Also the closest conceptual sibling — it frames panel selection as "pick K genes that reconstruct the genome-wide profile," exactly our imputation objective, but **static, non-spatial, no image, chosen from a reference before seeing the tissue.** Our differentiator is precisely the axes PERSIST fixes.

### Problem
FISH/imaging ST (MERFISH, seqFISH) can probe only ~hundreds of genes (<1% of genome). Which small panel? Frame as **feature selection**: choose K genes whose expression predicts the genome-wide expression profile, so downstream imputation of the rest is best. Reference data is scRNA-seq (spatially-resolved full-transcriptome references don't exist), which creates a **domain-shift** problem scRNA-seq↔FISH (nonlinear, noisy, dropout-heavy).

### Method (three ingredients)
1. **Differentiable feature selection via a binary mask layer.** An autoencoder-like net reconstructs ALL genes using only a masked subset of inputs. The mask is learned with the **Concrete / Gumbel-Softmax** relaxation: to select exactly K genes from d, use element-wise max of K Concrete random variables `A_i ~ Concrete(α_i, τ)`; temperature τ annealed 10→0.01 so each converges to a one-hot → exactly K genes selected. Alternative **binary gates layer** (`BinConcrete`, one gate/gene + L1 penalty λ·1ᵀb) for memory-efficient pre-filtering (used to narrow 10,000→~500 candidates first). Selection is by gradient descent, not combinatorial search — this is the scalability win.
2. **Hurdle loss (Eq. 1)** — models scRNA-seq **dropouts**: for each gene predict both a point value ŷ_i and a probability p̂_i of being expressed. `ℓ = (1/γ)(ŷ−y)² − log p̂` if y>0, else `−log(1−p̂)`. MSE component only when expressed + cross-entropy for detection. γ=10 balances the two. (Recent scRNA-seq best practice: zero-inflated losses beat plain MSE.)
3. **Binarization / expression quantization** — binarize gene counts (expressed vs not) so a model trained on scRNA-seq **transfers to FISH** despite domain shift (assumes monotonic transform between technologies; threshold-matching by quantile). This is the key robustness trick and the first quantitative demonstration that a scRNA-seq-trained predictor transfers to MERFISH.

### Flexibility
Default = unsupervised (reconstruct full profile). Swap the prediction target → **PERSIST-Classification** (cell-type labels, cross-entropy) or **PERSIST-Ephys** (electrophysiology, MSE). So it can be tuned to a downstream objective.

### Datasets & baselines
- scRNA-seq: **SSv4** (22,160 mouse V1/ALM neurons, SmartSeq) and **10X** (72,629 human M1 neurons). Restrict to top-10,000 high-variance genes; select panels of 8–256.
- Cross-tech validation: **MERFISH** MOp (258 genes, 280,327 cells) — select panel from scRNA-seq restricted to MERFISH genes, train imputer on scRNA-seq, test on MERFISH.
- Baselines: Seurat, Cell Ranger (variance/dispersion), **GeneBasis** (greedy manifold-preserving — most competitive), MutInfo, SMaSH, scGeneFit; plus Tasic marker set.

### Results / metrics (the exact eval protocol we can mirror)
- **Explained variance** of genome-wide profile by the panel (train NN to predict full profile, 1 − MSE/total-var). PERSIST > all unsupervised baselines at every panel size; GeneBasis closest but worse for small panels. Notably even 256 genes don't explain all variance (cell-type labels explain only 19%/11% — most variance is stochastic noise).
- **Expressed-gene prediction accuracy** (predict which genes are detected) — PERSIST best, aided by hurdle loss. E.g., 86.5% with a 32-gene panel transferring to MERFISH.
- Panels are **stochastic across runs** (SGD + Concrete) — run ~5 trials, keep best validation loss. Different runs give different genes but similar performance (correlated genes are substitutable).

### Confirmed limitations (= our openings)
- **One globally static panel**, chosen from a scRNA-seq reference **before seeing the target tissue**. No spatial coordinates, no histology image, no per-region adaptation, no sequential/conditional acquisition. Selection is a one-shot batch, not an MDP.
- Reconstructing the full profile from a few genes is inherently lossy; they note general-purpose panels are "good enough" past ~32 genes — but that's precisely because they can't adapt spatially. Our claim: image + spatial conditioning + sequential acquisition can beat a fixed panel at equal budget.

### Direct transfer to our project
- Our imputer (WS-B) plays the same role as PERSIST's reconstruction net; we can reuse the **hurdle loss** and **binarization/threshold-matching** for scRNA↔FISH transfer.
- The **Concrete/Gumbel-Softmax differentiable-mask** is a candidate mechanism for a *learned* (but still non-sequential) selection baseline, and a building block if we make selection amortized/conditional on the image.
- PERSIST's explained-variance and expressed-gene-accuracy-vs-panel-size curves are exactly the Pareto axes we plot in WS-E; run PERSIST as the static curve, our policy as the adaptive curve.
- Code: github.com/iancovert/persist (PyTorch). Datasets all public (Allen brain-map SSv4/10X, MERFISH MOp).

---

## 3. ReconST — "Optimal Gene Panel Selection for Targeted Spatial Transcriptomics Experiments" (Nucleic Acids Research 2026 / bioRxiv 2025.10.08.681071)

**Role for us:** The **current SOTA** panel selector and our second must-beat static baseline (WS-C). Critically: **its own paper names our exact contribution as its limitation** — see below. Read in depth via WebFetch (2026-07-05).

### Method
- **Gated autoencoder** with learnable per-gene importance weights + L1 regularization. A gating layer applies element-wise multiplicative masks to suppress uninformative genes while preserving those needed to reconstruct the full expression profile. End-to-end gradient-based (like PERSIST's differentiable selection, but a cleaner gating formulation).
- **Objective:** select the subset of genes that **optimally reconstruct the whole transcriptome** (reconstruction loss). After training, genes with nonzero gate weights = the panel.

### What it does NOT do (verbatim-confirmed limitations = our empty cell)
- Produces **one fixed panel per training run**; the panel "remains constant across all profiled locations" — **globally static across tissue.**
- Selection uses **only scRNA-seq** data; spatial data (MERFISH/Xenium) is used **post-hoc for validation only.**
- **No histology images. No spatial coordinates during selection. No sequential/active acquisition** (one-shot batch, not conditioned on already-measured spots).
- Stated limitations in paper: "Panel is globally static across tissue"; "No explicit incorporation of spatial structure ... during selection"; relies on representative scRNA-seq reference (biases propagate); L1 strength needs CV tuning.

### Datasets, baselines, metrics
- Datasets: mouse brain scRNA-seq + **MERFISH** (1,147 genes, 215,278 spatial cells); fetal lung scRNA-seq + **Xenium** (343-gene panel).
- Baselines (10): DE, HVG, Spapros, **PERSIST**, Triku, COSG, scPNMF, GeneBasis, NS-Forest, Vizgen default.
- Metrics: reconstruction loss, KL divergence, Pearson/Spearman, clustering ARI, **Moran's I** (spatial autocorrelation), runtime. Beats competitors on 9/10 metrics; **~40× faster than PERSIST** (12 min vs 8 h) — PERSIST's autoencoder is the slow point.

### Transfer / positioning for us
- ReconST is the strongest static reconstruction-objective baseline; run it as a second static Pareto curve alongside PERSIST in WS-E.
- Its "no spatial structure, globally static, no image" limitations are the literal justification for Direction 1. Quote these in the paper's intro/related-work.
- Uses Moran's I as a spatial-quality metric — worth adopting as a secondary eval (does our adaptive panel preserve spatial patterns better?).

---

## 4. HEST-1k — "A Dataset for Spatial Transcriptomics and Histology Image Analysis" (Jaume, Doucet, ..., Mahmood; NeurIPS 2024 Datasets & Benchmarks, Spotlight)

**Role for us:** THE benchmark + data infrastructure for WS-A. Gives (a) 1,229 paired ST+H&E samples with a unified `scanpy`/AnnData + pyramidal-TIFF format, (b) the HEST-Library toolkit (download/align/patch/segment/batch-correct), and (c) HEST-Benchmark — the standardized eval protocol we should mirror. Also supplies the UNI-family patch-encoder baselines and their exact PCC numbers.

### The dataset
- **1,229 samples** from 153 cohorts; 26 organs; 2 species (human/mouse); 367 cancer samples across 25 types. 2.1M expression–morphology pairs, 76M segmented nuclei (via CellViT, 5 classes: neoplastic/non-neoplastic epithelial, inflammatory, stromal, necrotic).
- Technologies: Visium 48% (n=547, sequencing-based, 55µm spots, whole-transcriptome), STv1 45%, Xenium 5.3% (n=65, imaging-based, exact 2D transcript positions, **panel-limited to ≤5,000 genes** — this is exactly the physical constraint motivating Direction 1), Visium HD 0.8%.
- Each sample: pyramidal TIFF WSI + AnnData raw counts (no normalization applied — users choose) + metadata (organ via OncoTree, technology, resolution, FFPE/frozen, spot coords). Patches = **224×224 px at 20× (112×112 µm)** centered on each spot. Xenium → "pseudo-Visium" by pooling transcripts on 55µm patches.
- Big practical wins for us: automatic Visium fiducial alignment (YOLOv8), auto pixel-resolution inference, unified format from messy legacy data, one-line filtered download (>1TB total; can pull just e.g. breast cancer). License CC BY-NC-SA 4.0. Hosted on HuggingFace. github.com/mahmoodlab/hest.

### HEST-Benchmark — the eval protocol to mirror (WS-A/E)
- **9 tasks**, 8 human cancers / 9 organs: IDC (breast), PRAD (prostate), PAAD (pancreas), SKCM (skin), COAD (colon), READ (rectum), ccRCC (kidney), LUAD (lung), LYMPH-IDC (metastatic lymph node). Mix of Xenium & Visium.
- **Task = multivariate regression:** predict the **top-50 most variable genes** (after excluding genes nonzero in <10% of spots), log1p-normalized, from the 224×224 patch. (Note: same top-50-HVG setup our old project used — so our simulator/metrics are directly comparable.)
- **Patient-stratified k-fold CV** (k = #patients; k/2 for ccRCC) to avoid leakage. Metric: **Pearson correlation** per gene, averaged, mean±std across folds.
- Downstream regressors benchmarked: PCA(256)+Ridge (main), Ridge, XGBoost(100 trees, depth 3).

### Baseline numbers (11 pathology foundation encoders → frozen embeddings → regression)
Average Pearson across 9 tasks (PCA+Ridge): **H-Optimus-0 0.415, UNIv1.5 0.409, Virchow2 0.398, Virchow 0.398, UNI 0.386**, GigaPath 0.385, REMEDIS 0.374, CONCH 0.371, Phikon 0.366, CTransPath 0.345, ResNet50(IN) 0.326 (only non-histology model, worst).
- Per-task spread is large: SKCM easy (~0.64 for H-Optimus-0), READ/ccRCC hard (~0.22–0.23). Confirms our midterm finding: **predictability is gene/tissue-specific, not decoder-limited.**
- **Model scaling law:** avg PCC grows logarithmically with encoder params (R=0.81). Data scaling weaker (R=0.48). Takeaway: bigger frozen encoder helps but with diminishing returns — reinforces that the bottleneck is the task/information, echoing our thesis.
- **UNI = ViT-Large, 307M params, DINOv2, 100M histology patches** — the encoder our pipeline already uses (UNI2-h). So our existing embeddings are competitive-but-not-top; H-Optimus-0/UNIv1.5 are stronger if we want to upgrade.

### Relevance / transfer to Direction 1
- **Use HEST-1k as our data substrate for the acquisition simulator** (WS-A): full-transcriptome Visium slides = ground truth to mask/reveal; Xenium panels = realistic panel-size constraint to justify the budget.
- **Adopt HEST-Benchmark's exact protocol** (top-50 HVG, patient-stratified CV, Pearson) so our numbers are field-comparable — fixing the "ad hoc 12-fold split" criticism from the TA.
- HEST-Library handles the painful plumbing (alignment, patching, AnnData) — big time save; budget WS-A around adopting it rather than rebuilding.
- Biomarker-exploration section (nuclear morphology ↔ gene expression correlations, e.g., GATA3↔nuclear area R=0.47) supports the premise that image morphology carries acquisition-relevant signal — evidence for image-conditioning in WS-D.
- Note the benchmark is *pure prediction* (frozen encoder → regress all 50 genes); nobody in HEST does *acquisition*. Our contribution sits on top of this benchmark, not inside it.

---

## 5. STFlow — "Scalable Generation of Spatial Transcriptomics from Histology Images via Whole-Slide Flow Matching" (Huang et al.; ICML 2025 Spotlight)

**Role for us:** The current SOTA prediction model and the strongest related-work baseline. Also the reference design for "joint distribution over all spots + spatial attention," which our WS-D policy/imputer can borrow structurally. NOT an acquisition method — pure prediction.

### Method
- **Flow matching** (rectified-flow style). Interpolate `Y_t = t·Y + (1−t)·Y_0` between true expression Y and a prior sample Y_0; train denoiser `f_θ(Y_t, I, C, t)` with MSE to the marginal vector field. Iterative (multi-step) generation, not one-shot regression.
- **Prior = zero-inflated negative binomial (ZINB)** to match ST data's sparsity + overdispersion (analogous role to PERSIST's hurdle loss — modeling dropouts/zeros matters).
- **Joint distribution over all spots (the headline contribution):** prior methods factorize `p(slide)=Π_spot p(spot)` and predict each spot independently; STFlow models the joint via a **slide-level encoder with local (k-NN) spatial attention**. Attention weights combine visual features, spatial geometry, AND current expression differences: `A_ij = softmax(MLP(Z_Q,i ‖ Z_K,j ‖ C'_ij ‖ (Y_t,i − Y_t,j)))` → explicitly models **cell-cell interaction**.
- **E(2)-invariant spatial attention via frame averaging:** neighbor direction vectors projected into PCA frames + averaged → rotation/translation invariance. Complexity O(Nk·d²) not O(N²) — scales to slides with >10k spots.

### Data, protocol, results
- Datasets: **HEST-1k (10 tissues, patient-stratified k-fold) + STImage-1K4M (8 tissues, 8:1:1)** = 17 benchmarks. Gene panel = **top-50 HVG, log1p** (same as HEST-Benchmark and our setup).
- Results: HEST avg **0.415 PCC** vs 0.395 (prior best TRIPLEX); STImage avg 0.293 vs 0.269 (GigaPath). **~18% relative improvement over pathology-foundation-model baselines.** Strong on biomarkers (GATA3 0.86, ERBB2 0.844).
- Inference: **only S=5 flow steps** (plateaus beyond 5); orders of magnitude faster + lower memory than slide-based baselines (TRIPLEX, HisToGene) due to k-NN attention. (Relevant to the *Direction-2* test-time-compute idea we shelved — 5 steps is already tuned, limited headroom there, further validating our choice of Direction 1.)
- Limitation: ZINB hyperparameters set by grid search, not learned (suggest VAE estimation as future work).

### Transfer to Direction 1
- STFlow's **local spatial attention over spot k-NN graph** is the exact structural prior we want for our image+spatial acquisition policy (WS-D) and/or our imputer (WS-B). We can reuse its E(2)-invariant frame-averaging attention.
- As a baseline: STFlow is a *predictor*, not an *acquirer*. In our simulator it can serve as the imputer (reconstruct held-out genes from measured ones + image), i.e., a strong WS-B backbone. Then our acquisition policy chooses what to feed it.
- Its joint-distribution framing supports our claim that spatial correlation is real and exploitable — measuring one spot informs neighbors (the submodularity our acquisition policy would exploit).
- Code: github.com/Graph-and-Geometric-Learning/STFlow.

---

## 6. TISSUE — "Uncertainty-calibrated prediction of single-cell spatial transcriptomics" (Sun et al.; 2023, bioRxiv/Cell Rep Methods)

**Role for us:** Candidate imputer + **the per-gene uncertainty signal** for our information-gain acquisition criterion (WS-B). This is the "how do we know where the model is unsure" piece that a greedy-IG acquisition policy needs.

### Method — model-agnostic conformal uncertainty wrapper
- Wraps any base spatial-imputation model and adds **calibrated per-gene, per-cell prediction intervals** via **conformal prediction**.
- **Cell-centric variability `U_ij`**: cosine-similarity-weighted deviation between a cell's predicted expression and its 15 nearest spatial neighbors — a local, spatially-aware uncertainty proxy (no ground truth needed at test time).
- **Calibration scores `s_ij = (X_ij − X̂_ij)/U_ij`**: computed on *measured* genes via 10-fold CV, then stratified into gene-groups and cell-groups (k-means) for heterogeneous calibration sets. Asymmetric upper/lower bounds (transcript counts are skewed).
- **Prediction intervals** for unseen genes: multiply cell-centric variability by quantiles of the calibration-score set. **Proposition 1**: coverage guarantee `P(X_ij ∈ [l_ij,u_ij]) ≥ 1−α`.
- Base models it wraps (model-agnostic): **Tangram, SpaGE, Harmony-kNN, gimVI**.

### Data / eval / downstream
- 11 benchmark ST datasets (seqFISH, MERFISH, STARmap, osmFISH, ISS, ExSeq...) paired with scRNA-seq references.
- Metrics: gene-wise Spearman, MAE, empirical interval coverage, calibration error.
- Downstream uses of the uncertainty: multiple-imputation hypothesis testing (Rubin's rules, lowers FDR); cell filtering by interval width (improves classification/clustering); weighted PCA. **Interval width positively correlates with actual prediction error** — exactly the property we need for acquisition.

### Transfer to Direction 1 (this is important for WS-B/WS-D)
- **The calibrated interval width = a ready-made "information need" / expected-error signal per gene per location.** Our greedy acquisition can target measurements where predicted uncertainty (interval width) is highest, or where measuring most reduces neighbors' uncertainty.
- Its core assumption — "spatially proximate cells with similar measured expression have similar unmeasured expression" — is precisely the spatial-correlation structure our acquisition thesis exploits (measuring one spot informs neighbors).
- Limitation to note: degrades for extremely sparse genes and rare non-co-localized cell types, and the CV calibration is k× compute — budget for this in WS-B.
- Practical: use TISSUE (or its uncertainty mechanism) on top of a base imputer (SpaGE/Tangram, or STFlow) to produce both the reconstruction (reward) and the uncertainty (acquisition signal).

---
---

# Part II — Summary-level entries

Entries below are synthesized from full abstracts + this session's targeted searches/WebFetch (and, for the RL gene-panel paper, an earlier full WebFetch deep read), rather than full-PDF line-by-line reads. They are accurate on method/venue/role but lighter on equation-level detail. Flagged `[summary]` or `[deep-fetch]`. Deep-read any on request.

## 7. RL gene-panel selection — "Knowledge-Guided Gene Panel Selection ... A Reinforcement Learning Perspective" (arXiv 2501.04718, 2025) `[deep-fetch]`
- **Role:** the closest existing RL formulation to Direction 1 → must position against. Confirmed by deep WebFetch to NOT occupy our cell.
- Problem: `G* = argmax_{G'⊆G} E(C(X[G']))` — pick gene subset maximizing a downstream metric. **MDP:** state = compressed descriptive stats (std/quartiles) of selected genes via autoencoder; action = per-gene binary select/discard; **reward = α·(NMI between expression & pseudo-labels) + (1−α)·(parsimony/size penalty)**. Actor-Critic + prioritized experience replay.
- **Does NOT use:** spatial data, histology images, per-tissue/per-region adaptation. Static single panel per dataset. Reward is clustering-quality (NMI), not whole-transcriptome imputation error. 24 scRNA-seq datasets; baselines CellRanger/Seurat/HRG/geneBasis/CellBRF/gpsFISH/scGIST; metrics NMI/ARI/Silhouette/accuracy/F1. Not a top-tier venue.
- **Our differentiation:** we add image + spatial conditioning, sequential/conditional acquisition (measured spots inform next), and an imputation-reconstruction objective. This paper is a scRNA-only, static, clustering-oriented feature selector wearing RL clothing.

## 8. BLEEP — "Spatially Resolved Gene Expression Prediction from H&E via Bi-modal Contrastive Learning" (Xie et al.; NeurIPS 2023) `[summary + abstract read]`
- **Role:** original retrieval baseline; our team already (mis-)reproduced it (broken local repro, gene PCC≈0.0001). Related-work.
- Method: CLIP-style **contrastive** joint embedding of H&E patches ↔ expression profiles (image encoder + expression encoder, InfoNCE). Inference = **query–reference retrieval**: embed a query patch, find top-k nearest reference spots in joint space, aggregate (average) their measured expression as the prediction. Verified top-1 < averaging.
- Data: human liver 10x Visium. Notes earlier methods "predict only ~200 genes"; task is ill-posed for most genes.
- Transfer: BLEEP-style retrieval is a candidate *imputer/baseline* in our simulator (retrieve from measured spots). Its contrastive image↔expression alignment is reusable to make the image embedding "expression-aware."

## 9. Stem — "Diffusion Generative Modeling for Spatially Resolved Gene Expression Inference from Histology Images" (Zhu, Zhu, Tao, Qiu, Georgia Tech; ICLR 2025) `[full PDF read]`
- **Role:** related-work SOTA-adjacent + candidate generative imputer backbone (WS-B). Our old project approximated it (badly) as a DiT+AdaLN transformer; this is the faithful description. GenAR's main comparison target.
- **Core thesis:** gene-expression-from-histology is a **one-to-many** problem — the same/similar H&E morphology can map to different expression profiles (biological + technical variation). Deterministic regressors (MLP/transformer) are trained with MSE and thus **regress to the conditional mean**, systematically under-dispersing predictions and washing out the very variance we care about. A **conditional generative model** instead learns the full conditional distribution `p(expression | image)`.
- **Method:** conditional **DDPM diffusion**. Forward process adds Gaussian noise to the (normalized) gene-expression vector over T steps; a denoiser `ε_θ(y_t, t, c)` is trained to predict noise, **conditioned on the H&E patch embedding** `c` from a (frozen or fine-tuned) image encoder. Conditioning injected via the denoiser backbone (a **DiT-style transformer with AdaLN** — the pooled image embedding is projected and fed to each block through adaptive layernorm; this is exactly what our old project mimicked but without real diffusion training/sampling). Reverse process samples an expression vector by iterative denoising from Gaussian noise.
- **Key operating modes / findings:**
  - Because it's generative, at inference you can **draw multiple samples** and average (→ point estimate that matches or beats deterministic models on PCC) OR keep the samples to recover **expression variance/heterogeneity** (its distinctive win — deterministic models can't).
  - Reports that generated expression better matches the true **distribution** (variance, gene-gene correlation structure), not just the mean.
- **Data/eval:** Visium + other ST; top-HVG panels; metrics PCC-10/50/200, MSE/MAE, and distribution-level metrics (e.g., RVD / variance recovery). SOTA-competitive on mean PCC while uniquely modeling dispersion.
- **Limitations:** diffusion sampling is multi-step → inference cost (the axis Direction 2 would have exploited); performance still bounded by image information content (most genes remain low-PCC, consistent with HEST/our findings).
- **Transfer to Direction 1:**
  - Strong **generative imputer** for WS-B: reconstruct held-out genes conditioned on measured genes + image; its samples give a **native uncertainty estimate** (sample variance) — an alternative to TISSUE's conformal intervals for the acquisition information-gain signal.
  - Confirms the **one-to-many** nature of the task → supports acquisition framing: when the image alone is ambiguous, *measuring* a few genes disambiguates the conditional distribution (exactly what our policy exploits). Stem quantifies the residual stochasticity that measurement removes.
  - Its AdaLN conditioning recipe is directly reusable for image-conditioning our policy/imputer. Code: github.com/SichenZhu/Stem.

## 10. GenAR — "Next-Scale Autoregressive Generation for Spatial Gene Expression Prediction" (arXiv 2510.04315, 2025, under review) `[full-text read]`
- **Role:** THE closest published work to our OLD "gene-token decoder" idea — and it is **not agentic**, just a well-engineered coarse-to-fine autoregressive generative model. This is the paper that "already did, rigorously, what we were attempting." Cite prominently to justify abandoning the old direction and to show our new (acquisition) contribution is orthogonal.
- **Multi-scale autoregressive formulation:** factorizes `p(y|H) = Π_{k=1}^K p(y^(k) | H, y^(<k))` over **K=6 scales** with gene groupings **(1, 4, 8, 40, 100, 200)** — scale 1 = single group = global transcriptional activity across all 200 genes; scales 2–5 progressively finer; scale 6 = individual genes. Each scale conditions on all coarser outputs ("cumulative context") → encodes cross-gene dependencies. This is *next-scale* prediction (à la VAR image models), NOT gene-by-gene left-to-right — a key distinction from our old token-by-token idea.
- **Codebook-free discrete tokenization:** exploits that gene counts are **already discrete integers** → map counts directly to token IDs via a learned embedding `E_gene ∈ ℝ^(vocab×768)`, vocab = max_count+1 (default 2048). No VQ-VAE codebook, no quantization bottleneck, end-to-end differentiable. Avoids log-transform biases; final scale uses **Gaussian NLL with heteroscedastic variance σ²=αμ+β** (preserves count semantics), intermediate scales use **soft KL divergence** via adaptive pooling.
- **Hierarchical gene clustering:** k-means on Z-scored spatial expression profiles (train set): 200 genes → 4 major clusters → ~12-gene subgroups. Groups are biologically coherent (e.g., HER2/17q amplicon ERBB2/GRB7/STARD3; luminal markers MUC1/TFF3/KRT7/19; CAF/ECM COL1A1/FN1/POSTN).
- **Conditioning:** H&E patch (224²) → **UNI** features (1024-d) → LN + 2 linear (GELU, dropout); spatial coords → sinusoidal PE; fused → `H ∈ ℝ^768`. Injected via **AdaLN** (modulates decoder activations) + **FiLM** with per-gene identity embeddings `E_identity ∈ ℝ^(200×768)`.
- **Architecture:** transformer decoder, 8 layers / 8 heads / 768-d / MLP-ratio 3.0, causal masking within scale; teacher forcing in training, greedy argmax autoregressive at inference.
- **Data:** 4 HEST-1k datasets — HER2ST (breast, 13,594 spots), PRAD (prostate), Kidney, Mouse Brain; +ccRCC validation. **Top-200 genes** (highly-expressed ∩ highly-variable). log₂ post-prediction for metric parity.
- **Results (vs STEM, the prev best):** HER2ST PCC-200 **0.663** vs 0.625 (+6.1%), MSE −9.8%; PRAD PCC-200 **0.512** vs 0.403 (**+27%**), MSE −18.3%; consistent gains on Kidney/MouseBrain. **Larger gains on cancer tissue.**
- **Ablations (PRAD, the useful part for us):** w/o multi-scale PCC-10 0.651 vs 0.702 (−7.3%, most critical); w/o gene-identity PCC-200 −6.1%; cross-entropy loss −6% vs their soft-KL+GaussNLL. Scale-count: single-scale (200) PCC-200 0.445 → K=6 (1,4,8,40,100,200) 0.512, plateaus beyond K=6. Encoder: UNI (0.383) >> CONCH (0.286) > ResNet-18 (0.273), GenAR lifts to 0.512.
- **Limitations:** weaker in extremely sparse regions (token rarity/gradient sparsity); fixed top-200 genes, full-transcriptome extension unclear; scale design needs tuning.
- **Transfer / positioning for us (critical):**
  - GenAR proves multi-scale autoregressive conditioning helps **without any agency** → definitively closes the door on our old "policy over gene order" framing. Our new contribution MUST be the *acquisition-under-budget* decision, which GenAR does not touch (it predicts all 200 genes from image alone; it never *chooses what to measure*).
  - But GenAR is a **superb imputer candidate** for WS-B: given measured genes as additional conditioning (prefix scales), it reconstructs the rest. Its hierarchical gene grouping = a natural structure for our acquisition action space (acquire at group granularity → resolves the exponential-action-space problem AFA-RL flagged).
  - Its heteroscedastic Gaussian-NLL head gives per-gene predictive variance → another ready **information-gain signal** for acquisition (like Stem samples / TISSUE intervals).
  - Reviewer-defense value: when a reviewer asks "isn't this just GenAR + selection?", the answer is our decision variable (what to physically measure under budget) is absent from GenAR entirely — GenAR assumes zero measurements and predicts everything; we assume a measurement budget and optimize its allocation. Code: github.com/oyjr/genar.

## 11. MERGE — "Multi-faceted Hierarchical Graph-based GNN for Gene Expression Prediction from WSIs" (CVPR 2025) `[summary]`
- **Role:** related-work; validates graph-structured spatial modeling (cf. our failed "spatial transformer").
- Method: cluster tissue patches by spatial + morphological features; build **intra- and inter-cluster edges** so distant tissue regions interact during **GNN** message passing; predict expression jointly. SOTA on breast/colorectal.
- Transfer: the spot-graph construction (spatial+morphological clustering) is a reusable structural prior for our acquisition policy's spatial reasoning.

## 12. PEaRL — "Pathway-Enhanced Representation Learning for Gene and Pathway Expression Prediction from Histology" (arXiv 2510.03455, 2025) `[summary]`
- **Role:** related-work; shows biologically-grounded supervision beats raw-HVG targets.
- Method: represent transcriptomics as **pathway activation scores (ssGSEA)**; transformer pathway encoder + image encoder aligned via **contrastive learning**. Up to +58.9% PCC (gene) / +20.4% (pathway) vs SOTA on breast/skin/lymph; best C-index on TCGA-BRCA survival.
- Transfer: suggests our acquisition objective could target *pathways* (biologically coherent gene groups) rather than individual HVGs — a possible refinement of the WS-A budget/target definition.

## 13. scGIST — "Gene panel design for spatial transcriptomics with prioritized gene sets" (Genome Biology 2024) `[summary]`
- **Role:** static-panel baseline (WS-C, secondary).
- Method: **constrained feature selection** (a differentiable "deep learning classifier as constraint") that designs a panel of a target size while letting users force-include prioritized genes; optimizes cell-type detectability. ~1,000-gene FISH ceiling is its motivating constraint.
- Vs us: static, scRNA-driven, classification-oriented, non-spatial, no image. Another incumbent whose limitations = our cell.

## 14. gpsFISH — "Gene panel selection for targeted spatial transcriptomics" (Genome Biology 2024) `[summary]`
- **Role:** static-panel baseline (WS-C, secondary).
- Method: optimizes gene panel for **cell-type classification**, explicitly **models & corrects platform effects** (scRNA↔FISH) via a probabilistic distortion model; outperforms prior on cell-type detection. Static, non-spatial, no image.
- Vs us: cell-typing objective, not imputation; static; confirms platform-effect modeling matters (echoes PERSIST binarization).

## 15. SpaIM — "Single-cell spatial transcriptomics imputation via style transfer" (Nature Communications 2025) `[summary]`
- **Role:** candidate imputer backbone (WS-B).
- Method: **style-transfer** framework — disentangles scRNA-seq "content" from ST "style" to impute unmeasured/missing genes in ST from a scRNA reference. Strong on imputation benchmarks.
- Transfer: an alternative WS-B imputer (reconstruct held-out genes from measured + reference).

## 16. PASTA — "Accurate imputation of pathway-specific gene expression in spatial transcriptomics" (Nature Communications 2025) `[summary]`
- **Role:** candidate imputer / target-definition input (WS-B/A).
- Method: imputes **pathway-specific** expression in ST (pathway-level rather than gene-level targets), improving robustness on sparse genes.
- Transfer: supports a pathway-level acquisition/reward variant (cf. PEaRL); alternative imputer.

## 17. Active Feature Acquisition — "Stochastic Encodings for Active Feature Acquisition" (arXiv 2508.01957, 2025) `[summary]`
- **Role:** AFA method-template variant (WS-D).
- Method: latent-variable AFA — learns stochastic encodings of partially-observed instances to drive which feature to acquire next; improves over greedy/RL AFA on standard benchmarks. Non-spatial, generic tabular/vision.
- Transfer: a more modern alternative to the 2011.00825 seq-VAE scaffold for our acquisition policy; worth benchmarking our spatial version against it conceptually.

## 18. NOCTA — "Non-Greedy Objective Cost-Tradeoff Acquisition for Longitudinal Data" (arXiv 2507.12412, 2025) `[summary]`
- **Role:** AFA method-template variant (WS-D), emphasizes non-greedy planning.
- Method: **non-greedy** acquisition that plans over future acquisitions (vs myopic per-step IG), trading objective vs cost on longitudinal/time-series data.
- Transfer: relevant if our greedy-IG acquisition (WS-D first cut) leaves headroom — motivates a non-greedy/lookahead policy as the "learned RL" escalation.

## 19. STImage-1K4M — "A histopathology image–gene expression dataset for spatial transcriptomics" (2024) `[summary]`
- **Role:** second benchmark dataset (WS-A), used alongside HEST-1k by STFlow.
- Content: 1,149 ST images → sub-tile image + gene-expression (15k–30k-dim) pairs. Larger per-image gene coverage than HEST in places.
- Transfer: secondary evaluation substrate for the acquisition simulator; STFlow's use of it (8 tissues, 8:1:1 split) gives a comparable protocol.

## 20. CellAgent — "An LLM-driven Multi-Agent Framework for Automated Single-cell Data Analysis" (Xiao et al.; ICLR 2025 poster) `[summary]`
- **Role:** the ONE agentic paper accepted at a top venue in this space → defines what "agentic" reviewers accept (workflow orchestration, not regression).
- Method: LLM roles (**planner / executor / evaluator**) with hierarchical decision-making orchestrate a scRNA-seq/ST **analysis pipeline** (QC, clustering, annotation) via natural language; sc-Omni tool library. Agency = choosing which tool/step to run.
- Transfer / positioning: shows agency-over-workflow is the accepted pattern; our Direction 1 is agency-over-*measurement*, a distinct and arguably deeper decision problem. Cite to frame "agentic AI in ST exists but does workflow, not acquisition."

## 21. SpatialAgent — "An autonomous AI agent for spatial biology" (Genentech; bioRxiv 2025) `[summary]`
- **Role:** agentic-context; NOT peer-reviewed at a top AI venue (bioRxiv only).
- Method: fully autonomous LLM agent spanning experimental design → multimodal analysis → hypothesis generation; adaptive reasoning + dynamic tool integration; tested on ~2M cells (brain/heart/mouse colitis); matches/beats human experts on some tasks. Agency over the *scientific workflow*.
- Positioning: closest in spirit to "agent decides experiments," but at the workflow level, not a trainable acquisition policy with an imputation objective. Reinforces the gap Direction 1 fills. Code: github.com/Genentech/SpatialAgent.

## 22. LLM4Cell — "A Survey of Large Language and Agentic Models for Single-Cell Biology" (arXiv 2510.07793, 2025) `[summary]`
- **Role:** survey / citation-mining source for the agentic-in-biology landscape.
- Content: taxonomy of LLM + agentic approaches across single-cell (and some spatial) biology; catalogs CellAgent, scBaseCount, BioInformatics-Agent, etc.
- Transfer: use for related-work breadth and to confirm (again) that no surveyed agentic method does image-conditioned sequential gene/spot **acquisition** — the negative-space evidence for our novelty.

---

## 23. SpatialFinder — "A human-in-the-loop vision-language framework for prioritizing high-value regions in spatial transcriptomics" (Frontiers in Bioinformatics 2026) `[deep-fetch]`
- **Role:** THE direct entrant into our Option-B (region-acquisition) space. Simultaneously proves the problem is real/publishable AND defines exactly what we must do differently. Added after the 2026-07-05 reality check (`lit_review/REALITY_CHECK.md`).
- **Task:** from H&E alone, rank ~500µm ROIs to **sequence**, to cut cost ("$15k/slide → multiple subjects/slide"). Visium HD.
- **Method:** VLM (MUSK) embeddings + active-learning cell classifier (StarDist nuclei + PLIP zero-shot + pathologist labels ~27min + XGBoost) + **hand-designed hybrid score**: Diversity (UMAP pairwise distance + Shannon entropy of cell composition) and Targeted (text-image cosine + target-cell abundance). α grid-searched per tissue.
- **Results:** 4 Visium HD tissues; Spearman ρ 0.37–0.80, Overlap@10% 25–48% vs cell-type-entropy ground truth. Expert labeling gives ~+50pt overlap.
- **What it does NOT do (= our precise, surviving gap):**
  1. **One-shot** ranking (all ROIs scored at once, greedy top-K) — **no closed loop, no conditioning on already-measured expression** → not an MDP/acquisition process.
  2. Objective = **diversity/relevance heuristic**, NOT whole-transcriptome imputation/reconstruction error.
  3. Budget **implicit**; no formal constrained objective.
  4. **Not a learned policy** — hand-designed scores + XGBoost + human labels; no RL/amortized policy.
  5. Per-tissue, no transfer; single annotator.
- **Positioning / transfer:** SpatialFinder is our nearest competitor and our best evidence of demand. Our differentiation must be, explicitly: **sequential closed-loop acquisition (condition on measured regions) + reconstruction/imputation objective + a learned policy.** The single decisive experiment: show closed-loop acquisition beats one-shot selection (SpatialFinder-style) at equal budget. If we drift to one-shot heuristic scoring, we reinvent this paper. Contrast in intro/related-work.

---

## Cross-paper synthesis (answers the "so what" for Direction 1)

1. **The method scaffold exists and is proven** (AFA-POMDP: seq-VAE over partial obs + RL policy trading reward vs cost). We instantiate it with genes/spots as features, imputation error as reward, budget as cost. [Paper 1]
2. **The imputer + uncertainty signal exist** (TISSUE conformal intervals; SpaIM/PASTA/STFlow/Stem as base imputers). Interval width = information-need signal for greedy acquisition. [6, 15, 16, 5, 9]
3. **The static-panel incumbents we must beat are strong but structurally limited** — PERSIST/ReconST/scGIST/gpsFISH: all static, scRNA-only, no image, no spatial adaptation, no sequential conditioning. Their own papers name these as limitations. [2, 3, 13, 14]
4. **The spatial-structure prior is available** (STFlow E(2)-invariant k-NN attention; MERGE cluster graph) to make acquisition spatially aware. [5, 11]
5. **Sequential/autoregressive prediction is already done non-agentically** (GenAR) → our novelty must be the *acquisition decision under budget*, not sequential prediction. [10]
6. **Agentic AI in ST = workflow orchestration only** (CellAgent, SpatialAgent) → image-conditioned sequential *measurement acquisition* is genuinely unoccupied. [20, 21, 22]
7. **The benchmark + protocol are standardized** (HEST-1k top-50-HVG patient-stratified Pearson; STImage-1K4M) → we evaluate there for comparability. [4, 19]

**Bottom line the KB supports:** every component of Direction 1 has a proven building block in the literature; the contribution is their *novel composition* into an image-conditioned, spatially-structured, budget-constrained acquisition policy — and the go/no-go remains whether that composition beats a static PERSIST/ReconST panel at equal budget (CHECKLIST WS-D gate).
