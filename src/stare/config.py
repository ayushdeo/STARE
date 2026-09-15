"""
STARE — pinned experiment parameters (Stage 0).

This is the SINGLE SOURCE for all experiment constants. Every script imports from
here so runs are reproducible and comparable. Values marked `CONFIRM_ON_MACHINE`
must be verified against the actual HEST-1k metadata on the 4070 laptop (we cannot
verify exact sample IDs without downloading HEST metadata) — see README "Next steps".

Rationale for each choice is in docs/SOURCE_OF_TRUTH.md (§4/§5) and
docs/REPRODUCTION_PLAN.md (§3). Do not change a value without updating those docs.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to repo root; the 4070 machine sets DATA_ROOT to its disk)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("data")          # HEST-1k download target (gitignored)
RESULTS_ROOT = Path("results")    # experiment outputs (gitignored except *.json summaries)
CACHE_ROOT = Path("cache")        # UNI embeddings, imputer checkpoints (gitignored)

# ---------------------------------------------------------------------------
# COMMON EVALUATION SUBSTRATE  (REPRODUCTION_PLAN §3)
# ---------------------------------------------------------------------------
# Platform: Visium HD — whole transcriptome + fine spatial resolution gives BOTH
# imputation ground truth AND region definition; also SpatialFinder's native platform
# so we can run it 1:1 (Tier-1). Fallback to standard Visium if HD sample count too thin.
PLATFORM = "visium_hd"            # {"visium_hd", "visium"} — fallback documented in README
PLATFORM_FALLBACK = "visium"

# Feasibility tissue = PROSTATE (PRAD): appears in BOTH SpatialFinder (4 tissues) and
# GenAR/HEST; cancer tissue with large reported acquisition headroom (GenAR PRAD gains).
# Validation tissue = KIDNEY (also in both). Scale later to all 4 SpatialFinder tissues.
FEASIBILITY_TISSUE = "prostate"
VALIDATION_TISSUE = "kidney"
SCALE_TISSUES = ["prostate", "kidney", "lung", "colon"]  # SpatialFinder's four

# Exact HEST-1k sample IDs. We validated against the live `MahmoodLab/hest` metadata CSV.
# For each tissue we keep 1-2 Visium HD samples to anchor the feasibility slice.
HEST_DATASET_REPO = "MahmoodLab/hest"
HEST_METADATA_FILE = "HEST_v1_3_0.csv"
HEST_SAMPLE_IDS: dict[str, list[str]] = {
    "prostate": ["TENX174"],
    "kidney": ["TENX186", "TENX173"],
    "lung": ["TENX171", "TENX170"],
    "colon": ["TENX175", "TENX156"],
}

# ---------------------------------------------------------------------------
# REGION / ACQUISITION UNIT  (SOURCE_OF_TRUTH §4; GeoMx ROI is 10-600 um)
# ---------------------------------------------------------------------------
# A "region" (acquisition unit) = one square tile of side REGION_UM microns.
# Pinned to 112 um so that ONE region == ONE 224x224 px patch at 20x magnification
# (HEST's patch convention == UNI/GenAR image-encoder input). This aligns the
# acquisition unit with the image-encoder patch exactly => no resampling mismatch,
# and sits mid-range of GeoMx ROI sizes (100-600 um). Visium HD 8um bins are pooled
# into these tiles.
REGION_UM = 112.0                 # region side length in microns
PATCH_PX = 224                    # image patch pixels per region (at 20x)
TARGET_MAG = 20                   # magnification for patch extraction
MICRONS_PER_PATCH = 112.0         # 224 px @ 20x ~= 112 um (HEST convention)

# ---------------------------------------------------------------------------
# BUDGET GRID  (SOURCE_OF_TRUTH §4/§6) — fraction of regions we may profile
# ---------------------------------------------------------------------------
# Evaluate imputation-error-vs-budget along this grid (the Pareto x-axis).
BUDGET_FRACTIONS = [0.01, 0.02, 0.05, 0.10, 0.20]
FEASIBILITY_BUDGET_FRACTION = 0.05   # single-point budget for the Stage-2 go/no-go
ACQUISITION_ROUND_SIZE = 1           # regions revealed per closed-loop round (1 = fully sequential)

# Budget sweep (scripts/03_budget_sweep.py). Include LOW budgets where region choice actually
# matters (at high budget the task saturates and every policy looks equal). This produces the
# imputation-error-vs-budget Pareto curve — the field-standard presentation (cf. PERSIST panel-size
# curves, AFA-RL reward-vs-#features). The diagnostic, not a single-point pass/fail.
SWEEP_BUDGET_FRACTIONS = [0.005, 0.01, 0.02, 0.05, 0.10, 0.20]
RANDOM_SEEDS = [0, 1, 2, 3, 4]       # random baseline is stochastic -> average over seeds (mean +/- std)
# The decisive contrast: run every policy under a STATIONARY imputer (kNN; theory => no closed-loop
# gain), a LEARNED non-stationary linear one (kriging), AND a STRONG STFlow-class one (graphflow;
# nonlinear + multi-hop spatial propagation, the imputer with the best chance of rewarding the loop).
# One file per imputer. If graphflow is ALSO no-gain, the negative result has no escape hatch left.
SWEEP_IMPUTERS = ["knn", "kriging", "graphflow"]

# --- Iteration 3 CI run (2026-09-15): significance across slides ----------------------------------
# The single prostate slide cannot separate a +0.003 PCC margin from noise. scripts/05_multislide_ci.py
# pools ALL pinned slides across these tissues and reports the paired margin (ours vs the strong
# non-adaptive opponent) with a percentile bootstrap CI + exact sign test. A claim of a closed-loop
# WIN requires the CI to exclude 0 across slides; otherwise the honest verdict is "tie / no gain".
CI_TISSUES = ["prostate", "kidney", "lung", "colon"]   # prostate(1)+kidney(2)+lung(2)+colon(2) = 7 slides
CI_BOOTSTRAP = 5000
CI_ALPHA = 0.05
CI_SEED = 2021

# ---------------------------------------------------------------------------
# GENE TARGETS  (HEST-Benchmark protocol)
# ---------------------------------------------------------------------------
# Primary metric on top-50 HVG (HEST-Benchmark standard, comparable to field + our old work);
# secondary on full transcriptome. HVGs computed per-slide, genes nonzero in <10% spots excluded.
N_HVG = 50
MIN_SPOT_FRACTION = 0.10          # drop genes expressed in <10% of spots (HEST rule)
EXPRESSION_TRANSFORM = "log1p"    # log1p-normalized counts (HEST rule)
EVAL_FULL_TRANSCRIPTOME = True    # also report all-gene imputation as secondary

# ---------------------------------------------------------------------------
# IMAGE ENCODER  (frozen; reuse existing pipeline)
# ---------------------------------------------------------------------------
IMAGE_ENCODER = "uni2h"           # UNI2-h ViT, 1536-d; frozen. {"uni2h","uni","resnet50"}
IMAGE_EMB_DIM = 1536

# ---------------------------------------------------------------------------
# IMPUTER  (WS-B; frozen environment/reward model)  REPRODUCTION_PLAN §2.4
# ---------------------------------------------------------------------------
# Feasibility imputer = transparent, cheap, 4070-friendly: kNN regression in the
# joint (image-embedding + measured-expression) space. Reconstructs an unprofiled
# region's expression from its k nearest PROFILED regions. Its neighbor-disagreement
# gives a free predictive-variance = information-gain signal for the greedy policy.
# Scale imputer = STFlow or SpaIM (upgrade after gate passes).
IMPUTER = "knn"                   # {"knn","stflow","spaim"}
IMPUTER_KNN_K = 10                # k for kNN imputer (matches BLEEP/our old kNN baseline)
IMPUTER_KNN_METRIC = "cosine"     # distance in image-embedding space
# Iteration-1 fix (2026-09-07): image-only kNN is a coverage interpolator that CANNOT reward the
# closed loop (see results/sweep_prostate.json: image-only kmeans wins, loop-on == loop-off).
# Blend image + physical-space so a measured region has LOCAL consequence -> the loop can help.
IMPUTER_SPACE = "joint"           # {"image","spatial","joint"} feature space for kNN neighbors
IMPUTER_JOINT_ALPHA = 0.5         # joint feats = sqrt(a)*L2norm(image) (+) sqrt(1-a)*unitbox(coords)

# Greedy acquisition criterion (WS-D). "uncertainty" (v0) picks the single most-uncertain region
# (often an outlier -> hurt coverage). "error_reduction" (v1) picks the region whose measurement
# reduces total imputation error over its NEIGHBOURHOOD (AFA / Bayesian experimental design).
GREEDY_CRITERION = "error_reduction"   # {"uncertainty","error_reduction"}
GREEDY_ER_K = 15                       # neighbourhood size for the error-reduction score

# --- Iteration 2 (2026-09-08): the decisive regime change --------------------------------------
# kNN is STATIONARY -> Krause 2008 says adaptivity gap = 0 (our World-2). To let the closed loop
# even be ABLE to help, the imputer's posterior must DEPEND ON MEASURED VALUES. Regression-kriging:
# an image->expression ridge REFIT on the profiled regions (so the predictor changes with what we
# measure => non-stationary) + spatial residual kriging. IMPUTER="kriging" selects it.
KRIGING_PCA_DIM = 64             # PCA-reduce UNI emb before ridge so per-step refit is cheap
KRIGING_RIDGE_ALPHA = 10.0       # ridge strength for the image->expression predictor
KRIGING_RESID_K = 10             # spatial-kNN for residual kriging (data-dependent uncertainty)

# --- Iteration 3 (2026-09-14): STFlow-class imputer — close the last gap --------------------------
# The kriging critique was "too weak a learned imputer." graphflow steel-mans the agent with a STRONG,
# STFlow-inspired imputer that keeps every STFlow inductive bias runnable IN-LOOP:
#   * NONLINEAR image->expr backbone via random Fourier features (RFF), ridge-refit on the profiled
#     regions each step (closed form => cheap => non-stationary, like kriging but nonlinear).
#   * ITERATIVE spatial message passing on a fixed kNN graph that CLAMPS measured regions to their
#     true values (Dirichlet boundary) — multi-hop neighbor-conditioning, STFlow's core mechanism.
#   * DATA-DEPENDENT uncertainty = 1 - (graph "influence" reaching a region from clamped nodes); it
#     shrinks near measured regions as they accumulate -> the fairest possible signal for the loop.
# If the closed loop cannot beat non-adaptive MI-greedy even HERE, adaptivity earns nothing (Krause).
FLOW_PCA_DIM = 64            # PCA-reduce UNI emb before the RFF map (cheap per-step ridge refit)
FLOW_RFF_DIM = 512          # random Fourier feature dimension (nonlinear image->expr backbone)
FLOW_RIDGE_ALPHA = 10.0     # ridge strength for the RFF image->expression predictor
FLOW_GRAPH_K = 8            # kNN degree of the fixed spatial+image message-passing graph
FLOW_STEPS = 20             # iterative refinement steps (the flow-matching analog: integrate T steps)
FLOW_STEP_SIZE = 0.5        # per-step mixing toward the graph-smoothed neighbour mean

# Principled NON-ADAPTIVE placement baselines [Krause, Guestrin & Singh, JMLR 2008]: submodular,
# (1-1/e)-optimal; the correct references for "select locations to reconstruct a field". RBF kernel
# on joint (PCA-image (+) unit-box coords) features. These use only the kernel, never measured
# values -> they are the honest open-loop opponents our closed-loop policy must beat.
PLACEMENT_PCA_DIM = 64
PLACEMENT_LENGTHSCALE = 1.0      # RBF lengthscale in standardized joint-feature space
PLACEMENT_NOISE = 1e-2           # jitter / observation noise for the kernel

# ---------------------------------------------------------------------------
# METRICS  (SOURCE_OF_TRUTH §5 WS-E)
# ---------------------------------------------------------------------------
# Scored over UNPROFILED regions only.
METRICS = ["pcc_mean", "pcc_top50", "mse", "morans_i"]
PRIMARY_METRIC = "pcc_top50"      # mean per-gene Pearson over top-50 HVG (maximize)

# ---------------------------------------------------------------------------
# SPLITS  (HEST patient-stratified; no slide leakage)
# ---------------------------------------------------------------------------
SPLIT = "patient_stratified"
SEED = 2021                       # matches GenAR for cross-comparability

# ---------------------------------------------------------------------------
# COMPUTE (4070 laptop: ~8-12 GB VRAM) — keep feasibility slice light
# ---------------------------------------------------------------------------
DEVICE = "cuda"
BATCH_SIZE_EMB = 128              # UNI embedding extraction batch
NUM_WORKERS = 4


@dataclass
class RunConfig:
    """Bundle for a single experiment run; scripts instantiate + log this to results."""
    tissue: str = FEASIBILITY_TISSUE
    budget_fraction: float = FEASIBILITY_BUDGET_FRACTION
    region_um: float = REGION_UM
    imputer: str = IMPUTER
    image_encoder: str = IMAGE_ENCODER
    n_hvg: int = N_HVG
    seed: int = SEED
    policy: str = "greedy_ig"        # {"random","grid","variance","kmeans","oneshot","greedy_ig","rl"}
    round_size: int = ACQUISITION_ROUND_SIZE
    extra: dict = field(default_factory=dict)


# Sanity-check tolerances for Tier-1 native reproduction (REPRODUCTION_PLAN §2)
TIER1_TOLERANCE = {
    "spatialfinder_spearman": 0.05,   # reproduce their rho within +/-0.05 on >=1 native tissue
    "imputer_pcc_rel": 0.10,          # imputer published PCC within +/-10% relative
}
