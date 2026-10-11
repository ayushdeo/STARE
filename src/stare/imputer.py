"""
Frozen imputer = the environment's reward + uncertainty model (WS-B).

Reconstructs UNPROFILED regions' expression from PROFILED regions (+ image). Shared,
frozen, and identical across every policy so comparisons are fair (REPRODUCTION_PLAN §6).

kNN regression in a chosen FEATURE SPACE (config.IMPUTER_SPACE):
  * "image"  : cosine-ish distance in frozen UNI embedding space (v0; a coverage interpolator).
  * "spatial": physical-coordinate distance (measurements have local consequence).
  * "joint"  : blend of both -> sqrt(a)*L2norm(image) concatenated with sqrt(1-a)*unitbox(coords).
An unprofiled region is imputed as the mean expression of its k nearest PROFILED regions in that
space; neighbour disagreement = predictive variance = the info-gain signal the greedy policy uses.

WHY joint (2026-09-07): image-only kNN made acquisition == image coverage, so a one-shot k-means
won and the closed loop added nothing (results/sweep_prostate.json). A spatial component gives a
measured region local consequence, which is the precondition for the closed loop to help at all.

Iteration 3 adds GraphFlowImputer: an STFlow-INSPIRED strong imputer (nonlinear RFF image->expr
backbone + iterative spatial message passing that clamps measured regions). It is the hardest test
of the adaptivity hypothesis. See docs/KNOWLEDGE_BASE.md #5/#6/#9 and config.py "Iteration 3".
"""
from __future__ import annotations
import numpy as np

from . import config
from .simulator import AcquisitionState, Slide


def _slide_cached(cache: dict, slide: Slide, build):
    """Per-slide cache for quantities that depend only on the slide's image embeddings and coordinates.

    Keyed by the identity of those two arrays, and the entry keeps references to them, so a key can never be
    reused by a different slide (keying by id(slide) could be: a discarded temporary slide's id may be reused
    by a new object, which returned another slide's cached graph). Copies made with dataclasses.replace -- the
    feedback-ablation view and the spiked slides -- share the arrays and therefore share the entry, while
    the cache never holds their (large) expression matrices."""
    emb, co = slide.image_embeddings, slide.region_coords
    key = (id(emb), id(co))
    hit = cache.get(key)
    if hit is not None and hit[0] is emb and hit[1] is co:
        return hit[2]
    val = build()
    cache[key] = (emb, co, val)
    return val


class BaseImputer:
    def impute(self, state: AcquisitionState) -> np.ndarray:
        raise NotImplementedError

    def uncertainty(self, state: AcquisitionState) -> np.ndarray:
        raise NotImplementedError


class KNNImputer(BaseImputer):
    """kNN in the configured feature space; predict = mean expression of k nearest PROFILED
    regions; uncertainty = mean-over-genes variance across those neighbours."""

    def __init__(self, k: int = config.IMPUTER_KNN_K, space: str = config.IMPUTER_SPACE,
                 alpha: float = config.IMPUTER_JOINT_ALPHA):
        self.k = k
        self.space = space
        self.alpha = alpha
        self._feat_cache: dict[int, np.ndarray] = {}   # per-slide feature matrix (frozen)

    # -- shared feature space (also used by the greedy policy) ---------------
    def features(self, slide: Slide) -> np.ndarray:
        """(R, d) per-region features in the configured space. Cached per slide (see _slide_cached)."""
        return _slide_cached(self._feat_cache, slide, lambda: self._build_features(slide))

    def _build_features(self, slide: Slide) -> np.ndarray:
        emb = slide.image_embeddings.astype(np.float64)
        emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)  # L2-normalize
        co = slide.region_coords.astype(np.float64)
        extent = np.clip(co.max(0) - co.min(0), 1e-9, None)
        co = (co - co.min(0)) / extent                                                # unit box
        if self.space == "image":
            feats = emb
        elif self.space == "spatial":
            feats = co
        elif self.space == "joint":
            feats = np.concatenate(
                [np.sqrt(self.alpha) * emb, np.sqrt(1.0 - self.alpha) * co], axis=1)
        else:
            raise ValueError(f"unknown IMPUTER_SPACE '{self.space}'")
        return feats

    def _neighbors(self, state: AcquisitionState) -> np.ndarray:
        """(n_unprofiled, k) indices INTO state.profiled of nearest profiled regions (euclidean
        in feature space; on L2-normalized emb this ranks the same as cosine)."""
        unpro, prof = state.unprofiled, state.profiled
        if unpro.size == 0 or prof.size == 0:
            return np.empty((0, 0), dtype=int)
        F = self.features(state.slide)
        q, p = F[unpro], F[prof]
        # squared euclidean distances (n_unpro, n_prof)
        d = ((q[:, None, :] - p[None, :, :]) ** 2).sum(axis=2)
        k_eff = min(self.k, prof.size)
        return np.argsort(d, axis=1)[:, :k_eff]

    def impute(self, state: AcquisitionState) -> np.ndarray:
        prof = state.profiled
        G = state.slide.expression.shape[1]
        if prof.size == 0:
            return np.zeros((state.unprofiled.size, G), dtype=float)
        nn = self._neighbors(state)
        if nn.size == 0:
            return np.zeros((state.unprofiled.size, G), dtype=float)
        return state.slide.expression[prof][nn].mean(axis=1)

    def uncertainty(self, state: AcquisitionState) -> np.ndarray:
        prof = state.profiled
        if prof.size == 0:
            return np.zeros(state.unprofiled.size, dtype=float)
        nn = self._neighbors(state)
        if nn.size == 0:
            return np.zeros(state.unprofiled.size, dtype=float)
        return np.var(state.slide.expression[prof][nn], axis=1).mean(axis=1)


class LearnedKrigingImputer(BaseImputer):
    """Regression-kriging (geostatistics): a LEARNED, NON-STATIONARY imputer whose posterior depends
    on the measured values — the precondition for a closed-loop adaptivity gap [Krause 2008].

      1. PCA-reduce UNI embeddings to KRIGING_PCA_DIM once per slide (cheap per-step refit).
      2. Ridge-fit image_pca -> expression on the PROFILED regions (closed form). Because this fit
         uses only measured regions, the predictor CHANGES with what we measure (non-stationary).
      3. Predict all regions from image; compute residuals on profiled; spatially krige residuals
         (mean over KRIGING_RESID_K nearest profiled regions in physical space).
      4. uncertainty(region) = variance of profiled residuals among its spatial neighbours -> a
         DATA-DEPENDENT signal (unlike stationary kNN neighbour-variance).
    """

    def __init__(self, pca_dim: int = config.KRIGING_PCA_DIM,
                 ridge_alpha: float = config.KRIGING_RIDGE_ALPHA,
                 resid_k: int = config.KRIGING_RESID_K):
        self.pca_dim = pca_dim
        self.ridge_alpha = ridge_alpha
        self.resid_k = resid_k
        self._pca_cache: dict[int, np.ndarray] = {}

    def _image_pca(self, slide: Slide) -> np.ndarray:
        def build():
            X = slide.image_embeddings.astype(np.float64)
            X = X - X.mean(0, keepdims=True)
            d = min(self.pca_dim, X.shape[1], max(1, X.shape[0] - 1))
            _, _, Vt = np.linalg.svd(X, full_matrices=False)
            Z = X @ Vt[:d].T
            return Z / np.clip(Z.std(0, keepdims=True), 1e-8, None)   # standardize
        return _slide_cached(self._pca_cache, slide, build)

    def _spatial_neighbors(self, slide: Slide, unpro, prof):
        co = slide.region_coords.astype(np.float64)
        d = ((co[unpro][:, None, :] - co[prof][None, :, :]) ** 2).sum(2)
        k = min(self.resid_k, prof.size)
        return np.argsort(d, axis=1)[:, :k]

    def _fit_predict(self, state: AcquisitionState):
        prof, unpro = state.profiled, state.unprofiled
        Y = state.slide.expression.astype(np.float64)
        G = Y.shape[1]
        if prof.size == 0:
            return np.zeros((unpro.size, G)), np.zeros(unpro.size)
        Z = self._image_pca(state.slide)
        Zp = Z[prof]
        # ridge image->expr on profiled (closed form on pca_dim => cheap): W = (ZpᵀZp+aI)^-1 ZpᵀYp
        A = Zp.T @ Zp + self.ridge_alpha * np.eye(Zp.shape[1])
        W = np.linalg.solve(A, Zp.T @ Y[prof])
        yhat_img = Z @ W                                   # image prediction for all regions
        resid_prof = Y[prof] - yhat_img[prof]              # residuals where measured
        nn = self._spatial_neighbors(state.slide, unpro, prof)
        resid_unpro = resid_prof[nn].mean(axis=1)          # krige residuals spatially
        pred = yhat_img[unpro] + resid_unpro
        unc = resid_prof[nn].var(axis=1).mean(axis=1)      # data-dependent uncertainty
        return pred, unc

    def features(self, slide: Slide) -> np.ndarray:
        """(R, d) joint features (PCA-image standardized (+) unit-box coords) — used by the greedy
        policy's error-reduction neighbourhood graph. Cached with the PCA."""
        Z = self._image_pca(slide)
        co = slide.region_coords.astype(np.float64)
        co = (co - co.min(0)) / np.clip(co.max(0) - co.min(0), 1e-9, None)
        return np.concatenate([Z, co], axis=1)

    def impute(self, state: AcquisitionState) -> np.ndarray:
        return self._fit_predict(state)[0]

    def uncertainty(self, state: AcquisitionState) -> np.ndarray:
        return self._fit_predict(state)[1]


class GraphFlowImputer(BaseImputer):
    """STFlow-INSPIRED imputer (iteration 3): the strong, non-stationary imputer that gives the
    closed loop its best possible chance. Keeps STFlow's inductive biases at a scale runnable inside
    the acquisition loop, WITHOUT a multi-week flow-matching reproduction:

      1. NONLINEAR backbone: PCA(image) -> random Fourier features (RFF) approx an RBF kernel; ridge
         image->expression refit on the PROFILED regions each step (closed form => cheap, and the
         predictor CHANGES with what we measure => non-stationary, like kriging but nonlinear).
      2. SPATIAL MESSAGE PASSING: a fixed row-normalized kNN graph on joint image+space features.
         Iterate  x <- (1-s) x + s (A x)  for FLOW_STEPS, CLAMPING profiled regions to their true
         expression every step (Dirichlet boundary). This is multi-hop neighbour-conditioning +
         iterative refinement — STFlow's core mechanism — so measured values propagate outward.
      3. DATA-DEPENDENT UNCERTAINTY: run the SAME diffusion on a profiled-indicator and read the
         "influence" reaching each region; uncertainty = 1 - influence. It shrinks near measured
         regions as they accumulate => a genuine, adaptive info signal (unlike stationary kNN).

    Honesty: this is "STFlow-class" (spatial graph, neighbour conditioning, iterative refinement,
    nonlinear backbone), NOT a 1:1 STFlow port. It is deliberately the STRONGEST imputer we can run
    in-loop, so a no-gain result here is the hardest possible test of the adaptivity hypothesis.
    """

    def __init__(self, pca_dim: int = config.FLOW_PCA_DIM, rff_dim: int = config.FLOW_RFF_DIM,
                 ridge_alpha: float = config.FLOW_RIDGE_ALPHA, graph_k: int = config.FLOW_GRAPH_K,
                 steps: int = config.FLOW_STEPS, step_size: float = config.FLOW_STEP_SIZE,
                 seed: int = config.SEED):
        self.pca_dim = pca_dim
        self.rff_dim = rff_dim
        self.ridge_alpha = ridge_alpha
        self.graph_k = graph_k
        self.steps = steps
        self.step_size = step_size
        self.seed = seed
        self._phi_cache: dict[int, np.ndarray] = {}   # RFF features per slide (frozen)
        self._adj_cache: dict = {}                      # row-normalized sparse kNN adjacency per slide
        self._feat_cache: dict[int, np.ndarray] = {}   # joint features per slide (for greedy graph)

    # -- shared joint features (also used by the greedy policy) --------------
    def features(self, slide: Slide) -> np.ndarray:
        def build():
            Z = self._image_pca(slide)
            co = slide.region_coords.astype(np.float64)
            co = (co - co.min(0)) / np.clip(co.max(0) - co.min(0), 1e-9, None)
            return np.concatenate([Z, co], axis=1)
        return _slide_cached(self._feat_cache, slide, build)

    def _image_pca(self, slide: Slide) -> np.ndarray:
        X = slide.image_embeddings.astype(np.float64)
        X = X - X.mean(0, keepdims=True)
        d = min(self.pca_dim, X.shape[1], max(1, X.shape[0] - 1))
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        Z = X @ Vt[:d].T
        return Z / np.clip(Z.std(0, keepdims=True), 1e-8, None)

    def _rff(self, slide: Slide) -> np.ndarray:
        """PCA(image) -> random Fourier features approximating an RBF kernel (nonlinear backbone).
        phi(x) = sqrt(2/D) cos(W x + b), W ~ N(0, 2*gamma), b ~ U(0, 2pi). Cached per slide."""
        return _slide_cached(self._phi_cache, slide, lambda: self._build_rff(slide))

    def _build_rff(self, slide: Slide) -> np.ndarray:
        Z = self._image_pca(slide)
        d = Z.shape[1]
        # median-heuristic bandwidth -> gamma = 1/(2 * median_dist^2)
        rng = np.random.default_rng(self.seed)
        n = Z.shape[0]
        idx = rng.choice(n, size=min(500, n), replace=False)
        S = Z[idx]
        dd = np.sqrt(np.clip(((S[:, None, :] - S[None, :, :]) ** 2).sum(2), 0, None))
        med = float(np.median(dd[np.triu_indices(len(idx), k=1)])) or 1.0
        gamma = 1.0 / (2.0 * med ** 2)
        W = rng.normal(0.0, np.sqrt(2.0 * gamma), size=(self.rff_dim, d))
        b = rng.uniform(0.0, 2.0 * np.pi, size=self.rff_dim)
        return np.sqrt(2.0 / self.rff_dim) * np.cos(Z @ W.T + b)

    def _adjacency(self, slide: Slide):
        """Fixed row-normalized kNN adjacency on joint features (data-INDEPENDENT graph; only the
        clamping is data-dependent). Symmetrized then row-normalized to a diffusion operator.
        Sparse (CSR) with a tree-based kNN: the same operator as the original dense build, which
        materialized an R x R x d distance tensor (~42 GB at 9k regions) and a dense R x R matrix."""
        return _slide_cached(self._adj_cache, slide, lambda: self._build_adjacency(slide))

    def _build_adjacency(self, slide: Slide):
        from scipy import sparse
        from sklearn.neighbors import NearestNeighbors
        F = self.features(slide)
        R = F.shape[0]
        k = int(min(self.graph_k, R - 1))
        _, idx = NearestNeighbors(n_neighbors=min(k + 1, R)).fit(F).kneighbors(F)
        notself = idx != np.arange(R)[:, None]               # drop self (first k non-self neighbours)
        keep = notself & (np.cumsum(notself, axis=1) <= k)
        nn = idx[keep].reshape(R, k)
        A = sparse.csr_matrix((np.ones(R * k), (np.repeat(np.arange(R), k), nn.ravel())), shape=(R, R))
        A = A.maximum(A.T).tocsr()                           # symmetrize
        A = sparse.diags(1.0 / np.clip(np.asarray(A.sum(1)).ravel(), 1e-12, None)) @ A   # row-normalize
        return A.tocsr()

    def _influence_uncertainty(self, state: AcquisitionState) -> np.ndarray:
        """Data-dependent uncertainty = 1 - graph influence of clamped (profiled) nodes. Needs only the
        graph, not the expression fit, so the policies' per-step uncertainty calls skip the ridge solve
        and the (R x G) diffusion."""
        prof, unpro = state.profiled, state.unprofiled
        if prof.size == 0:
            return np.zeros(unpro.size)
        A = self._adjacency(state.slide)
        s = self.step_size
        infl = np.zeros(A.shape[0]); infl[prof] = 1.0
        for _ in range(self.steps):
            infl = (1.0 - s) * infl + s * (A @ infl)
            infl[prof] = 1.0
        return np.clip(1.0 - infl[unpro], 0.0, None)

    def _fit_predict(self, state: AcquisitionState):
        prof, unpro = state.profiled, state.unprofiled
        Y = state.slide.expression.astype(np.float64)
        G = Y.shape[1]
        if prof.size == 0:
            return np.zeros((unpro.size, G)), np.zeros(unpro.size)
        phi = self._rff(state.slide)
        A = self._adjacency(state.slide)
        # (1) nonlinear ridge image->expr on profiled (closed form): W = (P^T P + aI)^-1 P^T Y
        Pp = phi[prof]
        M = Pp.T @ Pp + self.ridge_alpha * np.eye(phi.shape[1])
        Wt = np.linalg.solve(M, Pp.T @ Y[prof])
        prior = phi @ Wt                              # (R, G) image-based prior for all regions
        # (2) iterative graph refinement, clamping measured regions to truth (Dirichlet boundary)
        x = prior.copy()
        x[prof] = Y[prof]
        s = self.step_size
        for _ in range(self.steps):
            x = (1.0 - s) * x + s * (A @ x)
            x[prof] = Y[prof]                         # re-clamp measured every step
        pred = x[unpro]
        # (3) data-dependent uncertainty = 1 - graph influence of clamped nodes
        return pred, self._influence_uncertainty(state)

    def impute(self, state: AcquisitionState) -> np.ndarray:
        return self._fit_predict(state)[0]

    def uncertainty(self, state: AcquisitionState) -> np.ndarray:
        return self._influence_uncertainty(state)


def build_imputer(name: str = config.IMPUTER) -> BaseImputer:
    if name == "knn":
        return KNNImputer()
    if name == "kriging":
        return LearnedKrigingImputer()
    if name in ("graphflow", "stflow"):     # STFlow-class strong imputer (iteration 3)
        return GraphFlowImputer()
    if name == "spaim":
        raise NotImplementedError(f"{name} wrapper is a further upgrade (WS-B scale step)")
    raise ValueError(f"unknown imputer '{name}'")
