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

STFlow/SpaIM wrappers remain a post-diagnostic upgrade. See docs/KNOWLEDGE_BASE.md #5/#6/#9.
"""
from __future__ import annotations
import numpy as np

from . import config
from .simulator import AcquisitionState, Slide


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
        """(R, d) per-region features in the configured space. Cached per slide id()."""
        key = id(slide)
        if key in self._feat_cache:
            return self._feat_cache[key]
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
        self._feat_cache[key] = feats
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
        key = id(slide)
        if key in self._pca_cache:
            return self._pca_cache[key]
        X = slide.image_embeddings.astype(np.float64)
        X = X - X.mean(0, keepdims=True)
        d = min(self.pca_dim, X.shape[1], max(1, X.shape[0] - 1))
        _, _, Vt = np.linalg.svd(X, full_matrices=False)
        Z = X @ Vt[:d].T
        Z = Z / np.clip(Z.std(0, keepdims=True), 1e-8, None)   # standardize
        self._pca_cache[key] = Z
        return Z

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


def build_imputer(name: str = config.IMPUTER) -> BaseImputer:
    if name == "knn":
        return KNNImputer()
    if name == "kriging":
        return LearnedKrigingImputer()
    if name in ("stflow", "spaim"):
        raise NotImplementedError(f"{name} wrapper is a further upgrade (WS-B scale step)")
    raise ValueError(f"unknown imputer '{name}'")
