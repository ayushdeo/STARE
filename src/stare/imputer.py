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


def build_imputer(name: str = config.IMPUTER) -> BaseImputer:
    if name == "knn":
        return KNNImputer()
    if name in ("stflow", "spaim"):
        raise NotImplementedError(f"{name} wrapper is a post-diagnostic upgrade (WS-B scale step)")
    raise ValueError(f"unknown imputer '{name}'")
