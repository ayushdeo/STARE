"""
Greedy expected-information-gain policy — OUR contribution (WS-D).

CLOSED LOOP: at each round pick the unprofiled region whose measurement is expected to most
reduce whole-slide imputation error, conditioning on the image AND on the expression measured
so far (via the frozen imputer's uncertainty, computed in the imputer's feature space). This is
the differentiator vs the one-shot SpatialFinder competitor (SOURCE_OF_TRUTH §4, §6).

Two criteria (config.GREEDY_CRITERION):
  "uncertainty"     (v0): pick the single most-uncertain region. Tends to chase outliers ->
                          hurts coverage; the first sweep showed this loses to k-means.
  "error_reduction" (v1): pick the region that most reduces error over its NEIGHBOURHOOD, i.e.
                          argmax_r  sum_{j in unprofiled near r} uncertainty(j).  This is the
                          classic active-learning / Bayesian-experimental-design objective and
                          rewards measuring a region that anchors many uncertain neighbours.

Ablation switches drive the WS-E decisive experiments:
  use_measured : condition on measured expression (the CLOSED LOOP). False => image-only novelty
                 score (a one-shot-like control isolating the loop's value).
  use_image    : let image-embedding novelty contribute (uncertainty criterion only).
  use_spatial  : add a spatial-coverage term (uncertainty criterion only).
"""
from __future__ import annotations
import numpy as np
from .base import Policy
from ..simulator import AcquisitionState
from .. import config


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = x.std()
    return np.zeros_like(x) if s < 1e-12 else (x - x.mean()) / s


class GreedyInfoGainPolicy(Policy):
    closed_loop = True

    def __init__(self, imputer, use_image: bool = True, use_spatial: bool = True,
                 use_measured: bool = True, criterion: str = config.GREEDY_CRITERION,
                 er_k: int = config.GREEDY_ER_K):
        self.imputer = imputer
        self.use_image = use_image
        self.use_spatial = use_spatial
        self.use_measured = use_measured
        self.criterion = criterion
        self.er_k = er_k
        self.closed_loop = bool(use_measured)

    # -- image-only fallback terms (uncertainty criterion / measured off) ----
    def _image_novelty(self, state, unpro):
        emb = state.slide.image_embeddings
        q, p = emb[unpro], emb[state.profiled]
        qn = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
        pn = p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-12, None)
        return 1.0 - (qn @ pn.T).max(axis=1)

    def _spatial_coverage(self, state, unpro):
        co = state.slide.region_coords
        d = np.linalg.norm(co[unpro][:, None, :] - co[state.profiled][None, :, :], axis=2)
        return d.min(axis=1)

    def _cold_start(self, state, unpro):
        """No profiled region yet: seed at the image medoid (most representative region)."""
        emb = state.slide.image_embeddings
        en = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
        centrality = en @ en.mean(axis=0)
        return int(unpro[np.argmax(centrality[unpro])])

    # -- selection -----------------------------------------------------------
    def select(self, state: AcquisitionState):
        unpro = state.unprofiled
        if unpro.size == 0:
            raise RuntimeError("no unprofiled region left")
        if state.profiled.size == 0:
            return self._cold_start(state, unpro)

        if self.criterion == "error_reduction" and self.use_measured:
            return int(unpro[self._error_reduction_scores(state, unpro).argmax()])

        # --- "uncertainty" criterion (v0) / image-only control ---
        score = np.zeros(unpro.size)
        if self.use_measured:
            score += _zscore(self.imputer.uncertainty(state))
        elif self.use_image:
            score += _zscore(self._image_novelty(state, unpro))
        if self.use_image and self.use_measured:
            score += 0.25 * _zscore(self._image_novelty(state, unpro))
        if self.use_spatial:
            score += _zscore(self._spatial_coverage(state, unpro))
        return int(unpro[score.argmax()])

    def _error_reduction_scores(self, state, unpro):
        """score(r) = sum over unprofiled neighbours j of r (in the imputer's feature space) of
        uncertainty(j). Measuring r anchors those neighbours -> expected total error reduction.
        O(U * k) via a kNN graph on the unprofiled set."""
        from sklearn.neighbors import NearestNeighbors
        unc = self.imputer.uncertainty(state)               # (U,) aligned with unpro
        F = self.imputer.features(state.slide)[unpro]       # unprofiled features (same space as imputer)
        k = int(min(self.er_k, len(unpro)))
        nn = NearestNeighbors(n_neighbors=k).fit(F)
        _, idx = nn.kneighbors(F)                           # (U, k) neighbours among unprofiled
        # r's benefit = total uncertainty of the neighbours it would help impute
        return unc[idx].sum(axis=1)
