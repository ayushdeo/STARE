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
        self._nbr = None          # (slide, extended all-region neighbour lists) for error reduction
        self._acc = None          # running max-similarity / min-distance to the profiled set

    # -- image-only fallback terms (uncertainty criterion / measured off) ----
    def _running(self, state):
        """Max cosine similarity and min spatial distance of every region to the profiled set, updated
        once per newly revealed region (profiled only grows in an episode). Same values as the per-step
        U x t recomputation (whose image term is U x t x 1536)."""
        prof = [int(i) for i in state.profiled]
        acc = self._acc
        if acc is None or acc["slide"] is not state.slide or prof[:len(acc["done"])] != acc["done"]:
            emb = state.slide.image_embeddings          # original dtype, as before
            R = emb.shape[0]
            acc = self._acc = {"slide": state.slide, "done": [],
                               "en": emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None),
                               "co": state.slide.region_coords,
                               "maxsim": np.full(R, -np.inf), "mind": np.full(R, np.inf)}
        for j in prof[len(acc["done"]):]:
            np.maximum(acc["maxsim"], acc["en"] @ acc["en"][j], out=acc["maxsim"])
            np.minimum(acc["mind"], np.linalg.norm(acc["co"] - acc["co"][j], axis=1), out=acc["mind"])
            acc["done"].append(j)
        return acc

    def _image_novelty(self, state, unpro):
        return 1.0 - self._running(state)["maxsim"][unpro]

    def _spatial_coverage(self, state, unpro):
        return self._running(state)["mind"][unpro]

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

    def _neighbour_lists(self, slide, width):
        """All-region kNN lists in the imputer's feature space (self first), `width` long. Cached."""
        if self._nbr is not None and self._nbr[0] is slide and self._nbr[1].shape[1] >= width:
            return self._nbr[1]
        from sklearn.neighbors import NearestNeighbors
        F = self.imputer.features(slide)
        width = int(min(width, F.shape[0]))
        _, idx = NearestNeighbors(n_neighbors=width).fit(F).kneighbors(F)
        self._nbr = (slide, idx)
        return idx

    def _error_reduction_scores(self, state, unpro):
        """score(r) = sum over unprofiled neighbours j of r (in the imputer's feature space) of
        uncertainty(j). Measuring r anchors those neighbours -> expected total error reduction.
        Neighbours of r = its k nearest among the UNPROFILED set (r itself included, as before).
        Computed by filtering cached all-region kNN lists rather than refitting a kNN index on the
        unprofiled set every step; the lists are widened and recomputed whenever a row runs short,
        so the neighbour sets are the same as a fresh fit."""
        unc = self.imputer.uncertainty(state)               # (U,) aligned with unpro
        R = state.slide.n_regions
        k = int(min(self.er_k, len(unpro)))
        width = k + 64
        while True:
            L = self._neighbour_lists(state.slide, width)[unpro]          # (U, width) region ids
            ok = ~state.mask[L]
            c = np.cumsum(ok, axis=1)
            if c[:, -1].min() >= k or L.shape[1] >= R:
                break
            width *= 2
        take = ok & (c <= k)
        pos = np.full(R, -1); pos[unpro] = np.arange(unpro.size)
        idx = pos[L[take]].reshape(unpro.size, k)          # positions in unpro, nearest first
        return unc[idx].sum(axis=1)
