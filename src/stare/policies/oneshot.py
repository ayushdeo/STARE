"""
SpatialFinder-style one-shot region selection — THE primary competitor (WS-C / Tier-2 port).

Scores ALL regions once from the image (a diversity/representativeness heuristic), picks
top-K greedily, and does NOT condition on measured expression (no closed loop). This is the
method our closed-loop policy must beat at equal budget (SOURCE_OF_TRUTH §6 go/no-go).

Reference: SpatialFinder, Frontiers Bioinf. 2026 (docs/KNOWLEDGE_BASE.md #23). SpatialFinder's
native score = UMAP-embedding diversity + Shannon entropy of cell composition (from a
StarDist+PLIP+XGBoost cell classifier) + targeted text-image relevance.

HARMONIZED SIMPLIFICATION (log in experiments/deviations.md): for the Tier-2 port we replace
the cell-classifier composition-entropy term with an image-embedding diversity proxy —
k-means the frozen embeddings into C clusters and reward (a) rare-cluster membership and
(b) distance to the cluster centroid (within-cluster heterogeneity). This preserves
SpatialFinder's DIVERSITY intent and its ONE-SHOT, image-only, measured-expression-blind
nature (the properties our method is designed to beat), without the heavy cell pipeline.
A Tier-1 native sanity repro of their rho is done separately before trusting this port.
"""
from __future__ import annotations
import numpy as np
from .base import Policy
from ..simulator import AcquisitionState


class OneShotSpatialFinderPolicy(Policy):
    closed_loop = False   # the defining contrast vs GreedyInfoGainPolicy

    def __init__(self, n_clusters: int = 8, seed: int = 0):
        self.n_clusters = n_clusters
        self.seed = seed
        self._order: list[int] | None = None   # precomputed one-shot ranking (region indices)

    def _score_all_regions(self, state: AcquisitionState) -> np.ndarray:
        """(R,) diversity score from image embeddings ONLY (no measured expression)."""
        from sklearn.cluster import KMeans
        emb = state.slide.image_embeddings
        R = emb.shape[0]
        k = int(min(self.n_clusters, max(1, R)))
        km = KMeans(n_clusters=k, random_state=self.seed, n_init=10).fit(emb)
        labels = km.labels_
        centers = km.cluster_centers_
        counts = np.bincount(labels, minlength=k).astype(float)

        # (a) rare-cluster bonus: 1/size of the region's cluster (diverse/rare morphologies rank high)
        rare = 1.0 / counts[labels]
        # (b) within-cluster heterogeneity: distance to own cluster centroid
        dist = np.linalg.norm(emb - centers[labels], axis=1)

        def z(x):
            s = x.std()
            return (x - x.mean()) / s if s > 1e-12 else np.zeros_like(x)

        return z(rare) + z(dist)

    def select(self, state: AcquisitionState):
        if self._order is None:                       # score once, up front (ONE-SHOT)
            scores = self._score_all_regions(state)
            self._order = list(np.argsort(-scores))
        for i in self._order:
            if not state.mask[i]:
                return int(i)
        raise RuntimeError("no unprofiled region left")
