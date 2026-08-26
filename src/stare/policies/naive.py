"""Naive region-sampling baselines (no reproduction needed; implemented directly).

All are image-only / geometry-only and measured-expression-blind — floor references the
closed-loop policy should clear."""
from __future__ import annotations
import numpy as np
from .base import Policy
from ..simulator import AcquisitionState


class RandomPolicy(Policy):
    closed_loop = False

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def select(self, state: AcquisitionState):
        return int(self.rng.choice(state.unprofiled))


class GridPolicy(Policy):
    """Farthest-point (space-filling) sampling over region coordinates. Seeds at the region
    nearest the spatial centroid, then always adds the region farthest from all profiled ones."""
    closed_loop = False

    def select(self, state: AcquisitionState):
        unprofiled = state.unprofiled
        coords = state.slide.region_coords
        if state.profiled.size == 0:
            centroid = coords.mean(axis=0)
            d = np.linalg.norm(coords[unprofiled] - centroid, axis=1)
            return int(unprofiled[np.argmin(d)])
        d = np.linalg.norm(coords[unprofiled][:, None, :] - coords[state.profiled][None, :, :], axis=2)
        return int(unprofiled[np.argmax(d.min(axis=1))])


class VariancePolicy(Policy):
    """One-shot: rank regions by image-embedding variance (a morphological-complexity proxy).
    Image-only, measured-expression-blind."""
    closed_loop = False

    def __init__(self):
        self._order: list[int] | None = None

    def select(self, state: AcquisitionState):
        if self._order is None:
            score = state.slide.image_embeddings.var(axis=1)
            self._order = list(np.argsort(-score))
        for i in self._order:
            if not state.mask[i]:
                return int(i)
        raise RuntimeError("no unprofiled region left")


class KMeansImagePolicy(Policy):
    """One-shot: k-means the embeddings into (budget) clusters, visit cluster medoids in
    descending cluster-size order. Image-only, measured-expression-blind."""
    closed_loop = False

    def __init__(self, seed: int = 0):
        self.seed = seed
        self._order: list[int] | None = None

    def _build_order(self, state: AcquisitionState) -> list[int]:
        from sklearn.cluster import KMeans
        emb = state.slide.image_embeddings
        R = emb.shape[0]
        k = int(min(max(1, state.budget), R))
        km = KMeans(n_clusters=k, random_state=self.seed, n_init=10).fit(emb)
        order: list[int] = []
        # one medoid per cluster (nearest region to each centroid), largest clusters first
        for c in np.argsort(-np.bincount(km.labels_, minlength=k)):
            members = np.where(km.labels_ == c)[0]
            if members.size == 0:
                continue
            d = np.linalg.norm(emb[members] - km.cluster_centers_[c], axis=1)
            order.append(int(members[np.argmin(d)]))
        # append any leftover regions (space runs past #clusters) by index
        seen = set(order)
        order += [i for i in range(R) if i not in seen]
        return order

    def select(self, state: AcquisitionState):
        if self._order is None:
            self._order = self._build_order(state)
        for i in self._order:
            if not state.mask[i]:
                return int(i)
        raise RuntimeError("no unprofiled region left")
