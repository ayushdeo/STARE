"""Naive region-sampling baselines (no reproduction needed; implemented directly).

All are image-only / geometry-only and measured-expression-blind — floor references the
closed-loop policy should clear."""
from __future__ import annotations
import numpy as np
from .base import Policy
from ..simulator import AcquisitionState
from .. import config


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
        # min distance to the profiled set, updated once per newly revealed region (no U x t matrix)
        prof = [int(i) for i in state.profiled]
        if getattr(self, "_mind", None) is None or prof[:len(self._done)] != self._done:
            self._mind, self._done = np.full(coords.shape[0], np.inf), []
        for j in prof[len(self._done):]:
            np.minimum(self._mind, np.linalg.norm(coords - coords[j], axis=1), out=self._mind)
            self._done.append(j)
        return int(unprofiled[np.argmax(self._mind[unprofiled])])


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


_KMEANS_ORDERS: dict = {}     # (id(slide), budget, seed) -> (slide, order)


def _disk_memo(state, seed, build):
    """Persist k-means orders under cache/placements so separate scripts share them. Keyed by sample
    id, unit mode, region count, budget and seed; synthetic slides (no sample id) are not persisted."""
    import json
    from pathlib import Path
    sl = state.slide
    if not sl.patient_id:
        return build(state)
    mode = (sl.meta or {}).get("region_mode", "legacy")
    p = Path(config.CACHE_ROOT) / "placements" / f"{sl.patient_id}_{mode}_R{sl.n_regions}_kmeans_b{int(state.budget)}_s{seed}.json"
    if p.exists():
        return [int(i) for i in json.loads(p.read_text())]
    order = build(state)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps([int(i) for i in order]))
    return order


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
            # memoized per (slide, budget, seed): the order is measurement-blind, so it is identical
            # across imputers and repeated runs (k-means with n_init=10 dominates the sweep's cost)
            key = (id(state.slide), int(state.budget), self.seed)
            hit = _KMEANS_ORDERS.get(key)
            if hit is None or hit[0] is not state.slide:
                hit = _KMEANS_ORDERS[key] = (state.slide, _disk_memo(state, self.seed, self._build_order))
            self._order = hit[1]
        for i in self._order:
            if not state.mask[i]:
                return int(i)
        raise RuntimeError("no unprofiled region left")
