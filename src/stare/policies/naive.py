"""Naive region-sampling baselines (no reproduction needed; implement directly)."""
from __future__ import annotations
import numpy as np
from .base import Policy
from ..simulator import AcquisitionState


class RandomPolicy(Policy):
    closed_loop = False

    def __init__(self, seed: int = 0):
        self.rng = np.random.default_rng(seed)

    def select(self, state: AcquisitionState):
        u = state.unprofiled
        return int(self.rng.choice(u))


class GridPolicy(Policy):
    """Uniform/space-filling selection over the region grid. TODO: farthest-point order."""
    closed_loop = False

    def select(self, state: AcquisitionState):
        raise NotImplementedError("Stage-0 TODO: farthest-point / uniform grid order over coords")


class VariancePolicy(Policy):
    """Greedy by image-embedding-predicted expression variance (static, image-only)."""
    closed_loop = False

    def select(self, state: AcquisitionState):
        raise NotImplementedError("Stage-0 TODO: rank unprofiled by a static variance proxy")


class KMeansImagePolicy(Policy):
    """Cluster regions in image-embedding space; pick cluster medoids (one-shot, image-only)."""
    closed_loop = False

    def select(self, state: AcquisitionState):
        raise NotImplementedError("Stage-0 TODO: k-means medoids over image_embeddings")
