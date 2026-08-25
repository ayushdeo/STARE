"""
Frozen imputer = the environment's reward + uncertainty model (WS-B).

Reconstructs UNPROFILED regions' expression from PROFILED regions (+ image). Shared,
frozen, and identical across every policy so comparisons are fair (REPRODUCTION_PLAN §6).

Feasibility default (config.IMPUTER == "knn"): kNN regression in image-embedding space,
reading measured expression of the k nearest profiled regions. Neighbor disagreement =
predictive variance = the information-gain signal the greedy policy consumes.

STATUS: kNN skeleton is straightforward and 4070-friendly; STFlow/SpaIM wrappers are TODO
after the go/no-go gate. See docs/KNOWLEDGE_BASE.md #5 (STFlow), #6 (TISSUE), #9 (Stem).
"""
from __future__ import annotations
import numpy as np

from . import config
from .simulator import AcquisitionState


class BaseImputer:
    def impute(self, state: AcquisitionState) -> np.ndarray:
        """Return (n_unprofiled, G) predicted expression for state.unprofiled."""
        raise NotImplementedError

    def uncertainty(self, state: AcquisitionState) -> np.ndarray:
        """Return (n_unprofiled,) predictive uncertainty per unprofiled region
        (higher = model less sure). Consumed by greedy expected-information-gain policy."""
        raise NotImplementedError


class KNNImputer(BaseImputer):
    """kNN in frozen image-embedding space; predict = mean expression of the k nearest
    PROFILED regions; uncertainty = variance across those k neighbors."""

    def __init__(self, k: int = config.IMPUTER_KNN_K, metric: str = config.IMPUTER_KNN_METRIC):
        self.k = k
        self.metric = metric

    def _neighbors(self, state: AcquisitionState):
        """Return (n_unprofiled, k) indices into state.profiled of nearest profiled regions.
        TODO: cosine kNN between image_embeddings[unprofiled] and [profiled]."""
        raise NotImplementedError("Stage-0 TODO: cosine kNN in image-embedding space")

    def impute(self, state: AcquisitionState) -> np.ndarray:
        raise NotImplementedError("Stage-0 TODO: mean of measured_expression over neighbors")

    def uncertainty(self, state: AcquisitionState) -> np.ndarray:
        raise NotImplementedError("Stage-0 TODO: neighbor variance as info-gain signal")


def build_imputer(name: str = config.IMPUTER) -> BaseImputer:
    if name == "knn":
        return KNNImputer()
    if name in ("stflow", "spaim"):
        raise NotImplementedError(f"{name} wrapper is a post-gate upgrade (WS-B scale step)")
    raise ValueError(f"unknown imputer '{name}'")
