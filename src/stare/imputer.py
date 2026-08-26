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
        """Return (n_unprofiled, k) indices into state.profiled of nearest profiled regions."""
        unprofiled = state.unprofiled
        profiled = state.profiled
        if unprofiled.size == 0 or profiled.size == 0:
            return np.empty((0, 0), dtype=int)

        q = state.slide.image_embeddings[unprofiled]
        p = state.slide.image_embeddings[profiled]
        q_norm = np.linalg.norm(q, axis=1, keepdims=True)
        p_norm = np.linalg.norm(p, axis=1, keepdims=True)
        denom = q_norm * p_norm.T
        sims = np.divide(q @ p.T, np.clip(denom, a_min=1e-12, a_max=None), out=np.zeros_like(q @ p.T), where=denom > 0)
        d = 1.0 - sims
        k_eff = min(self.k, profiled.size)
        if k_eff <= 0:
            return np.empty((unprofiled.size, 0), dtype=int)
        order = np.argsort(d, axis=1)[:, :k_eff]
        return order

    def impute(self, state: AcquisitionState) -> np.ndarray:
        profiled = state.profiled
        if profiled.size == 0:
            return np.zeros((state.unprofiled.size, state.slide.expression.shape[1]), dtype=float)

        nn = self._neighbors(state)
        if nn.size == 0:
            return np.zeros((state.unprofiled.size, state.slide.expression.shape[1]), dtype=float)

        expr = state.slide.expression[profiled]
        preds = expr[nn]
        return preds.mean(axis=1)

    def uncertainty(self, state: AcquisitionState) -> np.ndarray:
        profiled = state.profiled
        if profiled.size == 0:
            return np.zeros(state.unprofiled.size, dtype=float)

        nn = self._neighbors(state)
        if nn.size == 0:
            return np.zeros(state.unprofiled.size, dtype=float)

        expr = state.slide.expression[profiled]
        neighbor_expr = expr[nn]
        var = np.var(neighbor_expr, axis=1)
        return var.mean(axis=1)


def build_imputer(name: str = config.IMPUTER) -> BaseImputer:
    if name == "knn":
        return KNNImputer()
    if name in ("stflow", "spaim"):
        raise NotImplementedError(f"{name} wrapper is a post-gate upgrade (WS-B scale step)")
    raise ValueError(f"unknown imputer '{name}'")
