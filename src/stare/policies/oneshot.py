"""
SpatialFinder-style one-shot region selection — THE primary competitor (WS-C / Tier-2 port).

Scores ALL regions once from the image (diversity + relevance heuristic), picks top-K greedily,
and does NOT condition on measured expression (no closed loop). This is the method our
closed-loop policy must beat at equal budget (SOURCE_OF_TRUTH §6 go/no-go).

Reference: SpatialFinder, Frontiers Bioinf. 2026 (docs/KNOWLEDGE_BASE.md #23).
Native code is an informal gist; Tier-1 sanity repro first, then this harmonized port.
STATUS: SKELETON.
"""
from __future__ import annotations
import numpy as np
from .base import Policy
from ..simulator import AcquisitionState


class OneShotSpatialFinderPolicy(Policy):
    closed_loop = False   # <- the defining contrast vs GreedyInfoGainPolicy

    def __init__(self, diversity_weight: float = 0.5):
        self.diversity_weight = diversity_weight
        self._order: list[int] | None = None   # precomputed one-shot ranking

    def _score_all_regions(self, state: AcquisitionState) -> np.ndarray:
        """(R,) SpatialFinder hybrid score from image embeddings ONLY:
        diversity (embedding-space distance + cell-composition entropy) + targeted relevance.
        TODO: port from paper/gist. Cell-composition needs a cell classifier (StarDist+PLIP+XGB)
        OR a simplified proxy for the harmonized comparison — log the choice in deviations.md."""
        raise NotImplementedError("Stage-0 TODO: SpatialFinder scoring (image-only, one-shot)")

    def select(self, state: AcquisitionState):
        if self._order is None:
            scores = self._score_all_regions(state)
            self._order = list(np.argsort(-scores))
        for i in self._order:
            if not state.mask[i]:
                return int(i)
        raise RuntimeError("no unprofiled region left")
