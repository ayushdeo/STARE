"""
Greedy expected-information-gain policy — OUR contribution, feasibility version (WS-D).

CLOSED LOOP: at each round, pick the unprofiled region whose measurement is expected to
most reduce whole-slide imputation uncertainty, conditioning on the image AND on the
expression measured so far (state.measured_expression()). This is the differentiator vs
the one-shot SpatialFinder competitor (SOURCE_OF_TRUTH §4, §6).

Feasibility criterion (cheap, 4070-friendly): use the frozen imputer's per-region
uncertainty (kNN neighbor variance) as the acquisition score; optionally spatial-diversity
penalty to avoid clustering picks. Later: proper expected-reduction / RL (AFA-RL template).
STATUS: SKELETON.
"""
from __future__ import annotations
import numpy as np
from .base import Policy
from ..simulator import AcquisitionState


class GreedyInfoGainPolicy(Policy):
    closed_loop = True   # <- reads measured expression; the whole point

    def __init__(self, imputer, use_image: bool = True, use_spatial: bool = True,
                 use_measured: bool = True):
        # Ablation switches for the WS-E decisive experiments:
        self.imputer = imputer
        self.use_image = use_image        # ablate image conditioning
        self.use_spatial = use_spatial    # ablate spatial structure
        self.use_measured = use_measured  # ablate the CLOSED LOOP (False == one-shot-like)
        if not use_measured:
            self.closed_loop = False

    def select(self, state: AcquisitionState):
        """Score each unprofiled region by expected information gain; return argmax.
        TODO: uncertainty = imputer.uncertainty(state); if use_spatial, add diversity term;
        if not use_measured, ignore state.measured_expression() (image-only fallback)."""
        raise NotImplementedError(
            "Stage-0 TODO: implement greedy EIG using imputer.uncertainty(state) "
            "(+ optional spatial-diversity term); honor the three ablation switches."
        )
