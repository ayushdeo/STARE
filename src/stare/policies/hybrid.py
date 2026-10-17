"""
Coverage-plus-feedback hybrid (revision round 3, point 1).

The first ceil(frac x budget) regions come from a fixed non-adaptive coverage rule; the remaining regions are
chosen by a feedback policy that conditions on everything measured so far. Because the coverage rules used here
(k-means, MI-greedy, variance-greedy, one-shot) select regions in a fixed order, the coverage part is exactly the
prefix of that rule's own full-budget selection. The hybrid and the coverage rule therefore differ only in how the
last (1 - frac) of the budget is spent, which is the contribution of feedback layered on a strong design.
"""
from __future__ import annotations
import math
from .base import Policy
from ..simulator import AcquisitionState


class HybridPolicy(Policy):
    closed_loop = True

    def __init__(self, coverage: Policy, feedback: Policy, frac: float):
        if not 0.0 < frac < 1.0:
            raise ValueError("frac must be in (0, 1)")
        self.coverage, self.feedback, self.frac = coverage, feedback, frac

    def select(self, state: AcquisitionState):
        n_cov = math.ceil(self.frac * state.budget)
        if len(state.profiled) < n_cov:
            return self.coverage.select(state)
        return self.feedback.select(state)
