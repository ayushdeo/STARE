"""
Feedback ablation (revision round 2, reviewer concern 4): run a policy unchanged, but let it see measured
values that carry no information about the regions it measured.

  mode="permute":  the policy sees the expression of a randomly chosen OTHER region for every region it
                   profiles (rows permuted once per slide with a fixed seed). The values keep their realistic
                   marginal distribution but are unrelated to where they were measured.
  mode="constant": the policy sees the same expression vector (the slide mean) everywhere, so any term built
                   on measured variation is constant and only the policy's exploration rule remains.

The simulator, imputer, and scoring still use the true slide; only the policy's view is replaced. The
difference between a policy and its permuted version is the contribution of informative feedback, with the
selection rule held fixed.
"""
from __future__ import annotations
import numpy as np
from dataclasses import replace
from .base import Policy
from ..simulator import AcquisitionState
from .. import config


class FeedbackAblation(Policy):
    closed_loop = False

    def __init__(self, inner: Policy, mode: str = "permute", seed: int = config.SEED):
        if mode not in ("permute", "constant"):
            raise ValueError(mode)
        self.inner, self.mode, self.seed = inner, mode, seed
        self._shadow = None      # (true slide, shadow slide)

    def _shadow_slide(self, slide):
        if self._shadow is not None and self._shadow[0] is slide:
            return self._shadow[1]
        X = slide.expression
        if self.mode == "permute":
            perm = np.random.default_rng(self.seed).permutation(X.shape[0])
            Xs = X[perm]
        else:
            Xs = np.broadcast_to(X.mean(0, keepdims=True), X.shape).copy()
        sh = replace(slide, expression=Xs)
        self._shadow = (slide, sh)
        return sh

    def select(self, state: AcquisitionState):
        sh = self._shadow_slide(state.slide)
        return self.inner.select(AcquisitionState(slide=sh, profiled=state.profiled, mask=state.mask,
                                                  budget=state.budget))
