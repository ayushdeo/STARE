"""
Value-conditioned acquisition policy (revision, reviewer point 2 + positive control).

The reviewer correctly noted that our earlier "closed-loop" policies did not actually condition the
SELECTION on the measured expression VALUES: the graphflow uncertainty (1 - graph influence) depends
only on WHICH regions are profiled, not on their values, and the image-only ablation swapped the whole
selection rule rather than toggling measured values. This policy makes the dependence on measured
expression explicit and auditable.

Selection score for a candidate region i, given the profiled set A_t and its measured expression y_{A_t}:

    s(i | A_t, y_{A_t}, X) = zscore_i[ exploit(i) ]  +  beta * zscore_i[ explore(i) ]

    exploit(i) = local weighted VARIANCE of measured expression around i                # USES y_{A_t}
               = mean_g [ (K y^2 / K1)_i - ((K y / K1)_i)^2 ]
    explore(i) = min_{j in A_t} dist(coord_i, coord_j)                          # spatial coverage
    K(x_i, x_j) = exp( - ||coord_i - coord_j||^2 / (2 ell^2) )                  # spatial kernel
    K1_i       = sum_{j in A_t} K(x_i, x_j)

`exploit` is large where the already-measured field is locally ROUGH (high local variance), i.e. where
the reconstruction is hardest and dense sampling helps most. It enters through y_{A_t}, the measured
values. `explore` keeps coverage so the policy does not collapse onto one region. The next region is
arg max_i s(i). With beta large this reduces to farthest-point coverage (a value-independent control);
with beta small it is driven by the measured values (roughness-seeking).

This is the policy used for the positive control (scripts/07_positive_control.py): in a synthetic
setting with image-unpredictable, spatially clustered structure, a value-conditioned policy should
allocate measurements toward that structure and reconstruct it better than value-independent placement.
"""
from __future__ import annotations
import numpy as np
from .base import Policy
from ..simulator import AcquisitionState
from .. import config


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = x.std()
    return np.zeros_like(x) if s < 1e-12 else (x - x.mean()) / s


class ValueGreedyPolicy(Policy):
    """Adaptive policy whose selection score depends explicitly on measured expression values."""
    closed_loop = True

    def __init__(self, beta: float = 0.5, lengthscale_frac: float = 0.1, seed: int = 0):
        self.beta = beta                       # weight on the coverage (explore) term
        self.lengthscale_frac = lengthscale_frac  # spatial kernel width as a fraction of slide extent
        self.rng = np.random.default_rng(seed)

    def _cold_start(self, state, unpro):
        co = state.slide.region_coords
        centroid = co.mean(0)
        d = np.linalg.norm(co[unpro] - centroid, axis=1)
        return int(unpro[np.argmin(d)])

    def select(self, state: AcquisitionState):
        unpro = state.unprofiled
        if unpro.size == 0:
            raise RuntimeError("no unprofiled region left")
        if state.profiled.size == 0:
            return self._cold_start(state, unpro)

        co = state.slide.region_coords.astype(np.float64)
        extent = float(np.linalg.norm(co.max(0) - co.min(0)))
        ell = max(extent * self.lengthscale_frac, 1e-6)

        prof = state.profiled
        y = state.measured_expression().astype(np.float64)      # (t, G) measured values  <- y_{A_t}

        # spatial kernel between every unprofiled candidate and every profiled region
        d2 = ((co[unpro][:, None, :] - co[prof][None, :, :]) ** 2).sum(2)   # (U, t)
        K = np.exp(-d2 / (2.0 * ell ** 2))
        K1 = np.clip(K.sum(1, keepdims=True), 1e-12, None)      # (U,1)
        Ey = (K @ y) / K1                                       # (U,G) local weighted mean of measured
        Ey2 = (K @ (y ** 2)) / K1                               # (U,G) local weighted 2nd moment
        exploit = np.clip(Ey2 - Ey ** 2, 0.0, None).mean(1)     # (U,) local measured VARIANCE (uses y)
        explore = np.sqrt(d2).min(axis=1)                       # (U,) coverage term

        score = _zscore(exploit) + self.beta * _zscore(explore)
        return int(unpro[int(np.argmax(score))])
