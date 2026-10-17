"""
Value-conditioned acquisition policy (revision point 2 + positive control).

A critique of the draft correctly noted that our earlier "closed-loop" policies did not actually condition the
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
        self._acc = None                       # incremental kernel sums (see _accumulate)

    def _accumulate(self, state):
        """Running kernel sums over the profiled set, updated by one rank-1 term per newly revealed
        region (the profiled list only grows within an episode):
            S0_i = sum_j K_ij,  S1_i = sum_j K_ij y_j,  S2_i = sum_j K_ij y_j^2,  mind2_i = min_j d_ij^2.
        Same quantities as recomputing K (U x t) @ y (t x G) every step, at O(R G) per step instead of
        O(R t G) -- the difference between minutes and days on 6-9k-spot Visium slides."""
        prof = [int(i) for i in state.profiled]
        acc = self._acc
        if acc is None or acc["slide"] is not state.slide or prof[:len(acc["done"])] != acc["done"]:
            co = state.slide.region_coords.astype(np.float64)
            R, G = state.slide.expression.shape
            acc = self._acc = {"slide": state.slide, "done": [], "co": co,
                               "ell": max(float(np.linalg.norm(co.max(0) - co.min(0))) * self.lengthscale_frac, 1e-6),
                               "S0": np.zeros(R), "S1": np.zeros((R, G)), "S2": np.zeros((R, G)),
                               "mind2": np.full(R, np.inf)}
        co, ell = acc["co"], acc["ell"]
        for j in prof[len(acc["done"]):]:
            d2 = ((co - co[j]) ** 2).sum(1)                   # (R,)
            k = np.exp(-d2 / (2.0 * ell ** 2))
            y = state.slide.expression[j].astype(np.float64)  # measured y_j  <- y_{A_t}
            acc["S0"] += k
            acc["S1"] += k[:, None] * y[None, :]
            acc["S2"] += k[:, None] * (y ** 2)[None, :]
            np.minimum(acc["mind2"], d2, out=acc["mind2"])
            acc["done"].append(j)
        return acc

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

        acc = self._accumulate(state)
        S0 = np.clip(acc["S0"][unpro], 1e-12, None)[:, None]   # (U,1)  K1
        Ey = acc["S1"][unpro] / S0                              # (U,G) local weighted mean of measured
        Ey2 = acc["S2"][unpro] / S0                             # (U,G) local weighted 2nd moment
        exploit = np.clip(Ey2 - Ey ** 2, 0.0, None).mean(1)     # (U,) local measured VARIANCE (uses y)
        explore = np.sqrt(acc["mind2"][unpro])                  # (U,) coverage term

        score = _zscore(exploit) + self.beta * _zscore(explore)
        return int(unpro[int(np.argmax(score))])
