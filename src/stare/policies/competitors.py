"""
Harmonized reimplementations of the published adaptive-sampling competitors, ported into our simulator
so they can be compared head-to-head under matched imputer + matched objectives (reconstruction AND
detection). These follow each method's STATED acquisition objective (quoted below); they are NOT the
authors' trained models. For native reproduction of the authors' code and reported numbers, see
docs/COMPETITOR_REPRODUCTION.md.

SCR2-ST (Zhu et al., MIDL 2026; github hrlblab/SCR2ST). Its RL sampler reward combines, verbatim from
the repo: "single-cell manifold coverage", "cell type diversity", and "spatial uniformity". This is a
DIVERSITY + COVERAGE objective and is NON-ADAPTIVE (it does not condition on measured expression). We
port it as farthest-point coverage in the JOINT (image-embedding (+) spatial) feature space -- image
embeddings stand in for the single-cell manifold / cell-state axis, coordinates for spatial uniformity.

SOFisher (Ng et al., Nat. Commun. 2026; github Heathcliff-Ng/SOFisher). A DQN that "harnesses the
knowledge gained from the sequence of previously sampled FOVs to guide the selection of the next FOV" to
"capture more regions of interest". This is an ROI-DETECTION objective and IS ADAPTIVE. We port it as a
greedy ROI-seeking policy: from the measured FOVs, estimate where region-of-interest signal (local
expression heterogeneity) is high and move toward it, with an exploration term (the DQN explore/exploit
trade-off). This conditions on measured expression.
"""
from __future__ import annotations
import numpy as np
from .base import Policy
from ..simulator import AcquisitionState
from .. import config


def _joint_feats(slide):
    emb = slide.image_embeddings.astype(np.float64)
    emb = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
    co = slide.region_coords.astype(np.float64)
    co = (co - co.min(0)) / np.clip(co.max(0) - co.min(0), 1e-9, None)
    return np.concatenate([emb, co], axis=1)


class SCR2STDiversityPolicy(Policy):
    """Harmonized SCR2-ST sampler: farthest-point coverage in joint image+space (manifold coverage +
    cell-type diversity + spatial uniformity). NON-ADAPTIVE."""
    closed_loop = False

    def __init__(self):
        self._F = None
        self._mind2, self._done = None, []

    def select(self, state: AcquisitionState):
        unpro = state.unprofiled
        if unpro.size == 0:
            raise RuntimeError("no unprofiled region left")
        if self._F is None:
            self._F = _joint_feats(state.slide)
        F = self._F
        if state.profiled.size == 0:
            # seed at the medoid (most representative point)
            c = F.mean(0, keepdims=True)
            return int(unpro[np.argmin(((F[unpro] - c) ** 2).sum(1))])
        # min squared distance to the profiled set, updated once per newly revealed region (same values
        # as the full U x t x d recomputation, without the temporary)
        prof = [int(i) for i in state.profiled]
        if self._mind2 is None or prof[:len(self._done)] != self._done:
            self._mind2, self._done = np.full(F.shape[0], np.inf), []
        for j in prof[len(self._done):]:
            np.minimum(self._mind2, ((F - F[j]) ** 2).sum(1), out=self._mind2)
            self._done.append(j)
        return int(unpro[np.argmax(self._mind2[unpro])])   # farthest-point: maximize diversity+spread


class SOFisherROIPolicy(Policy):
    """Harmonized SOFisher: ROI-seeking adaptive policy. Uses measured expression to estimate where
    region-of-interest signal (local heterogeneity) is high and moves toward it, with an exploration
    term. ADAPTIVE (conditions on measured values)."""
    closed_loop = True

    def __init__(self, beta: float = 0.5, lengthscale_frac: float = 0.1):
        self.beta = beta
        self.lengthscale_frac = lengthscale_frac

    def _kernel_columns(self, state):
        """Kernel columns K(:, j) and min squared distance for the profiled set, appended once per
        newly revealed region (profiled only grows in an episode) instead of rebuilding U x t each step."""
        prof = [int(i) for i in state.profiled]
        acc = getattr(self, "_acc", None)
        if acc is None or acc["slide"] is not state.slide or prof[:len(acc["done"])] != acc["done"]:
            co = state.slide.region_coords.astype(np.float64)
            R = co.shape[0]
            acc = self._acc = {"slide": state.slide, "done": [], "co": co,
                               "ell": max(float(np.linalg.norm(co.max(0) - co.min(0))) * self.lengthscale_frac, 1e-6),
                               "K": np.zeros((R, 64)), "mind2": np.full(R, np.inf)}
        co, ell = acc["co"], acc["ell"]
        for j in prof[len(acc["done"]):]:
            m = len(acc["done"])
            if m >= acc["K"].shape[1]:
                acc["K"] = np.hstack([acc["K"], np.zeros_like(acc["K"])])
            d2 = ((co - co[j]) ** 2).sum(1)
            acc["K"][:, m] = np.exp(-d2 / (2.0 * ell ** 2))
            np.minimum(acc["mind2"], d2, out=acc["mind2"])
            acc["done"].append(j)
        return acc

    def _cold_start(self, state, unpro):
        co = state.slide.region_coords
        return int(unpro[np.argmin(np.linalg.norm(co[unpro] - co.mean(0), axis=1))])

    def select(self, state: AcquisitionState):
        unpro = state.unprofiled
        if unpro.size == 0:
            raise RuntimeError("no unprofiled region left")
        if state.profiled.size == 0:
            return self._cold_start(state, unpro)
        prof = state.profiled
        acc = self._kernel_columns(state)
        y = state.measured_expression().astype(np.float64)
        # ROI signal at each measured FOV = its expression deviation from the measured mean (heterogeneity)
        roi_sig = np.abs(y - y.mean(0, keepdims=True)).mean(1)          # (t,)  uses measured values
        exploit = acc["K"][unpro][:, :len(prof)] @ roi_sig             # move toward high-ROI measured FOVs
        explore = np.sqrt(acc["mind2"][unpro])                         # DQN-style exploration
        def z(x):
            s = x.std(); return np.zeros_like(x) if s < 1e-12 else (x - x.mean()) / s
        score = z(exploit) + self.beta * z(explore)
        return int(unpro[int(np.argmax(score))])
