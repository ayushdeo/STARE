"""
Greedy expected-information-gain policy — OUR contribution, feasibility version (WS-D).

CLOSED LOOP: at each round, pick the unprofiled region whose measurement is expected to
most reduce whole-slide imputation uncertainty, conditioning on the image AND on the
expression measured so far (via imputer.uncertainty(state)). This is the differentiator
vs the one-shot SpatialFinder competitor (SOURCE_OF_TRUTH §4, §6).

Feasibility criterion (cheap, 4070-friendly): score each unprofiled region by the frozen
imputer's predictive uncertainty (kNN neighbor variance = expression-derived, hence
closed-loop), optionally plus a spatial-diversity term to spread early picks.

The three ablation switches drive the WS-E decisive experiments:
  use_measured : condition on measured expression (the CLOSED LOOP). False => image-only
                 novelty score, i.e. a one-shot-like control that isolates the loop's value.
  use_image    : let image-embedding novelty contribute.
  use_spatial  : add a spatial-coverage term (distance to nearest profiled region).
"""
from __future__ import annotations
import numpy as np
from .base import Policy
from ..simulator import AcquisitionState


def _zscore(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    s = x.std()
    if s < 1e-12:
        return np.zeros_like(x)
    return (x - x.mean()) / s


class GreedyInfoGainPolicy(Policy):
    closed_loop = True   # reads measured expression via imputer.uncertainty — the whole point

    def __init__(self, imputer, use_image: bool = True, use_spatial: bool = True,
                 use_measured: bool = True):
        self.imputer = imputer
        self.use_image = use_image
        self.use_spatial = use_spatial
        self.use_measured = use_measured
        # If we do NOT condition on measured expression, this is no longer the closed loop.
        self.closed_loop = bool(use_measured)

    # -- scoring terms -------------------------------------------------------
    def _image_novelty(self, state: AcquisitionState, unprofiled: np.ndarray) -> np.ndarray:
        """Per-unprofiled: cosine distance to the NEAREST profiled region in embedding space
        (higher = more morphologically novel => image-only info signal)."""
        emb = state.slide.image_embeddings
        prof = state.profiled
        q = emb[unprofiled]
        p = emb[prof]
        qn = q / np.clip(np.linalg.norm(q, axis=1, keepdims=True), 1e-12, None)
        pn = p / np.clip(np.linalg.norm(p, axis=1, keepdims=True), 1e-12, None)
        sims = qn @ pn.T                      # (n_unprof, n_prof)
        nearest_sim = sims.max(axis=1)
        return 1.0 - nearest_sim              # distance to nearest profiled

    def _spatial_coverage(self, state: AcquisitionState, unprofiled: np.ndarray) -> np.ndarray:
        """Per-unprofiled: Euclidean distance to the NEAREST profiled region (higher = fills a gap)."""
        coords = state.slide.region_coords
        prof = state.profiled
        d = np.linalg.norm(coords[unprofiled][:, None, :] - coords[prof][None, :, :], axis=2)
        return d.min(axis=1)

    # -- selection -----------------------------------------------------------
    def select(self, state: AcquisitionState):
        unprofiled = state.unprofiled
        if unprofiled.size == 0:
            raise RuntimeError("no unprofiled region left")

        # Cold start (nothing profiled yet): seed deterministically with the image medoid
        # (the most representative region) so the imputer has a sensible anchor.
        if state.profiled.size == 0:
            emb = state.slide.image_embeddings
            en = emb / np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
            centrality = (en @ en.mean(axis=0))     # similarity to the global mean direction
            return int(unprofiled[np.argmax(centrality[unprofiled])])

        score = np.zeros(unprofiled.size, dtype=float)
        if self.use_measured:
            # expression-derived predictive uncertainty => THE closed-loop signal
            score = score + _zscore(self.imputer.uncertainty(state))
        elif self.use_image:
            # one-shot-like control: image novelty only, no measured-expression conditioning
            score = score + _zscore(self._image_novelty(state, unprofiled))

        if self.use_image and self.use_measured:
            # small novelty nudge on top of uncertainty when both are on
            score = score + 0.25 * _zscore(self._image_novelty(state, unprofiled))
        if self.use_spatial:
            score = score + _zscore(self._spatial_coverage(state, unprofiled))

        return int(unprofiled[np.argmax(score)])
