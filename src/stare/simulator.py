"""
Closed-loop region-acquisition simulator (WS-A).

The environment for Option B: given a slide's full ST ground truth, the policy reveals
regions one round at a time under a budget; after each round (and at the end) we impute
the UNPROFILED regions and score reconstruction error. Wet-lab-free.

STATUS: SKELETON. Interfaces are fixed; method bodies are TODO for the 4070 machine.
See docs/SOURCE_OF_TRUTH.md §4 (formal objective) and docs/REPRODUCTION_PLAN.md §4 Stage 0.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
import numpy as np

from . import config


@dataclass
class Slide:
    """One slide's data on the common region grid."""
    tissue: str
    region_coords: np.ndarray          # (R, 2) region centroids in microns
    image_embeddings: np.ndarray       # (R, IMAGE_EMB_DIM) frozen UNI patch embeddings
    expression: np.ndarray             # (R, G) log1p ground-truth expression (all genes)
    hvg_idx: np.ndarray                # (N_HVG,) indices into G of top-50 HVG
    gene_names: Optional[list[str]] = None
    patient_id: Optional[str] = None

    @property
    def n_regions(self) -> int:
        return self.region_coords.shape[0]


class AcquisitionSimulator:
    """
    Closed-loop environment.

    Usage:
        sim = AcquisitionSimulator(slide, budget_fraction=0.05)
        state = sim.reset()
        while not sim.done:
            region = policy.select(state)      # policy picks next region(s)
            state = sim.reveal(region)         # reveals measured expression
        score = sim.score()                    # impute unprofiled + score
    """

    def __init__(self, slide: Slide, imputer, budget_fraction: float,
                 round_size: int = config.ACQUISITION_ROUND_SIZE):
        self.slide = slide
        self.imputer = imputer                 # frozen; see imputer.py
        self.budget = int(round(budget_fraction * slide.n_regions))
        self.round_size = round_size
        self.profiled: list[int] = []          # region indices measured so far (ordered)
        self._mask = np.zeros(slide.n_regions, dtype=bool)

    # -- lifecycle -----------------------------------------------------------
    def reset(self) -> "AcquisitionState":
        self.profiled = []
        self._mask[:] = False
        return self._state()

    @property
    def done(self) -> bool:
        return len(self.profiled) >= self.budget

    def reveal(self, region_idx: int | list[int]) -> "AcquisitionState":
        idxs = [region_idx] if isinstance(region_idx, (int, np.integer)) else list(region_idx)
        for i in idxs:
            if self._mask[i]:
                raise ValueError(f"region {i} already profiled")
            if len(self.profiled) >= self.budget:
                break
            self._mask[i] = True
            self.profiled.append(int(i))
        return self._state()

    # -- state exposed to the policy ----------------------------------------
    def _state(self) -> "AcquisitionState":
        return AcquisitionState(
            slide=self.slide,
            profiled=np.array(self.profiled, dtype=int),
            mask=self._mask.copy(),
            budget=self.budget,
        )

    # -- scoring -------------------------------------------------------------
    def score(self) -> dict:
        """Impute unprofiled regions from profiled ones, score vs ground truth.
        Returns dict of metrics (see metrics.py). TODO: wire to imputer + metrics."""
        raise NotImplementedError("Stage-0 TODO: call imputer.impute(state) then metrics.evaluate(...)")


@dataclass
class AcquisitionState:
    """Everything a policy may condition on. The CLOSED LOOP = policy sees measured
    expression of profiled regions (slide.expression[profiled]), not just the image."""
    slide: Slide
    profiled: np.ndarray               # (t,) indices measured so far
    mask: np.ndarray                   # (R,) bool, True = profiled
    budget: int

    @property
    def unprofiled(self) -> np.ndarray:
        return np.where(~self.mask)[0]

    def measured_expression(self) -> np.ndarray:
        """(t, G) expression of profiled regions — the closed-loop signal."""
        return self.slide.expression[self.profiled]
