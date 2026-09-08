"""
Principled NON-ADAPTIVE placement baselines [Krause, Guestrin & Singh, JMLR 2008].

These are the CORRECT open-loop opponents for our closed-loop policy — the field's named methods
for "choose locations to reconstruct a spatial field", with submodular (1-1/e)-optimality. They use
ONLY the kernel over regions (a function of the image+space features), never the measured values —
so they are genuinely non-adaptive. If our closed-loop policy cannot beat these, the agentic claim
is honestly dead; if it can (under a learned imputer), the win is meaningful.

Kernel: RBF on joint features = [PCA(image) standardized] (+) [unit-box coords], lengthscale L.
Two criteria, both via an efficient incremental conditional-variance recursion (O(R*|A|) total):
  * VarianceGreedy : add argmax posterior variance given selected (A-optimal-style space filling).
  * MIGreedy       : add argmax  Var(y|A) / Var(y|all-others)  (Krause's mutual-information rule).

STATUS: functional scaffold. For very large R the O(R^2) init inverse (MI only) may need the
locality trick from Krause 2008; a MAX_REGIONS subsample knob keeps it tractable meanwhile.
"""
from __future__ import annotations
import numpy as np
from .base import Policy
from ..simulator import AcquisitionState
from .. import config


def _joint_features(slide, pca_dim: int) -> np.ndarray:
    X = slide.image_embeddings.astype(np.float64)
    X = X - X.mean(0, keepdims=True)
    d = min(pca_dim, X.shape[1], max(1, X.shape[0] - 1))
    _, _, Vt = np.linalg.svd(X, full_matrices=False)
    Z = X @ Vt[:d].T
    Z = Z / np.clip(Z.std(0, keepdims=True), 1e-8, None)
    co = slide.region_coords.astype(np.float64)
    co = (co - co.min(0)) / np.clip(co.max(0) - co.min(0), 1e-9, None)
    return np.concatenate([Z, co], axis=1)


def _median_lengthscale(F: np.ndarray, cap: int = 1000) -> float:
    """Median-heuristic bandwidth: median pairwise Euclidean distance. Keeps the RBF kernel
    meaningful at any feature dimensionality (a fixed lengthscale collapses K to the identity
    in high dim -> trivial placement)."""
    n = F.shape[0]
    idx = np.random.default_rng(0).choice(n, size=min(cap, n), replace=False)
    S = F[idx]
    sq = ((S[:, None, :] - S[None, :, :]) ** 2).sum(2)
    d = np.sqrt(sq[np.triu_indices(len(idx), k=1)])
    m = float(np.median(d))
    return m if m > 1e-8 else 1.0


def _rbf_kernel(F: np.ndarray, lengthscale_scale: float) -> np.ndarray:
    ell = _median_lengthscale(F) * float(lengthscale_scale)   # scale factor x median heuristic
    sq = ((F[:, None, :] - F[None, :, :]) ** 2).sum(2)
    return np.exp(-sq / (2.0 * ell ** 2))


class _PlacementBase(Policy):
    """Shared incremental engine. NON-ADAPTIVE: ignores measured expression entirely."""
    closed_loop = False

    def __init__(self, lengthscale: float = config.PLACEMENT_LENGTHSCALE,
                 noise: float = config.PLACEMENT_NOISE, pca_dim: int = config.PLACEMENT_PCA_DIM):
        self.lengthscale, self.noise, self.pca_dim = lengthscale, noise, pca_dim
        self._order: list[int] | None = None   # full offline placement order (non-adaptive)

    def _build_order(self, slide) -> list[int]:
        raise NotImplementedError

    def select(self, state: AcquisitionState):
        if self._order is None:
            self._order = self._build_order(state.slide)     # computed once, offline
        for i in self._order:
            if not state.mask[i]:
                return int(i)
        raise RuntimeError("no unprofiled region left")

    # incremental conditional-variance greedy shared by both criteria
    def _greedy(self, K: np.ndarray, budget: int, mi: bool, v_all: np.ndarray | None):
        R = K.shape[0]
        v = np.diag(K).astype(np.float64).copy()   # conditional variance given selected (init: prior)
        U: list[np.ndarray] = []                   # rank-1 downdate vectors
        selected: list[int] = []
        avail = np.ones(R, dtype=bool)
        n = min(budget, R)
        for _ in range(n):
            crit = (v / v_all) if mi else v         # MI ratio vs raw variance
            crit = np.where(avail, crit, -np.inf)
            a = int(np.argmax(crit))
            selected.append(a); avail[a] = False
            # rank-1 downdate of conditional variance after observing a
            u = K[:, a].astype(np.float64).copy()
            for uj in U:
                u -= uj[a] * uj
            denom = max(v[a] + self.noise, 1e-9)
            u /= np.sqrt(denom)
            v = np.clip(v - u ** 2, 1e-12, None)
            U.append(u)
        return selected

    def _order_from_kernel(self, slide, mi: bool) -> list[int]:
        F = _joint_features(slide, self.pca_dim)
        K = _rbf_kernel(F, self.lengthscale)
        R = K.shape[0]
        v_all = None
        if mi:
            # Var(y | all-others): 1/diag(inv(K+noise I)). One O(R^3) inverse, offline.
            P = np.linalg.inv(K + self.noise * np.eye(R))
            v_all = np.clip(1.0 / np.clip(np.diag(P), 1e-12, None), 1e-12, None)
        return self._greedy(K, budget=R, mi=mi, v_all=v_all)   # full order; select() consumes prefix


class VarianceGreedyPolicy(_PlacementBase):
    """Greedy maximum-posterior-variance placement (A-optimal-style space filling)."""
    def _build_order(self, slide):
        return self._order_from_kernel(slide, mi=False)


class MIGreedyPolicy(_PlacementBase):
    """Krause mutual-information greedy: argmax Var(y|A)/Var(y|V\\A\\y). Beats pure variance."""
    def _build_order(self, slide):
        return self._order_from_kernel(slide, mi=True)
