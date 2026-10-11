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

Cost: one R x R kernel and (MI only) one O(R^3) inverse per slide, shared across policy instances;
the order is extended lazily, so only the prefix the largest budget needs is computed.
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
    n2 = (F * F).sum(1)                                       # Gram form: no R x R x d temporary
    sq = np.clip(n2[:, None] + n2[None, :] - 2.0 * (F @ F.T), 0.0, None)
    np.fill_diagonal(sq, 0.0)                                 # exact zeros (prior variance ties as before)
    return np.exp(-sq / (2.0 * ell ** 2))


class _GreedyEngine:
    """Incremental conditional-variance greedy for one slide, extended lazily and shared by every
    policy instance on that slide (the order is non-adaptive, so it does not depend on the imputer,
    the budget, or what has been revealed). Rank-1 downdates are applied as one matrix product."""

    def __init__(self, slide, mi: bool, lengthscale: float, noise: float, pca_dim: int):
        # keep only the inputs of the kernel (not the slide, whose expression matrix may be a large copy)
        self.emb, self.co, self.mi, self.noise = slide.image_embeddings, slide.region_coords, mi, noise
        K = _rbf_kernel(_joint_features(slide, pca_dim), lengthscale)
        self.K, R = K, K.shape[0]
        self.v = np.diag(K).astype(np.float64).copy()   # conditional variance given selected
        self.v_all = None
        if mi:
            # Var(y | all-others): 1/diag(inv(K+noise I)). One O(R^3) inverse, offline, once per slide.
            P = np.linalg.inv(K + noise * np.eye(R))
            self.v_all = np.clip(1.0 / np.clip(np.diag(P), 1e-12, None), 1e-12, None)
        self.U = np.zeros((max(16, R // 50), R))           # rank-1 downdate vectors (grown on demand)
        self.order: list[int] = []
        self.avail = np.ones(R, dtype=bool)

    def extend(self, n: int) -> None:
        R = self.K.shape[0]
        while len(self.order) < min(n, R):
            crit = (self.v / self.v_all) if self.mi else self.v   # MI ratio vs raw variance
            a = int(np.argmax(np.where(self.avail, crit, -np.inf)))
            m = len(self.order)
            self.order.append(a); self.avail[a] = False
            u = self.K[:, a] - self.U[:m].T @ self.U[:m, a]
            u = u / np.sqrt(max(self.v[a] + self.noise, 1e-9))
            self.v = np.clip(self.v - u ** 2, 1e-12, None)
            if m >= self.U.shape[0]:
                self.U = np.vstack([self.U, np.zeros_like(self.U)])
            self.U[m] = u


_ENGINES: dict = {}          # (id(embeddings), id(coords), mi, lengthscale, noise, pca_dim) -> _GreedyEngine
_MAX_ENGINES = 2             # each holds an R x R kernel; keep memory bounded


def _engine(slide, mi, lengthscale, noise, pca_dim) -> _GreedyEngine:
    # keyed by the kernel's inputs (embeddings, coordinates); the engine holds them, so ids cannot be reused
    key = (id(slide.image_embeddings), id(slide.region_coords), mi, lengthscale, noise, pca_dim)
    eng = _ENGINES.get(key)
    if eng is None or eng.emb is not slide.image_embeddings or eng.co is not slide.region_coords:
        while len(_ENGINES) >= _MAX_ENGINES:
            _ENGINES.pop(next(iter(_ENGINES)))
        eng = _ENGINES[key] = _GreedyEngine(slide, mi, lengthscale, noise, pca_dim)
    return eng


class _PlacementBase(Policy):
    """Shared incremental engine. NON-ADAPTIVE: ignores measured expression entirely."""
    closed_loop = False

    _mi = False

    def __init__(self, lengthscale: float = config.PLACEMENT_LENGTHSCALE,
                 noise: float = config.PLACEMENT_NOISE, pca_dim: int = config.PLACEMENT_PCA_DIM):
        self.lengthscale, self.noise, self.pca_dim = lengthscale, noise, pca_dim
        self._ptr = 0             # order positions before this are already profiled (mask only grows)

    def select(self, state: AcquisitionState):
        eng = _engine(state.slide, self._mi, self.lengthscale, self.noise, self.pca_dim)
        R = state.slide.n_regions
        while True:
            if self._ptr >= len(eng.order):
                if len(eng.order) >= R:
                    raise RuntimeError("no unprofiled region left")
                eng.extend(len(eng.order) + max(32, R // 20))
            a = eng.order[self._ptr]
            if not state.mask[a]:
                return int(a)
            self._ptr += 1


class VarianceGreedyPolicy(_PlacementBase):
    """Greedy maximum-posterior-variance placement (A-optimal-style space filling)."""
    _mi = False


class MIGreedyPolicy(_PlacementBase):
    """Krause mutual-information greedy: argmax Var(y|A)/Var(y|V\\A\\y). Beats pure variance."""
    _mi = True
