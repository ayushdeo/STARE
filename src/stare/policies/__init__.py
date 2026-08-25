"""Acquisition policies. `select(state)` returns the next region index/indices to profile."""
from .base import Policy
from .naive import RandomPolicy, GridPolicy, VariancePolicy, KMeansImagePolicy
from .oneshot import OneShotSpatialFinderPolicy
from .greedy_ig import GreedyInfoGainPolicy

REGISTRY = {
    "random": RandomPolicy,
    "grid": GridPolicy,
    "variance": VariancePolicy,
    "kmeans": KMeansImagePolicy,
    "oneshot": OneShotSpatialFinderPolicy,   # SpatialFinder-style — PRIMARY competitor
    "greedy_ig": GreedyInfoGainPolicy,       # ours (closed-loop)
}


def build_policy(name: str, **kw) -> Policy:
    if name not in REGISTRY:
        raise ValueError(f"unknown policy '{name}'; options: {list(REGISTRY)}")
    return REGISTRY[name](**kw)
