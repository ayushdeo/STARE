"""Acquisition policies. `select(state)` returns the next region index/indices to profile."""
from .base import Policy
from .naive import RandomPolicy, GridPolicy, VariancePolicy, KMeansImagePolicy
from .oneshot import OneShotSpatialFinderPolicy
from .greedy_ig import GreedyInfoGainPolicy
from .placement import VarianceGreedyPolicy, MIGreedyPolicy
from .value_greedy import ValueGreedyPolicy

REGISTRY = {
    "random": RandomPolicy,
    "grid": GridPolicy,
    "variance": VariancePolicy,
    "kmeans": KMeansImagePolicy,
    "oneshot": OneShotSpatialFinderPolicy,       # SpatialFinder-style — PRIMARY competitor
    "var_greedy": VarianceGreedyPolicy,          # Krause non-adaptive placement (variance)
    "mi_greedy": MIGreedyPolicy,                 # Krause non-adaptive placement (mutual information)
    "greedy_ig": GreedyInfoGainPolicy,           # ours (closed-loop, uncertainty/error-reduction)
    "value_greedy": ValueGreedyPolicy,           # value-conditioned adaptive (selection uses measured y)
}


def build_policy(name: str, **kw) -> Policy:
    if name not in REGISTRY:
        raise ValueError(f"unknown policy '{name}'; options: {list(REGISTRY)}")
    return REGISTRY[name](**kw)
