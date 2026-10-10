"""
Residual-guided error reduction: a value-conditioned policy aligned with reconstruction error
(revision round 2, reviewer concern 2).

The error-reduction policy sums the imputer's uncertainty over a candidate's unprofiled neighbourhood. Under
graphflow that uncertainty (1 - graph influence) depends only on WHICH regions are profiled, so the policy is a
sequential fixed design. This policy keeps the same error-reduction form but replaces the uncertainty with an
estimate of the imputer's actual reconstruction error that is computed from the MEASURED values:

  1. Fit graphflow's image model (random-Fourier-feature ridge) to the profiled regions and take each profiled
     region's leave-one-out residual, r_j = mean_g ((y_jg - yhat_{-j,g}) / s_g)^2, over the G' genes with the
     highest variance among the profiled regions (s_g = their profiled SD). Leave-one-out uses the ridge hat
     matrix, so no refits are needed. Gene choice uses measured values only (no evaluation genes).
  2. Expected error at an unprofiled region i = (mean r_j over its k nearest profiled regions in the imputer's
     feature space) x (1 - graph influence at i): large where the image model fails nearby AND message passing
     does not yet reach i.
  3. Score each candidate by the summed expected error of its 15 nearest unprofiled neighbours (as in
     error-reduction) and select the maximum.

Its choices depend on the measured values through the residuals, which is checked by the permuted-feedback
ablation (policies/feedback.py). Requires the graphflow imputer.
"""
from __future__ import annotations
import numpy as np
from .greedy_ig import GreedyInfoGainPolicy
from .. import config


class ResidualGuidedPolicy(GreedyInfoGainPolicy):
    closed_loop = True

    def __init__(self, imputer, n_genes: int = 256, k_prof: int = 10, er_k: int = config.GREEDY_ER_K):
        super().__init__(imputer, criterion="error_reduction", er_k=er_k)
        if not hasattr(imputer, "_rff") or not hasattr(imputer, "_influence_uncertainty"):
            raise ValueError("ResidualGuidedPolicy requires the graphflow imputer")
        self.n_genes, self.k_prof = n_genes, k_prof

    def _loo_residuals(self, state) -> np.ndarray:
        prof = np.asarray(state.profiled, dtype=int)
        Y = state.slide.expression[prof].astype(np.float64)            # measured values  <- y_{A_t}
        sd = Y.std(0)
        g = np.argsort(-sd)[: min(self.n_genes, Y.shape[1])]
        Yg = (Y[:, g] - Y[:, g].mean(0)) / np.clip(sd[g], 1e-8, None)
        P = self.imputer._rff(state.slide)[prof]                       # (t, D) image features
        Minv = np.linalg.inv(P.T @ P + self.imputer.ridge_alpha * np.eye(P.shape[1]))
        PM = P @ Minv
        h = np.einsum("ij,ij->i", PM, P)                                # hat-matrix diagonal
        resid = Yg - PM @ (P.T @ Yg)
        loo = resid / np.clip(1.0 - h, 1e-6, None)[:, None]
        return (loo ** 2).mean(1)                                       # (t,)

    def _unit_error(self, state):
        unpro, prof = state.unprofiled, np.asarray(state.profiled, dtype=int)
        infl_unc = self.imputer._influence_uncertainty(state)          # (U,) 1 - graph influence
        if prof.size < 2:
            return infl_unc
        r = self._loo_residuals(state)
        from sklearn.neighbors import NearestNeighbors
        F = self.imputer.features(state.slide)
        k = int(min(self.k_prof, prof.size))
        _, nn = NearestNeighbors(n_neighbors=k).fit(F[prof]).kneighbors(F[unpro])
        return r[nn].mean(1) * infl_unc                                  # (U,)
