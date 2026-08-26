"""
Evaluation metrics, scored over UNPROFILED regions only (WS-E).

Kept dependency-light and unit-testable offline. See config.METRICS / PRIMARY_METRIC.
"""
from __future__ import annotations
import numpy as np


def per_gene_pcc(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    """(G,) Pearson per gene across regions; genes with zero variance -> nan (skipped in means)."""
    yt = y_true - y_true.mean(0, keepdims=True)
    yp = y_pred - y_pred.mean(0, keepdims=True)
    num = (yt * yp).sum(0)
    den = np.sqrt((yt**2).sum(0) * (yp**2).sum(0))
    with np.errstate(invalid="ignore", divide="ignore"):
        r = np.where(den > 0, num / den, np.nan)
    return r


def evaluate(y_true: np.ndarray, y_pred: np.ndarray, hvg_idx: np.ndarray,
             coords: np.ndarray | None = None) -> dict:
    """y_true/y_pred: (n_unprofiled, G). Returns the config.METRICS dict."""
    r = per_gene_pcc(y_true, y_pred)
    out = {
        "pcc_mean": float(np.nanmean(r)),
        "pcc_top50": float(np.nanmean(r[hvg_idx])),
        "mse": float(np.mean((y_true - y_pred) ** 2)),
    }
    if coords is not None:
        out["morans_i"] = float(morans_i(y_pred, coords))
    return out


def morans_i(values: np.ndarray, coords: np.ndarray, k: int = 6) -> float:
    """Mean Moran's I across genes (spatial-pattern preservation; cf. ReconST).

    Builds a symmetric binary k-NN spatial-weight matrix on `coords`, computes Moran's I
    per gene, averages over genes with nonzero variance.
        I = (N / W) * (sum_ij w_ij (x_i - x_bar)(x_j - x_bar)) / sum_i (x_i - x_bar)^2
    `values`: (N_regions, G). Returns nan if fewer than k+1 regions.
    """
    values = np.asarray(values, dtype=float)
    n = coords.shape[0]
    if n < k + 1 or values.shape[0] != n:
        return float("nan")

    # k-NN adjacency (exclude self), symmetrized to binary weights
    d = np.linalg.norm(coords[:, None, :] - coords[None, :, :], axis=2)
    np.fill_diagonal(d, np.inf)
    nn = np.argsort(d, axis=1)[:, :k]
    W = np.zeros((n, n), dtype=float)
    rows = np.repeat(np.arange(n), k)
    W[rows, nn.ravel()] = 1.0
    W = np.maximum(W, W.T)
    w_sum = W.sum()
    if w_sum == 0:
        return float("nan")

    xs = values - values.mean(axis=0, keepdims=True)      # (N, G)
    denom = (xs ** 2).sum(axis=0)                          # (G,)
    num = np.einsum("ij,ig,jg->g", W, xs, xs)              # (G,)
    with np.errstate(invalid="ignore", divide="ignore"):
        gene_I = np.where(denom > 0, (n / w_sum) * (num / denom), np.nan)
    return float(np.nanmean(gene_I))
