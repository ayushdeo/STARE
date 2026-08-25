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
    TODO(4070): build k-NN spatial weights on coords and compute Moran's I per gene, average.
    Stub returns nan until implemented (not needed for the Stage-2 gate)."""
    return float("nan")
