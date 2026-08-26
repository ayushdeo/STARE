import numpy as np

from stare.imputer import KNNImputer
from stare.metrics import evaluate
from stare.simulator import AcquisitionSimulator, Slide


def test_build_region_grid_groups_tiles():
    coords = np.array([
        [0.1, 0.1],
        [0.4, 0.2],
        [0.9, 0.8],
        [1.2, 0.9],
    ])
    inv, centroids = __import__('stare.data', fromlist=['build_region_grid']).build_region_grid(coords, region_um=1.0)
    assert inv.tolist() == [0, 0, 0, 1]
    assert centroids.shape == (2, 2)


def test_knn_imputer_missing_neighbors_and_uncertainty():
    slide = Slide(
        tissue='test',
        region_coords=np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]),
        image_embeddings=np.array([
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 1.0],
            [0.0, 0.0],
        ], dtype=float),
        expression=np.array([
            [5.0, 1.0],
            [1.0, 5.0],
            [4.0, 2.0],
            [2.0, 4.0],
        ], dtype=float),
        hvg_idx=np.array([0, 1]),
    )
    state = AcquisitionSimulator(slide, KNNImputer(k=2), budget_fraction=0.5).reset()
    state = AcquisitionSimulator(slide, KNNImputer(k=2), budget_fraction=0.5).reveal([0, 1])
    pred = KNNImputer(k=2).impute(state)
    assert pred.shape == (2, 2)
    assert np.allclose(pred[0], np.array([3.0, 3.0]), atol=1e-6)
    unc = KNNImputer(k=2).uncertainty(state)
    assert unc.shape == (2,)
    assert np.all(unc >= 0)


def test_evaluate_scores_top50_pcc():
    y_true = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=float)
    y_pred = np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 3.0]], dtype=float)
    scores = evaluate(y_true, y_pred, np.array([0, 1]), coords=np.array([[0, 0], [1, 0], [0, 1]]))
    assert scores['pcc_mean'] == 1.0
    assert scores['pcc_top50'] == 1.0
    assert scores['mse'] == 0.0
