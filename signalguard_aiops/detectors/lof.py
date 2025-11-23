from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.neighbors import LocalOutlierFactor

from .base import BaseDetector


class LOFDetector(BaseDetector):
    """
    Local Outlier Factor anomaly detector for 1D time series.

    Uses LocalOutlierFactor in unsupervised mode (novelty=False).
    LOF gives negative_outlier_factor_: lower = more anomalous.

    Parameters
    ----------
    n_neighbors : int
        Number of neighbors used by LOF.
    contamination : float
        Proportion of outliers expected in the data.
    """

    def __init__(self, n_neighbors: int = 20, contamination: float = 0.05):
        self.n_neighbors = n_neighbors
        self.contamination = contamination

    def detect(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=float).reshape(-1, 1)
        n = values.shape[0]
        if n == 0:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=float)

        # LOF in unsupervised mode
        lof = LocalOutlierFactor(
            n_neighbors=min(self.n_neighbors, max(2, n - 1)),
            contamination=self.contamination,
        )
        labels = lof.fit_predict(values)  # -1 = outlier, 1 = inlier

        # negative_outlier_factor_: lower (more negative) = more anomalous
        lof_scores = lof.negative_outlier_factor_
        # Convert to positive anomaly score
        raw_scores = -lof_scores
        raw_scores = raw_scores - raw_scores.min()
        norm_scores = raw_scores / (raw_scores.max() + 1e-8)

        anomaly_labels = (labels == -1).astype(int)
        return anomaly_labels, norm_scores.ravel()
