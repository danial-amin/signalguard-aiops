from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
from sklearn.ensemble import IsolationForest

from .base import BaseDetector


class IsolationForestDetector(BaseDetector):
    """
    Isolation Forest anomaly detector for 1D time series.

    Fits an IsolationForest on the value distribution and uses the
    decision_function as an anomaly score.

    Parameters
    ----------
    n_estimators : int
        Number of base estimators in the ensemble.
    contamination : float or 'auto'
        Proportion of anomalies in the data.
    random_state : Optional[int]
        Random seed for reproducibility.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        contamination: float | str = "auto",
        random_state: Optional[int] = 42,
    ):
        self.model = IsolationForest(
            n_estimators=n_estimators,
            contamination=contamination,
            random_state=random_state,
        )

    def detect(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=float).reshape(-1, 1)

        if values.shape[0] == 0:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=float)

        # Fit on all values (unsupervised)
        self.model.fit(values)

        # decision_function: higher = more normal, lower = more anomalous
        decision_scores = self.model.decision_function(values)

        # Convert to anomaly score: invert and normalize roughly to [0, 1+]
        raw_scores = -decision_scores
        raw_scores = raw_scores - raw_scores.min()
        norm_scores = raw_scores / (raw_scores.max() + 1e-8)

        # Label: simple threshold on normalized score
        # You can tune this later, or compute based on contamination.
        labels = (norm_scores > 0.8).astype(int)

        return labels, norm_scores.ravel()
