from __future__ import annotations

from typing import Tuple

import numpy as np

from .base import BaseDetector


class ZScoreDetector(BaseDetector):
    """
    Simple rolling z-score detector.

    For each point, compute (x - mean(window)) / std(window) and flag as anomalous
    if |z| >= z_thresh.

    Parameters
    ----------
    window : int
        Number of points to use as the rolling history window.
    z_thresh : float
        Threshold on |z-score| to mark an anomaly.
    min_history : int
        Minimum number of points before starting to mark anomalies.
    """

    def __init__(self, window: int = 30, z_thresh: float = 3.0, min_history: int = 10):
        self.window = int(window)
        self.z_thresh = float(z_thresh)
        self.min_history = int(min_history)

    def detect(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=float)
        n = len(values)
        labels = np.zeros(n, dtype=int)
        scores = np.zeros(n, dtype=float)

        if n == 0:
            return labels, scores

        for i in range(n):
            start = max(0, i - self.window)
            history = values[start:i]

            # Not enough history to compute a stable baseline
            if len(history) < self.min_history:
                labels[i] = 0
                scores[i] = 0.0
                continue

            mean = history.mean()
            std = history.std() or 1e-8  # avoid division by zero
            z = (values[i] - mean) / std
            score = abs(z)

            labels[i] = int(score >= self.z_thresh)
            scores[i] = score

        return labels, scores
