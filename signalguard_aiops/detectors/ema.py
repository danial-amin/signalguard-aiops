from __future__ import annotations

from typing import Tuple

import numpy as np

from .base import BaseDetector


class EMADetector(BaseDetector):
    """
    Exponential moving average anomaly detector.

    Maintains an exponentially weighted moving average (EMA) and deviation,
    flags points whose deviation from EMA exceeds a threshold.

    Parameters
    ----------
    alpha : float
        Smoothing factor in (0, 1]. Higher = more weight on recent values.
    k_sigma : float
        Number of "sigma" deviations away from EMA to flag anomalies.
    warmup : int
        Number of initial points to treat as baseline without anomalies.
    """

    def __init__(self, alpha: float = 0.2, k_sigma: float = 3.0, warmup: int = 10):
        self.alpha = float(alpha)
        self.k_sigma = float(k_sigma)
        self.warmup = int(warmup)

    def detect(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=float)
        n = len(values)
        labels = np.zeros(n, dtype=int)
        scores = np.zeros(n, dtype=float)

        if n == 0:
            return labels, scores

        ema = values[0]
        ema_dev = 0.0

        for i, x in enumerate(values):
            if i == 0:
                labels[i] = 0
                scores[i] = 0.0
                continue

            # Update EMA and deviation
            ema_prev = ema
            ema = self.alpha * x + (1 - self.alpha) * ema_prev

            dev = abs(x - ema)
            ema_dev = self.alpha * dev + (1 - self.alpha) * ema_dev

            # Warm-up period: don't flag anomalies yet
            if i < self.warmup or ema_dev == 0:
                labels[i] = 0
                scores[i] = 0.0
                continue

            score = dev / (ema_dev + 1e-8)
            labels[i] = int(score >= self.k_sigma)
            scores[i] = score

        return labels, scores
