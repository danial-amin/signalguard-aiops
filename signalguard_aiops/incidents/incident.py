from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np


@dataclass
class Incident:
    """
    Represents an incident derived from anomaly detection.

    Attributes
    ----------
    service : str
        Logical service name (e.g. "orders-service").
    metric : str
        Metric name (e.g. "error_rate").
    timestamps : np.ndarray
        1D array of timestamps.
    scores : np.ndarray
        1D array of anomaly scores.
    labels : np.ndarray
        1D array of 0/1 labels.
    note : str
        Optional free-text note or explanation.
    """

    service: str
    metric: str
    timestamps: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    note: str = ""

    @classmethod
    def from_detector_output(
        cls,
        service: str,
        metric: str,
        timestamps: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        note: str = "",
    ) -> "Incident":
        timestamps = np.asarray(timestamps, dtype=float)
        scores = np.asarray(scores, dtype=float)
        labels = np.asarray(labels, dtype=int)

        if not (len(timestamps) == len(scores) == len(labels)):
            raise ValueError("timestamps, scores, and labels must have the same length")

        return cls(
            service=service,
            metric=metric,
            timestamps=timestamps,
            scores=scores,
            labels=labels,
            note=note,
        )

    def anomaly_indices(self) -> np.ndarray:
        return np.where(self.labels == 1)[0]

    def max_score(self) -> float:
        return float(self.scores.max()) if len(self.scores) else 0.0

    def duration(self) -> float:
        if len(self.timestamps) == 0:
            return 0.0
        return float(self.timestamps[-1] - self.timestamps[0])
