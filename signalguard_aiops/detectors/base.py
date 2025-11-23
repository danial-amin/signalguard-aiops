from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

import numpy as np


class BaseDetector(ABC):
    """
    Abstract base class for anomaly detectors.

    All detectors take a 1D array of float values and return:
      - labels: np.ndarray of shape (n,), values in {0, 1}
      - scores: np.ndarray of shape (n,), anomaly score (higher = more anomalous)
    """

    @abstractmethod
    def detect(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Run anomaly detection on a 1D array of values."""
        raise NotImplementedError
