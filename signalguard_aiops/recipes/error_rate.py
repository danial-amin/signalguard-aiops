from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from ..metrics import TimeSeries
from ..detectors import ZScoreDetector, IsolationForestDetector
from ..incidents import Incident
from ..incidents import IncidentScorer
from .base import BaseRecipe


@dataclass
class ErrorRateZScoreRecipe(BaseRecipe):
    """
    Opinionated recipe for detecting anomalies in error-rate metrics using
    a rolling z-score detector.

    Good for:
      - Simple, interpretable baselines
      - Fast experimentation on error_rate / failure_rate series
    """

    service: str
    metric: str = "error_rate"
    window: int = 30
    z_thresh: float = 3.0
    min_history: int = 10

    def run(self, series: TimeSeries) -> Incident:
        detector = ZScoreDetector(
            window=self.window,
            z_thresh=self.z_thresh,
            min_history=self.min_history,
        )
        labels, scores = detector.detect(series.values)

        incident = Incident.from_detector_output(
            service=self.service,
            metric=self.metric,
            timestamps=series.timestamps,
            scores=scores,
            labels=labels,
            note=f"ErrorRateZScoreRecipe (window={self.window}, z={self.z_thresh})",
        )
        return incident


@dataclass
class ErrorRateIForestRecipe(BaseRecipe):
    """
    Error-rate recipe using Isolation Forest.

    Good for:
      - Non-Gaussian error distributions
      - Complex patterns where simple thresholds are not enough
    """

    service: str
    metric: str = "error_rate"
    contamination: float = 0.05
    n_estimators: int = 100

    def run(self, series: TimeSeries) -> Incident:
        detector = IsolationForestDetector(
            n_estimators=self.n_estimators,
            contamination=self.contamination,
        )
        labels, scores = detector.detect(series.values)

        incident = Incident.from_detector_output(
            service=self.service,
            metric=self.metric,
            timestamps=series.timestamps,
            scores=scores,
            labels=labels,
            note=f"ErrorRateIForestRecipe (contamination={self.contamination})",
        )
        return incident
