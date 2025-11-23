from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Tuple

import numpy as np

from ..metrics import TimeSeries
from ..detectors import ZScoreDetector, IsolationForestDetector, LOFDetector
from ..incidents import Incident
from .base import BaseRecipe


@dataclass
class EnsembleErrorRateRecipe(BaseRecipe):
    """
    Ensemble recipe for error-rate anomalies.

    Uses multiple detectors and aggregates their votes:
      - ZScoreDetector
      - IsolationForestDetector
      - LOFDetector

    Label rule:
      - A point is anomalous if at least `min_votes` detectors mark it.

    Score rule:
      - Average of normalized scores from all detectors.
    """

    service: str
    metric: str = "error_rate"
    min_votes: int = 2

    # Configs for internal detectors (optional override)
    z_window: int = 30
    z_thresh: float = 3.0

    iforest_contamination: float = 0.05
    lof_contamination: float = 0.05

    def run(self, series: TimeSeries) -> Incident:
        values = series.values

        # 1) Run detectors
        z_det = ZScoreDetector(window=self.z_window, z_thresh=self.z_thresh)
        z_labels, z_scores = z_det.detect(values)

        iforest_det = IsolationForestDetector(contamination=self.iforest_contamination)
        i_labels, i_scores = iforest_det.detect(values)

        lof_det = LOFDetector(contamination=self.lof_contamination)
        l_labels, l_scores = lof_det.detect(values)

        # 2) Normalize scores to [0,1]
        def _norm(x: np.ndarray) -> np.ndarray:
            x = np.asarray(x, dtype=float)
            if x.size == 0:
                return x
            x = x - x.min()
            maxv = x.max() or 1e-8
            return x / maxv

        z_scores_n = _norm(z_scores)
        i_scores_n = _norm(i_scores)
        l_scores_n = _norm(l_scores)

        # 3) Majority vote on labels
        votes = z_labels + i_labels + l_labels
        ensemble_labels = (votes >= self.min_votes).astype(int)

        # 4) Average scores
        ensemble_scores = (z_scores_n + i_scores_n + l_scores_n) / 3.0

        incident = Incident.from_detector_output(
            service=self.service,
            metric=self.metric,
            timestamps=series.timestamps,
            scores=ensemble_scores,
            labels=ensemble_labels,
            note=f"EnsembleErrorRateRecipe (min_votes={self.min_votes})",
        )
        return incident
