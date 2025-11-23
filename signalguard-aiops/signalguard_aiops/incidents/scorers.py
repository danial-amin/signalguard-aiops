from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .incident import Incident


@dataclass
class IncidentScorer:
    """
    Utility class for computing simple severity scores and levels
    from an Incident object.

    The idea is to have a pluggable scoring mechanism that can later be
    replaced with more advanced models or rules.
    """

    # weights for different contributing factors
    w_max_score: float = 0.5
    w_anomaly_count: float = 0.3
    w_duration: float = 0.2

    def score(self, incident: Incident) -> float:
        if len(incident.timestamps) == 0:
            return 0.0

        max_score = incident.max_score()
        anomaly_count = len(incident.anomaly_indices())
        duration = incident.duration()

        # Normalize factors crudely for now
        norm_max_score = min(max_score / 10.0, 1.0)      # assume 10 is "very high"
        norm_count = min(anomaly_count / 50.0, 1.0)      # 50 anomalies = max
        norm_duration = min(duration / 600.0, 1.0)       # 10 minutes window

        composite = (
            self.w_max_score * norm_max_score
            + self.w_anomaly_count * norm_count
            + self.w_duration * norm_duration
        )
        return float(composite)

    def severity_level(self, incident: Incident) -> str:
        s = self.score(incident)
        if s < 0.2:
            return "info"
        if s < 0.4:
            return "low"
        if s < 0.7:
            return "medium"
        if s < 0.9:
            return "high"
        return "critical"

    @staticmethod
    def simple_severity(incident: Incident) -> str:
        """
        Stateless convenience wrapper with default weighting.
        """
        return IncidentScorer().severity_level(incident)
