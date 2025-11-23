from __future__ import annotations

from dataclasses import dataclass

from ..metrics import TimeSeries
from ..detectors import EMADetector
from ..incidents import Incident
from .base import BaseRecipe


@dataclass
class LatencySLORecipe(BaseRecipe):
    """
    Recipe for latency-based SLO breaches.

    Typical use: p95 latency time series for a given service.

    Logic:
      - Track EMA-based deviation to detect sharp jumps.
      - Apply a hard SLO threshold (e.g. 300ms).
      - Mark points anomalous if EITHER SLO is violated or EMA deviation is large.
    """

    service: str
    metric: str = "latency_p95"
    slo_ms: float = 300.0
    alpha: float = 0.2
    k_sigma: float = 3.0
    warmup: int = 10

    def run(self, series: TimeSeries) -> Incident:
        # EMADetector on latency values
        detector = EMADetector(alpha=self.alpha, k_sigma=self.k_sigma, warmup=self.warmup)
        labels_ema, scores_ema = detector.detect(series.values)

        # SLO threshold check (assumes series.values are in seconds)
        slo_sec = self.slo_ms / 1000.0
        slo_breach_labels = (series.values > slo_sec).astype(int)

        # Combine EMA anomaly and SLO breach
        combined_labels = ((labels_ema == 1) | (slo_breach_labels == 1)).astype(int)

        # Combine scores: emphasize SLO breaches
        slo_boost = slo_breach_labels * 2.0
        combined_scores = scores_ema + slo_boost

        incident = Incident.from_detector_output(
            service=self.service,
            metric=self.metric,
            timestamps=series.timestamps,
            scores=combined_scores,
            labels=combined_labels,
            note=f"LatencySLORecipe (SLO={self.slo_ms}ms, alpha={self.alpha}, k_sigma={self.k_sigma})",
        )
        return incident
