"""
Example:
- Fetch error rate metric from Prometheus
- Run Z-score detector
- Build an Incident and compute severity
"""

import time

from signalguard_aiops.detectors import ZScoreDetector
from signalguard_aiops.incidents import Incident, IncidentScorer
from signalguard_aiops.pipelines import PrometheusSeriesFetcher


def main():
    prom = PrometheusSeriesFetcher(base_url="http://localhost:9090")

    end_ts = time.time()
    start_ts = end_ts - 15 * 60  # last 15 minutes

    # Example metric from SignalGuard project
    query = 'rate(app_request_errors_total[5m])'

    series = prom.fetch_range(query=query, start_ts=start_ts, end_ts=end_ts, step="30s")

    detector = ZScoreDetector(window=30, z_thresh=3.0, min_history=10)
    labels, scores = detector.detect(series.values)

    incident = Incident.from_detector_output(
        service="orders",
        metric="error_rate",
        timestamps=series.timestamps,
        scores=scores,
        labels=labels,
        note="Demo incident from Z-score detector",
    )

    severity = IncidentScorer.simple_severity(incident)

    print(f"Incident severity: {severity}")
    print(f"Max score: {incident.max_score():.2f}")
    print(f"Anomaly count: {len(incident.anomaly_indices())}")


if __name__ == "__main__":
    main()
