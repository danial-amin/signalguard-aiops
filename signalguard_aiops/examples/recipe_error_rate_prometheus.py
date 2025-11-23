"""
Demo: ErrorRateZScoreRecipe using real Prometheus data.

Assumes:
  - SignalGuard or another app exports a metric compatible with:
      rate(app_request_errors_total[5m])
  - Prometheus is reachable at http://localhost:9090
"""

import time

from signalguard_aiops.pipelines import PrometheusSeriesFetcher
from signalguard_aiops.metrics import TimeSeries
from signalguard_aiops.recipes import ErrorRateZScoreRecipe
from signalguard_aiops.incidents import IncidentScorer


def main():
    prom = PrometheusSeriesFetcher(base_url="http://localhost:9090")

    end_ts = time.time()
    start_ts = end_ts - 15 * 60  # last 15 minutes

    query = 'rate(app_request_errors_total[5m])'
    ts = prom.fetch_range(query=query, start_ts=start_ts, end_ts=end_ts, step="30s")

    recipe = ErrorRateZScoreRecipe(
        service="signalguard-app",
        metric="error_rate",
        window=10,
        z_thresh=2.5,
        min_history=5,
    )

    incident = recipe.run(ts)
    severity = IncidentScorer.simple_severity(incident)

    print("=== Prometheus Error Rate Recipe Demo ===")
    print(f"Metric    : {ts.name}")
    print(f"Service   : {incident.service}")
    print(f"Severity  : {severity}")
    print(f"Max score : {incident.max_score():.3f}")
    print(f"Anomalies : {len(incident.anomaly_indices())} points")


if __name__ == "__main__":
    main()
