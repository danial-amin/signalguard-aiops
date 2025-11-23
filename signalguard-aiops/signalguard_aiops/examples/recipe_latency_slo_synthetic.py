"""
Demo: LatencySLORecipe on synthetic p95 latency series.
"""

import numpy as np

from signalguard_aiops.metrics import TimeSeries
from signalguard_aiops.recipes import LatencySLORecipe
from signalguard_aiops.incidents import IncidentScorer


def generate_synthetic_latency(n: int = 300):
    timestamps = np.arange(n)

    # baseline p95 latency ~ 120–180ms
    base_ms = 150 + np.random.normal(scale=20, size=n)

    # incident: spike to 400–500ms
    base_ms[80:100] += 250
    base_ms[200:210] += 200

    # convert to seconds
    values_sec = base_ms / 1000.0

    return TimeSeries.from_lists(timestamps, values_sec, name="latency_p95")


def main():
    ts = generate_synthetic_latency()

    recipe = LatencySLORecipe(
        service="payments-service",
        metric="latency_p95",
        slo_ms=300.0,
        alpha=0.2,
        k_sigma=3.0,
    )

    incident = recipe.run(ts)
    severity = IncidentScorer.simple_severity(incident)

    print("=== LatencySLORecipe Demo ===")
    print(f"Service   : {incident.service}")
    print(f"Metric    : {incident.metric}")
    print(f"Severity  : {severity}")
    print(f"Max score : {incident.max_score():.3f}")
    print(f"Anomalies : {len(incident.anomaly_indices())} points")


if __name__ == "__main__":
    main()
