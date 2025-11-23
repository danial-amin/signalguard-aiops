"""
Demo: EnsembleErrorRateRecipe on synthetic error-rate series.

Shows how combining detectors (Z-score, IsolationForest, LOF) can give
a more robust anomaly signal.
"""

import numpy as np

from signalguard_aiops.metrics import TimeSeries
from signalguard_aiops.recipes import EnsembleErrorRateRecipe
from signalguard_aiops.incidents import IncidentScorer


def generate_synthetic_error_rate(n: int = 400):
    timestamps = np.arange(n)

    base = 0.03 + np.random.normal(scale=0.004, size=n)

    # multiple anomaly regimes
    base[100:130] += 0.12
    base[220:260] += 0.18
    base[330:345] += 0.25

    values = np.clip(base, 0, 1)
    return TimeSeries.from_lists(timestamps, values, name="error_rate")


def main():
    ts = generate_synthetic_error_rate()

    recipe = EnsembleErrorRateRecipe(
        service="orders-service",
        metric="error_rate",
        min_votes=2,
    )

    incident = recipe.run(ts)
    severity = IncidentScorer.simple_severity(incident)

    print("=== EnsembleErrorRateRecipe Demo ===")
    print(f"Service   : {incident.service}")
    print(f"Metric    : {incident.metric}")
    print(f"Severity  : {severity}")
    print(f"Max score : {incident.max_score():.3f}")
    print(f"Anomalies : {len(incident.anomaly_indices())} points")


if __name__ == "__main__":
    main()
