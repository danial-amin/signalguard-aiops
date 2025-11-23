"""
Demo: ErrorRateZScoreRecipe on synthetic error-rate series.
"""

import numpy as np

from signalguard_aiops.metrics import TimeSeries
from signalguard_aiops.recipes import ErrorRateZScoreRecipe
from signalguard_aiops.incidents import IncidentScorer


def generate_synthetic_error_rate(n: int = 300):
    timestamps = np.arange(n)

    # base error rate around 3–5%
    base = 0.04 + np.random.normal(scale=0.005, size=n)

    # incident window with much higher error rate
    base[120:150] += 0.15
    base[220:235] += 0.2

    # clip to [0, 1]
    values = np.clip(base, 0, 1)
    return TimeSeries.from_lists(timestamps, values, name="error_rate")


def main():
    ts = generate_synthetic_error_rate()

    recipe = ErrorRateZScoreRecipe(service="checkout-service", metric="error_rate")
    incident = recipe.run(ts)

    severity = IncidentScorer.simple_severity(incident)

    print("=== ErrorRateZScoreRecipe Demo ===")
    print(f"Service   : {incident.service}")
    print(f"Metric    : {incident.metric}")
    print(f"Severity  : {severity}")
    print(f"Max score : {incident.max_score():.3f}")
    print(f"Anomalies : {len(incident.anomaly_indices())} points")


if __name__ == "__main__":
    main()
