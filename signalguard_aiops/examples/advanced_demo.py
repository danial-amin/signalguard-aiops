import time
import numpy as np

from signalguard_aiops.detectors import (
    IsolationForestDetector,
    LOFDetector,
    ProphetResidualDetector,
    LSTMAutoencoderDetector,
)
from signalguard_aiops.incidents import Incident, IncidentScorer
from signalguard_aiops.metrics import TimeSeries


def main():
    # Fake series with spikes
    n = 300
    timestamps = np.arange(n)
    base = np.sin(np.linspace(0, 6, n)) * 0.1 + 0.05
    noise = np.random.normal(scale=0.02, size=n)
    values = base + noise
    values[100:105] += 0.6  # anomaly burst
    values[220:225] += 0.5

    ts = TimeSeries.from_lists(timestamps, values, name="demo_metric")

    detectors = {
        "IsolationForest": IsolationForestDetector(contamination=0.05),
        "LOF": LOFDetector(contamination=0.05),
        "ProphetResidual": ProphetResidualDetector(z_thresh=2.5),
        "LSTM-AE": LSTMAutoencoderDetector(window_size=30, epochs=5),
    }

    for name, detector in detectors.items():
        print(f"\n=== {name} ===")
        if isinstance(detector, ProphetResidualDetector):
            labels, scores = detector.detect(ts.values, timestamps=ts.timestamps)
        else:
            labels, scores = detector.detect(ts.values)

        incident = Incident.from_detector_output(
            service="demo-service",
            metric="demo_metric",
            timestamps=ts.timestamps,
            scores=scores,
            labels=labels,
            note=f"{name} detector demo",
        )
        severity = IncidentScorer.simple_severity(incident)
        print(f"Severity: {severity}")
        print(f"Max score: {incident.max_score():.3f}")
        print(f"Anomaly points: {len(incident.anomaly_indices())}")


if __name__ == "__main__":
    main()
