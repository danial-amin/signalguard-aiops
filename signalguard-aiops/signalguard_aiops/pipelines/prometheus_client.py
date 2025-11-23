from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Any

import requests
import numpy as np

from ..metrics import TimeSeries


@dataclass
class PrometheusSeriesFetcher:
    """
    Very small wrapper around the Prometheus HTTP API
    to fetch range vectors and convert them into TimeSeries.

    Parameters
    ----------
    base_url : str
        Base URL of the Prometheus server, e.g. "http://localhost:9090".
    """

    base_url: str

    def fetch_range(
        self,
        query: str,
        start_ts: float,
        end_ts: float,
        step: str = "30s",
    ) -> TimeSeries:
        """
        Fetch a range vector for the given query and time span.

        Returns the first series in the response as a TimeSeries.
        """
        params = {
            "query": query,
            "start": start_ts,
            "end": end_ts,
            "step": step,
        }
        resp = requests.get(f"{self.base_url}/api/v1/query_range", params=params, timeout=10)
        resp.raise_for_status()
        data = resp.json()

        results = data.get("data", {}).get("result", [])
        if not results:
            return TimeSeries.from_lists([], [], name=query)

        # For simplicity, take the first result
        series = results[0]
        values = series["values"]  # list of [timestamp, value_str]

        timestamps = np.array([float(t) for t, _ in values], dtype=float)
        vals = np.array([float(v) for _, v in values], dtype=float)

        return TimeSeries.from_lists(timestamps, vals, name=query)
