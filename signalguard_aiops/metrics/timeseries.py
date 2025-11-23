from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, List, Optional

import numpy as np
import pandas as pd


@dataclass
class TimeSeries:
    """
    Simple time series container used across detectors and pipelines.

    Attributes
    ----------
    timestamps : np.ndarray
        1D array of timestamps (seconds since epoch or pandas-compatible).
    values : np.ndarray
        1D array of float values.
    name : str
        Optional name, e.g. metric name.
    """
    timestamps: np.ndarray
    values: np.ndarray
    name: str = ""

    @classmethod
    def from_pandas(cls, series: pd.Series, name: Optional[str] = None) -> "TimeSeries":
        idx = series.index
        if isinstance(idx, pd.DatetimeIndex):
            timestamps = idx.view("int64") // 10**9
        else:
            timestamps = idx.to_numpy()

        return cls(
            timestamps=np.asarray(timestamps, dtype=float),
            values=series.to_numpy(dtype=float),
            name=name or getattr(series, "name", "") or "",
        )

    @classmethod
    def from_lists(
        cls,
        timestamps: Sequence[float],
        values: Sequence[float],
        name: str = "",
    ) -> "TimeSeries":
        return cls(
            timestamps=np.asarray(timestamps, dtype=float),
            values=np.asarray(values, dtype=float),
            name=name,
        )

    def to_pandas(self) -> pd.Series:
        return pd.Series(self.values, index=pd.to_datetime(self.timestamps, unit="s"), name=self.name)

    def window(self, start_ts: float, end_ts: float) -> "TimeSeries":
        mask = (self.timestamps >= start_ts) & (self.timestamps <= end_ts)
        return TimeSeries(
            timestamps=self.timestamps[mask],
            values=self.values[mask],
            name=self.name,
        )
