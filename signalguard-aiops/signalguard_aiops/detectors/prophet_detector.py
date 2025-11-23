from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import pandas as pd
from prophet import Prophet

from .base import BaseDetector


class ProphetResidualDetector(BaseDetector):
    """
    Anomaly detector based on forecasting with Prophet and inspecting residuals.

    Pipeline:
      1. Fit Prophet on (ds, y) time series.
      2. Forecast over the observed time span.
      3. Compute residual = y - yhat.
      4. Standardize residuals and flag large deviations.

    Parameters
    ----------
    z_thresh : float
        Threshold on standardized residuals to mark anomalies.
    weekly_seasonality : bool
    daily_seasonality : bool
    yearly_seasonality : bool
        Seasonality switches for Prophet.
    """

    def __init__(
        self,
        z_thresh: float = 3.0,
        weekly_seasonality: bool = False,
        daily_seasonality: bool = False,
        yearly_seasonality: bool = False,
    ):
        self.z_thresh = float(z_thresh)
        self.weekly_seasonality = weekly_seasonality
        self.daily_seasonality = daily_seasonality
        self.yearly_seasonality = yearly_seasonality

    def detect(self, values: np.ndarray, timestamps: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Detect anomalies based on Prophet residuals.

        Parameters
        ----------
        values : np.ndarray
            1D array of values.
        timestamps : np.ndarray, optional
            1D array of timestamps in seconds. If None, uses a simple index.

        Returns
        -------
        labels, scores : tuple of np.ndarray
            labels: 0/1 anomalies
            scores: standardized residual magnitude
        """
        values = np.asarray(values, dtype=float)
        n = len(values)
        if n == 0:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=float)

        if timestamps is None:
            # Build a simple time index
            ds = pd.date_range(start="2025-01-01", periods=n, freq="T")
        else:
            timestamps = np.asarray(timestamps, dtype=float)
            ds = pd.to_datetime(timestamps, unit="s")

        df = pd.DataFrame({"ds": ds, "y": values})

        m = Prophet(
            weekly_seasonality=self.weekly_seasonality,
            daily_seasonality=self.daily_seasonality,
            yearly_seasonality=self.yearly_seasonality,
        )
        m.fit(df)

        forecast = m.predict(df)
        yhat = forecast["yhat"].to_numpy()
        residuals = values - yhat

        mean_res = residuals.mean()
        std_res = residuals.std() or 1e-8

        z = (residuals - mean_res) / std_res
        scores = np.abs(z)
        labels = (scores >= self.z_thresh).astype(int)

        return labels, scores
