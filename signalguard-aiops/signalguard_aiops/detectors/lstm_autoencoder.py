from __future__ import annotations

from typing import Tuple, Optional

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models

from .base import BaseDetector


class LSTMAutoencoderDetector(BaseDetector):
    """
    LSTM autoencoder for 1D time-series anomaly detection.

    Workflow:
      - Build sliding windows from the series.
      - Train LSTM autoencoder to reconstruct "normal" windows.
      - Compute reconstruction error for each window.
      - Map window-level anomaly scores back to point-level scores.

    NOTE:
      This detector has a `fit` step. For convenience, if you call detect()
      before fit(), it will perform a quick fit on the provided values.

    Parameters
    ----------
    window_size : int
        Number of timesteps per sequence window.
    latent_dim : int
        Dimensionality of LSTM latent representation.
    epochs : int
        Training epochs.
    batch_size : int
        Batch size for training.
    """

    def __init__(
        self,
        window_size: int = 30,
        latent_dim: int = 16,
        epochs: int = 10,
        batch_size: int = 32,
    ):
        self.window_size = int(window_size)
        self.latent_dim = int(latent_dim)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)

        self.model: Optional[tf.keras.Model] = None
        self._trained = False

    def _build_model(self):
        inputs = layers.Input(shape=(self.window_size, 1))
        x = layers.LSTM(self.latent_dim, activation="tanh", return_sequences=False)(inputs)
        encoded = layers.RepeatVector(self.window_size)(x)
        x = layers.LSTM(self.latent_dim, activation="tanh", return_sequences=True)(encoded)
        outputs = layers.TimeDistributed(layers.Dense(1))(x)

        model = models.Model(inputs, outputs)
        model.compile(optimizer="adam", loss="mse")
        self.model = model

    def _create_windows(self, values: np.ndarray) -> np.ndarray:
        """
        Build overlapping windows [n_windows, window_size, 1]
        """
        n = len(values)
        if n < self.window_size:
            return np.empty((0, self.window_size, 1))

        windows = []
        for i in range(n - self.window_size + 1):
            window = values[i : i + self.window_size]
            windows.append(window)
        windows = np.array(windows, dtype=float)
        return windows[..., np.newaxis]  # add feature dim

    def fit(self, values: np.ndarray):
        """
        Fit the LSTM autoencoder on the given series.
        """
        values = np.asarray(values, dtype=float)
        windows = self._create_windows(values)
        if windows.shape[0] == 0:
            return

        if self.model is None:
            self._build_model()

        self.model.fit(
            windows,
            windows,
            epochs=self.epochs,
            batch_size=self.batch_size,
            verbose=0,
        )
        self._trained = True

    def detect(self, values: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        values = np.asarray(values, dtype=float)
        n = len(values)
        if n == 0:
            return np.zeros(0, dtype=int), np.zeros(0, dtype=float)

        if not self._trained:
            # Quick self-fit on the provided values
            self.fit(values)

        windows = self._create_windows(values)
        if windows.shape[0] == 0:
            return np.zeros(n, dtype=int), np.zeros(n, dtype=float)

        recon = self.model.predict(windows, verbose=0)
        # MSE per window
        window_errors = np.mean((windows - recon) ** 2, axis=(1, 2))

        # Map window error to point-level scores:
        # assign each window error to its center point
        point_scores = np.zeros(n)
        counts = np.zeros(n)

        half = self.window_size // 2
        for i, err in enumerate(window_errors):
            center = i + half
            if center < n:
                point_scores[center] += err
                counts[center] += 1

        # average where multiple windows overlap
        mask = counts > 0
        point_scores[mask] /= counts[mask]

        # normalize scores to [0, 1+]
        if point_scores.max() > 0:
            point_scores = point_scores / point_scores.max()

        # simple threshold for labels
        labels = (point_scores > 0.8).astype(int)

        return labels, point_scores
