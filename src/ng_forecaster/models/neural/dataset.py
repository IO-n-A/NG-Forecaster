"""Dataset utilities for real LSTM training loops."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ng_forecaster.errors import ContractViolation


@dataclass(frozen=True)
class LSTMSupervisedDataset:
    """Supervised windows used by LSTM training."""

    x: np.ndarray
    y: np.ndarray
    mean: float
    std: float


def _as_numeric(values: Sequence[float] | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ContractViolation(
            "insufficient_training_data",
            key="component_values",
            detail="component series must contain at least one value",
        )
    if not np.isfinite(array).all():
        raise ContractViolation(
            "invalid_model_policy",
            key="component_values",
            detail="component series contains non-finite values",
        )
    return array


def build_lstm_supervised_dataset(
    values: Sequence[float] | pd.Series,
    *,
    lookback: int,
) -> LSTMSupervisedDataset:
    """Build normalized supervised windows with shape [n, lookback, 1]."""

    if int(lookback) < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm.lookback",
            detail="lookback must be >= 1",
        )
    raw = _as_numeric(values)
    mean = float(np.mean(raw))
    std = float(np.std(raw))
    if std <= 1e-12:
        std = 1.0
    normalized = (raw - mean) / std
    n_samples = int(len(normalized) - int(lookback))
    if n_samples < 2:
        raise ContractViolation(
            "insufficient_training_data",
            key="lstm.lookback",
            detail=(
                "real LSTM requires at least lookback+2 points; "
                f"received={len(normalized)} lookback={lookback}"
            ),
        )
    x = np.stack(
        [normalized[idx : idx + int(lookback)] for idx in range(n_samples)],
        axis=0,
    ).astype(float)
    y = np.asarray(
        [normalized[idx + int(lookback)] for idx in range(n_samples)],
        dtype=float,
    )
    return LSTMSupervisedDataset(
        x=x[:, :, np.newaxis],
        y=y,
        mean=mean,
        std=std,
    )

