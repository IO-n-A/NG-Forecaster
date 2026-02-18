"""Forecast metric computation utilities."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.lag_guard import require_columns


def _as_numeric_series(values: pd.Series, *, key: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ContractViolation(
            "invalid_metric_payload",
            key=key,
            detail="metric inputs must be numeric and non-null",
        )
    return numeric.astype(float)


def mean_absolute_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    true = _as_numeric_series(y_true, key="y_true")
    pred = _as_numeric_series(y_pred, key="y_pred")
    return float((true - pred).abs().mean())


def root_mean_squared_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    true = _as_numeric_series(y_true, key="y_true")
    pred = _as_numeric_series(y_pred, key="y_pred")
    return float(np.sqrt(((true - pred) ** 2).mean()))


def mean_absolute_percentage_error(y_true: pd.Series, y_pred: pd.Series) -> float:
    true = _as_numeric_series(y_true, key="y_true")
    pred = _as_numeric_series(y_pred, key="y_pred")
    denominator = true.abs().replace(0.0, np.nan)
    ratio = ((true - pred).abs() / denominator).dropna()
    if ratio.empty:
        raise ContractViolation(
            "invalid_metric_payload",
            key="y_true",
            detail="MAPE denominator is zero for all rows",
        )
    return float(ratio.mean())


def score_point_forecasts(
    frame: pd.DataFrame,
    *,
    actual_col: str = "actual",
    forecast_col: str = "forecast",
    group_cols: list[str] | None = None,
) -> pd.DataFrame:
    """Compute MAE/RMSE/MAPE scorecards for forecast outputs."""

    require_columns(frame, (actual_col, forecast_col), key="point_forecast_frame")
    group_cols = group_cols or []

    data = frame.copy()
    data[actual_col] = _as_numeric_series(data[actual_col], key=actual_col)
    data[forecast_col] = _as_numeric_series(data[forecast_col], key=forecast_col)

    def _compute(group: pd.DataFrame) -> dict[str, Any]:
        return {
            "mae": mean_absolute_error(group[actual_col], group[forecast_col]),
            "rmse": root_mean_squared_error(group[actual_col], group[forecast_col]),
            "mape": mean_absolute_percentage_error(
                group[actual_col], group[forecast_col]
            ),
            "n_obs": int(len(group)),
        }

    if group_cols:
        require_columns(data, tuple(group_cols), key="group_cols")
        rows: list[dict[str, Any]] = []
        for keys, group in data.groupby(group_cols, sort=True):
            metrics = _compute(group)
            if not isinstance(keys, tuple):
                keys = (keys,)
            for col_name, value in zip(group_cols, keys):
                metrics[col_name] = value
            rows.append(metrics)
        ordered_cols = group_cols + ["n_obs", "mae", "rmse", "mape"]
        return (
            pd.DataFrame(rows)
            .sort_values(group_cols)
            .reset_index(drop=True)[ordered_cols]
        )

    metrics = _compute(data)
    return pd.DataFrame([metrics])[["n_obs", "mae", "rmse", "mape"]]
