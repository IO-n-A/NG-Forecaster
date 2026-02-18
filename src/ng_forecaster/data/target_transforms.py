"""Deterministic target transforms for month-length normalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
import pandas as pd

from ng_forecaster.errors import ContractViolation
from ng_forecaster.utils.calendar import (
    days_in_month,
    horizons_to_month_end_map,
    parse_month_end,
)


@dataclass(frozen=True)
class MonthLengthContext:
    """Month-length context used to map forecast horizons to calendar months."""

    last_observed_month: pd.Period


def ensure_monthly_timestamp(frame: pd.DataFrame, *, timestamp_col: str) -> pd.Series:
    """Validate and return the timestamp series required for month-length transforms."""

    if timestamp_col not in frame.columns:
        raise ContractViolation(
            "missing_column",
            key=timestamp_col,
            detail=f"{timestamp_col} is required for target transforms",
        )
    timestamps = pd.to_datetime(frame[timestamp_col], errors="coerce")
    if timestamps.isna().any():
        raise ContractViolation(
            "invalid_timestamp",
            key=timestamp_col,
            detail="target transforms require parseable timestamps",
        )
    if timestamps.empty:
        raise ContractViolation(
            "insufficient_training_data",
            key=timestamp_col,
            detail="target transforms require at least one timestamp row",
        )
    return timestamps


def days_in_month_from_timestamps(timestamps: Iterable[object]) -> pd.Series:
    """Return month lengths for each timestamp in the input iterable."""

    parsed = pd.to_datetime(list(timestamps), errors="coerce")
    if pd.isna(parsed).any():
        raise ContractViolation(
            "invalid_timestamp",
            key="timestamps",
            detail="timestamps contain unparseable values",
        )
    return pd.Series([days_in_month(value, key="timestamps") for value in parsed])


def monthly_total_to_daily_average(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    value_col: str,
    out_col: str = "target_value_per_day",
) -> tuple[pd.DataFrame, MonthLengthContext]:
    """Convert monthly totals to daily-average units with month-length metadata."""

    if value_col not in frame.columns:
        raise ContractViolation(
            "missing_column",
            key=value_col,
            detail=f"{value_col} is required for target transforms",
        )

    converted = frame.copy()
    timestamps = ensure_monthly_timestamp(converted, timestamp_col=timestamp_col)
    values = pd.to_numeric(converted[value_col], errors="coerce")
    if values.isna().any():
        raise ContractViolation(
            "invalid_model_policy",
            key=value_col,
            detail="target transform received non-numeric values",
        )

    values_float = values.astype(float)
    if not np.isfinite(values_float.to_numpy(dtype=float)).all():
        raise ContractViolation(
            "invalid_model_policy",
            key=value_col,
            detail="target transform received non-finite values",
        )

    month_days = timestamps.dt.days_in_month.astype(int)
    converted["days_in_month"] = month_days
    converted[out_col] = values_float / month_days.astype(float)

    context = MonthLengthContext(last_observed_month=timestamps.max().to_period("M"))
    return converted, context


def horizons_to_month_ends(
    *,
    context: MonthLengthContext,
    horizons: Iterable[object],
) -> dict[int, pd.Timestamp]:
    """Map horizon integers to calendar month-end timestamps."""

    return horizons_to_month_end_map(
        last_observed_month_end=context.last_observed_month.to_timestamp("M"),
        horizons=horizons,
    )


def daily_average_to_monthly_total(
    per_day_value: float,
    *,
    month_end: object,
) -> float:
    """Convert a daily-average value to a monthly total using month_end day count."""

    parsed_value = float(per_day_value)
    if not np.isfinite(parsed_value):
        raise ContractViolation(
            "invalid_model_policy",
            key="per_day_value",
            detail="per_day_value must be finite",
        )

    parsed_month = parse_month_end(month_end, key="month_end")
    return parsed_value * float(int(parsed_month.day))
