from __future__ import annotations

import pandas as pd
import pytest

from ng_forecaster.data.target_transforms import (
    daily_average_to_monthly_total,
    days_in_month_from_timestamps,
    horizons_to_month_ends,
    monthly_total_to_daily_average,
)
from ng_forecaster.errors import ContractViolation


def test_month_length_round_trip_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-01-31", "2024-02-29", "2024-04-30"],
            "target_value": [3100.0, 2900.0, 3000.0],
        }
    )

    converted, context = monthly_total_to_daily_average(
        frame,
        timestamp_col="timestamp",
        value_col="target_value",
    )

    assert converted["days_in_month"].tolist() == [31, 29, 30]
    assert converted["target_value_per_day"].round(8).tolist() == [100.0, 100.0, 100.0]
    assert str(context.last_observed_month) == "2024-04"

    month_map = horizons_to_month_ends(context=context, horizons=[1, 2, 3])
    assert month_map[1].date().isoformat() == "2024-05-31"
    assert month_map[2].date().isoformat() == "2024-06-30"
    assert month_map[3].date().isoformat() == "2024-07-31"

    totals = [
        daily_average_to_monthly_total(100.0, month_end=month_map[horizon])
        for horizon in (1, 2, 3)
    ]
    assert totals == [3100.0, 3000.0, 3100.0]


def test_days_in_month_from_timestamps_handles_leap_boundaries() -> None:
    days = days_in_month_from_timestamps(["2025-02-28", "2024-02-29", "2025-03-31"])
    assert days.tolist() == [28, 29, 31]


def test_monthly_total_to_daily_average_rejects_invalid_timestamp() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-01-31", "not-a-date"],
            "target_value": [3100.0, 2900.0],
        }
    )
    with pytest.raises(ContractViolation, match="reason_code=invalid_timestamp"):
        monthly_total_to_daily_average(
            frame,
            timestamp_col="timestamp",
            value_col="target_value",
        )


def test_monthly_total_to_daily_average_rejects_non_finite_values() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-01-31", "2024-02-29"],
            "target_value": [3100.0, float("inf")],
        }
    )
    with pytest.raises(ContractViolation, match="reason_code=invalid_model_policy"):
        monthly_total_to_daily_average(
            frame,
            timestamp_col="timestamp",
            value_col="target_value",
        )


def test_horizons_to_month_ends_rejects_non_positive_horizons() -> None:
    frame = pd.DataFrame(
        {
            "timestamp": ["2024-01-31", "2024-02-29"],
            "target_value": [3100.0, 2900.0],
        }
    )
    _, context = monthly_total_to_daily_average(
        frame,
        timestamp_col="timestamp",
        value_col="target_value",
    )
    with pytest.raises(ContractViolation, match="reason_code=invalid_model_policy"):
        horizons_to_month_ends(context=context, horizons=[0, 1])
