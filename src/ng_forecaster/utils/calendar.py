"""Canonical calendar helpers for month-end and leap-year-safe runtime logic."""

from __future__ import annotations

import calendar as _calendar
from typing import Iterable

import pandas as pd

from ng_forecaster.errors import ContractViolation


def parse_month_end(value: object, *, key: str = "month_end") -> pd.Timestamp:
    """Parse value into calendar month-end timestamp."""

    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ContractViolation(
            "invalid_timestamp",
            key=key,
            detail=f"{key} could not be parsed",
        )
    return ts.to_period("M").to_timestamp("M")


def days_in_month(value: object, *, key: str = "month_end") -> int:
    """Return calendar days in month for the supplied date-like value."""

    return int(parse_month_end(value, key=key).day)


def is_leap_year(year: int) -> bool:
    """Return True when calendar year is leap year."""

    return bool(_calendar.isleap(int(year)))


def is_leap_february(value: object, *, key: str = "month_end") -> bool:
    """Return True when value resolves to February in a leap year."""

    month_end = parse_month_end(value, key=key)
    return bool(month_end.month == 2 and is_leap_year(int(month_end.year)))


def shift_month_end(month_end: object, *, months: int) -> pd.Timestamp:
    """Shift month-end timestamp by integer month offsets and return month-end."""

    anchor = parse_month_end(month_end, key="month_end")
    return (anchor.to_period("M") + int(months)).to_timestamp("M")


def horizons_to_month_end_map(
    *,
    last_observed_month_end: object,
    horizons: Iterable[object],
) -> dict[int, pd.Timestamp]:
    """Map positive horizon integers to month-end timestamps."""

    anchor = parse_month_end(last_observed_month_end, key="last_observed_month_end")
    parsed: list[int] = []
    for value in horizons:
        try:
            horizon = int(str(value).strip())
        except (TypeError, ValueError) as exc:
            raise ContractViolation(
                "invalid_model_policy",
                key="forecast.horizons",
                detail="horizons must be integers",
            ) from exc
        if horizon < 1:
            raise ContractViolation(
                "invalid_model_policy",
                key="forecast.horizons",
                detail="horizons must be positive integers",
            )
        parsed.append(horizon)
    if not parsed:
        raise ContractViolation(
            "invalid_model_policy",
            key="forecast.horizons",
            detail="at least one horizon is required",
        )
    return {
        int(horizon): shift_month_end(anchor, months=int(horizon))
        for horizon in sorted(set(parsed))
    }


def clamp_day_in_month(*, year: int, month: int, preferred_day: int) -> int:
    """Clamp preferred day to a valid day in the specified year/month."""

    month_days = _calendar.monthrange(int(year), int(month))[1]
    return min(max(1, int(preferred_day)), int(month_days))
