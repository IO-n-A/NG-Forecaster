from __future__ import annotations

import pandas as pd

from ng_forecaster.utils.calendar import (
    clamp_day_in_month,
    days_in_month,
    horizons_to_month_end_map,
    is_leap_february,
    shift_month_end,
)


def test_calendar_days_and_leap_year_flags_are_correct() -> None:
    assert days_in_month("2024-02-10") == 29
    assert days_in_month("2025-02-10") == 28
    assert is_leap_february("2024-02-29") is True
    assert is_leap_february("2025-02-28") is False


def test_month_end_roll_logic_is_deterministic() -> None:
    anchor = pd.Timestamp("2025-11-30")
    assert shift_month_end(anchor, months=1).date().isoformat() == "2025-12-31"
    assert shift_month_end(anchor, months=2).date().isoformat() == "2026-01-31"
    mapping = horizons_to_month_end_map(
        last_observed_month_end=anchor, horizons=[1, 2, 2]
    )
    assert mapping[1].date().isoformat() == "2025-12-31"
    assert mapping[2].date().isoformat() == "2026-01-31"


def test_clamp_day_in_month_handles_short_months() -> None:
    assert clamp_day_in_month(year=2024, month=2, preferred_day=31) == 29
    assert clamp_day_in_month(year=2025, month=2, preferred_day=31) == 28
    assert clamp_day_in_month(year=2025, month=1, preferred_day=0) == 1
