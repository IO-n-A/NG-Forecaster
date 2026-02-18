from __future__ import annotations

import pandas as pd

from ng_forecaster.models.corrections.calendar_calibration import (
    apply_calendar_calibration,
)


def test_calendar_calibration_is_bounded() -> None:
    frame = pd.DataFrame(
        [
            {
                "horizon": 1,
                "fused_point": 1000.0,
                "target_month_days": 31,
                "release_anchor_point": 1200.0,
                "anchor_month_end": "2025-01-31",
            }
        ]
    )
    calibrated, _ = apply_calendar_calibration(
        frame,
        calibration_config={
            "enabled": True,
            "max_abs_adjustment": 15.0,
            "day_weights": {"31": 1.0},
        },
        regime_label="normal",
    )
    assert float(calibrated.iloc[0]["calendar_calibration_delta"]) == 15.0
    assert float(calibrated.iloc[0]["fused_point"]) == 1015.0


def test_calendar_calibration_day_direction_and_leap_flag() -> None:
    frame = pd.DataFrame(
        [
            {
                "horizon": 1,
                "fused_point": 1000.0,
                "target_month_days": 31,
                "release_anchor_point": 1100.0,
                "anchor_month_end": "2025-01-31",
            },
            {
                "horizon": 2,
                "fused_point": 1000.0,
                "target_month_days": 30,
                "release_anchor_point": 1100.0,
                "anchor_month_end": "2025-04-30",
            },
            {
                "horizon": 3,
                "fused_point": 1000.0,
                "target_month_days": 29,
                "release_anchor_point": 1000.0,
                "anchor_month_end": "2024-02-29",
            },
        ]
    )
    calibrated, _ = apply_calendar_calibration(
        frame,
        calibration_config={
            "enabled": True,
            "max_abs_adjustment": 100.0,
            "day_weights": {"31": 0.1, "30": -0.1, "29": 0.0},
            "leap_february_bonus": -3.0,
        },
        regime_label="normal",
    )
    row_31 = calibrated[calibrated["target_month_days"] == 31].iloc[0]
    row_30 = calibrated[calibrated["target_month_days"] == 30].iloc[0]
    row_29 = calibrated[calibrated["target_month_days"] == 29].iloc[0]
    assert float(row_31["calendar_calibration_delta"]) > 0
    assert float(row_30["calendar_calibration_delta"]) < 0
    assert bool(row_29["is_leap_february"]) is True
    assert float(row_29["calendar_calibration_delta"]) == -3.0
