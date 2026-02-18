from __future__ import annotations

import pandas as pd

from ng_forecaster.models.corrections.weather_shock_state import (
    apply_weather_shock_state_adjustment,
)


def test_weather_shock_state_adjustment_is_bounded_and_stateful() -> None:
    forecast = pd.DataFrame(
        {
            "horizon": [1, 2],
            "point_forecast": [1000.0, 1020.0],
        }
    )
    adjusted, diag = apply_weather_shock_state_adjustment(
        forecast,
        exogenous_features={
            "freeze_intensity_mtd_weighted": 2.0,
            "freeze_days_mtd_weighted": 8.0,
        },
        cfg={
            "exogenous": {
                "weather_shock_state": {
                    "enabled": True,
                    "persistence": 0.65,
                    "impact_weight": 5000.0,
                    "recovery_weight": 0.55,
                    "cap_abs_adjustment": 1500.0,
                }
            }
        },
    )
    assert diag["enabled"] is True
    assert diag["applied"] is True
    assert (adjusted["weather_state_adjustment"].abs() <= 1500.0 + 1e-9).all()
    # First horizon should carry stronger shock than the second due to recovery.
    assert abs(float(adjusted.iloc[0]["weather_state_adjustment"])) >= abs(
        float(adjusted.iloc[1]["weather_state_adjustment"])
    )
