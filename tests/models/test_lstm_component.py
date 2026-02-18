from __future__ import annotations

import pandas as pd
import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.models.neural.lstm_component import forecast_component_with_lstm


def _signal() -> pd.Series:
    return pd.Series([100.0 + idx * 0.5 + ((idx % 6) - 3) * 0.3 for idx in range(60)])


def test_component_lstm_is_deterministic_for_same_seed() -> None:
    cfg = {
        "lookback": 36,
        "hidden_units": 8,
        "batch_size": 16,
        "learning_rate": 0.001,
        "repeat_runs": 5,
    }
    first = forecast_component_with_lstm(
        _signal(),
        horizons=[1, 2],
        lstm_config=cfg,
        component_name="PF1::mode_1",
        base_seed=17,
    )
    second = forecast_component_with_lstm(
        _signal(),
        horizons=[1, 2],
        lstm_config=cfg,
        component_name="PF1::mode_1",
        base_seed=17,
    )

    assert first.point_forecast.equals(second.point_forecast)
    assert first.run_forecasts.equals(second.run_forecasts)
    assert first.diagnostics == second.diagnostics
    assert first.diagnostics["input_shape"] == [36, 1]


def test_component_lstm_requires_sufficient_points_for_lookback() -> None:
    with pytest.raises(
        ContractViolation, match="reason_code=insufficient_training_data"
    ):
        forecast_component_with_lstm(
            pd.Series([1.0] * 12),
            horizons=[1],
            lstm_config={
                "lookback": 12,
                "hidden_units": 8,
                "batch_size": 16,
                "learning_rate": 0.001,
                "repeat_runs": 2,
            },
            component_name="PF2::mode_1",
        )
