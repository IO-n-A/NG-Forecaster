from __future__ import annotations

from pathlib import Path

import pandas as pd

from ng_forecaster.models.prototypes import (
    build_cohort_kernel_forecast,
    fit_kernel_parameters,
    load_drilling_metrics_history,
)


def _build_release_history(periods: int = 24) -> pd.DataFrame:
    timestamps = pd.date_range("2023-01-31", periods=periods, freq="ME")
    values = [3_000_000.0 + idx * 12_000.0 for idx in range(periods)]
    return pd.DataFrame({"timestamp": timestamps, "target_value": values})


def _build_drilling_history(periods: int = 24) -> pd.DataFrame:
    timestamps = pd.date_range("2023-01-31", periods=periods, freq="ME")
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "available_timestamp": [ts + pd.Timedelta(days=6) for ts in timestamps],
            "new_wells_completed": [900 + idx * 2 for idx in range(periods)],
            "new_well_gas_production": [3400 + idx * 8 for idx in range(periods)],
            "existing_gas_production_change": [
                -3200 + idx * 3 for idx in range(periods)
            ],
            "duc_inventory": [5200 - idx * 5 for idx in range(periods)],
            "completion_to_duc_ratio": [0.16 + idx * 0.0005 for idx in range(periods)],
        }
    )


def test_load_drilling_metrics_history_selects_latest_available_vintage(
    tmp_path: Path,
) -> None:
    rows = _build_drilling_history(periods=14)
    duplicate = rows.iloc[[5]].copy()
    duplicate["available_timestamp"] = duplicate["available_timestamp"] + pd.Timedelta(
        days=20
    )
    duplicate["new_wells_completed"] = duplicate["new_wells_completed"] + 99
    panel = pd.concat([rows, duplicate], ignore_index=True)
    panel_path = tmp_path / "steo_drilling_metrics_panel.parquet"
    panel.to_parquet(panel_path, index=False)

    loaded = load_drilling_metrics_history(
        asof="2024-06-30",
        panel_path=str(panel_path),
        lookback_months=12,
    )
    assert len(loaded) == 12
    selected = loaded[loaded["timestamp"] == pd.Timestamp("2023-06-30")]
    assert len(selected) == 1
    assert float(selected.iloc[0]["new_wells_completed"]) == float(
        rows.iloc[5]["new_wells_completed"] + 99
    )


def test_fit_kernel_parameters_returns_constrained_payload() -> None:
    params = fit_kernel_parameters(
        release_history=_build_release_history(periods=24),
        drilling_history=_build_drilling_history(periods=24),
    )
    assert params["training_rows"] >= 12
    assert 1.0 <= float(params["half_life_months"]) <= 6.0
    assert float(params["completion_coef"]) >= 0.0
    assert float(params["duc_coef"]) >= 0.0


def test_build_cohort_kernel_forecast_emits_two_horizons() -> None:
    release_history = _build_release_history(periods=24)
    drilling_history = _build_drilling_history(periods=24)
    params = fit_kernel_parameters(
        release_history=release_history,
        drilling_history=drilling_history,
    )
    result = build_cohort_kernel_forecast(
        drilling_history=drilling_history,
        target_month="2024-12-31",
        release_history=release_history,
        horizons=[1, 2],
        kernel_params=params,
    )
    assert list(result.forecast["horizon"]) == [1, 2]
    assert {
        "prototype_point_forecast",
        "prototype_lower_95",
        "prototype_upper_95",
    }.issubset(set(result.forecast.columns))
    assert result.diagnostics["training_rows"] >= 12
