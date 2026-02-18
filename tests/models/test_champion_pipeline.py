from __future__ import annotations

import pandas as pd
import pytest

from ng_forecaster.errors import ContractViolation
import ng_forecaster.models.champion_wpd_vmd_lstm as champion_module
from ng_forecaster.models.champion_wpd_vmd_lstm import (
    run_champion_pipeline,
    run_champion_seed_repeats,
)
from ng_forecaster.models.neural import ComponentLSTMForecast


def _training_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-31", periods=60, freq="M"),
            "target_value": [
                90 + idx * 0.4 + ((idx % 6) - 2) * 0.7 for idx in range(60)
            ],
        }
    )


def test_champion_pipeline_is_deterministic_for_same_seed() -> None:
    config = {
        "training": {"lookback": 36, "seed": 7, "seed_noise_scale": 0.0},
        "forecast": {"horizons": [1, 2]},
    }

    frame = _training_frame()
    first = run_champion_pipeline(frame, config)
    second = run_champion_pipeline(frame, config)

    assert first.point_forecast.equals(second.point_forecast)
    assert {"horizon_month_offset", "horizon_label"}.issubset(
        set(first.point_forecast.columns)
    )
    assert first.diagnostics == second.diagnostics
    assert first.diagnostics["horizon_month_offset"]["1"] == 0
    assert first.diagnostics["component_lstm_input_shape"] == [36, 1]
    assert first.diagnostics["lstm_repeat_runs"] == 5
    assert len(first.wpd_components) > 0
    assert len(first.vmd_components) > 0


def test_champion_seed_repeats_return_seed_dimension() -> None:
    config = {
        "training": {"lookback": 36, "seed_noise_scale": 0.03},
        "forecast": {"horizons": [1, 2]},
    }

    repeated = run_champion_seed_repeats(_training_frame(), config, seeds=[3, 5, 7])
    assert set(repeated.forecasts["seed"]) == {3, 5, 7}
    assert set(repeated.forecasts["horizon"]) == {1, 2}


def test_champion_one_layer_variant_runs_without_vmd_redecomposition() -> None:
    config = {
        "model": {"variant": "wpd_lstm_one_layer"},
        "training": {"lookback": 36, "seed": 13, "seed_noise_scale": 0.0},
        "forecast": {"horizons": [1, 2]},
    }

    result = run_champion_pipeline(_training_frame(), config)
    assert result.diagnostics["decomposition_depth"] == "one_layer"
    assert result.diagnostics["vmd_applied"] is False
    assert result.diagnostics["vmd_modes"] == 1

    mode_counts = result.vmd_components.groupby("component_name")["mode_name"].nunique()
    assert not mode_counts.empty
    assert (mode_counts == 1).all()
    assert "seed_repeat_stats" in result.diagnostics


def test_champion_two_layer_strategy2_uses_grouped_components() -> None:
    config = {
        "model": {"variant": "wpd_vmd_lstm2"},
        "training": {"lookback": 36, "seed": 19, "seed_noise_scale": 0.0},
        "forecast": {"horizons": [1, 2]},
    }
    result = run_champion_pipeline(_training_frame(), config)
    groups = sorted(result.vmd_components["component_name"].unique().tolist())
    assert groups
    assert all(name.startswith("group_") for name in groups)


def test_champion_requires_lookback_observations() -> None:
    short_frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-31", periods=6, freq="M"),
            "target_value": [1, 2, 3, 4, 5, 6],
        }
    )
    with pytest.raises(
        ContractViolation, match="reason_code=insufficient_training_data"
    ):
        run_champion_pipeline(short_frame, {"training": {"lookback": 12}})


def test_champion_reconverts_daily_forecast_to_month_totals(monkeypatch) -> None:
    def _fake_forecast_component_with_lstm(
        values,
        *,
        horizons,
        lstm_config,
        component_name,
        base_seed=42,
    ) -> ComponentLSTMForecast:
        _ = (values, component_name, base_seed)
        repeat_runs = int(lstm_config.get("repeat_runs", 1))
        point_forecast = pd.DataFrame(
            {
                "horizon": [int(h) for h in horizons],
                "point_forecast": [100.0 for _ in horizons],
            }
        )
        run_forecasts = pd.DataFrame(
            [
                {
                    "horizon": int(horizon),
                    "run_index": int(run_index),
                    "point_forecast": 100.0,
                }
                for horizon in horizons
                for run_index in range(repeat_runs)
            ]
        )
        return ComponentLSTMForecast(
            point_forecast=point_forecast,
            run_forecasts=run_forecasts,
            diagnostics={"readout_rmse_mean": 0.0},
        )

    monkeypatch.setattr(
        champion_module,
        "forecast_component_with_lstm",
        _fake_forecast_component_with_lstm,
    )

    timestamps = pd.date_range("2020-01-31", "2025-01-31", freq="M")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "target_value": [
                float(ts.days_in_month * 100.0 + idx * 0.1)
                for idx, ts in enumerate(timestamps)
            ],
        }
    )
    result = run_champion_pipeline(
        frame,
        {
            "model": {"variant": "wpd_lstm_one_layer"},
            "lstm": {"lookback": 36, "repeat_runs": 2},
            "forecast": {"horizons": [1, 2]},
            "target_transform": {"normalize_by_days_in_month": True},
        },
    )

    points = {
        int(row["horizon"]): float(row["point_forecast"])
        for _, row in result.point_forecast.iterrows()
    }
    assert points[2] > points[1]
    assert points[1] / 28.0 == points[2] / 31.0
    assert result.diagnostics["target_transform"]["normalize_by_days_in_month"] is True


def test_champion_rejects_disabled_optional_log_transform() -> None:
    with pytest.raises(ContractViolation, match="target_transform.log"):
        run_champion_pipeline(
            _training_frame(),
            {
                "target_transform": {
                    "normalize_by_days_in_month": True,
                    "log": True,
                }
            },
        )


def test_champion_applies_transfer_prior_exogenous_adjustments_when_enabled() -> None:
    frame = _training_frame()
    base = run_champion_pipeline(
        frame,
        {
            "training": {"lookback": 36, "seed": 21, "seed_noise_scale": 0.0},
            "forecast": {"horizons": [1, 2]},
        },
    )
    adjusted = run_champion_pipeline(
        frame,
        {
            "training": {"lookback": 36, "seed": 21, "seed_noise_scale": 0.0},
            "forecast": {"horizons": [1, 2]},
            "exogenous": {
                "transfer_priors": {
                    "enabled": True,
                    "prior_weight": 5000.0,
                    "dispersion_weight": 1000.0,
                    "prior_scale": 1000.0,
                    "dispersion_scale": 100.0,
                }
            },
        },
        exogenous_features={
            "transfer_prior_us_bcfd_t": 3200.0,
            "transfer_prior_dispersion_t": 200.0,
            "transfer_prior_us_bcfd_t_plus_1": 3300.0,
            "transfer_prior_dispersion_t_plus_1": 220.0,
        },
    )

    assert "transfer_prior_adjustment" in adjusted.point_forecast.columns
    assert adjusted.diagnostics["exogenous_transfer_priors"]["enabled"] is True
    assert not adjusted.point_forecast["point_forecast"].equals(
        base.point_forecast["point_forecast"]
    )
    assert (adjusted.point_forecast["transfer_prior_adjustment"].abs() > 0).all()


def test_champion_requires_exogenous_features_when_transfer_block_enabled() -> None:
    with pytest.raises(ContractViolation, match="exogenous_features"):
        run_champion_pipeline(
            _training_frame(),
            {
                "training": {"lookback": 36},
                "exogenous": {"transfer_priors": {"enabled": True}},
            },
        )


def test_champion_dispatches_to_pytorch_engine_when_configured(monkeypatch) -> None:
    def _fake_forecast_component_with_pytorch_lstm(
        values,
        *,
        horizons,
        lstm_config,
        component_name,
        base_seed=42,
        artifact_dir=None,
    ) -> ComponentLSTMForecast:
        _ = (values, lstm_config, component_name, base_seed, artifact_dir)
        point_forecast = pd.DataFrame(
            {
                "horizon": [int(h) for h in horizons],
                "point_forecast": [101.0 for _ in horizons],
            }
        )
        run_forecasts = pd.DataFrame(
            [
                {
                    "horizon": int(horizon),
                    "run_index": 0,
                    "point_forecast": 101.0,
                }
                for horizon in horizons
            ]
        )
        return ComponentLSTMForecast(
            point_forecast=point_forecast,
            run_forecasts=run_forecasts,
            diagnostics={"engine": "pytorch"},
        )

    monkeypatch.setattr(
        champion_module,
        "forecast_component_with_pytorch_lstm",
        _fake_forecast_component_with_pytorch_lstm,
    )

    result = run_champion_pipeline(
        _training_frame(),
        {
            "model": {"variant": "wpd_lstm_one_layer"},
            "lstm": {
                "engine": "pytorch",
                "lookback": 36,
                "repeat_runs": 1,
            },
            "forecast": {"horizons": [1, 2]},
        },
    )
    assert result.diagnostics["lstm_engine"] == "pytorch"
