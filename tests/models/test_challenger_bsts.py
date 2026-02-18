from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.models.challenger_bsts import run_challenger_model


def _frame_with_trailing_nan() -> pd.DataFrame:
    values = [80 + i * 0.5 + ((i % 4) - 1.5) * 0.2 for i in range(52)] + [None, None]
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2021-01-31", periods=54, freq="M"),
            "target_value": values,
        }
    )


def test_challenger_outputs_student_t_intervals() -> None:
    result = run_challenger_model(
        _frame_with_trailing_nan(),
        {
            "training": {"lookback": 24},
            "forecast": {"horizons": [1, 2], "alpha": 0.95},
        },
    )

    forecast = result.forecast
    assert {
        "horizon",
        "horizon_month_offset",
        "horizon_label",
        "mean_forecast",
        "lower_95",
        "upper_95",
        "residual_scale",
    }.issubset(set(forecast.columns))
    assert (forecast["lower_95"] < forecast["mean_forecast"]).all()
    assert (forecast["mean_forecast"] < forecast["upper_95"]).all()
    assert result.diagnostics["trailing_nan_slots"] == 2
    assert result.diagnostics["horizon_label"]["1"] == "horizon_1"


def test_challenger_requires_sufficient_history() -> None:
    short = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-01-31", periods=8, freq="M"),
            "target_value": [float(i) for i in range(8)],
        }
    )
    with pytest.raises(
        ContractViolation, match="reason_code=insufficient_training_data"
    ):
        run_challenger_model(short, {"training": {"lookback": 12}})


def test_challenger_reconverts_daily_forecast_to_month_totals() -> None:
    timestamps = pd.date_range("2020-01-31", "2025-01-31", freq="M")
    frame = pd.DataFrame(
        {
            "timestamp": timestamps,
            "target_value": [float(ts.days_in_month * 100.0) for ts in timestamps],
        }
    )
    result = run_challenger_model(
        frame,
        {
            "training": {"lookback": 24},
            "forecast": {"horizons": [1, 2], "alpha": 0.95},
            "target_transform": {"normalize_by_days_in_month": True},
        },
    )
    points = {
        int(row["horizon"]): float(row["mean_forecast"])
        for _, row in result.forecast.iterrows()
    }
    assert points[1] == 2800.0  # February 2025
    assert points[2] == 3100.0  # March 2025
    assert result.diagnostics["target_transform"]["normalize_by_days_in_month"] is True


def test_challenger_rejects_disabled_optional_log_transform() -> None:
    with pytest.raises(ContractViolation, match="target_transform.log"):
        run_challenger_model(
            _frame_with_trailing_nan(),
            {
                "training": {"lookback": 24},
                "target_transform": {
                    "normalize_by_days_in_month": True,
                    "log": True,
                },
            },
        )


def test_challenger_applies_transfer_prior_exogenous_adjustments_when_enabled() -> None:
    base = run_challenger_model(
        _frame_with_trailing_nan(),
        {
            "training": {"lookback": 24},
            "forecast": {"horizons": [1, 2], "alpha": 0.95},
        },
    )
    adjusted = run_challenger_model(
        _frame_with_trailing_nan(),
        {
            "training": {"lookback": 24},
            "forecast": {"horizons": [1, 2], "alpha": 0.95},
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

    assert "transfer_prior_adjustment" in adjusted.forecast.columns
    assert adjusted.diagnostics["exogenous_transfer_priors"]["enabled"] is True
    assert not adjusted.forecast["mean_forecast"].equals(base.forecast["mean_forecast"])
    assert (adjusted.forecast["transfer_prior_adjustment"].abs() > 0).all()
    assert (adjusted.forecast["lower_95"] < adjusted.forecast["mean_forecast"]).all()
    assert (adjusted.forecast["mean_forecast"] < adjusted.forecast["upper_95"]).all()


def test_challenger_requires_exogenous_features_when_transfer_block_enabled() -> None:
    with pytest.raises(ContractViolation, match="exogenous_features"):
        run_challenger_model(
            _frame_with_trailing_nan(),
            {
                "training": {"lookback": 24},
                "exogenous": {"transfer_priors": {"enabled": True}},
            },
        )


def test_challenger_pymc_engine_requires_optional_dependency() -> None:
    if importlib.util.find_spec("pymc") is not None:
        pytest.skip("pymc installed; dependency check path not applicable")
    with pytest.raises(ContractViolation, match="reason_code=missing_dependency"):
        run_challenger_model(
            _frame_with_trailing_nan(),
            {
                "model": {"engine": "pymc"},
                "training": {"lookback": 24},
                "forecast": {"horizons": [1, 2], "alpha": 0.95},
            },
        )


def test_challenger_pymc_engine_writes_posterior_artifact(tmp_path) -> None:
    if importlib.util.find_spec("pymc") is None:
        pytest.skip("pymc not installed")
    result = run_challenger_model(
        _frame_with_trailing_nan(),
        {
            "model": {"engine": "pymc"},
            "training": {"lookback": 24},
            "state_space": {"pymc_draws": 100, "pymc_tune": 100, "pymc_chains": 1},
            "forecast": {"horizons": [1, 2], "alpha": 0.95},
        },
        artifact_root=tmp_path,
        artifact_tag="pymc-smoke",
    )
    backend = result.diagnostics["backend_diagnostics"]
    posterior_path = backend.get("posterior_path", "")
    assert result.diagnostics["model_engine"] == "pymc"
    assert backend.get("engine") == "pymc"
    assert posterior_path
    assert Path(str(posterior_path)).exists()
    assert (result.forecast["lower_95"] < result.forecast["mean_forecast"]).all()
    assert (result.forecast["mean_forecast"] < result.forecast["upper_95"]).all()
