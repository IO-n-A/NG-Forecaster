from __future__ import annotations

import pandas as pd
import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.models.fusion import (
    build_fusion_result,
    fuse_forecasts,
    summarize_seed_stability,
)
from ng_forecaster.models.selection import (
    enforce_divergence_gate,
    enforce_stability_gate,
    select_model_by_metric,
)


def _champion() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "horizon": [1, 2],
            "point_forecast": [101.0, 102.5],
            "seed": [42, 42],
        }
    )


def _challenger() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "horizon": [1, 2],
            "mean_forecast": [100.5, 102.0],
            "lower_95": [99.2, 100.1],
            "upper_95": [101.8, 103.9],
            "residual_scale": [0.8, 1.0],
        }
    )


def _seed_runs() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "seed": [1, 2, 3, 1, 2, 3],
            "horizon": [1, 1, 1, 2, 2, 2],
            "point_forecast": [100.8, 101.0, 100.9, 102.1, 102.6, 102.4],
        }
    )


def _realized() -> pd.DataFrame:
    return pd.DataFrame({"horizon": [1, 2], "realized_value": [100.9, 102.2]})


def test_fusion_bundle_produces_divergence_stability_calibration() -> None:
    fused = build_fusion_result(
        _champion(),
        _challenger(),
        _seed_runs(),
        _realized(),
        champion_weight=0.6,
    )

    assert set(fused.forecast["horizon"]) == {1, 2}
    assert "mean_abs_divergence" in fused.divergence_summary
    assert "coverage_rate" in fused.calibration_summary
    assert "spread" in fused.stability_summary.columns


def test_selection_and_gates_use_metric_outputs() -> None:
    metrics = pd.DataFrame(
        {
            "candidate_model": ["champion", "challenger"],
            "rmse": [0.45, 0.61],
        }
    )
    decision = select_model_by_metric(metrics)
    assert decision.selected_model == "champion"

    stability = summarize_seed_stability(_seed_runs())
    stability_pass, _ = enforce_stability_gate(stability, max_std=0.5)
    assert stability_pass

    fused = fuse_forecasts(_champion(), _challenger(), champion_weight=0.6)
    divergence_pass, _ = enforce_divergence_gate(
        {"mean_abs_divergence": float(fused["abs_divergence"].mean())},
        max_mean_abs_divergence=1.0,
    )
    assert divergence_pass


def test_fusion_raises_when_interval_order_contract_is_violated() -> None:
    champion = pd.DataFrame(
        {
            "horizon": [1],
            "point_forecast": [200.0],
            "seed": [42],
        }
    )
    challenger = pd.DataFrame(
        {
            "horizon": [1],
            "mean_forecast": [100.0],
            "lower_95": [99.0],
            "upper_95": [101.0],
            "residual_scale": [1.0],
        }
    )

    with pytest.raises(ContractViolation, match="reason_code=interval_order_violation"):
        fuse_forecasts(champion, challenger, champion_weight=0.7)


def test_fusion_applies_release_anchor_when_history_is_available() -> None:
    release_history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-31", periods=24, freq="ME"),
            "target_value": [3000.0 + idx * 5.0 for idx in range(24)],
        }
    )

    fused = fuse_forecasts(
        _champion(),
        _challenger(),
        champion_weight=0.6,
        release_history=release_history,
        release_anchor_weight=0.2,
    )

    assert "release_anchor_point" in fused.columns
    assert fused["release_anchor_applied"].all()


def test_fusion_release_anchor_uses_month_length_normalized_delta() -> None:
    release_history = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(
                ["2024-10-31", "2024-11-30", "2024-12-31", "2025-01-31"]
            ),
            "target_value": [3100.0, 3000.0, 3100.0, 3100.0],
        }
    )
    fused = fuse_forecasts(
        _champion(),
        _challenger(),
        champion_weight=0.6,
        release_history=release_history,
        release_anchor_weight=0.2,
    )

    anchor_lookup = {
        int(row["horizon"]): float(row["release_anchor_point"])
        for _, row in fused.iterrows()
    }
    assert anchor_lookup[1] == 2800.0  # February 2025
    assert anchor_lookup[2] == 3100.0  # March 2025


def test_fusion_supports_horizon_specific_weights_and_regime_label() -> None:
    fused = fuse_forecasts(
        _champion(),
        _challenger(),
        champion_weight=0.7,
        horizon_weights={1: 0.8, 2: 0.6},
        regime_label="freeze_off",
    )

    lookup = {
        int(row["horizon"]): float(row["applied_champion_weight"])
        for _, row in fused.iterrows()
    }
    assert lookup == {1: 0.8, 2: 0.6}
    assert set(fused["regime_label"]) == {"freeze_off"}


def test_fusion_supports_optional_steo_stream_blending() -> None:
    steo = pd.DataFrame(
        {
            "horizon": [1, 2],
            "steo_point_forecast": [100.2, 101.9],
            "steo_lower_95": [99.8, 101.2],
            "steo_upper_95": [100.8, 102.6],
        }
    )
    fused = fuse_forecasts(
        _champion(),
        _challenger(),
        champion_weight=0.6,
        steo_forecast=steo,
        steo_weight=0.1,
    )

    assert set(fused["steo_applied"]) == {True}
    assert (fused["applied_steo_weight"] == 0.1).all()
    assert "steo_point_forecast" in fused.columns


def test_fusion_supports_optional_prototype_stream_blending() -> None:
    prototype = pd.DataFrame(
        {
            "horizon": [1, 2],
            "prototype_point_forecast": [100.6, 102.3],
            "prototype_lower_95": [99.9, 101.4],
            "prototype_upper_95": [101.5, 103.1],
        }
    )
    fused = fuse_forecasts(
        _champion(),
        _challenger(),
        champion_weight=0.6,
        prototype_forecast=prototype,
        prototype_weight=0.08,
    )

    assert set(fused["prototype_applied"]) == {True}
    assert (fused["applied_prototype_weight"] == 0.08).all()
    assert "prototype_point_forecast" in fused.columns


def test_fusion_applies_month_length_bias_weight_to_31_day_targets() -> None:
    release_history = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(["2024-11-30", "2024-12-31", "2025-01-31"]),
            "target_value": [3000.0, 3100.0, 3200.0],
        }
    )
    fused = fuse_forecasts(
        _champion(),
        _challenger(),
        champion_weight=0.6,
        release_history=release_history,
        release_anchor_weight=0.2,
        month_length_bias_weight=0.1,
    )

    lookup = {
        int(row["horizon"]): (
            int(row["target_month_days"]),
            bool(row["month_length_bias_applied"]),
            float(row["applied_month_length_bias_weight"]),
        )
        for _, row in fused.iterrows()
    }
    assert lookup[1][0] == 28
    assert lookup[1][1] is False
    assert lookup[1][2] == 0.0
    assert lookup[2][0] == 31
    assert lookup[2][1] is True
    assert lookup[2][2] == 0.1
