from __future__ import annotations

import pandas as pd
import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.models.steo_forecaster import build_steo_forecast


def test_build_steo_forecast_emits_horizon_rows_with_intervals() -> None:
    features = pd.DataFrame(
        [
            {
                "feature_name": "steo_dry_prod_bcfd_t",
                "feature_timestamp": pd.Timestamp("2025-12-31"),
                "available_timestamp": pd.Timestamp("2026-01-10"),
                "value": 100.0,
            },
            {
                "feature_name": "steo_dry_prod_bcfd_t_plus_1",
                "feature_timestamp": pd.Timestamp("2026-01-31"),
                "available_timestamp": pd.Timestamp("2026-01-10"),
                "value": 101.0,
            },
        ]
    )
    release_history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-31", periods=36, freq="ME"),
            "target_value": [3000.0 + idx * 2.0 for idx in range(36)],
        }
    )
    result = build_steo_forecast(
        features,
        target_month="2025-12-31",
        release_history=release_history,
        horizons=[1, 2],
    )

    assert len(result.forecast) == 2
    assert set(result.forecast["horizon"]) == {1, 2}
    point_lookup = {
        int(row["horizon"]): float(row["steo_point_forecast"])
        for _, row in result.forecast.iterrows()
    }
    assert point_lookup[1] == 3100.0  # 100 Bcf/d * 31 days -> Bcf/month scaling
    assert point_lookup[2] == 3131.0  # 101 Bcf/d * 31 days -> Bcf/month scaling
    assert result.diagnostics["target_unit"] == "bcf_per_month"


def test_build_steo_forecast_requires_observation_features() -> None:
    features = pd.DataFrame(
        [
            {
                "feature_name": "steo_driver_active_rigs",
                "feature_timestamp": pd.Timestamp("2025-12-31"),
                "available_timestamp": pd.Timestamp("2026-01-10"),
                "value": 500.0,
            }
        ]
    )
    with pytest.raises(ContractViolation, match="reason_code=missing_column"):
        build_steo_forecast(
            features,
            target_month="2025-12-31",
            horizons=[1],
        )
