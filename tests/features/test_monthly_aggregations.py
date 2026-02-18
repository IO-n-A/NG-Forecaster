from __future__ import annotations

import pandas as pd
import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.monthly_aggregations import (
    build_monthly_features,
    build_weighted_freezeoff_features,
)


def _daily_price() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-12-29", periods=14, freq="D"),
            "value": [
                2.0,
                2.1,
                2.2,
                2.4,
                2.5,
                2.4,
                2.6,
                2.8,
                2.7,
                2.9,
                3.1,
                3.0,
                3.2,
                3.3,
            ],
        }
    )


def _weekly_storage() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "timestamp": [
                "2023-12-08",
                "2023-12-15",
                "2023-12-22",
                "2023-12-29",
                "2024-01-05",
                "2024-01-12",
            ],
            "value": [3200, 3210, 3230, 3250, 3260, 3275],
        }
    )


def test_build_monthly_features_is_lag_safe_and_deterministic() -> None:
    features = build_monthly_features(
        _daily_price(), _weekly_storage(), asof="2024-01-10"
    )

    assert len(features) == 8
    assert set(features["feature_name"]) == {
        "hh_mtd_mean",
        "hh_last",
        "hh_vol_7d",
        "hh_diff_1d",
        "stor_last",
        "stor_mean_4w",
        "stor_slope_4w",
        "stor_mom_change",
    }

    hh_last = features.loc[features["feature_name"] == "hh_last", "value"].item()
    stor_last = features.loc[features["feature_name"] == "stor_last", "value"].item()
    stor_mom = features.loc[
        features["feature_name"] == "stor_mom_change", "value"
    ].item()
    assert hh_last == 3.2
    assert stor_last == 3260
    assert stor_mom == 10


def test_build_monthly_features_fails_when_daily_input_missing() -> None:
    with pytest.raises(ContractViolation, match="reason_code=missing_feature_input"):
        build_monthly_features(
            pd.DataFrame(columns=["timestamp", "value"]),
            _weekly_storage(),
            asof="2024-01-10",
        )


def test_build_weighted_freezeoff_features_produces_weighted_rows() -> None:
    weather_panel = pd.DataFrame(
        {
            "basin_id": ["a", "b"],
            "timestamp": [pd.Timestamp("2024-01-31"), pd.Timestamp("2024-01-31")],
            "available_timestamp": [
                pd.Timestamp("2024-01-10"),
                pd.Timestamp("2024-01-10"),
            ],
            "freeze_days": [4, 2],
            "freeze_event_share": [0.4, 0.2],
            "freeze_intensity_c": [2.0, 1.0],
            "coverage_fraction": [1.0, 0.5],
        }
    )

    features = build_weighted_freezeoff_features(weather_panel, asof="2024-01-10")

    assert set(features["feature_name"]) == {
        "coverage_fraction_mtd",
        "extreme_min_mtd",
        "freeze_days_mtd",
        "freeze_days_mtd_weighted",
        "freeze_event_flag",
        "freeze_event_intensity",
        "freeze_event_share_mtd_weighted",
        "freeze_intensity_mtd_weighted",
    }
    assert (features["block_id"] == "weather_freezeoff").all()
