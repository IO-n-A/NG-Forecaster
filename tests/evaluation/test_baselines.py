from __future__ import annotations

import pandas as pd

from ng_forecaster.evaluation.baselines import build_baseline_point_estimates


def test_build_baseline_point_estimates_emits_three_deterministic_baselines() -> None:
    history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-31", periods=30, freq="ME"),
            "target_value": [3000.0 + idx * 5.0 for idx in range(30)],
        }
    )
    features = pd.DataFrame(
        [
            {
                "feature_name": "steo_dry_prod_bcfd_t",
                "feature_timestamp": pd.Timestamp("2025-06-30"),
                "available_timestamp": pd.Timestamp("2025-05-31"),
                "value": 102.0,
            }
        ]
    )

    estimates = build_baseline_point_estimates(
        history,
        target_month="2025-06-30",
        feature_rows=features,
    )
    assert len(estimates) == 3
    lookup = {item.model_variant: item for item in estimates}
    assert set(lookup) == {
        "baseline_seasonal_naive",
        "baseline_drift",
        "baseline_steo_anchor_naive",
    }
    assert lookup["baseline_seasonal_naive"].fused_point == 3085.0
    assert lookup["baseline_steo_anchor_naive"].fused_point == 3060.0
    assert lookup["baseline_steo_anchor_naive"].metadata["anchor_unit_out"] == "bcf_per_month"


def test_baseline_steo_anchor_falls_back_to_seasonal_without_feature_rows() -> None:
    history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-31", periods=30, freq="ME"),
            "target_value": [3000.0 + idx * 5.0 for idx in range(30)],
        }
    )

    estimates = build_baseline_point_estimates(
        history,
        target_month="2025-06-30",
        feature_rows=pd.DataFrame(),
    )
    lookup = {item.model_variant: item for item in estimates}
    assert (
        lookup["baseline_steo_anchor_naive"].fused_point
        == lookup["baseline_seasonal_naive"].fused_point
    )
