from __future__ import annotations

import pandas as pd

from ng_forecaster.evaluation.baselines import build_baseline_point_estimates


def _lookup_steo_baseline(
    history: pd.DataFrame,
    *,
    target_month: str,
    bcfd_value: float,
):
    features = pd.DataFrame(
        [
            {
                "feature_name": "steo_dry_prod_bcfd_t",
                "feature_timestamp": pd.Timestamp(target_month),
                "available_timestamp": pd.Timestamp(target_month)
                - pd.offsets.MonthEnd(1),
                "value": bcfd_value,
            }
        ]
    )
    estimates = build_baseline_point_estimates(
        history,
        target_month=target_month,
        feature_rows=features,
    )
    return {item.model_variant: item for item in estimates}[
        "baseline_steo_anchor_naive"
    ]


def test_steo_anchor_baseline_uses_mmcf_per_month_when_target_is_mmcf() -> None:
    history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-31", periods=36, freq="ME"),
            "target_value": [3_000_000.0 + idx * 1000.0 for idx in range(36)],
        }
    )
    baseline = _lookup_steo_baseline(
        history,
        target_month="2025-02-28",
        bcfd_value=100.0,
    )
    assert baseline.metadata["anchor_unit_out"] == "mmcf_per_month"
    assert baseline.fused_point == 2_800_000.0


def test_steo_anchor_baseline_uses_bcf_per_month_when_target_is_bcf() -> None:
    history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-31", periods=36, freq="ME"),
            "target_value": [3_000.0 + idx * 2.0 for idx in range(36)],
        }
    )
    baseline = _lookup_steo_baseline(
        history,
        target_month="2025-02-28",
        bcfd_value=100.0,
    )
    assert baseline.metadata["anchor_unit_out"] == "bcf_per_month"
    assert baseline.fused_point == 2_800.0
