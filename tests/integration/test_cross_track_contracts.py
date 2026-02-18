"""Cross-track fixture/runtime compatibility checks."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ng_forecaster.data.preprocess import run_preprocessing
from ng_forecaster.data.validators import (
    load_and_validate_feature_policy,
    load_and_validate_preprocessing_policy,
)
from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.lag_guard import (
    ensure_no_future_feature_rows,
    ensure_target_lag_eligibility,
)
from ng_forecaster.features.monthly_aggregations import build_monthly_features
from ng_forecaster.features.vintage_builder import build_vintage_panel
from ng_forecaster.qa.artifact_schema import validate_preprocess_artifact_bundle

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"


def test_fixture_contracts_align_with_runtime_interfaces() -> None:
    leakage = pd.read_csv(FIXTURE_ROOT / "leakage" / "asof_valid.csv")
    for asof, group in leakage.groupby("asof", sort=True):
        ensure_no_future_feature_rows(
            group,
            asof=asof,
            feature_timestamp_col="feature_timestamp",
            key_cols=("row_key",),
        )
        ensure_target_lag_eligibility(
            group,
            asof=asof,
            target_timestamp_col="target_timestamp",
            min_lag_days=1,
            key_cols=("row_key",),
        )

    preprocess_policy = load_and_validate_preprocessing_policy(
        "configs/preprocessing.yaml"
    )
    preprocess_frame = pd.read_csv(FIXTURE_ROOT / "preprocess" / "mixed_case.csv")
    preprocess_result = run_preprocessing(preprocess_frame, preprocess_policy)
    assert preprocess_result.status == "passed"

    artifact_root = Path("data/artifacts/nowcast/2024-01-12")
    validate_preprocess_artifact_bundle(
        artifact_root / "preprocess_summary.json",
        artifact_root / "missing_value_flags.csv",
        artifact_root / "outlier_flags.csv",
    )

    daily = pd.read_csv(FIXTURE_ROOT / "features" / "input_daily_price.csv").rename(
        columns={"hh_price": "value"}
    )
    weekly = pd.read_csv(FIXTURE_ROOT / "features" / "input_weekly_storage.csv").rename(
        columns={"storage_bcf": "value"}
    )
    features = build_monthly_features(daily, weekly, asof="2025-02-28")

    target = leakage[["target_timestamp", "target_value"]].drop_duplicates().copy()
    target["target_value"] = target["target_value"].astype(float)

    policy = load_and_validate_feature_policy("configs/features.yaml")
    vintage = build_vintage_panel(
        features[["feature_name", "feature_timestamp", "value"]],
        target,
        asof="2025-03-01",
        feature_policy=policy,
    )

    assert sorted(vintage.slices.keys()) == ["T", "T-1"]


def test_runtime_detects_fixture_interface_drift() -> None:
    daily = pd.read_csv(FIXTURE_ROOT / "features" / "input_daily_price.csv")
    weekly = pd.read_csv(FIXTURE_ROOT / "features" / "input_weekly_storage.csv").rename(
        columns={"storage_bcf": "value"}
    )

    with pytest.raises(ContractViolation, match="reason_code=missing_column"):
        build_monthly_features(daily, weekly, asof="2025-02-28")
