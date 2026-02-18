#!/usr/bin/env python3
"""Cross-track compatibility checker for Sprint 1A fixtures vs Sprint 1B runtime."""

# ruff: noqa: E402

from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ng_forecaster.data.preprocess import run_preprocessing  # noqa: E402
from ng_forecaster.data.validators import (
    load_and_validate_feature_policy,
    load_and_validate_preprocessing_policy,
)  # noqa: E402
from ng_forecaster.features.lag_guard import (
    ensure_no_future_feature_rows,
    ensure_target_lag_eligibility,
)  # noqa: E402
from ng_forecaster.features.monthly_aggregations import (
    build_monthly_features,
)  # noqa: E402
from ng_forecaster.features.vintage_builder import build_vintage_panel  # noqa: E402
from ng_forecaster.qa.artifact_schema import (
    validate_preprocess_artifact_bundle,
)  # noqa: E402
from ng_forecaster.qa.preprocess_gate import (
    resolve_latest_nowcast_artifact_dir,
)  # noqa: E402


def _load_fixture(path: str) -> pd.DataFrame:
    return pd.read_csv(ROOT / path)


def main() -> int:
    try:
        leakage = _load_fixture("tests/fixtures/leakage/asof_valid.csv")
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
                key_cols=("row_key",),
                min_lag_days=1,
            )

        preprocess_fixture = _load_fixture("tests/fixtures/preprocess/mixed_case.csv")
        preprocess_policy = load_and_validate_preprocessing_policy(
            ROOT / "configs/preprocessing.yaml"
        )
        preprocess_result = run_preprocessing(preprocess_fixture, preprocess_policy)

        latest_artifacts = resolve_latest_nowcast_artifact_dir(
            ROOT / "data/artifacts/nowcast"
        )
        validate_preprocess_artifact_bundle(
            latest_artifacts / "preprocess_summary.json",
            latest_artifacts / "missing_value_flags.csv",
            latest_artifacts / "outlier_flags.csv",
        )

        daily = _load_fixture("tests/fixtures/features/input_daily_price.csv").rename(
            columns={"hh_price": "value"}
        )
        weekly = _load_fixture(
            "tests/fixtures/features/input_weekly_storage.csv"
        ).rename(columns={"storage_bcf": "value"})
        feature_rows = build_monthly_features(daily, weekly, asof="2025-02-28")

        target = leakage[["target_timestamp", "target_value"]].drop_duplicates().copy()
        target["target_value"] = target["target_value"].astype(float)

        feature_policy = load_and_validate_feature_policy(
            ROOT / "configs/features.yaml"
        )
        vintage = build_vintage_panel(
            feature_rows[["feature_name", "feature_timestamp", "value"]],
            target,
            asof="2025-03-01",
            feature_policy=feature_policy,
        )

    except Exception as exc:  # noqa: BLE001
        print(f"FAIL: {exc}")
        return 1

    payload = {
        "status": "passed",
        "checks": {
            "leakage_fixture_contract": "passed",
            "preprocess_fixture_contract": preprocess_result.status,
            "artifact_schema_contract": "passed",
            "feature_runtime_contract": "passed",
            "vintage_horizons": sorted(vintage.slices.keys()),
        },
    }
    print("PASS: cross-track fixture/runtime contracts are compatible")
    print(json.dumps(payload, sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
