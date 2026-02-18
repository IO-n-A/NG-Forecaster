from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ng_forecaster.data.preprocess import run_preprocessing
from ng_forecaster.data.validators import (
    load_and_validate_feature_policy,
    load_and_validate_preprocessing_policy,
    validate_preprocessing_policy,
)
from ng_forecaster.errors import ContractViolation
from ng_forecaster.reporting.exporters import export_preprocess_artifacts


def _sample_preprocess_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "series_id": ["A"] * 12,
            "timestamp": pd.date_range("2024-01-01", periods=12, freq="D"),
            "value": [1.0, 2.0, None, 4.0, 100.0, 6.0, 7.0, None, None, None, 11.0, 12.0],
        }
    )


def test_preprocess_is_deterministic_and_exports_required_artifacts(tmp_path: Path) -> None:
    policy = load_and_validate_preprocessing_policy("configs/preprocessing.yaml")
    policy["min_non_null_ratio"] = 0.5

    frame = _sample_preprocess_frame()
    first = run_preprocessing(frame, policy)
    second = run_preprocessing(frame, policy)

    assert first.status == "passed"
    assert first.data.equals(second.data)
    assert first.summary == second.summary
    assert len(first.missing_flags) == 4
    assert len(first.outlier_flags) >= 1

    paths = export_preprocess_artifacts(first, tmp_path)
    assert set(paths) == {
        "preprocess_summary",
        "missing_value_flags",
        "outlier_flags",
    }
    for path in paths.values():
        assert path.exists()

    with (tmp_path / "preprocess_summary.json").open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert payload["status"] == "passed"


def test_validate_preprocessing_policy_rejects_invalid_schema() -> None:
    with pytest.raises(ContractViolation, match="reason_code=invalid_preprocess_policy"):
        validate_preprocessing_policy({"short_gap_limit": -1})


def test_feature_policy_yaml_contract_loads() -> None:
    policy = load_and_validate_feature_policy("configs/features.yaml")
    assert "features" in policy
    assert "hh_last" in policy["features"]
