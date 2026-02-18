"""Schema contract tests for preprocessing artifacts."""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from tests.helpers.artifact_schema import (
    ArtifactSchemaError,
    validate_preprocess_artifact_bundle,
)


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("rows must be non-empty")

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_valid_preprocess_artifact_bundle_passes(tmp_path: Path) -> None:
    summary_path = tmp_path / "preprocess_summary.json"
    missing_path = tmp_path / "missing_value_flags.csv"
    outlier_path = tmp_path / "outlier_flags.csv"

    summary_path.write_text(
        json.dumps(
            {
                "asof": "2025-02-28",
                "status": "passed",
                "row_count": 5,
                "missing_short_gap_count": 1,
                "missing_long_gap_count": 0,
                "outlier_count": 1,
            }
        ),
        encoding="utf-8",
    )

    _write_csv(
        missing_path,
        [
            {
                "asof": "2025-02-28",
                "series_id": "prod",
                "timestamp": "2025-02-02",
                "gap_type": "short",
                "gap_length": "1",
                "imputation_policy": "forward_fill",
            }
        ],
    )
    _write_csv(
        outlier_path,
        [
            {
                "asof": "2025-02-28",
                "series_id": "prod",
                "timestamp": "2025-02-04",
                "method": "robust_zscore_mad",
                "z_score": "12.500000",
                "action": "winsorize_to_median",
            }
        ],
    )

    validate_preprocess_artifact_bundle(summary_path, missing_path, outlier_path)


def test_summary_missing_required_key_fails(tmp_path: Path) -> None:
    summary_path = tmp_path / "preprocess_summary.json"
    missing_path = tmp_path / "missing_value_flags.csv"
    outlier_path = tmp_path / "outlier_flags.csv"

    summary_path.write_text(
        json.dumps(
            {
                "asof": "2025-02-28",
                "row_count": 5,
                "missing_short_gap_count": 1,
                "missing_long_gap_count": 0,
                "outlier_count": 1,
            }
        ),
        encoding="utf-8",
    )

    _write_csv(
        missing_path,
        [
            {
                "asof": "2025-02-28",
                "series_id": "prod",
                "timestamp": "2025-02-02",
                "gap_type": "short",
                "gap_length": "1",
                "imputation_policy": "forward_fill",
            }
        ],
    )
    _write_csv(
        outlier_path,
        [
            {
                "asof": "2025-02-28",
                "series_id": "prod",
                "timestamp": "2025-02-04",
                "method": "robust_zscore_mad",
                "z_score": "12.500000",
                "action": "winsorize_to_median",
            }
        ],
    )

    with pytest.raises(ArtifactSchemaError) as exc_info:
        validate_preprocess_artifact_bundle(summary_path, missing_path, outlier_path)

    assert "artifact=preprocess_summary.json" in str(exc_info.value)
    assert "reason_code=missing_required_key" in str(exc_info.value)


def test_missing_flags_invalid_gap_type_fails(tmp_path: Path) -> None:
    summary_path = tmp_path / "preprocess_summary.json"
    missing_path = tmp_path / "missing_value_flags.csv"
    outlier_path = tmp_path / "outlier_flags.csv"

    summary_path.write_text(
        json.dumps(
            {
                "asof": "2025-02-28",
                "status": "passed",
                "row_count": 5,
                "missing_short_gap_count": 1,
                "missing_long_gap_count": 0,
                "outlier_count": 1,
            }
        ),
        encoding="utf-8",
    )

    _write_csv(
        missing_path,
        [
            {
                "asof": "2025-02-28",
                "series_id": "prod",
                "timestamp": "2025-02-02",
                "gap_type": "unexpected_gap",
                "gap_length": "1",
                "imputation_policy": "forward_fill",
            }
        ],
    )
    _write_csv(
        outlier_path,
        [
            {
                "asof": "2025-02-28",
                "series_id": "prod",
                "timestamp": "2025-02-04",
                "method": "robust_zscore_mad",
                "z_score": "12.500000",
                "action": "winsorize_to_median",
            }
        ],
    )

    with pytest.raises(ArtifactSchemaError) as exc_info:
        validate_preprocess_artifact_bundle(summary_path, missing_path, outlier_path)

    assert "artifact=missing_value_flags.csv" in str(exc_info.value)
    assert "reason_code=invalid_gap_type" in str(exc_info.value)
