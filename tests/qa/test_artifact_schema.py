from __future__ import annotations

import json
from pathlib import Path

import pytest

from ng_forecaster.qa.artifact_schema import (
    ArtifactValidationError,
    validate_preprocess_artifact_bundle,
)


def _write(path: Path, text: str) -> Path:
    path.write_text(text, encoding="utf-8")
    return path


def _valid_summary() -> dict[str, object]:
    return {
        "status": "passed",
        "row_count": 12,
        "series_count": 1,
        "missing_flag_count": 4,
        "outlier_flag_count": 1,
        "unresolved_missing_count": 0,
        "low_coverage_series_count": 0,
        "policy": {
            "short_gap_limit": 2,
            "short_gap_method": "ffill",
            "long_gap_method": "median",
            "outlier_zscore_threshold": 3.0,
            "outlier_method": "winsorize",
            "min_non_null_ratio": 0.5,
        },
    }


def _valid_missing_csv() -> str:
    return "\n".join(
        [
            "series_id,timestamp,gap_length,treatment",
            "A,2024-01-03,1,ffill",
            "A,2024-01-08,3,median",
        ]
    )


def _valid_outlier_csv() -> str:
    return "\n".join(
        [
            "series_id,timestamp,zscore,original_value,treated_value,treatment",
            "A,2024-01-05,3.2,100.0,92.0,winsorize",
        ]
    )


def test_artifact_schema_passes_for_valid_bundle(tmp_path: Path) -> None:
    summary = _write(
        tmp_path / "preprocess_summary.json",
        json.dumps(_valid_summary()),
    )
    missing = _write(tmp_path / "missing_value_flags.csv", _valid_missing_csv())
    outlier = _write(tmp_path / "outlier_flags.csv", _valid_outlier_csv())

    validate_preprocess_artifact_bundle(summary, missing, outlier)


def test_artifact_schema_fails_on_missing_summary_key(tmp_path: Path) -> None:
    summary_payload = _valid_summary()
    del summary_payload["status"]

    summary = _write(tmp_path / "preprocess_summary.json", json.dumps(summary_payload))
    missing = _write(tmp_path / "missing_value_flags.csv", _valid_missing_csv())
    outlier = _write(tmp_path / "outlier_flags.csv", _valid_outlier_csv())

    with pytest.raises(ArtifactValidationError, match="missing_required_key"):
        validate_preprocess_artifact_bundle(summary, missing, outlier)


def test_artifact_schema_fails_on_unsorted_missing_rows(tmp_path: Path) -> None:
    summary = _write(
        tmp_path / "preprocess_summary.json",
        json.dumps(_valid_summary()),
    )
    missing = _write(
        tmp_path / "missing_value_flags.csv",
        "\n".join(
            [
                "series_id,timestamp,gap_length,treatment",
                "A,2024-01-08,3,median",
                "A,2024-01-03,1,ffill",
            ]
        ),
    )
    outlier = _write(tmp_path / "outlier_flags.csv", _valid_outlier_csv())

    with pytest.raises(ArtifactValidationError, match="unsorted_rows"):
        validate_preprocess_artifact_bundle(summary, missing, outlier)
