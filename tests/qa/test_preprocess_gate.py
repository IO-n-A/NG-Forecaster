from __future__ import annotations

import json
from pathlib import Path

import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.qa.preprocess_gate import check_preprocess_gate


def _write_valid_bundle(path: Path, *, status: str = "passed") -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "preprocess_summary.json").write_text(
        json.dumps(
            {
                "status": status,
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
        ),
        encoding="utf-8",
    )
    (path / "missing_value_flags.csv").write_text(
        "\n".join(
            [
                "series_id,timestamp,gap_length,treatment",
                "A,2024-01-03,1,ffill",
                "A,2024-01-08,3,median",
            ]
        ),
        encoding="utf-8",
    )
    (path / "outlier_flags.csv").write_text(
        "\n".join(
            [
                "series_id,timestamp,zscore,original_value,treated_value,treatment",
                "A,2024-01-05,3.2,100.0,92.0,winsorize",
            ]
        ),
        encoding="utf-8",
    )


def test_preprocess_gate_passes_for_valid_artifacts(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "run"
    _write_valid_bundle(artifact_dir)

    result = check_preprocess_gate(artifact_dir)
    assert result.passed
    assert result.summary_status == "passed"


def test_preprocess_gate_fails_when_artifacts_are_missing(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "run"
    artifact_dir.mkdir()

    with pytest.raises(
        ContractViolation, match="reason_code=missing_preprocess_artifacts"
    ):
        check_preprocess_gate(artifact_dir)


def test_preprocess_gate_fails_when_status_is_failed(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "run"
    _write_valid_bundle(artifact_dir, status="failed")

    with pytest.raises(ContractViolation, match="reason_code=preprocess_status_failed"):
        check_preprocess_gate(artifact_dir)
