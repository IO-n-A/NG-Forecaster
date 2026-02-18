from __future__ import annotations

from pathlib import Path

import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.qa.preprocess_gate import check_preprocess_gate


def test_n3_gate_accepts_valid_runtime_artifacts() -> None:
    artifact_dir = Path("data/artifacts/nowcast/2024-01-12")
    result = check_preprocess_gate(artifact_dir)

    assert result.passed
    assert result.summary_status == "passed"


def test_n3_gate_blocks_when_required_artifact_is_missing(tmp_path: Path) -> None:
    artifact_dir = tmp_path / "run"
    artifact_dir.mkdir()

    with pytest.raises(
        ContractViolation, match="reason_code=missing_preprocess_artifacts"
    ):
        check_preprocess_gate(artifact_dir)
