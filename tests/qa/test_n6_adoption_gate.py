from __future__ import annotations

from pathlib import Path

import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.qa.n6_adoption_gate import check_n6_adoption_readiness


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _valid_scorecard() -> str:
    return "\n".join(
        [
            "experiment_id,mae,rmse,mape,dm_vs_baseline_p_value,runtime_seconds,lineage_id",
            "B0_baseline,1.20,1.60,2.80,1.000,41,b0abc",
            "B1_plus_preprocessing,1.15,1.55,2.70,0.120,48,b1abc",
            "B4_full_method,1.05,1.40,2.30,0.040,62,b4abc",
        ]
    )


def test_n6_gate_passes_for_adoption_ready_scorecard(tmp_path: Path) -> None:
    scorecard = tmp_path / "ablation_scorecard.csv"
    _write(scorecard, _valid_scorecard())

    result = check_n6_adoption_readiness(scorecard)
    assert result.passed
    assert result.full_method_mae <= result.baseline_mae


def test_n6_gate_fails_when_full_method_regresses(tmp_path: Path) -> None:
    scorecard = tmp_path / "ablation_scorecard.csv"
    _write(
        scorecard,
        _valid_scorecard().replace("B4_full_method,1.05", "B4_full_method,1.50"),
    )

    with pytest.raises(ContractViolation, match="reason_code=ablation_regression"):
        check_n6_adoption_readiness(scorecard)
