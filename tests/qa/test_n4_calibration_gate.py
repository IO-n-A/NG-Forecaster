from __future__ import annotations

from pathlib import Path

import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.qa.n4_calibration_gate import check_n4_acceptance


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_valid_bundle(tmp_path: Path) -> tuple[Path, Path, Path]:
    seed = tmp_path / "seed.csv"
    calibration = tmp_path / "calibration.csv"
    outputs = tmp_path / "outputs.csv"

    _write(
        seed,
        "\n".join(
            [
                "model,seed_count,point_mean,point_std,interval_width_mean",
                "champion,5,100.2,1.1,6.2",
                "challenger,5,99.8,1.4,6.4",
            ]
        ),
    )
    _write(
        calibration,
        "\n".join(
            [
                "model,nominal_coverage,empirical_coverage,calibration_error",
                "champion,0.8,0.78,-0.02",
                "challenger,0.8,0.81,0.01",
            ]
        ),
    )
    _write(
        outputs,
        "\n".join(
            [
                "asof,model,point_forecast,interval_low,interval_high",
                "2026-02-14,champion,100.2,96.0,104.0",
                "2026-02-14,challenger,99.8,95.5,103.9",
            ]
        ),
    )

    return seed, calibration, outputs


def test_n4_gate_passes_for_valid_inputs(tmp_path: Path) -> None:
    seed, calibration, outputs = _write_valid_bundle(tmp_path)
    result = check_n4_acceptance(seed, calibration, outputs)
    assert result.passed
    assert result.model_count == 2


def test_n4_gate_fails_on_seed_instability(tmp_path: Path) -> None:
    seed, calibration, outputs = _write_valid_bundle(tmp_path)
    _write(
        seed,
        "\n".join(
            [
                "model,seed_count,point_mean,point_std,interval_width_mean",
                "champion,5,100.2,9.1,6.2",
                "challenger,5,99.8,1.4,6.4",
            ]
        ),
    )

    with pytest.raises(ContractViolation, match="reason_code=seed_instability"):
        check_n4_acceptance(seed, calibration, outputs)
