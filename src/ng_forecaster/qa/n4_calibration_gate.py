"""N4 acceptance gate for seed stability and calibration diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ng_forecaster.errors import ContractViolation

_REQUIRED_SEED_COLUMNS = {
    "model",
    "seed_count",
    "point_mean",
    "point_std",
    "interval_width_mean",
}
_REQUIRED_CALIBRATION_COLUMNS = {
    "model",
    "nominal_coverage",
    "empirical_coverage",
    "calibration_error",
}
_REQUIRED_OUTPUT_COLUMNS = {
    "asof",
    "model",
    "point_forecast",
    "interval_low",
    "interval_high",
}
_REQUIRED_MODELS = ("champion", "challenger")


@dataclass(frozen=True)
class N4GateResult:
    """Result payload for N4 acceptance checks."""

    passed: bool
    seed_path: Path
    calibration_path: Path
    output_path: Path
    model_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "seed_path": str(self.seed_path),
            "calibration_path": str(self.calibration_path),
            "output_path": str(self.output_path),
            "model_count": self.model_count,
        }


def _require_columns(frame: pd.DataFrame, required: set[str], *, key: str) -> None:
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ContractViolation(
            "missing_column",
            key=key,
            detail="missing columns: " + ", ".join(missing),
        )


def check_n4_acceptance(
    seed_stability_csv: str | Path,
    calibration_csv: str | Path,
    output_csv: str | Path,
    *,
    min_seed_runs: int = 3,
    max_point_std: float = 5.0,
    max_calibration_error: float = 0.15,
) -> N4GateResult:
    """Validate N4 seed stability, calibration evidence, and reporting completeness."""

    seed_path = Path(seed_stability_csv)
    calibration_path = Path(calibration_csv)
    output_path = Path(output_csv)

    for path, key in (
        (seed_path, "seed_stability"),
        (calibration_path, "calibration_summary"),
        (output_path, "nowcast_outputs"),
    ):
        if not path.exists():
            raise ContractViolation(
                "missing_artifact",
                key=key,
                detail=f"expected artifact at {path}",
            )

    seed = pd.read_csv(seed_path)
    calibration = pd.read_csv(calibration_path)
    outputs = pd.read_csv(output_path)

    _require_columns(seed, _REQUIRED_SEED_COLUMNS, key="seed_stability")
    _require_columns(
        calibration, _REQUIRED_CALIBRATION_COLUMNS, key="calibration_summary"
    )
    _require_columns(outputs, _REQUIRED_OUTPUT_COLUMNS, key="nowcast_outputs")

    observed_models = sorted(
        set(seed["model"].astype(str))
        | set(calibration["model"].astype(str))
        | set(outputs["model"].astype(str))
    )
    for model in _REQUIRED_MODELS:
        if model not in observed_models:
            raise ContractViolation(
                "missing_model_evidence",
                key="model",
                detail=f"required model missing from N4 artifacts: {model}",
            )

    for _, row in seed.iterrows():
        if int(row["seed_count"]) < min_seed_runs:
            raise ContractViolation(
                "insufficient_seed_runs",
                key=str(row["model"]),
                detail=f"seed_count={int(row['seed_count'])} < {min_seed_runs}",
            )
        if float(row["point_std"]) > max_point_std:
            raise ContractViolation(
                "seed_instability",
                key=str(row["model"]),
                detail=f"point_std={float(row['point_std']):.6f} > {max_point_std}",
            )

    for _, row in calibration.iterrows():
        nominal = float(row["nominal_coverage"])
        empirical = float(row["empirical_coverage"])
        error = float(row["calibration_error"])

        if not 0.0 <= nominal <= 1.0:
            raise ContractViolation(
                "invalid_calibration_value",
                key="nominal_coverage",
                detail=f"{nominal}",
            )
        if not 0.0 <= empirical <= 1.0:
            raise ContractViolation(
                "invalid_calibration_value",
                key="empirical_coverage",
                detail=f"{empirical}",
            )
        if abs(error) > max_calibration_error:
            raise ContractViolation(
                "calibration_drift",
                key=str(row["model"]),
                detail=f"abs(calibration_error)={abs(error):.6f} > {max_calibration_error}",
            )

    invalid_interval = outputs[outputs["interval_low"] > outputs["interval_high"]]
    if not invalid_interval.empty:
        first = invalid_interval.iloc[0]
        raise ContractViolation(
            "invalid_interval",
            key=str(first["model"]),
            detail=(
                f"interval_low={float(first['interval_low'])} > "
                f"interval_high={float(first['interval_high'])}"
            ),
        )

    return N4GateResult(
        passed=True,
        seed_path=seed_path,
        calibration_path=calibration_path,
        output_path=output_path,
        model_count=len(observed_models),
    )
