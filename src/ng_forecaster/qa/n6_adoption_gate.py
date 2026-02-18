"""N6 governance gate for ablation adoption readiness evidence."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from ng_forecaster.errors import ContractViolation

_REQUIRED_COLUMNS = {
    "experiment_id",
    "mae",
    "rmse",
    "mape",
    "dm_vs_baseline_p_value",
    "runtime_seconds",
    "lineage_id",
}


@dataclass(frozen=True)
class N6AdoptionResult:
    """Result payload for N6 adoption-readiness review."""

    passed: bool
    scorecard_path: Path
    baseline_id: str
    full_method_id: str
    baseline_mae: float
    full_method_mae: float
    full_method_dm_p_value: float

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "scorecard_path": str(self.scorecard_path),
            "baseline_id": self.baseline_id,
            "full_method_id": self.full_method_id,
            "baseline_mae": self.baseline_mae,
            "full_method_mae": self.full_method_mae,
            "full_method_dm_p_value": self.full_method_dm_p_value,
        }


def check_n6_adoption_readiness(
    scorecard_csv: str | Path,
    *,
    baseline_id: str = "B0_baseline",
    full_method_id: str = "B4_full_method",
    max_dm_p_value: float = 0.10,
) -> N6AdoptionResult:
    """Validate ablation scorecard has adoption-ready deltas and evidence."""

    scorecard_path = Path(scorecard_csv)
    if not scorecard_path.exists():
        raise ContractViolation(
            "missing_ablation_scorecard",
            key="scorecard_csv",
            detail=f"missing file: {scorecard_path}",
        )

    frame = pd.read_csv(scorecard_path)
    missing = sorted(_REQUIRED_COLUMNS - set(frame.columns))
    if missing:
        raise ContractViolation(
            "missing_column",
            key="ablation_scorecard",
            detail="missing columns: " + ", ".join(missing),
        )
    if frame.empty:
        raise ContractViolation(
            "empty_ablation_scorecard",
            key="ablation_scorecard",
            detail="scorecard cannot be empty",
        )

    if baseline_id not in set(frame["experiment_id"].astype(str)):
        raise ContractViolation(
            "missing_baseline_experiment",
            key="experiment_id",
            detail=f"missing {baseline_id}",
        )
    if full_method_id not in set(frame["experiment_id"].astype(str)):
        raise ContractViolation(
            "missing_full_method_experiment",
            key="experiment_id",
            detail=f"missing {full_method_id}",
        )

    for _, row in frame.iterrows():
        runtime = float(row["runtime_seconds"])
        if runtime <= 0:
            raise ContractViolation(
                "invalid_runtime",
                key=str(row["experiment_id"]),
                detail=f"runtime_seconds={runtime}",
            )
        lineage = str(row["lineage_id"]).strip()
        if not lineage:
            raise ContractViolation(
                "missing_lineage_id",
                key=str(row["experiment_id"]),
                detail="lineage_id cannot be empty",
            )

    baseline_row = frame[frame["experiment_id"].astype(str) == baseline_id].iloc[0]
    full_method_row = frame[frame["experiment_id"].astype(str) == full_method_id].iloc[
        0
    ]

    baseline_mae = float(baseline_row["mae"])
    full_method_mae = float(full_method_row["mae"])
    dm_p_value = float(full_method_row["dm_vs_baseline_p_value"])

    if full_method_mae > baseline_mae:
        raise ContractViolation(
            "ablation_regression",
            key=full_method_id,
            detail=f"full_method_mae={full_method_mae} > baseline_mae={baseline_mae}",
        )

    if dm_p_value > max_dm_p_value:
        raise ContractViolation(
            "insufficient_ablation_significance",
            key=full_method_id,
            detail=f"dm_vs_baseline_p_value={dm_p_value} > {max_dm_p_value}",
        )

    return N6AdoptionResult(
        passed=True,
        scorecard_path=scorecard_path,
        baseline_id=baseline_id,
        full_method_id=full_method_id,
        baseline_mae=baseline_mae,
        full_method_mae=full_method_mae,
        full_method_dm_p_value=dm_p_value,
    )
