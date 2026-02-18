"""N5 governance audit for DM decision policy compliance."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ng_forecaster.data.validators import load_yaml
from ng_forecaster.errors import ContractViolation

_REQUIRED_POLICY_KEYS = {
    "loss",
    "sidedness",
    "alpha_levels",
    "small_sample_adjustment",
    "benchmark_by_target",
    "multiple_comparison",
}
_REQUIRED_DM_COLUMNS = {
    "target",
    "candidate_model",
    "benchmark_model",
    "d_bar",
    "dm_stat",
    "p_value",
    "significant_0_05",
    "significant_0_01",
    "adjusted_p_value",
}


@dataclass(frozen=True)
class N5AuditResult:
    """Result payload for N5 policy audit."""

    passed: bool
    policy_path: Path
    dm_results_path: Path
    target_count: int

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "policy_path": str(self.policy_path),
            "dm_results_path": str(self.dm_results_path),
            "target_count": self.target_count,
        }


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid bool-like value: {value}")


def audit_n5_policy(
    policy_yaml: str | Path,
    dm_results_csv: str | Path,
) -> N5AuditResult:
    """Validate N5 DM policy config and runtime reporting evidence."""

    policy_path = Path(policy_yaml)
    dm_path = Path(dm_results_csv)

    if not policy_path.exists():
        raise ContractViolation(
            "missing_policy_config",
            key="evaluation_yaml",
            detail=f"missing file: {policy_path}",
        )
    if not dm_path.exists():
        raise ContractViolation(
            "missing_dm_results",
            key="dm_results_csv",
            detail=f"missing file: {dm_path}",
        )

    policy = load_yaml(policy_path)
    missing_policy_keys = sorted(_REQUIRED_POLICY_KEYS - set(policy.keys()))
    if missing_policy_keys:
        raise ContractViolation(
            "invalid_dm_policy",
            key="policy_keys",
            detail="missing keys: " + ", ".join(missing_policy_keys),
        )

    alpha_levels = policy["alpha_levels"]
    if not isinstance(alpha_levels, list) or not alpha_levels:
        raise ContractViolation(
            "invalid_dm_policy",
            key="alpha_levels",
            detail="alpha_levels must be a non-empty list",
        )

    benchmark_by_target = policy["benchmark_by_target"]
    if not isinstance(benchmark_by_target, dict) or not benchmark_by_target:
        raise ContractViolation(
            "invalid_dm_policy",
            key="benchmark_by_target",
            detail="benchmark_by_target must be a non-empty mapping",
        )
    comparison_pairs_by_target = policy.get("comparison_pairs_by_target", {})
    if not isinstance(comparison_pairs_by_target, dict):
        raise ContractViolation(
            "invalid_dm_policy",
            key="comparison_pairs_by_target",
            detail="comparison_pairs_by_target must be a mapping",
        )

    allowed_pairs: dict[str, set[tuple[str, str]]] = {}
    for target_name, pairs in comparison_pairs_by_target.items():
        if not isinstance(pairs, list):
            raise ContractViolation(
                "invalid_dm_policy",
                key=f"comparison_pairs_by_target.{target_name}",
                detail="pair payload must be a list",
            )
        normalized: set[tuple[str, str]] = set()
        for pair in pairs:
            if not isinstance(pair, list) or len(pair) != 2:
                raise ContractViolation(
                    "invalid_dm_policy",
                    key=f"comparison_pairs_by_target.{target_name}",
                    detail="each pair must contain [candidate, benchmark]",
                )
            normalized.add((str(pair[0]).strip(), str(pair[1]).strip()))
        if normalized:
            allowed_pairs[str(target_name)] = normalized

    frame = pd.read_csv(dm_path)
    missing_dm_columns = sorted(_REQUIRED_DM_COLUMNS - set(frame.columns))
    if missing_dm_columns:
        raise ContractViolation(
            "missing_column",
            key="dm_results",
            detail="missing columns: " + ", ".join(missing_dm_columns),
        )

    if frame.empty:
        raise ContractViolation(
            "empty_dm_results",
            key="dm_results",
            detail="dm results cannot be empty",
        )

    for _, row in frame.iterrows():
        target = str(row["target"])
        candidate = str(row["candidate_model"]).strip()
        benchmark = str(row["benchmark_model"]).strip()
        if not benchmark:
            raise ContractViolation(
                "missing_benchmark",
                key=target,
                detail="benchmark_model cannot be empty",
            )

        if target in allowed_pairs:
            if (candidate, benchmark) not in allowed_pairs[target]:
                raise ContractViolation(
                    "benchmark_mismatch",
                    key=target,
                    detail=(
                        "dm result pair is not declared in comparison_pairs_by_target: "
                        f"candidate={candidate} benchmark={benchmark}"
                    ),
                )
        else:
            expected_benchmark = str(benchmark_by_target.get(target, "")).strip()
            if expected_benchmark and benchmark != expected_benchmark:
                raise ContractViolation(
                    "benchmark_mismatch",
                    key=target,
                    detail=f"expected={expected_benchmark} observed={benchmark}",
                )

        p_value = float(row["p_value"])
        adjusted_p = float(row["adjusted_p_value"])
        if not 0.0 <= p_value <= 1.0:
            raise ContractViolation(
                "invalid_p_value",
                key=target,
                detail=f"p_value={p_value}",
            )
        if not 0.0 <= adjusted_p <= 1.0:
            raise ContractViolation(
                "invalid_p_value",
                key=target,
                detail=f"adjusted_p_value={adjusted_p}",
            )
        if adjusted_p < p_value:
            raise ContractViolation(
                "invalid_adjusted_p_value",
                key=target,
                detail=f"adjusted_p_value={adjusted_p} < p_value={p_value}",
            )

        try:
            sig_005 = _coerce_bool(row["significant_0_05"])
            sig_001 = _coerce_bool(row["significant_0_01"])
        except ValueError as exc:
            raise ContractViolation(
                "invalid_significance_flag",
                key=target,
                detail=str(exc),
            ) from exc

        if sig_001 and not sig_005:
            raise ContractViolation(
                "invalid_significance_flag",
                key=target,
                detail="significant_0_01 requires significant_0_05",
            )

        expected_005 = adjusted_p <= 0.05
        expected_001 = adjusted_p <= 0.01
        if sig_005 != expected_005 or sig_001 != expected_001:
            raise ContractViolation(
                "significance_mismatch",
                key=target,
                detail=(
                    f"adjusted_p={adjusted_p}; expected_0_05={expected_005}; "
                    f"expected_0_01={expected_001}"
                ),
            )

    return N5AuditResult(
        passed=True,
        policy_path=policy_path,
        dm_results_path=dm_path,
        target_count=int(frame["target"].nunique()),
    )
