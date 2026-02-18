"""Target-month and interval integrity gate for nowcast artifacts."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd

from ng_forecaster.errors import ContractViolation


@dataclass(frozen=True)
class TargetMonthGateResult:
    """Result payload for target-month QA checks."""

    passed: bool
    asof: str
    target_month: str
    expected_target_month: str
    nowcast_count: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "asof": self.asof,
            "target_month": self.target_month,
            "expected_target_month": self.expected_target_month,
            "nowcast_count": self.nowcast_count,
        }


def _resolve_latest_nowcast_json(root: str | Path) -> Path:
    base = Path(root)
    if not base.exists():
        raise ContractViolation(
            "missing_target_month_artifact",
            key="artifact_root",
            detail=f"artifact root does not exist: {base}",
        )
    candidates = sorted(path for path in base.iterdir() if path.is_dir())
    if not candidates:
        raise ContractViolation(
            "missing_target_month_artifact",
            key="artifact_root",
            detail=f"no nowcast run directories found under {base}",
        )
    return candidates[-1] / "nowcast.json"


def check_target_month_gate(
    nowcast_json_path: str | Path | None = None,
    *,
    artifact_root: str | Path = "data/artifacts/nowcast",
    lag_months: int = 2,
) -> TargetMonthGateResult:
    """Validate nowcast.json contains expected target-month and interval contracts."""

    if lag_months < 1:
        raise ContractViolation(
            "invalid_lag_policy",
            key="lag_months",
            detail="lag_months must be >= 1",
        )

    path = (
        Path(nowcast_json_path)
        if nowcast_json_path is not None
        else _resolve_latest_nowcast_json(artifact_root)
    )
    if not path.exists():
        raise ContractViolation(
            "missing_target_month_artifact",
            key="nowcast_json",
            detail=f"missing nowcast artifact: {path}",
        )

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ContractViolation(
            "invalid_target_month_artifact",
            key="nowcast_json",
            detail="nowcast.json root must be an object",
        )

    asof_raw = payload.get("asof")
    target_raw = payload.get("target_month")
    if asof_raw is None or target_raw is None:
        raise ContractViolation(
            "invalid_target_month_artifact",
            key="asof,target_month",
            detail="nowcast.json must include asof and target_month",
        )

    asof_ts = pd.Timestamp(asof_raw)
    target_ts = pd.Timestamp(target_raw).to_period("M").to_timestamp("M")
    if pd.isna(asof_ts) or pd.isna(target_ts):
        raise ContractViolation(
            "invalid_timestamp",
            key="asof,target_month",
            detail="asof and target_month must be parseable timestamps",
        )

    expected_target = (asof_ts.to_period("M") - lag_months).to_timestamp("M")
    if target_ts != expected_target:
        raise ContractViolation(
            "target_month_mismatch",
            asof=asof_ts.to_pydatetime(),
            key="target_month",
            detail=(
                f"expected {expected_target.date().isoformat()} but found "
                f"{target_ts.date().isoformat()}"
            ),
        )

    nowcasts = payload.get("nowcasts")
    if not isinstance(nowcasts, list) or not nowcasts:
        raise ContractViolation(
            "invalid_target_month_artifact",
            key="nowcasts",
            detail="nowcasts must be a non-empty list",
        )

    parsed_months: list[pd.Timestamp] = []
    for idx, item in enumerate(nowcasts):
        if not isinstance(item, dict):
            raise ContractViolation(
                "invalid_target_month_artifact",
                key=f"nowcasts[{idx}]",
                detail="nowcast entries must be objects",
            )
        for key in ("fused_lower_95", "fused_point", "fused_upper_95"):
            if key not in item:
                raise ContractViolation(
                    "invalid_target_month_artifact",
                    key=f"nowcasts[{idx}].{key}",
                    detail="missing interval field",
                )
            try:
                item[key] = float(item[key])
            except (TypeError, ValueError) as exc:
                raise ContractViolation(
                    "invalid_target_month_artifact",
                    key=f"nowcasts[{idx}].{key}",
                    detail="interval field must be numeric",
                ) from exc

        if not (
            item["fused_lower_95"] <= item["fused_point"] <= item["fused_upper_95"]
        ):
            raise ContractViolation(
                "interval_order_violation",
                asof=asof_ts.to_pydatetime(),
                key=f"nowcasts[{idx}]",
                detail=(
                    f"lower={item['fused_lower_95']}, "
                    f"point={item['fused_point']}, "
                    f"upper={item['fused_upper_95']}"
                ),
            )

        month_raw = item.get("target_month")
        if month_raw is None:
            raise ContractViolation(
                "invalid_target_month_artifact",
                key=f"nowcasts[{idx}].target_month",
                detail="target_month field is required per nowcast row",
            )
        month_ts = pd.Timestamp(month_raw).to_period("M").to_timestamp("M")
        if pd.isna(month_ts):
            raise ContractViolation(
                "invalid_timestamp",
                key=f"nowcasts[{idx}].target_month",
                detail="row target_month must be parseable",
            )
        parsed_months.append(month_ts)

    if parsed_months[0] != target_ts:
        raise ContractViolation(
            "target_month_mismatch",
            asof=asof_ts.to_pydatetime(),
            key="nowcasts[0].target_month",
            detail=(
                f"first nowcast target_month should equal root target_month "
                f"{target_ts.date().isoformat()}"
            ),
        )

    sorted_months = sorted(parsed_months)
    if parsed_months != sorted_months:
        raise ContractViolation(
            "invalid_target_month_artifact",
            asof=asof_ts.to_pydatetime(),
            key="nowcasts.target_month",
            detail="target_month rows must be sorted ascending",
        )

    return TargetMonthGateResult(
        passed=True,
        asof=asof_ts.date().isoformat(),
        target_month=target_ts.date().isoformat(),
        expected_target_month=expected_target.date().isoformat(),
        nowcast_count=len(nowcasts),
    )
