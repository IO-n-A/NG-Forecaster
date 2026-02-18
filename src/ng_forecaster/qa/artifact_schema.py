"""Schema validators for Sprint 1 preprocessing artifacts."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class ArtifactValidationError(ValueError):
    """Structured schema error for preprocessing artifacts."""

    artifact: str
    key: str
    reason_code: str
    detail: str = ""

    def __str__(self) -> str:
        message = f"artifact={self.artifact} | key={self.key} | reason_code={self.reason_code}"
        if self.detail:
            message = f"{message} | detail={self.detail}"
        return message


_SUMMARY_REQUIRED_INT_KEYS = (
    "row_count",
    "series_count",
    "missing_flag_count",
    "outlier_flag_count",
    "unresolved_missing_count",
    "low_coverage_series_count",
)


def _load_csv_rows(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = [str(name) for name in (reader.fieldnames or []) if name is not None]
        rows: list[dict[str, str]] = []
        for raw_row in reader:
            row = {
                key: ("" if raw_row.get(key) is None else str(raw_row.get(key)))
                for key in fieldnames
            }
            rows.append(row)
    return fieldnames, rows


def _load_json(path: Path) -> Mapping[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ArtifactValidationError(
            artifact=path.name,
            key="root",
            reason_code="invalid_type",
            detail="summary payload must be an object",
        )
    return payload


def _assert_columns(
    *,
    artifact: str,
    actual: Sequence[str],
    required: Sequence[str],
) -> None:
    missing = sorted(set(required) - set(actual))
    if missing:
        raise ArtifactValidationError(
            artifact=artifact,
            key=",".join(missing),
            reason_code="missing_required_column",
        )


def _assert_date(value: str, *, artifact: str, key: str) -> None:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ArtifactValidationError(
            artifact=artifact,
            key=key,
            reason_code="invalid_date",
            detail=value,
        ) from exc


def _assert_sorted(
    rows: Sequence[Mapping[str, str]],
    *,
    artifact: str,
    sort_keys: Sequence[str],
) -> None:
    tuples = [tuple(str(row[key]) for key in sort_keys) for row in rows]
    if tuples != sorted(tuples):
        raise ArtifactValidationError(
            artifact=artifact,
            key=",".join(sort_keys),
            reason_code="unsorted_rows",
            detail="rows must be sorted deterministically",
        )


def validate_preprocess_summary(summary: Mapping[str, Any]) -> None:
    """Validate preprocess_summary.json schema."""

    required = {"status", "policy", *set(_SUMMARY_REQUIRED_INT_KEYS)}
    missing = sorted(required - set(summary.keys()))
    if missing:
        raise ArtifactValidationError(
            artifact="preprocess_summary.json",
            key=",".join(missing),
            reason_code="missing_required_key",
        )

    status = str(summary["status"])
    if status not in {"passed", "failed"}:
        raise ArtifactValidationError(
            artifact="preprocess_summary.json",
            key="status",
            reason_code="invalid_status",
            detail=status,
        )

    policy = summary["policy"]
    if not isinstance(policy, Mapping):
        raise ArtifactValidationError(
            artifact="preprocess_summary.json",
            key="policy",
            reason_code="invalid_type",
        )

    for key in _SUMMARY_REQUIRED_INT_KEYS:
        value = summary[key]
        if not isinstance(value, int):
            raise ArtifactValidationError(
                artifact="preprocess_summary.json",
                key=key,
                reason_code="invalid_type",
            )
        if value < 0:
            raise ArtifactValidationError(
                artifact="preprocess_summary.json",
                key=key,
                reason_code="negative_value",
            )


def validate_missing_value_flags(
    rows: Sequence[Mapping[str, str]], columns: Sequence[str]
) -> None:
    """Validate missing_value_flags.csv schema and sort contract."""

    artifact = "missing_value_flags.csv"
    required = ("series_id", "timestamp", "gap_length", "treatment")
    _assert_columns(artifact=artifact, actual=columns, required=required)

    for row in rows:
        _assert_date(str(row["timestamp"]), artifact=artifact, key="timestamp")
        try:
            gap_length = int(str(row["gap_length"]))
        except ValueError as exc:
            raise ArtifactValidationError(
                artifact=artifact,
                key="gap_length",
                reason_code="invalid_integer",
                detail=str(row["gap_length"]),
            ) from exc
        if gap_length < 1:
            raise ArtifactValidationError(
                artifact=artifact,
                key="gap_length",
                reason_code="invalid_integer",
            )
        if not str(row["treatment"]).strip():
            raise ArtifactValidationError(
                artifact=artifact,
                key="treatment",
                reason_code="empty_value",
            )

    _assert_sorted(rows, artifact=artifact, sort_keys=("series_id", "timestamp"))


def validate_outlier_flags(
    rows: Sequence[Mapping[str, str]], columns: Sequence[str]
) -> None:
    """Validate outlier_flags.csv schema and sort contract."""

    artifact = "outlier_flags.csv"
    required = (
        "series_id",
        "timestamp",
        "zscore",
        "original_value",
        "treated_value",
        "treatment",
    )
    _assert_columns(artifact=artifact, actual=columns, required=required)

    for row in rows:
        _assert_date(str(row["timestamp"]), artifact=artifact, key="timestamp")
        for numeric_key in ("zscore", "original_value", "treated_value"):
            try:
                float(str(row[numeric_key]))
            except ValueError as exc:
                raise ArtifactValidationError(
                    artifact=artifact,
                    key=numeric_key,
                    reason_code="invalid_float",
                    detail=str(row[numeric_key]),
                ) from exc
        if not str(row["treatment"]).strip():
            raise ArtifactValidationError(
                artifact=artifact,
                key="treatment",
                reason_code="empty_value",
            )

    _assert_sorted(rows, artifact=artifact, sort_keys=("series_id", "timestamp"))


def validate_preprocess_artifact_bundle(
    summary_path: str | Path,
    missing_flags_path: str | Path,
    outlier_flags_path: str | Path,
) -> None:
    """Validate all required preprocessing artifacts together."""

    summary = _load_json(Path(summary_path))
    validate_preprocess_summary(summary)

    missing_columns, missing_rows = _load_csv_rows(Path(missing_flags_path))
    validate_missing_value_flags(missing_rows, missing_columns)

    outlier_columns, outlier_rows = _load_csv_rows(Path(outlier_flags_path))
    validate_outlier_flags(outlier_rows, outlier_columns)
