"""Schema guards for preprocessing audit artifacts."""

from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping, Sequence

_ALLOWED_STATUS = {"passed", "failed"}
_ALLOWED_GAP_TYPES = {"short", "long"}
_ALLOWED_OUTLIER_ACTIONS = {"winsorize_to_median", "clip", "flag_only"}


class ArtifactSchemaError(ValueError):
    """Structured schema contract error for preprocess artifacts."""

    def __init__(self, *, artifact: str, key: str, reason_code: str, detail: str = "") -> None:
        self.artifact = artifact
        self.key = key
        self.reason_code = reason_code
        self.detail = detail
        message = f"artifact={artifact} | key={key} | reason_code={reason_code}"
        if detail:
            message = f"{message} | detail={detail}"
        super().__init__(message)


def _parse_date(value: str, *, artifact: str, key: str) -> None:
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ArtifactSchemaError(
            artifact=artifact,
            key=key,
            reason_code="invalid_date",
            detail=value,
        ) from exc


def _load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _require_columns(rows: Sequence[Mapping[str, str]], *, artifact: str, required: set[str]) -> None:
    if not rows:
        raise ArtifactSchemaError(artifact=artifact, key="rows", reason_code="empty_artifact")
    row_keys = set(rows[0].keys())
    missing = sorted(required - row_keys)
    if missing:
        raise ArtifactSchemaError(
            artifact=artifact,
            key=",".join(missing),
            reason_code="missing_required_column",
        )


def validate_preprocess_summary(summary: Mapping[str, Any]) -> None:
    """Validate preprocess summary contract fields and value ranges."""
    required = {
        "asof",
        "status",
        "row_count",
        "missing_short_gap_count",
        "missing_long_gap_count",
        "outlier_count",
    }
    missing = sorted(required - set(summary.keys()))
    if missing:
        raise ArtifactSchemaError(
            artifact="preprocess_summary.json",
            key=",".join(missing),
            reason_code="missing_required_key",
        )

    asof = str(summary["asof"])
    status = str(summary["status"])
    _parse_date(asof, artifact="preprocess_summary.json", key="asof")

    if status not in _ALLOWED_STATUS:
        raise ArtifactSchemaError(
            artifact="preprocess_summary.json",
            key="status",
            reason_code="invalid_status",
            detail=status,
        )

    for int_key in (
        "row_count",
        "missing_short_gap_count",
        "missing_long_gap_count",
        "outlier_count",
    ):
        value = summary[int_key]
        if not isinstance(value, int):
            raise ArtifactSchemaError(
                artifact="preprocess_summary.json",
                key=int_key,
                reason_code="invalid_type",
            )
        if value < 0:
            raise ArtifactSchemaError(
                artifact="preprocess_summary.json",
                key=int_key,
                reason_code="negative_value",
            )


def validate_missing_value_flags(rows: Sequence[Mapping[str, str]]) -> None:
    """Validate missing-value flag artifact schema."""
    required = {
        "asof",
        "series_id",
        "timestamp",
        "gap_type",
        "gap_length",
        "imputation_policy",
    }
    _require_columns(rows, artifact="missing_value_flags.csv", required=required)

    for row in rows:
        _parse_date(row["asof"], artifact="missing_value_flags.csv", key="asof")
        _parse_date(row["timestamp"], artifact="missing_value_flags.csv", key="timestamp")

        gap_type = row["gap_type"]
        if gap_type not in _ALLOWED_GAP_TYPES:
            raise ArtifactSchemaError(
                artifact="missing_value_flags.csv",
                key="gap_type",
                reason_code="invalid_gap_type",
                detail=gap_type,
            )

        try:
            gap_length = int(row["gap_length"])
        except ValueError as exc:
            raise ArtifactSchemaError(
                artifact="missing_value_flags.csv",
                key="gap_length",
                reason_code="invalid_gap_length",
                detail=row["gap_length"],
            ) from exc

        if gap_length < 1:
            raise ArtifactSchemaError(
                artifact="missing_value_flags.csv",
                key="gap_length",
                reason_code="invalid_gap_length",
            )

        if not row["imputation_policy"]:
            raise ArtifactSchemaError(
                artifact="missing_value_flags.csv",
                key="imputation_policy",
                reason_code="empty_policy",
            )


def validate_outlier_flags(rows: Sequence[Mapping[str, str]]) -> None:
    """Validate outlier-flag artifact schema."""
    required = {
        "asof",
        "series_id",
        "timestamp",
        "method",
        "z_score",
        "action",
    }
    _require_columns(rows, artifact="outlier_flags.csv", required=required)

    for row in rows:
        _parse_date(row["asof"], artifact="outlier_flags.csv", key="asof")
        _parse_date(row["timestamp"], artifact="outlier_flags.csv", key="timestamp")

        if not row["method"]:
            raise ArtifactSchemaError(
                artifact="outlier_flags.csv",
                key="method",
                reason_code="empty_method",
            )

        try:
            float(row["z_score"])
        except ValueError as exc:
            raise ArtifactSchemaError(
                artifact="outlier_flags.csv",
                key="z_score",
                reason_code="invalid_z_score",
                detail=row["z_score"],
            ) from exc

        action = row["action"]
        if action not in _ALLOWED_OUTLIER_ACTIONS:
            raise ArtifactSchemaError(
                artifact="outlier_flags.csv",
                key="action",
                reason_code="invalid_action",
                detail=action,
            )


def validate_preprocess_artifact_bundle(
    summary_path: Path,
    missing_flags_path: Path,
    outlier_flags_path: Path,
) -> None:
    """Validate all required preprocessing artifacts in one call."""
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    validate_preprocess_summary(summary)

    validate_missing_value_flags(_load_csv_rows(missing_flags_path))
    validate_outlier_flags(_load_csv_rows(outlier_flags_path))
