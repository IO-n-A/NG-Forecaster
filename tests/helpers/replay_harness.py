"""Replay helper utilities for leakage-safe fixture validation."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Mapping, Sequence

REASON_FUTURE_FEATURE = "future_feature_timestamp"
REASON_TARGET_NOT_LAGGED = "target_not_lagged"
REASON_MISSING_FIELD = "missing_required_field"
REASON_INVALID_HORIZON = "invalid_horizon"
REASON_CHECKPOINT_GRID_INCOMPLETE = "checkpoint_grid_incomplete"

_VALID_HORIZONS = ("T-1", "T")


class LeakageContractError(ValueError):
    """Contract violation for leakage-safe fixtures and replay grids."""

    def __init__(self, *, asof: str, key: str, reason_code: str, detail: str = "") -> None:
        self.asof = asof
        self.key = key
        self.reason_code = reason_code
        self.detail = detail
        message = f"asof={asof} | key={key} | reason_code={reason_code}"
        if detail:
            message = f"{message} | detail={detail}"
        super().__init__(message)


@dataclass(frozen=True)
class ReplayCheckpoint:
    """Single replay checkpoint and forecast horizon."""

    checkpoint_date: str
    horizon: str
    asof: str


def _parse_date(value: str, *, asof: str, key: str, field_name: str) -> date:
    try:
        return datetime.strptime(value, "%Y-%m-%d").date()
    except ValueError as exc:
        raise LeakageContractError(
            asof=asof,
            key=key,
            reason_code=REASON_MISSING_FIELD,
            detail=f"invalid {field_name}",
        ) from exc


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    """Load a CSV fixture as row dictionaries."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def validate_asof_rows(
    rows: Sequence[Mapping[str, str]],
    *,
    asof_key: str = "asof",
    feature_ts_key: str = "feature_timestamp",
    target_ts_key: str = "target_timestamp",
    key_key: str = "row_key",
) -> list[dict[str, str]]:
    """Validate strict as-of information-set contracts for a fixture row set."""
    validated: list[dict[str, str]] = []
    for row in rows:
        key_value = row.get(key_key, "<missing_key>")
        asof_value = row.get(asof_key)
        feature_value = row.get(feature_ts_key)
        target_value = row.get(target_ts_key)

        if not asof_value or not feature_value or not target_value:
            missing_field = asof_key
            if not feature_value:
                missing_field = feature_ts_key
            if not target_value:
                missing_field = target_ts_key
            raise LeakageContractError(
                asof=asof_value or "<missing_asof>",
                key=key_value,
                reason_code=REASON_MISSING_FIELD,
                detail=f"missing {missing_field}",
            )

        asof_date = _parse_date(asof_value, asof=asof_value, key=key_value, field_name=asof_key)
        feature_date = _parse_date(
            feature_value,
            asof=asof_value,
            key=key_value,
            field_name=feature_ts_key,
        )
        target_date = _parse_date(
            target_value,
            asof=asof_value,
            key=key_value,
            field_name=target_ts_key,
        )

        if feature_date > asof_date:
            raise LeakageContractError(
                asof=asof_value,
                key=key_value,
                reason_code=REASON_FUTURE_FEATURE,
            )

        if target_date >= asof_date:
            raise LeakageContractError(
                asof=asof_value,
                key=key_value,
                reason_code=REASON_TARGET_NOT_LAGGED,
            )

        validated.append(dict(row))

    return sorted(
        validated,
        key=lambda row: (
            row.get(asof_key, ""),
            row.get(key_key, ""),
            row.get(feature_ts_key, ""),
            row.get(target_ts_key, ""),
        ),
    )


def load_and_validate_asof_fixture(path: Path) -> list[dict[str, str]]:
    """Read an as-of fixture from disk and validate leakage contracts."""
    return validate_asof_rows(read_csv_rows(path))


def load_replay_checkpoints(path: Path) -> list[ReplayCheckpoint]:
    """Load replay checkpoints and verify horizon vocabulary."""
    checkpoints: list[ReplayCheckpoint] = []
    for row in read_csv_rows(path):
        key_value = row.get("checkpoint_date", "<missing_checkpoint>")
        asof_value = row.get("asof", "<missing_asof>")
        horizon = row.get("horizon", "")
        if horizon not in _VALID_HORIZONS:
            raise LeakageContractError(
                asof=asof_value,
                key=key_value,
                reason_code=REASON_INVALID_HORIZON,
                detail=f"horizon={horizon}",
            )

        _parse_date(key_value, asof=asof_value, key=key_value, field_name="checkpoint_date")
        _parse_date(asof_value, asof=asof_value, key=key_value, field_name="asof")
        checkpoints.append(ReplayCheckpoint(key_value, horizon, asof_value))

    return sorted(
        checkpoints,
        key=lambda item: (item.checkpoint_date, 0 if item.horizon == "T-1" else 1),
    )


def assert_complete_checkpoint_grid(checkpoints: Sequence[ReplayCheckpoint]) -> None:
    """Require both T-1 and T entries for each checkpoint date."""
    horizon_map: dict[str, set[str]] = {}
    for checkpoint in checkpoints:
        horizon_map.setdefault(checkpoint.checkpoint_date, set()).add(checkpoint.horizon)

    for checkpoint_date, horizons in horizon_map.items():
        if set(_VALID_HORIZONS) != horizons:
            raise LeakageContractError(
                asof=checkpoint_date,
                key=checkpoint_date,
                reason_code=REASON_CHECKPOINT_GRID_INCOMPLETE,
            )


def build_replay_schedule(path: Path) -> list[ReplayCheckpoint]:
    """Load and validate replay schedule from a checkpoint CSV."""
    checkpoints = load_replay_checkpoints(path)
    assert_complete_checkpoint_grid(checkpoints)
    return checkpoints
