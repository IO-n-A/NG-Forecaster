"""Leakage and lag contract guards."""

from __future__ import annotations

from datetime import timedelta
from typing import Iterable, Sequence

import pandas as pd

from ng_forecaster.errors import ContractViolation


def _as_timestamp(value: object, *, field_name: str) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ContractViolation(
            "invalid_timestamp",
            key=field_name,
            detail=f"timestamp for {field_name} cannot be NaT",
        )
    return ts


def _stringify_key(frame: pd.DataFrame, row_index: int, key_cols: Sequence[str]) -> str:
    if not key_cols:
        return str(row_index)
    parts: list[str] = []
    for col in key_cols:
        if col in frame.columns:
            parts.append(f"{col}={frame.at[row_index, col]}")
    if not parts:
        return str(row_index)
    return ",".join(parts)


def ensure_no_future_feature_rows(
    frame: pd.DataFrame,
    *,
    asof: object,
    feature_timestamp_col: str = "feature_timestamp",
    key_cols: Sequence[str] = ("feature_name",),
) -> None:
    """Fail if any feature observation uses information after ``asof``."""

    if feature_timestamp_col not in frame.columns:
        raise ContractViolation(
            "missing_column",
            key=feature_timestamp_col,
            detail="feature timestamp column is required",
        )

    asof_ts = _as_timestamp(asof, field_name="asof")
    observed_ts = pd.to_datetime(frame[feature_timestamp_col], errors="coerce")
    invalid_idx = frame.index[observed_ts > asof_ts]
    if len(invalid_idx) == 0:
        return

    idx = int(invalid_idx[0])
    offending_ts = pd.Timestamp(frame.at[idx, feature_timestamp_col])
    raise ContractViolation(
        "feature_after_asof",
        asof=asof_ts.to_pydatetime(),
        key=_stringify_key(frame, idx, key_cols),
        detail=f"{feature_timestamp_col}={offending_ts.isoformat()} is after asof",
    )


def ensure_target_lag_eligibility(
    frame: pd.DataFrame,
    *,
    asof: object,
    target_timestamp_col: str = "target_timestamp",
    min_lag_days: int = 1,
    key_cols: Sequence[str] = ("target_name",),
) -> None:
    """Fail if any target observation violates the minimum lag from ``asof``."""

    if min_lag_days < 0:
        raise ContractViolation(
            "invalid_lag_policy",
            key="min_lag_days",
            detail="min_lag_days must be >= 0",
        )
    if target_timestamp_col not in frame.columns:
        raise ContractViolation(
            "missing_column",
            key=target_timestamp_col,
            detail="target timestamp column is required",
        )

    asof_ts = _as_timestamp(asof, field_name="asof")
    cutoff = asof_ts - timedelta(days=min_lag_days)
    observed_ts = pd.to_datetime(frame[target_timestamp_col], errors="coerce")
    invalid_idx = frame.index[observed_ts > cutoff]
    if len(invalid_idx) == 0:
        return

    idx = int(invalid_idx[0])
    offending_ts = pd.Timestamp(frame.at[idx, target_timestamp_col])
    raise ContractViolation(
        "target_lag_violation",
        asof=asof_ts.to_pydatetime(),
        key=_stringify_key(frame, idx, key_cols),
        detail=(
            f"{target_timestamp_col}={offending_ts.isoformat()} exceeds lag cutoff"
            f" {cutoff.isoformat()}"
        ),
    )


def ensure_available_before_asof(
    frame: pd.DataFrame,
    *,
    asof: object,
    available_timestamp_col: str = "available_timestamp",
    key_cols: Sequence[str] = ("feature_name",),
) -> None:
    """Fail when feature availability metadata is after the runtime ``asof``."""

    if available_timestamp_col not in frame.columns:
        raise ContractViolation(
            "missing_column",
            key=available_timestamp_col,
            detail="available timestamp column is required",
        )

    asof_ts = _as_timestamp(asof, field_name="asof")
    available_ts = pd.to_datetime(frame[available_timestamp_col], errors="coerce")
    invalid_idx = frame.index[available_ts > asof_ts]
    if len(invalid_idx) == 0:
        return

    idx = int(invalid_idx[0])
    offending_ts = pd.Timestamp(frame.at[idx, available_timestamp_col])
    raise ContractViolation(
        "availability_after_asof",
        asof=asof_ts.to_pydatetime(),
        key=_stringify_key(frame, idx, key_cols),
        detail=f"{available_timestamp_col}={offending_ts.isoformat()} is after asof",
    )


def require_columns(frame: pd.DataFrame, required: Iterable[str], *, key: str) -> None:
    """Fail fast when required columns are missing."""

    missing = [name for name in required if name not in frame.columns]
    if missing:
        raise ContractViolation(
            "missing_column",
            key=key,
            detail=f"missing required columns: {','.join(sorted(missing))}",
        )
