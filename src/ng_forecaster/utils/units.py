"""Canonical unit conversions for natural gas production series."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ng_forecaster.errors import ContractViolation


def _as_finite_float(value: object, *, key: str) -> float:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        raise ContractViolation(
            "invalid_model_policy",
            key=key,
            detail=f"{key} must be numeric",
        )
    parsed = float(numeric)
    if not np.isfinite(parsed):
        raise ContractViolation(
            "invalid_model_policy",
            key=key,
            detail=f"{key} must be finite",
        )
    return parsed


def _days_in_month(month_end: object) -> int:
    try:
        ts = pd.Timestamp(month_end)
    except Exception as exc:
        raise ContractViolation(
            "invalid_timestamp",
            key="month_end",
            detail="month_end could not be parsed",
        ) from exc
    if pd.isna(ts):
        raise ContractViolation(
            "invalid_timestamp",
            key="month_end",
            detail="month_end could not be parsed",
        )
    return int(ts.to_period("M").to_timestamp("M").days_in_month)


def bcfd_to_mmcf_per_day(value_bcfd: object) -> float:
    """Convert Bcf/d to MMcf/day."""

    return _as_finite_float(value_bcfd, key="value_bcfd") * 1000.0


def bcfd_to_mmcf_per_month(*, value_bcfd: object, month_end: object) -> float:
    """Convert Bcf/d to monthly MMcf using calendar month length."""

    return bcfd_to_mmcf_per_day(value_bcfd) * float(_days_in_month(month_end))


def mmcf_per_month_to_bcfd(*, value_mmcf_per_month: object, month_end: object) -> float:
    """Convert monthly MMcf to Bcf/d using calendar month length."""

    days = float(_days_in_month(month_end))
    value = _as_finite_float(value_mmcf_per_month, key="value_mmcf_per_month")
    return value / (1000.0 * days)
