"""Deterministic baseline generators for rolling validation integrity checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.lag_guard import require_columns
from ng_forecaster.utils.units import bcfd_to_mmcf_per_month

BASELINE_VARIANTS: tuple[str, ...] = (
    "baseline_seasonal_naive",
    "baseline_drift",
    "baseline_steo_anchor_naive",
)

_STEO_ANCHOR_FEATURE_PRIORITY: tuple[str, ...] = (
    "steo_dry_prod_bcfd_t",
    "steo_marketed_prod_bcfd_t",
)


@dataclass(frozen=True)
class BaselinePointEstimate:
    """Single deterministic baseline estimate for a target month."""

    model_variant: str
    fused_point: float
    metadata: dict[str, Any]


def _month_end(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ContractViolation(
            "invalid_timestamp",
            key="target_month",
            detail="target month could not be parsed",
        )
    return ts.to_period("M").to_timestamp("M")


def _prepare_history(history: pd.DataFrame, *, key: str) -> pd.DataFrame:
    require_columns(history, ("timestamp", "target_value"), key=key)
    prepared = history[["timestamp", "target_value"]].copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
    prepared["target_value"] = pd.to_numeric(prepared["target_value"], errors="coerce")
    prepared = prepared[
        prepared["timestamp"].notna() & prepared["target_value"].notna()
    ]
    prepared["timestamp"] = prepared["timestamp"].dt.to_period("M").dt.to_timestamp("M")
    prepared = prepared.sort_values("timestamp").drop_duplicates(
        "timestamp", keep="last"
    )
    prepared = prepared.reset_index(drop=True)
    if prepared.empty:
        raise ContractViolation(
            "missing_target_history_rows",
            key=key,
            detail="history is empty after timestamp/value normalization",
        )
    return prepared


def _align_steo_anchor_unit(
    steo_mmcf_per_month: float,
    *,
    history: pd.DataFrame,
) -> tuple[float, str]:
    """Align STEO monthly totals to observed target scale (MMcf vs Bcf)."""

    median_target = float(pd.to_numeric(history["target_value"], errors="coerce").median())
    if median_target >= 100000.0:
        return float(steo_mmcf_per_month), "mmcf_per_month"
    return float(steo_mmcf_per_month / 1000.0), "bcf_per_month"


def _seasonal_naive(
    history: pd.DataFrame, *, target_month: pd.Timestamp
) -> BaselinePointEstimate:
    seasonal_month = (target_month.to_period("M") - 12).to_timestamp("M")
    row = history[history["timestamp"] == seasonal_month]
    if row.empty:
        raise ContractViolation(
            "insufficient_release_history",
            key="baseline_seasonal_naive",
            detail=(
                "seasonal naive requires at least one same-month-last-year value; "
                f"missing={seasonal_month.date().isoformat()}"
            ),
        )
    return BaselinePointEstimate(
        model_variant="baseline_seasonal_naive",
        fused_point=float(row.iloc[-1]["target_value"]),
        metadata={
            "reference_timestamp": seasonal_month.date().isoformat(),
            "method": "same_month_last_year",
        },
    )


def _drift(
    history: pd.DataFrame, *, target_month: pd.Timestamp
) -> BaselinePointEstimate:
    if len(history) < 2:
        raise ContractViolation(
            "insufficient_release_history",
            key="baseline_drift",
            detail="drift baseline requires at least two released observations",
        )
    lookback = min(12, int(len(history)))
    window = history.tail(lookback).reset_index(drop=True)
    first = window.iloc[0]
    last = window.iloc[-1]
    first_ts = pd.Timestamp(first["timestamp"]).to_period("M").to_timestamp("M")
    last_ts = pd.Timestamp(last["timestamp"]).to_period("M").to_timestamp("M")
    span_months = max(
        1,
        int((last_ts.year - first_ts.year) * 12 + (last_ts.month - first_ts.month)),
    )
    monthly_delta = float(last["target_value"] - first["target_value"]) / float(
        span_months
    )
    horizon_months = max(
        1,
        int(
            (target_month.year - last_ts.year) * 12
            + (target_month.month - last_ts.month)
        ),
    )
    estimate = float(last["target_value"]) + monthly_delta * float(horizon_months)
    return BaselinePointEstimate(
        model_variant="baseline_drift",
        fused_point=estimate,
        metadata={
            "window_rows": int(len(window)),
            "monthly_delta": float(monthly_delta),
            "horizon_months": int(horizon_months),
            "method": "rolling_drift",
        },
    )


def _prepare_feature_rows(features: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        features,
        ("feature_name", "feature_timestamp", "available_timestamp", "value"),
        key="baseline_features",
    )
    prepared = features[
        ["feature_name", "feature_timestamp", "available_timestamp", "value"]
    ].copy()
    prepared["feature_timestamp"] = pd.to_datetime(
        prepared["feature_timestamp"],
        errors="coerce",
    )
    prepared["available_timestamp"] = pd.to_datetime(
        prepared["available_timestamp"],
        errors="coerce",
    )
    prepared["value"] = pd.to_numeric(prepared["value"], errors="coerce")
    prepared = prepared[
        prepared["feature_timestamp"].notna()
        & prepared["available_timestamp"].notna()
        & prepared["value"].notna()
    ].copy()
    return prepared.sort_values(
        ["feature_name", "feature_timestamp", "available_timestamp"]
    ).reset_index(drop=True)


def _steo_anchor_naive(
    history: pd.DataFrame,
    *,
    target_month: pd.Timestamp,
    features: pd.DataFrame | None,
) -> BaselinePointEstimate:
    if features is None or features.empty:
        fallback = _seasonal_naive(history, target_month=target_month)
        return BaselinePointEstimate(
            model_variant="baseline_steo_anchor_naive",
            fused_point=fallback.fused_point,
            metadata={
                "method": "steo_anchor_fallback_to_seasonal",
                "fallback_reference_timestamp": fallback.metadata[
                    "reference_timestamp"
                ],
            },
        )

    prepared = _prepare_feature_rows(features)
    if prepared.empty:
        fallback = _seasonal_naive(history, target_month=target_month)
        return BaselinePointEstimate(
            model_variant="baseline_steo_anchor_naive",
            fused_point=fallback.fused_point,
            metadata={
                "method": "steo_anchor_fallback_to_seasonal",
                "fallback_reference_timestamp": fallback.metadata[
                    "reference_timestamp"
                ],
            },
        )

    target_mask = (
        prepared["feature_timestamp"].dt.to_period("M").dt.to_timestamp("M")
        == target_month
    )
    target_rows = prepared[target_mask].copy()
    for feature_name in _STEO_ANCHOR_FEATURE_PRIORITY:
        candidate = target_rows[target_rows["feature_name"] == feature_name]
        if candidate.empty:
            continue
        row = candidate.sort_values("available_timestamp").iloc[-1]
        monthly_total_mmcf = bcfd_to_mmcf_per_month(
            value_bcfd=float(row["value"]),
            month_end=target_month,
        )
        monthly_total, aligned_unit = _align_steo_anchor_unit(
            monthly_total_mmcf,
            history=history,
        )
        return BaselinePointEstimate(
            model_variant="baseline_steo_anchor_naive",
            fused_point=float(monthly_total),
            metadata={
                "method": "steo_anchor_naive",
                "anchor_unit_in": "bcf_per_day",
                "anchor_unit_out": str(aligned_unit),
                "anchor_feature": feature_name,
                "anchor_feature_timestamp": pd.Timestamp(row["feature_timestamp"])
                .date()
                .isoformat(),
                "anchor_available_timestamp": pd.Timestamp(row["available_timestamp"])
                .date()
                .isoformat(),
            },
        )

    fallback = _seasonal_naive(history, target_month=target_month)
    return BaselinePointEstimate(
        model_variant="baseline_steo_anchor_naive",
        fused_point=fallback.fused_point,
        metadata={
            "method": "steo_anchor_fallback_to_seasonal",
            "fallback_reference_timestamp": fallback.metadata["reference_timestamp"],
        },
    )


def build_baseline_point_estimates(
    target_history_released: pd.DataFrame,
    *,
    target_month: object,
    feature_rows: pd.DataFrame | None = None,
) -> list[BaselinePointEstimate]:
    """Generate deterministic CP0 baseline estimates for a target month."""

    history = _prepare_history(target_history_released, key="target_history_released")
    target = _month_end(target_month)
    return [
        _seasonal_naive(history, target_month=target),
        _drift(history, target_month=target),
        _steo_anchor_naive(history, target_month=target, features=feature_rows),
    ]
