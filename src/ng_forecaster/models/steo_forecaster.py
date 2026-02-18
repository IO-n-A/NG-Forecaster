"""STEO anchor forecaster used as an explicit fusion stream (Sprint 6 Option A)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd

from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.lag_guard import require_columns
from ng_forecaster.utils.units import bcfd_to_mmcf_per_month

_STEO_FEATURE_PRIORITY = (
    "steo_dry_prod_bcfd",
    "steo_marketed_prod_bcfd",
)


@dataclass(frozen=True)
class SteoForecastResult:
    """Forecast payload and diagnostics for STEO-derived point estimates."""

    forecast: pd.DataFrame
    diagnostics: dict[str, Any]


def _suffix_for_horizon(horizon: int) -> str:
    if int(horizon) == 1:
        return "t"
    if int(horizon) == 2:
        return "t_plus_1"
    return f"t_plus_{int(horizon) - 1}"


def _target_unit_scale(release_history: pd.DataFrame | None) -> tuple[float, str]:
    if release_history is None or release_history.empty:
        return 1000.0, "bcf_per_month"
    require_columns(release_history, ("target_value",), key="release_history")
    target = pd.to_numeric(release_history["target_value"], errors="coerce").dropna()
    if target.empty:
        return 1000.0, "bcf_per_month"
    median_target = float(target.median())
    if median_target >= 100000.0:
        return 1.0, "mmcf_per_month"
    return 1000.0, "bcf_per_month"


def _resolve_feature_point(
    features: pd.DataFrame,
    *,
    horizon: int,
) -> tuple[float, str, pd.Timestamp]:
    suffix = _suffix_for_horizon(horizon)
    candidates = [f"{base}_{suffix}" for base in _STEO_FEATURE_PRIORITY]
    for feature_name in candidates:
        match = features[features["feature_name"] == feature_name].copy()
        if match.empty:
            continue
        match = match.sort_values(["available_timestamp", "feature_timestamp"])
        row = match.iloc[-1]
        return (
            float(row["value"]),
            feature_name,
            pd.Timestamp(row["available_timestamp"]),
        )
    raise ContractViolation(
        "missing_column",
        key=f"steo_feature_h{horizon}",
        detail="missing STEO observation feature for horizon",
    )


def _uncertainty_half_width(
    release_history: pd.DataFrame | None,
    *,
    point_forecast: float,
    horizon: int,
) -> float:
    if release_history is None or release_history.empty:
        return max(abs(point_forecast) * 0.04, 1.0)
    require_columns(release_history, ("target_value",), key="release_history")
    target = pd.to_numeric(release_history["target_value"], errors="coerce").dropna()
    if len(target) < 6:
        return max(abs(point_forecast) * 0.04, 1.0)
    absolute_delta = target.diff().abs().dropna()
    if absolute_delta.empty:
        return max(abs(point_forecast) * 0.04, 1.0)
    width = float(absolute_delta.tail(12).mean() * np.sqrt(float(max(1, horizon))))
    return max(width, abs(point_forecast) * 0.02, 1.0)


def build_steo_forecast(
    feature_rows: pd.DataFrame,
    *,
    target_month: object,
    horizons: Sequence[int] = (1, 2),
    release_history: pd.DataFrame | None = None,
) -> SteoForecastResult:
    """Build STEO point/interval forecasts aligned to target unit scale."""

    require_columns(
        feature_rows,
        ("feature_name", "feature_timestamp", "available_timestamp", "value"),
        key="feature_rows",
    )
    features = feature_rows.copy()
    features["feature_name"] = features["feature_name"].astype(str)
    features["feature_timestamp"] = pd.to_datetime(
        features["feature_timestamp"],
        errors="coerce",
    )
    features["available_timestamp"] = pd.to_datetime(
        features["available_timestamp"],
        errors="coerce",
    )
    features["value"] = pd.to_numeric(features["value"], errors="coerce")
    features = features[
        features["feature_timestamp"].notna()
        & features["available_timestamp"].notna()
        & features["value"].notna()
    ].copy()
    if features.empty:
        raise ContractViolation(
            "missing_feature_input",
            key="steo_features",
            detail="feature rows are empty after timestamp/value normalization",
        )

    target = pd.Timestamp(target_month).to_period("M").to_timestamp("M")
    parsed_horizons = sorted({int(item) for item in horizons})
    if not parsed_horizons or parsed_horizons[0] < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="horizons",
            detail="horizons must include positive integers",
        )

    scale_divisor, target_unit = _target_unit_scale(release_history)
    rows: list[dict[str, Any]] = []
    for horizon in parsed_horizons:
        month = (target.to_period("M") + horizon - 1).to_timestamp("M")
        bcfd_value, source_feature, available_ts = _resolve_feature_point(
            features, horizon=int(horizon)
        )
        mmcf_month = bcfd_to_mmcf_per_month(value_bcfd=bcfd_value, month_end=month)
        point = float(mmcf_month / scale_divisor)
        half_width = _uncertainty_half_width(
            release_history,
            point_forecast=point,
            horizon=int(horizon),
        )
        rows.append(
            {
                "horizon": int(horizon),
                "steo_point_forecast": float(point),
                "steo_lower_95": float(point - 1.96 * half_width),
                "steo_upper_95": float(point + 1.96 * half_width),
                "steo_source_feature": source_feature,
                "steo_available_timestamp": available_ts,
                "steo_target_unit": target_unit,
                "steo_raw_bcfd": float(bcfd_value),
            }
        )

    frame = pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)
    diagnostics = {
        "target_month": target.date().isoformat(),
        "target_unit": target_unit,
        "source_feature_priority": list(_STEO_FEATURE_PRIORITY),
        "n_horizons": int(len(frame)),
    }
    return SteoForecastResult(forecast=frame, diagnostics=diagnostics)

