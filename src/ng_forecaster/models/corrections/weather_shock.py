"""Bounded weather shock correction for monthly production point forecasts."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.lag_guard import require_columns


def _feature_value(
    exogenous_features: Mapping[str, Any],
    *,
    keys: tuple[str, ...],
    required: bool,
    default: float = 0.0,
) -> float:
    for key in keys:
        if key not in exogenous_features:
            continue
        try:
            value = float(exogenous_features[key])
        except (TypeError, ValueError) as exc:
            raise ContractViolation(
                "invalid_model_policy",
                key=key,
                detail="weather-shock exogenous feature must be numeric",
            ) from exc
        if np.isnan(value):
            raise ContractViolation(
                "invalid_model_policy",
                key=key,
                detail="weather-shock exogenous feature cannot be NaN",
            )
        return value
    if required:
        raise ContractViolation(
            "missing_column",
            key="|".join(keys),
            detail="required weather-shock exogenous feature is missing",
        )
    return float(default)


def apply_weather_shock_adjustment(
    forecast: pd.DataFrame,
    *,
    exogenous_features: Mapping[str, Any] | None,
    cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply bounded weather correction to point forecasts when configured."""

    require_columns(forecast, ("horizon", "point_forecast"), key="champion_forecast")
    weather_cfg = dict(cfg.get("exogenous", {}).get("weather_shock", {}))
    enabled = bool(weather_cfg.get("enabled", False))

    adjusted = forecast.copy()
    adjusted["weather_shock_adjustment"] = 0.0
    adjusted["weather_shock_signal"] = 0.0
    adjusted["weather_coverage_fraction"] = np.nan

    if not enabled:
        return adjusted, {"enabled": False}
    if exogenous_features is None:
        raise ContractViolation(
            "missing_column",
            key="exogenous_features",
            detail="weather_shock is enabled but no exogenous features were passed",
        )

    intercept = float(weather_cfg.get("intercept", 0.0))
    beta_intensity = float(weather_cfg.get("beta_freeze_intensity", 5000.0))
    beta_days = float(weather_cfg.get("beta_freeze_days", 3000.0))
    beta_extreme = float(weather_cfg.get("beta_extreme_min", 2000.0))
    intensity_scale = float(weather_cfg.get("freeze_intensity_scale", 1.0))
    days_scale = float(weather_cfg.get("freeze_days_scale", 5.0))
    extreme_min_reference = float(weather_cfg.get("extreme_min_reference_f", 20.0))
    extreme_min_scale = float(weather_cfg.get("extreme_min_scale", 15.0))
    min_coverage = float(weather_cfg.get("min_coverage_fraction", 0.75))
    cap = float(weather_cfg.get("cap_abs_adjustment", 25000.0))

    if intensity_scale <= 0 or days_scale <= 0 or extreme_min_scale <= 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="exogenous.weather_shock",
            detail="weather shock scales must all be > 0",
        )
    if min_coverage < 0 or min_coverage > 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="exogenous.weather_shock.min_coverage_fraction",
            detail="min_coverage_fraction must be in [0, 1]",
        )
    if cap < 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="exogenous.weather_shock.cap_abs_adjustment",
            detail="cap_abs_adjustment must be >= 0",
        )

    coverage_fraction = _feature_value(
        exogenous_features,
        keys=("coverage_fraction_mtd", "coverage_fraction_mtd_weighted"),
        required=False,
        default=1.0,
    )
    adjusted["weather_coverage_fraction"] = float(coverage_fraction)

    if coverage_fraction < min_coverage:
        return adjusted, {
            "enabled": True,
            "applied": False,
            "status": "insufficient_coverage",
            "coverage_fraction": float(coverage_fraction),
            "min_coverage_fraction": float(min_coverage),
            "cap_abs_adjustment": float(cap),
        }

    required_feature_groups = {
        "freeze_intensity": ("freeze_intensity_mtd_weighted", "freeze_event_intensity"),
        "freeze_days": ("freeze_days_mtd_weighted", "freeze_days_mtd"),
    }
    missing_groups = [
        group
        for group, keys in required_feature_groups.items()
        if not any(key in exogenous_features for key in keys)
    ]
    if missing_groups:
        return adjusted, {
            "enabled": True,
            "applied": False,
            "status": "missing_features",
            "missing_feature_groups": missing_groups,
            "coverage_fraction": float(coverage_fraction),
            "min_coverage_fraction": float(min_coverage),
            "cap_abs_adjustment": float(cap),
        }

    freeze_intensity = _feature_value(
        exogenous_features,
        keys=("freeze_intensity_mtd_weighted", "freeze_event_intensity"),
        required=True,
    )
    freeze_days = _feature_value(
        exogenous_features,
        keys=("freeze_days_mtd_weighted", "freeze_days_mtd"),
        required=True,
    )
    extreme_min = _feature_value(
        exogenous_features,
        keys=("extreme_min_mtd", "extreme_min_mtd_weighted"),
        required=False,
        default=extreme_min_reference,
    )

    freeze_intensity_signal = float(np.tanh(freeze_intensity / intensity_scale))
    freeze_days_signal = float(np.tanh(freeze_days / days_scale))
    extreme_min_gap = max(0.0, extreme_min_reference - extreme_min)
    extreme_min_signal = float(np.tanh(extreme_min_gap / extreme_min_scale))
    raw_delta = (
        intercept
        + beta_intensity * freeze_intensity_signal
        + beta_days * freeze_days_signal
        + beta_extreme * extreme_min_signal
    )
    delta = float(np.clip(raw_delta, -cap, cap))

    adjusted["point_forecast"] = (
        pd.to_numeric(adjusted["point_forecast"], errors="coerce").astype(float) + delta
    )
    adjusted["weather_shock_adjustment"] = delta
    adjusted["weather_shock_signal"] = (
        freeze_intensity_signal + freeze_days_signal + extreme_min_signal
    ) / 3.0

    return adjusted, {
        "enabled": True,
        "applied": True,
        "status": "applied",
        "coverage_fraction": float(coverage_fraction),
        "min_coverage_fraction": float(min_coverage),
        "freeze_intensity": float(freeze_intensity),
        "freeze_days": float(freeze_days),
        "extreme_min": float(extreme_min),
        "freeze_intensity_signal": freeze_intensity_signal,
        "freeze_days_signal": freeze_days_signal,
        "extreme_min_signal": extreme_min_signal,
        "raw_delta": float(raw_delta),
        "applied_delta": float(delta),
        "cap_abs_adjustment": float(cap),
    }
