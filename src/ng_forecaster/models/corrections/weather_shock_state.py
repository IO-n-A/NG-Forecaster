"""Stateful freeze-off weather correction with bounded recovery dynamics."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.lag_guard import require_columns

_DEFAULT_STATE_CFG: dict[str, Any] = {
    "enabled": False,
    "persistence": 0.65,
    "impact_weight": 12000.0,
    "recovery_weight": 0.55,
    "cap_abs_adjustment": 20000.0,
}


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
                detail="weather-shock-state exogenous feature must be numeric",
            ) from exc
        if np.isnan(value):
            raise ContractViolation(
                "invalid_model_policy",
                key=key,
                detail="weather-shock-state exogenous feature cannot be NaN",
            )
        return value
    if required:
        raise ContractViolation(
            "missing_column",
            key="|".join(keys),
            detail="required weather-shock-state exogenous feature is missing",
        )
    return float(default)


def _merge_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    cfg = {**_DEFAULT_STATE_CFG, **dict(payload)}
    persistence = float(cfg["persistence"])
    impact_weight = float(cfg["impact_weight"])
    recovery_weight = float(cfg["recovery_weight"])
    cap = float(cfg["cap_abs_adjustment"])
    if persistence < 0 or persistence >= 1.0:
        raise ContractViolation(
            "invalid_model_policy",
            key="exogenous.weather_shock_state.persistence",
            detail="persistence must be in [0, 1)",
        )
    if impact_weight < 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="exogenous.weather_shock_state.impact_weight",
            detail="impact_weight must be >= 0",
        )
    if recovery_weight < 0 or recovery_weight > 1.5:
        raise ContractViolation(
            "invalid_model_policy",
            key="exogenous.weather_shock_state.recovery_weight",
            detail="recovery_weight must be in [0, 1.5]",
        )
    if cap < 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="exogenous.weather_shock_state.cap_abs_adjustment",
            detail="cap_abs_adjustment must be >= 0",
        )
    return {
        "enabled": bool(cfg["enabled"]),
        "persistence": persistence,
        "impact_weight": impact_weight,
        "recovery_weight": recovery_weight,
        "cap_abs_adjustment": cap,
    }


def apply_weather_shock_state_adjustment(
    forecast: pd.DataFrame,
    *,
    exogenous_features: Mapping[str, Any] | None,
    cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply a bounded AR(1)-style lost-production/recovery correction."""

    require_columns(forecast, ("horizon", "point_forecast"), key="champion_forecast")
    weather_state_cfg = _merge_config(
        dict(cfg.get("exogenous", {}).get("weather_shock_state", {}))
    )
    adjusted = forecast.copy()
    adjusted["weather_state_adjustment"] = 0.0
    adjusted["weather_state_lost_production"] = 0.0
    adjusted["weather_state_signal"] = 0.0

    if not bool(weather_state_cfg["enabled"]):
        return adjusted, {"enabled": False}
    if exogenous_features is None:
        raise ContractViolation(
            "missing_column",
            key="exogenous_features",
            detail="weather_shock_state is enabled but no exogenous features were passed",
        )

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
    intensity_signal = float(np.tanh(freeze_intensity))
    days_signal = float(np.tanh(freeze_days / 5.0))
    shock_signal = max(0.0, (intensity_signal + days_signal) / 2.0)

    persistence = float(weather_state_cfg["persistence"])
    impact_weight = float(weather_state_cfg["impact_weight"])
    recovery_weight = float(weather_state_cfg["recovery_weight"])
    cap = float(weather_state_cfg["cap_abs_adjustment"])

    lost_state = 0.0
    sorted_idx = adjusted.sort_values("horizon").index.tolist()
    for step, idx in enumerate(sorted_idx, start=1):
        horizon_scale = 1.0 / float(step)
        impact = impact_weight * shock_signal * horizon_scale
        previous_lost = lost_state
        lost_state = max(0.0, persistence * lost_state + impact)
        recovery = recovery_weight * previous_lost
        delta = float(np.clip((-lost_state + recovery), -cap, cap))
        adjusted.at[idx, "point_forecast"] = (
            float(adjusted.at[idx, "point_forecast"]) + delta
        )
        adjusted.at[idx, "weather_state_adjustment"] = delta
        adjusted.at[idx, "weather_state_lost_production"] = float(lost_state)
        adjusted.at[idx, "weather_state_signal"] = float(shock_signal)

    return adjusted, {
        "enabled": True,
        "applied": True,
        "status": "applied",
        "freeze_intensity": float(freeze_intensity),
        "freeze_days": float(freeze_days),
        "shock_signal": float(shock_signal),
        "persistence": persistence,
        "impact_weight": impact_weight,
        "recovery_weight": recovery_weight,
        "cap_abs_adjustment": cap,
    }
