"""Direct calendar calibration for fused monthly point forecasts."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd

from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.lag_guard import require_columns
from ng_forecaster.utils.calendar import is_leap_february, parse_month_end

DEFAULT_CALENDAR_CALIBRATION: dict[str, Any] = {
    "enabled": False,
    "max_abs_adjustment": 0.0,
    "day_weights": {
        "28": 0.0,
        "29": 0.0,
        "30": 0.0,
        "31": 0.0,
    },
    "leap_february_bonus": 0.0,
    "regime_scale": {},
}


def _merge_dict(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            merged[key] = _merge_dict(base[key], value)
        else:
            merged[key] = value
    return merged


def validate_calendar_calibration_config(
    payload: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Validate calendar calibration configuration."""

    merged = _merge_dict(DEFAULT_CALENDAR_CALIBRATION, payload or {})
    enabled = bool(merged.get("enabled", False))
    max_abs_adjustment = float(merged.get("max_abs_adjustment", 0.0))
    if max_abs_adjustment < 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="calendar_calibration.max_abs_adjustment",
            detail="max_abs_adjustment must be >= 0",
        )

    day_weights_raw = merged.get("day_weights", {})
    if not isinstance(day_weights_raw, Mapping):
        raise ContractViolation(
            "invalid_model_policy",
            key="calendar_calibration.day_weights",
            detail="day_weights must be a mapping",
        )
    day_weights: dict[str, float] = {}
    for day in (28, 29, 30, 31):
        key = str(day)
        day_weights[key] = float(day_weights_raw.get(key, day_weights_raw.get(day, 0.0)))

    leap_february_bonus = float(merged.get("leap_february_bonus", 0.0))
    regime_scale_raw = merged.get("regime_scale", {})
    if not isinstance(regime_scale_raw, Mapping):
        raise ContractViolation(
            "invalid_model_policy",
            key="calendar_calibration.regime_scale",
            detail="regime_scale must be a mapping",
        )
    regime_scale: dict[str, float] = {
        str(label): float(value) for label, value in regime_scale_raw.items()
    }

    return {
        "enabled": enabled,
        "max_abs_adjustment": max_abs_adjustment,
        "day_weights": day_weights,
        "leap_february_bonus": leap_february_bonus,
        "regime_scale": regime_scale,
    }


def _resolve_regime_multiplier(
    cfg: Mapping[str, Any],
    *,
    regime_label: str,
) -> float:
    regime_scale = cfg.get("regime_scale", {})
    if not isinstance(regime_scale, Mapping):
        return 1.0
    return float(regime_scale.get(regime_label, 1.0))


def apply_calendar_calibration(
    frame: pd.DataFrame,
    *,
    calibration_config: Mapping[str, Any] | None,
    regime_label: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Apply direct day-count and leap-year calibration to fused points."""

    require_columns(
        frame,
        ("fused_point", "target_month_days"),
        key="calendar_calibration_input",
    )
    cfg = validate_calendar_calibration_config(calibration_config)
    output = frame.copy()
    output["fused_point_pre_calendar_calibration"] = pd.to_numeric(
        output["fused_point"], errors="coerce"
    )
    if output["fused_point_pre_calendar_calibration"].isna().any():
        raise ContractViolation(
            "invalid_model_policy",
            key="fused_point",
            detail="fused_point must be numeric for calendar calibration",
        )
    output["calendar_calibration_delta"] = 0.0
    output["calendar_calibration_applied"] = False
    output["is_leap_february"] = False

    if not bool(cfg["enabled"]):
        output["fused_point"] = output["fused_point_pre_calendar_calibration"].astype(float)
        return output, {"enabled": False, "applied_rows": 0}

    day_series = pd.to_numeric(output["target_month_days"], errors="coerce")
    if day_series.isna().any():
        raise ContractViolation(
            "invalid_model_policy",
            key="target_month_days",
            detail="target_month_days must be numeric for calendar calibration",
        )
    day_series = day_series.astype(int)
    regime_multiplier = _resolve_regime_multiplier(cfg, regime_label=regime_label)
    leap_bonus = float(cfg["leap_february_bonus"])
    day_weights = dict(cfg["day_weights"])
    max_abs_adjustment = float(cfg["max_abs_adjustment"])

    if "anchor_month_end" in output.columns:
        leap_flags = output["anchor_month_end"].map(
            lambda value: bool(is_leap_february(value))
            if pd.notna(value)
            else False
        )
    else:
        leap_flags = pd.Series([False] * len(output), index=output.index)
    output["is_leap_february"] = leap_flags.astype(bool)

    base_series = output["fused_point_pre_calendar_calibration"].astype(float)
    if "release_anchor_point" in output.columns:
        anchor_series = pd.to_numeric(output["release_anchor_point"], errors="coerce")
        anchor_series = anchor_series.where(anchor_series.notna(), base_series)
    else:
        anchor_series = base_series

    deltas: list[float] = []
    applied_rows = 0
    for idx, day_count in day_series.items():
        day_weight = float(day_weights.get(str(int(day_count)), 0.0))
        day_shift = day_weight * float(anchor_series.at[idx] - base_series.at[idx])
        leap_shift = 0.0
        if bool(leap_flags.at[idx]):
            leap_shift = leap_bonus
        delta = (day_shift + leap_shift) * regime_multiplier
        delta = float(np.clip(delta, -max_abs_adjustment, max_abs_adjustment))
        deltas.append(delta)
        if abs(delta) > 1e-12:
            applied_rows += 1

    output["calendar_calibration_delta"] = deltas
    output["calendar_calibration_applied"] = (
        output["calendar_calibration_delta"].abs() > 1e-12
    )
    output["fused_point"] = (
        output["fused_point_pre_calendar_calibration"] + output["calendar_calibration_delta"]
    )

    return output, {
        "enabled": True,
        "applied_rows": int(applied_rows),
        "total_rows": int(len(output)),
        "max_abs_adjustment": max_abs_adjustment,
        "regime_multiplier": float(regime_multiplier),
        "regime_label": str(regime_label),
    }


def apply_calendar_calibration_to_nowcast_row(
    *,
    fused_point: float,
    target_month: object,
    calibration_config: Mapping[str, Any] | None,
    regime_label: str,
    release_anchor_point: float | None = None,
) -> dict[str, float | bool | int | str]:
    """Convenience wrapper for single-row calibration payload generation."""

    month_end = parse_month_end(target_month, key="target_month")
    payload = pd.DataFrame(
        [
            {
                "fused_point": float(fused_point),
                "target_month_days": int(month_end.day),
                "anchor_month_end": month_end,
                "release_anchor_point": (
                    float(release_anchor_point)
                    if release_anchor_point is not None
                    else float(fused_point)
                ),
            }
        ]
    )
    calibrated, _ = apply_calendar_calibration(
        payload,
        calibration_config=calibration_config,
        regime_label=regime_label,
    )
    row = calibrated.iloc[0]
    return {
        "fused_point": float(row["fused_point"]),
        "fused_point_pre_calendar_calibration": float(
            row["fused_point_pre_calendar_calibration"]
        ),
        "calendar_calibration_delta": float(row["calendar_calibration_delta"]),
        "calendar_calibration_applied": bool(row["calendar_calibration_applied"]),
        "target_month_days": int(row["target_month_days"]),
        "is_leap_february": bool(row["is_leap_february"]),
        "regime_label": str(regime_label),
    }
