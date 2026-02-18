"""Champion/challenger fusion with stability and calibration diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import pandas as pd

from ng_forecaster.data.target_transforms import daily_average_to_monthly_total
from ng_forecaster.data.validators import load_yaml
from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.lag_guard import require_columns
from ng_forecaster.models.corrections.calendar_calibration import (
    apply_calendar_calibration,
    validate_calendar_calibration_config,
)

DEFAULT_FUSION_POLICY: dict[str, Any] = {
    "version": 1,
    "base": {
        "champion_weight": 0.70,
        "horizon_weights": {"1": 0.70, "2": 0.70},
        "release_anchor_weight": 0.15,
        "steo_weight": 0.0,
        "prototype_weight": 0.0,
        "month_length_bias_weight": 0.0,
    },
    "regime_overrides": {},
    "weight_search": {
        "champion_weight_grid": [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        "release_anchor_weight_grid": [0.00, 0.05, 0.10, 0.15, 0.20, 0.25],
        "steo_weight_grid": [0.00, 0.05, 0.10],
        "prototype_weight_grid": [0.00, 0.05, 0.10],
        "calendar_calibration_enabled_grid": [0, 1],
        "calendar_calibration_weight_grid": [0.5, 1.0, 1.5],
        "calendar_calibration_cap_abs_grid": [5000.0, 10000.0, 15000.0],
    },
    "anchor_ablation": {
        "release_anchor_weight_grid": [0.00, 0.05, 0.10, 0.15, 0.20, 0.25],
    },
    "calendar_calibration": {
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
    },
}


@dataclass(frozen=True)
class FusionResult:
    """Fusion output with diagnostics."""

    forecast: pd.DataFrame
    divergence_summary: dict[str, float]
    stability_summary: pd.DataFrame
    calibration_summary: dict[str, float]


def _merge_dict(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            merged[key] = _merge_dict(base[key], value)
        else:
            merged[key] = value
    return merged


def _validate_champion_weight(weight: float, *, key: str) -> float:
    value = float(weight)
    if value <= 0 or value > 1:
        raise ContractViolation(
            "invalid_model_policy",
            key=key,
            detail="champion_weight must be in the half-open interval (0, 1]",
        )
    return value


def _validate_anchor_weight(weight: float, *, key: str) -> float:
    value = float(weight)
    if value < 0 or value >= 1:
        raise ContractViolation(
            "invalid_model_policy",
            key=key,
            detail="release_anchor_weight must be in the half-open interval [0, 1)",
        )
    return value


def _validate_steo_weight(weight: float, *, key: str) -> float:
    value = float(weight)
    if value < 0 or value >= 1:
        raise ContractViolation(
            "invalid_model_policy",
            key=key,
            detail="steo_weight must be in the half-open interval [0, 1)",
        )
    return value


def _validate_prototype_weight(weight: float, *, key: str) -> float:
    value = float(weight)
    if value < 0 or value >= 1:
        raise ContractViolation(
            "invalid_model_policy",
            key=key,
            detail="prototype_weight must be in the half-open interval [0, 1)",
        )
    return value


def _validate_month_length_bias_weight(weight: float, *, key: str) -> float:
    value = float(weight)
    if value < 0 or value >= 1:
        raise ContractViolation(
            "invalid_model_policy",
            key=key,
            detail="month_length_bias_weight must be in the half-open interval [0, 1)",
        )
    return value


def _parse_horizon_weights(
    value: Mapping[str, Any] | Mapping[int, Any] | None,
    *,
    key: str,
) -> dict[int, float]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ContractViolation(
            "invalid_model_policy",
            key=key,
            detail="horizon_weights must be a mapping",
        )
    parsed: dict[int, float] = {}
    for raw_horizon, raw_weight in value.items():
        try:
            horizon = int(raw_horizon)
        except (TypeError, ValueError) as exc:
            raise ContractViolation(
                "invalid_model_policy",
                key=f"{key}.{raw_horizon}",
                detail="horizon key must be an integer",
            ) from exc
        if horizon < 1:
            raise ContractViolation(
                "invalid_model_policy",
                key=f"{key}.{raw_horizon}",
                detail="horizon key must be >= 1",
            )
        parsed[horizon] = _validate_champion_weight(
            float(raw_weight),
            key=f"{key}.{horizon}",
        )
    return parsed


def load_fusion_policy(config_path: str = "configs/fusion.yaml") -> dict[str, Any]:
    """Load and validate fusion policy used by evaluation and runtime."""

    payload = load_yaml(config_path)
    merged = _merge_dict(DEFAULT_FUSION_POLICY, payload)

    base = dict(merged.get("base", {}))
    base_champion = _validate_champion_weight(
        float(base.get("champion_weight", 0.70)),
        key="base.champion_weight",
    )
    base_anchor = _validate_anchor_weight(
        float(base.get("release_anchor_weight", 0.15)),
        key="base.release_anchor_weight",
    )
    base_steo = _validate_steo_weight(
        float(base.get("steo_weight", 0.0)),
        key="base.steo_weight",
    )
    base_prototype = _validate_prototype_weight(
        float(base.get("prototype_weight", 0.0)),
        key="base.prototype_weight",
    )
    base_month_length_bias = _validate_month_length_bias_weight(
        float(base.get("month_length_bias_weight", 0.0)),
        key="base.month_length_bias_weight",
    )
    if base_champion + base_steo + base_prototype > 1.0:
        raise ContractViolation(
            "invalid_model_policy",
            key="base.prototype_weight",
            detail="champion_weight + steo_weight + prototype_weight must be <= 1",
        )
    base_horizon = _parse_horizon_weights(
        base.get("horizon_weights"),
        key="base.horizon_weights",
    )
    if not base_horizon:
        base_horizon = {1: base_champion, 2: base_champion}

    regime_raw = merged.get("regime_overrides", {})
    if not isinstance(regime_raw, Mapping):
        raise ContractViolation(
            "invalid_model_policy",
            key="regime_overrides",
            detail="regime_overrides must be a mapping",
        )

    regime_overrides: dict[str, dict[str, Any]] = {}
    for regime_label, regime_cfg_raw in regime_raw.items():
        if not isinstance(regime_cfg_raw, Mapping):
            raise ContractViolation(
                "invalid_model_policy",
                key=f"regime_overrides.{regime_label}",
                detail="regime override must be a mapping",
            )
        regime_cfg = dict(regime_cfg_raw)
        horizon_weights = _parse_horizon_weights(
            regime_cfg.get("horizon_weights"),
            key=f"regime_overrides.{regime_label}.horizon_weights",
        )
        anchor_weight = _validate_anchor_weight(
            float(regime_cfg.get("release_anchor_weight", base_anchor)),
            key=f"regime_overrides.{regime_label}.release_anchor_weight",
        )
        steo_weight = _validate_steo_weight(
            float(regime_cfg.get("steo_weight", base_steo)),
            key=f"regime_overrides.{regime_label}.steo_weight",
        )
        prototype_weight = _validate_prototype_weight(
            float(regime_cfg.get("prototype_weight", base_prototype)),
            key=f"regime_overrides.{regime_label}.prototype_weight",
        )
        month_length_bias_weight = _validate_month_length_bias_weight(
            float(
                regime_cfg.get(
                    "month_length_bias_weight",
                    base_month_length_bias,
                )
            ),
            key=f"regime_overrides.{regime_label}.month_length_bias_weight",
        )
        if base_champion + steo_weight + prototype_weight > 1.0:
            raise ContractViolation(
                "invalid_model_policy",
                key=f"regime_overrides.{regime_label}.prototype_weight",
                detail=(
                    "champion_weight + steo_weight + prototype_weight must be <= 1 "
                    "for each regime override"
                ),
            )
        regime_overrides[str(regime_label)] = {
            "horizon_weights": horizon_weights,
            "release_anchor_weight": anchor_weight,
            "steo_weight": steo_weight,
            "prototype_weight": prototype_weight,
            "month_length_bias_weight": month_length_bias_weight,
        }

    search_cfg = dict(merged.get("weight_search", {}))
    search_champion = sorted(
        {
            _validate_champion_weight(
                float(value),
                key="weight_search.champion_weight_grid",
            )
            for value in search_cfg.get("champion_weight_grid", [base_champion])
        }
    )
    search_anchor = sorted(
        {
            _validate_anchor_weight(
                float(value),
                key="weight_search.release_anchor_weight_grid",
            )
            for value in search_cfg.get("release_anchor_weight_grid", [base_anchor])
        }
    )
    search_steo = sorted(
        {
            _validate_steo_weight(
                float(value),
                key="weight_search.steo_weight_grid",
            )
            for value in search_cfg.get("steo_weight_grid", [base_steo])
        }
    )
    search_prototype = sorted(
        {
            _validate_prototype_weight(
                float(value),
                key="weight_search.prototype_weight_grid",
            )
            for value in search_cfg.get("prototype_weight_grid", [base_prototype])
        }
    )
    if not search_champion:
        search_champion = [base_champion]
    if not search_anchor:
        search_anchor = [base_anchor]
    if not search_steo:
        search_steo = [base_steo]
    if not search_prototype:
        search_prototype = [base_prototype]
    calendar_calibration = validate_calendar_calibration_config(
        merged.get("calendar_calibration", {})
    )

    enabled_grid_raw = search_cfg.get(
        "calendar_calibration_enabled_grid",
        [int(calendar_calibration.get("enabled", False))],
    )
    if not isinstance(enabled_grid_raw, list):
        enabled_grid_raw = [enabled_grid_raw]
    search_calibration_enabled = sorted(
        {int(bool(value)) for value in enabled_grid_raw if int(bool(value)) in (0, 1)}
    )
    if not search_calibration_enabled:
        search_calibration_enabled = [int(calendar_calibration.get("enabled", False))]

    weight_grid_raw = search_cfg.get("calendar_calibration_weight_grid", [1.0])
    if not isinstance(weight_grid_raw, list):
        weight_grid_raw = [weight_grid_raw]
    search_calibration_weight = sorted(
        {float(value) for value in weight_grid_raw if float(value) >= 0.0}
    )
    if not search_calibration_weight:
        search_calibration_weight = [1.0]

    cap_grid_raw = search_cfg.get(
        "calendar_calibration_cap_abs_grid",
        [float(calendar_calibration.get("max_abs_adjustment", 0.0))],
    )
    if not isinstance(cap_grid_raw, list):
        cap_grid_raw = [cap_grid_raw]
    search_calibration_cap_abs = sorted(
        {float(value) for value in cap_grid_raw if float(value) >= 0.0}
    )
    if not search_calibration_cap_abs:
        search_calibration_cap_abs = [
            float(calendar_calibration.get("max_abs_adjustment", 0.0))
        ]

    anchor_cfg = dict(merged.get("anchor_ablation", {}))
    anchor_grid = sorted(
        {
            _validate_anchor_weight(
                float(value),
                key="anchor_ablation.release_anchor_weight_grid",
            )
            for value in anchor_cfg.get("release_anchor_weight_grid", [base_anchor])
        }
    )
    if not anchor_grid:
        anchor_grid = [base_anchor]

    return {
        "version": int(merged.get("version", 1)),
        "base": {
            "champion_weight": base_champion,
            "horizon_weights": base_horizon,
            "release_anchor_weight": base_anchor,
            "steo_weight": base_steo,
            "prototype_weight": base_prototype,
            "month_length_bias_weight": base_month_length_bias,
        },
        "regime_overrides": regime_overrides,
        "weight_search": {
            "champion_weight_grid": search_champion,
            "release_anchor_weight_grid": search_anchor,
            "steo_weight_grid": search_steo,
            "prototype_weight_grid": search_prototype,
            "calendar_calibration_enabled_grid": search_calibration_enabled,
            "calendar_calibration_weight_grid": search_calibration_weight,
            "calendar_calibration_cap_abs_grid": search_calibration_cap_abs,
        },
        "anchor_ablation": {
            "release_anchor_weight_grid": anchor_grid,
        },
        "calendar_calibration": calendar_calibration,
    }


def resolve_fusion_weights_for_regime_full(
    fusion_policy: Mapping[str, Any],
    *,
    regime_label: str,
) -> tuple[dict[int, float], float, float, float, float]:
    """Resolve horizon and anchor weights for a given regime label."""

    base = dict(fusion_policy.get("base", {}))
    base_horizon = _parse_horizon_weights(
        base.get("horizon_weights"),
        key="base.horizon_weights",
    )
    if not base_horizon:
        champion = _validate_champion_weight(
            float(base.get("champion_weight", 0.70)),
            key="base.champion_weight",
        )
        base_horizon = {1: champion, 2: champion}
    anchor = _validate_anchor_weight(
        float(base.get("release_anchor_weight", 0.15)),
        key="base.release_anchor_weight",
    )
    steo_weight = _validate_steo_weight(
        float(base.get("steo_weight", 0.0)),
        key="base.steo_weight",
    )
    prototype_weight = _validate_prototype_weight(
        float(base.get("prototype_weight", 0.0)),
        key="base.prototype_weight",
    )
    month_length_bias_weight = _validate_month_length_bias_weight(
        float(base.get("month_length_bias_weight", 0.0)),
        key="base.month_length_bias_weight",
    )

    overrides = fusion_policy.get("regime_overrides", {})
    if isinstance(overrides, Mapping) and regime_label in overrides:
        regime_cfg = overrides[regime_label]
        if isinstance(regime_cfg, Mapping):
            override_horizon = _parse_horizon_weights(
                regime_cfg.get("horizon_weights"),
                key=f"regime_overrides.{regime_label}.horizon_weights",
            )
            if override_horizon:
                base_horizon = {**base_horizon, **override_horizon}
            anchor = _validate_anchor_weight(
                float(regime_cfg.get("release_anchor_weight", anchor)),
                key=f"regime_overrides.{regime_label}.release_anchor_weight",
            )
            steo_weight = _validate_steo_weight(
                float(regime_cfg.get("steo_weight", steo_weight)),
                key=f"regime_overrides.{regime_label}.steo_weight",
            )
            prototype_weight = _validate_prototype_weight(
                float(regime_cfg.get("prototype_weight", prototype_weight)),
                key=f"regime_overrides.{regime_label}.prototype_weight",
            )
            month_length_bias_weight = _validate_month_length_bias_weight(
                float(
                    regime_cfg.get(
                        "month_length_bias_weight",
                        month_length_bias_weight,
                    )
                ),
                key=f"regime_overrides.{regime_label}.month_length_bias_weight",
            )
    if base.get("champion_weight") is not None:
        champion_weight = _validate_champion_weight(
            float(base.get("champion_weight", 0.70)),
            key="base.champion_weight",
        )
        if champion_weight + steo_weight + prototype_weight > 1.0:
            raise ContractViolation(
                "invalid_model_policy",
                key="base.prototype_weight",
                detail="champion_weight + steo_weight + prototype_weight must be <= 1",
            )
    return (
        base_horizon,
        anchor,
        steo_weight,
        prototype_weight,
        month_length_bias_weight,
    )


def resolve_calendar_calibration_for_regime(
    fusion_policy: Mapping[str, Any],
    *,
    regime_label: str,
) -> dict[str, Any]:
    """Resolve calendar calibration payload for the supplied regime."""

    base_cfg = validate_calendar_calibration_config(
        fusion_policy.get("calendar_calibration", {})
    )
    overrides = fusion_policy.get("regime_overrides", {})
    if isinstance(overrides, Mapping):
        regime_cfg = overrides.get(regime_label)
        if isinstance(regime_cfg, Mapping) and isinstance(
            regime_cfg.get("calendar_calibration"), Mapping
        ):
            merged = _merge_dict(base_cfg, regime_cfg["calendar_calibration"])
            return validate_calendar_calibration_config(merged)
    return base_cfg


def resolve_fusion_weights_for_regime(
    fusion_policy: Mapping[str, Any],
    *,
    regime_label: str,
) -> tuple[dict[int, float], float]:
    """Resolve horizon and anchor weights for a given regime label."""

    horizon, anchor, _, _, _ = resolve_fusion_weights_for_regime_full(
        fusion_policy,
        regime_label=regime_label,
    )
    return horizon, anchor


def _ensure_interval_order(frame: pd.DataFrame) -> None:
    invalid = frame[
        (frame["fused_lower_95"] > frame["fused_upper_95"])
        | (frame["fused_lower_95"] > frame["fused_point"])
        | (frame["fused_point"] > frame["fused_upper_95"])
    ]
    if invalid.empty:
        return

    first = invalid.iloc[0]
    raise ContractViolation(
        "interval_order_violation",
        key=f"horizon={int(first['horizon'])}",
        detail=(
            f"lower={float(first['fused_lower_95'])}, "
            f"point={float(first['fused_point'])}, "
            f"upper={float(first['fused_upper_95'])}"
        ),
    )


def _build_release_anchor(
    release_history: pd.DataFrame,
    *,
    horizons: pd.Series,
) -> pd.DataFrame:
    require_columns(
        release_history, ("timestamp", "target_value"), key="release_history"
    )

    prepared = release_history[["timestamp", "target_value"]].copy()
    prepared["timestamp"] = pd.to_datetime(prepared["timestamp"], errors="coerce")
    prepared["target_value"] = pd.to_numeric(prepared["target_value"], errors="coerce")
    prepared = prepared[
        prepared["timestamp"].notna() & prepared["target_value"].notna()
    ].copy()
    prepared = prepared.sort_values("timestamp").drop_duplicates(
        "timestamp", keep="last"
    )
    prepared = prepared.reset_index(drop=True)
    if prepared.empty:
        return pd.DataFrame(
            columns=["horizon", "release_anchor_point", "target_month_days"]
        )

    prepared["days_in_month"] = prepared["timestamp"].dt.days_in_month.astype(int)
    prepared["value_per_day"] = prepared["target_value"].astype(float) / prepared[
        "days_in_month"
    ].astype(float)

    if len(prepared) < 2:
        daily_delta = 0.0
    else:
        daily_delta = float(prepared["value_per_day"].diff().dropna().mean())

    last_row = prepared.iloc[-1]
    last_per_day_value = float(last_row["value_per_day"])
    last_release_month = pd.Timestamp(last_row["timestamp"]).to_period("M")
    anchor = pd.DataFrame({"horizon": horizons.astype(int)})
    anchor["anchor_month_end"] = anchor["horizon"].map(
        lambda h: (last_release_month + int(h)).to_timestamp("M")
    )
    anchor["target_month_days"] = pd.to_datetime(
        anchor["anchor_month_end"], errors="coerce"
    ).dt.days_in_month.astype("Int64")
    anchor["release_anchor_point"] = anchor.apply(
        lambda row: daily_average_to_monthly_total(
            last_per_day_value + daily_delta * float(row["horizon"]),
            month_end=pd.Timestamp(row["anchor_month_end"]),
        ),
        axis=1,
    )
    anchor["is_leap_february"] = pd.to_datetime(
        anchor["anchor_month_end"], errors="coerce"
    ).map(lambda value: bool(value.month == 2 and value.day == 29))
    return anchor[
        [
            "horizon",
            "anchor_month_end",
            "target_month_days",
            "release_anchor_point",
            "is_leap_february",
        ]
    ]


def fuse_forecasts(
    champion_forecast: pd.DataFrame,
    challenger_forecast: pd.DataFrame,
    *,
    champion_weight: float = 0.7,
    release_history: pd.DataFrame | None = None,
    release_anchor_weight: float = 0.15,
    steo_forecast: pd.DataFrame | None = None,
    steo_weight: float = 0.0,
    prototype_forecast: pd.DataFrame | None = None,
    prototype_weight: float = 0.0,
    month_length_bias_weight: float = 0.0,
    calendar_calibration: Mapping[str, Any] | None = None,
    horizon_weights: Mapping[str, float] | Mapping[int, float] | None = None,
    regime_label: str = "normal",
) -> pd.DataFrame:
    """Fuse champion point forecasts and challenger intervals by horizon."""

    resolved_champion_weight = _validate_champion_weight(
        champion_weight,
        key="champion_weight",
    )
    resolved_anchor_weight = _validate_anchor_weight(
        release_anchor_weight,
        key="release_anchor_weight",
    )
    resolved_steo_weight = _validate_steo_weight(
        steo_weight,
        key="steo_weight",
    )
    resolved_prototype_weight = _validate_prototype_weight(
        prototype_weight,
        key="prototype_weight",
    )
    resolved_month_length_bias_weight = _validate_month_length_bias_weight(
        month_length_bias_weight,
        key="month_length_bias_weight",
    )
    if (
        resolved_champion_weight + resolved_steo_weight + resolved_prototype_weight
        > 1.0
    ):
        raise ContractViolation(
            "invalid_model_policy",
            key="prototype_weight",
            detail="champion_weight + steo_weight + prototype_weight must be <= 1",
        )
    parsed_horizon_weights = _parse_horizon_weights(
        horizon_weights,
        key="horizon_weights",
    )

    require_columns(
        champion_forecast,
        ("horizon", "point_forecast"),
        key="champion_forecast",
    )
    require_columns(
        challenger_forecast,
        ("horizon", "mean_forecast", "lower_95", "upper_95"),
        key="challenger_forecast",
    )

    merged = champion_forecast[["horizon", "point_forecast"]].merge(
        challenger_forecast[["horizon", "mean_forecast", "lower_95", "upper_95"]],
        on="horizon",
        how="inner",
    )
    if merged.empty:
        raise ContractViolation(
            "missing_fusion_overlap",
            key="horizon",
            detail="champion and challenger forecasts must share at least one horizon",
        )

    merged = merged.sort_values("horizon").reset_index(drop=True)
    merged["abs_divergence"] = (
        merged["point_forecast"] - merged["mean_forecast"]
    ).abs()
    merged["applied_champion_weight"] = merged["horizon"].map(
        lambda horizon: parsed_horizon_weights.get(
            int(horizon),
            resolved_champion_weight,
        )
    )
    merged["fused_point"] = (
        merged["applied_champion_weight"] * merged["point_forecast"]
        + (1 - merged["applied_champion_weight"]) * merged["mean_forecast"]
    )
    merged["applied_release_anchor_weight"] = resolved_anchor_weight
    merged["applied_steo_weight"] = 0.0
    merged["applied_prototype_weight"] = 0.0
    merged["applied_month_length_bias_weight"] = 0.0
    merged["month_length_bias_applied"] = False
    merged["regime_label"] = str(regime_label)
    merged["fused_point_pre_calendar_calibration"] = pd.NA
    merged["calendar_calibration_delta"] = 0.0
    merged["calendar_calibration_applied"] = False
    merged["is_leap_february"] = False

    if steo_forecast is not None and not steo_forecast.empty:
        require_columns(
            steo_forecast, ("horizon", "steo_point_forecast"), key="steo_forecast"
        )
        steo_payload = steo_forecast.copy()
        for column in ("steo_point_forecast", "steo_lower_95", "steo_upper_95"):
            if column in steo_payload.columns:
                steo_payload[column] = pd.to_numeric(
                    steo_payload[column], errors="coerce"
                )
        merged = merged.merge(
            (
                steo_payload[
                    ["horizon", "steo_point_forecast", "steo_lower_95", "steo_upper_95"]
                ]
                if {"steo_lower_95", "steo_upper_95"}.issubset(steo_payload.columns)
                else steo_payload[["horizon", "steo_point_forecast"]]
            ),
            on="horizon",
            how="left",
        )
        for column in ("steo_lower_95", "steo_upper_95"):
            if column not in merged.columns:
                merged[column] = float("nan")
        has_steo = merged["steo_point_forecast"].notna()
        merged["steo_applied"] = has_steo
        merged.loc[has_steo, "fused_point"] = (1 - resolved_steo_weight) * merged.loc[
            has_steo, "fused_point"
        ] + resolved_steo_weight * merged.loc[has_steo, "steo_point_forecast"]
        merged.loc[has_steo, "applied_steo_weight"] = resolved_steo_weight
    else:
        merged["steo_point_forecast"] = pd.NA
        merged["steo_lower_95"] = pd.NA
        merged["steo_upper_95"] = pd.NA
        merged["steo_applied"] = False

    if prototype_forecast is not None and not prototype_forecast.empty:
        require_columns(
            prototype_forecast,
            ("horizon", "prototype_point_forecast"),
            key="prototype_forecast",
        )
        prototype_payload = prototype_forecast.copy()
        for column in (
            "prototype_point_forecast",
            "prototype_lower_95",
            "prototype_upper_95",
        ):
            if column in prototype_payload.columns:
                prototype_payload[column] = pd.to_numeric(
                    prototype_payload[column],
                    errors="coerce",
                )
        merged = merged.merge(
            (
                prototype_payload[
                    [
                        "horizon",
                        "prototype_point_forecast",
                        "prototype_lower_95",
                        "prototype_upper_95",
                    ]
                ]
                if {"prototype_lower_95", "prototype_upper_95"}.issubset(
                    prototype_payload.columns
                )
                else prototype_payload[["horizon", "prototype_point_forecast"]]
            ),
            on="horizon",
            how="left",
        )
        for column in ("prototype_lower_95", "prototype_upper_95"):
            if column not in merged.columns:
                merged[column] = float("nan")
        has_prototype = merged["prototype_point_forecast"].notna()
        merged["prototype_applied"] = has_prototype
        merged.loc[has_prototype, "fused_point"] = (
            1 - resolved_prototype_weight
        ) * merged.loc[
            has_prototype, "fused_point"
        ] + resolved_prototype_weight * merged.loc[
            has_prototype, "prototype_point_forecast"
        ]
        merged.loc[has_prototype, "applied_prototype_weight"] = (
            resolved_prototype_weight
        )
    else:
        merged["prototype_point_forecast"] = pd.NA
        merged["prototype_lower_95"] = pd.NA
        merged["prototype_upper_95"] = pd.NA
        merged["prototype_applied"] = False

    if release_history is not None and not release_history.empty:
        anchor = _build_release_anchor(
            release_history,
            horizons=merged["horizon"],
        )
        merged = merged.merge(anchor, on="horizon", how="left")
        merged["release_anchor_applied"] = merged["release_anchor_point"].notna()
        merged.loc[merged["release_anchor_applied"], "fused_point"] = (
            1 - resolved_anchor_weight
        ) * merged.loc[
            merged["release_anchor_applied"], "fused_point"
        ] + resolved_anchor_weight * merged.loc[
            merged["release_anchor_applied"], "release_anchor_point"
        ]
        if resolved_month_length_bias_weight > 0:
            day_31_rows = merged["release_anchor_applied"].astype(bool) & (
                merged["target_month_days"] == 31
            )
            merged.loc[day_31_rows, "fused_point"] = (
                1 - resolved_month_length_bias_weight
            ) * merged.loc[
                day_31_rows, "fused_point"
            ] + resolved_month_length_bias_weight * merged.loc[
                day_31_rows, "release_anchor_point"
            ]
            merged.loc[day_31_rows, "applied_month_length_bias_weight"] = (
                resolved_month_length_bias_weight
            )
            merged.loc[day_31_rows, "month_length_bias_applied"] = True
    else:
        merged["release_anchor_point"] = pd.NA
        merged["release_anchor_applied"] = False
        merged["anchor_month_end"] = pd.NaT
        merged["target_month_days"] = pd.NA

    calendar_cfg = validate_calendar_calibration_config(calendar_calibration or {})
    if (
        bool(calendar_cfg["enabled"])
        and "target_month_days" in merged.columns
        and merged["target_month_days"].notna().all()
    ):
        calibrated, _ = apply_calendar_calibration(
            merged,
            calibration_config=calendar_cfg,
            regime_label=str(regime_label),
        )
        merged = calibrated
    else:
        merged["fused_point_pre_calendar_calibration"] = merged["fused_point"]
        merged["calendar_calibration_delta"] = 0.0
        merged["calendar_calibration_applied"] = False
        if "is_leap_february" not in merged.columns:
            merged["is_leap_february"] = False

    merged["fused_lower_95"] = merged[["lower_95"]].min(axis=1)
    merged["fused_upper_95"] = merged[["upper_95"]].max(axis=1)

    widened = merged["abs_divergence"] * 0.5
    merged["fused_lower_95"] = merged["fused_lower_95"] - widened
    merged["fused_upper_95"] = merged["fused_upper_95"] + widened
    steo_rows = merged["steo_applied"].astype(bool)
    if steo_rows.any():
        if "steo_lower_95" in merged.columns:
            merged.loc[steo_rows, "fused_lower_95"] = merged.loc[
                steo_rows, ["fused_lower_95", "steo_lower_95"]
            ].min(axis=1)
        if "steo_upper_95" in merged.columns:
            merged.loc[steo_rows, "fused_upper_95"] = merged.loc[
                steo_rows, ["fused_upper_95", "steo_upper_95"]
            ].max(axis=1)
    prototype_rows = merged["prototype_applied"].astype(bool)
    if prototype_rows.any():
        if "prototype_lower_95" in merged.columns:
            merged.loc[prototype_rows, "fused_lower_95"] = merged.loc[
                prototype_rows, ["fused_lower_95", "prototype_lower_95"]
            ].min(axis=1)
        if "prototype_upper_95" in merged.columns:
            merged.loc[prototype_rows, "fused_upper_95"] = merged.loc[
                prototype_rows, ["fused_upper_95", "prototype_upper_95"]
            ].max(axis=1)
    anchored_rows = merged["release_anchor_applied"].astype(bool)
    if anchored_rows.any():
        merged.loc[anchored_rows, "fused_lower_95"] = merged.loc[
            anchored_rows, ["fused_lower_95", "fused_point"]
        ].min(axis=1)
        merged.loc[anchored_rows, "fused_upper_95"] = merged.loc[
            anchored_rows, ["fused_upper_95", "fused_point"]
        ].max(axis=1)

    ordered = merged[
        [
            "horizon",
            "point_forecast",
            "mean_forecast",
            "fused_point",
            "fused_lower_95",
            "fused_upper_95",
            "abs_divergence",
            "applied_champion_weight",
            "applied_release_anchor_weight",
            "applied_steo_weight",
            "applied_prototype_weight",
            "applied_month_length_bias_weight",
            "regime_label",
            "target_month_days",
            "is_leap_february",
            "release_anchor_point",
            "anchor_month_end",
            "release_anchor_applied",
            "month_length_bias_applied",
            "fused_point_pre_calendar_calibration",
            "calendar_calibration_delta",
            "calendar_calibration_applied",
            "steo_point_forecast",
            "steo_lower_95",
            "steo_upper_95",
            "steo_applied",
            "prototype_point_forecast",
            "prototype_lower_95",
            "prototype_upper_95",
            "prototype_applied",
        ]
    ]
    _ensure_interval_order(ordered)
    return ordered


def summarize_divergence(fused_forecast: pd.DataFrame) -> dict[str, float]:
    """Compute divergence diagnostics for champion/challenger fusion."""

    require_columns(fused_forecast, ("abs_divergence",), key="fused_forecast")
    return {
        "mean_abs_divergence": float(fused_forecast["abs_divergence"].mean()),
        "max_abs_divergence": float(fused_forecast["abs_divergence"].max()),
        "min_abs_divergence": float(fused_forecast["abs_divergence"].min()),
    }


def summarize_seed_stability(seed_forecasts: pd.DataFrame) -> pd.DataFrame:
    """Summarize repeated-seed stability by horizon."""

    require_columns(
        seed_forecasts, ("seed", "horizon", "point_forecast"), key="seed_forecasts"
    )
    summary = (
        seed_forecasts.groupby("horizon", sort=True)["point_forecast"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )
    summary["std"] = summary["std"].fillna(0.0)
    summary["spread"] = summary["max"] - summary["min"]
    return summary


def summarize_calibration(
    fused_forecast: pd.DataFrame,
    realized: pd.DataFrame,
    *,
    realized_col: str = "realized_value",
) -> dict[str, float]:
    """Compute interval coverage and error diagnostics by horizon."""

    require_columns(
        fused_forecast,
        ("horizon", "fused_point", "fused_lower_95", "fused_upper_95"),
        key="fused_forecast",
    )
    require_columns(realized, ("horizon", realized_col), key="realized")

    merged = fused_forecast.merge(
        realized[["horizon", realized_col]], on="horizon", how="inner"
    )
    if merged.empty:
        raise ContractViolation(
            "missing_calibration_overlap",
            key="horizon",
            detail="realized observations must overlap fusion horizons",
        )

    within = (merged[realized_col] >= merged["fused_lower_95"]) & (
        merged[realized_col] <= merged["fused_upper_95"]
    )
    abs_error = (merged[realized_col] - merged["fused_point"]).abs()
    interval_width = merged["fused_upper_95"] - merged["fused_lower_95"]

    return {
        "coverage_rate": float(within.mean()),
        "mean_abs_error": float(abs_error.mean()),
        "mean_interval_width": float(interval_width.mean()),
    }


def build_fusion_result(
    champion_forecast: pd.DataFrame,
    challenger_forecast: pd.DataFrame,
    seed_forecasts: pd.DataFrame,
    realized: pd.DataFrame,
    *,
    champion_weight: float = 0.7,
    realized_col: str = "realized_value",
    release_history: pd.DataFrame | None = None,
    release_anchor_weight: float = 0.15,
    steo_forecast: pd.DataFrame | None = None,
    steo_weight: float = 0.0,
    prototype_forecast: pd.DataFrame | None = None,
    prototype_weight: float = 0.0,
    month_length_bias_weight: float = 0.0,
    calendar_calibration: Mapping[str, Any] | None = None,
    horizon_weights: Mapping[str, float] | Mapping[int, float] | None = None,
    regime_label: str = "normal",
) -> FusionResult:
    """Build complete fusion output bundle with diagnostics."""

    fused = fuse_forecasts(
        champion_forecast,
        challenger_forecast,
        champion_weight=champion_weight,
        release_history=release_history,
        release_anchor_weight=release_anchor_weight,
        steo_forecast=steo_forecast,
        steo_weight=steo_weight,
        prototype_forecast=prototype_forecast,
        prototype_weight=prototype_weight,
        month_length_bias_weight=month_length_bias_weight,
        calendar_calibration=calendar_calibration,
        horizon_weights=horizon_weights,
        regime_label=regime_label,
    )
    divergence = summarize_divergence(fused)
    stability = summarize_seed_stability(seed_forecasts)
    calibration = summarize_calibration(fused, realized, realized_col=realized_col)

    return FusionResult(
        forecast=fused,
        divergence_summary=divergence,
        stability_summary=stability,
        calibration_summary=calibration,
    )
