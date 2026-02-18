"""Prototype cohort-kernel forecasting stream for drilling/completion dynamics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ng_forecaster.data.target_transforms import daily_average_to_monthly_total
from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.lag_guard import require_columns
from ng_forecaster.models.prototypes.fit_kernel import fit_kernel_parameters
from ng_forecaster.models.prototypes.kernels import (
    bounded_value,
    cumulative_kernel_weight,
)

_DRILLING_SIGNAL_COLUMNS = (
    "new_wells_completed",
    "new_well_gas_production",
    "existing_gas_production_change",
    "duc_inventory",
    "completion_to_duc_ratio",
)


@dataclass(frozen=True)
class PrototypeKernelForecastResult:
    """Forecast payload and diagnostics for prototype kernel stream."""

    forecast: pd.DataFrame
    diagnostics: dict[str, Any]


def _prepare_release_history(release_history: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        release_history, ("timestamp", "target_value"), key="release_history"
    )
    frame = release_history[["timestamp", "target_value"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["target_value"] = pd.to_numeric(frame["target_value"], errors="coerce")
    frame = frame[frame["timestamp"].notna() & frame["target_value"].notna()].copy()
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    if frame.empty:
        raise ContractViolation(
            "missing_target_history_rows",
            key="release_history",
            detail="release history has no valid rows",
        )
    frame["days_in_month"] = frame["timestamp"].dt.days_in_month.astype(float)
    frame["daily_target"] = frame["target_value"] / frame["days_in_month"]
    return frame.reset_index(drop=True)


def _prepare_drilling_history(drilling_history: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        drilling_history,
        ("timestamp", "available_timestamp", *_DRILLING_SIGNAL_COLUMNS),
        key="drilling_history",
    )
    frame = drilling_history[
        ["timestamp", "available_timestamp", *_DRILLING_SIGNAL_COLUMNS]
    ].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["available_timestamp"] = pd.to_datetime(
        frame["available_timestamp"],
        errors="coerce",
    )
    for column in _DRILLING_SIGNAL_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        frame["timestamp"].notna() & frame["available_timestamp"].notna()
    ].copy()
    frame = frame.sort_values(["timestamp", "available_timestamp"])
    frame = frame.groupby("timestamp", as_index=False).tail(1)
    frame = frame.dropna(subset=list(_DRILLING_SIGNAL_COLUMNS))
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    if frame.empty:
        raise ContractViolation(
            "missing_feature_input",
            key="drilling_history",
            detail="drilling history has no valid rows after normalization",
        )
    return frame


def _resolve_latest_signal_row(
    drilling_history: pd.DataFrame,
    *,
    target_month: pd.Timestamp,
) -> pd.Series:
    latest_allowed = drilling_history[drilling_history["timestamp"] <= target_month]
    if latest_allowed.empty:
        latest_allowed = drilling_history
    return latest_allowed.sort_values("timestamp").iloc[-1]


def _compute_signals(row: Mapping[str, Any]) -> dict[str, float]:
    completed = float(
        pd.to_numeric(
            pd.Series([row.get("new_wells_completed")]), errors="coerce"
        ).iloc[0]
    )
    new_well = float(
        pd.to_numeric(
            pd.Series([row.get("new_well_gas_production")]), errors="coerce"
        ).iloc[0]
    )
    legacy = float(
        pd.to_numeric(
            pd.Series([row.get("existing_gas_production_change")]),
            errors="coerce",
        ).iloc[0]
    )
    duc_inventory = float(
        pd.to_numeric(pd.Series([row.get("duc_inventory")]), errors="coerce").iloc[0]
    )
    completion_ratio = float(
        pd.to_numeric(
            pd.Series([row.get("completion_to_duc_ratio")]), errors="coerce"
        ).iloc[0]
    )
    completion_signal = completed * new_well / max(duc_inventory, 1.0)
    return {
        "completion_signal": float(completion_signal),
        "legacy_signal": float(legacy),
        "duc_signal": float(completion_ratio),
    }


def _zscore_signal(
    signal_name: str,
    value: float,
    *,
    params: Mapping[str, Any],
) -> float:
    signal_mean = dict(params.get("signal_mean", {})).get(signal_name, 0.0)
    signal_std = dict(params.get("signal_std", {})).get(signal_name, 1.0)
    std_value = float(signal_std)
    if not np.isfinite(std_value) or std_value < 1e-9:
        std_value = 1.0
    return float((float(value) - float(signal_mean)) / std_value)


def _interval_half_width_daily(
    release_history: pd.DataFrame,
    *,
    horizon: int,
) -> float:
    daily_delta = (
        pd.to_numeric(release_history["daily_target"], errors="coerce")
        .diff()
        .abs()
        .dropna()
    )
    horizon_scale = float(np.sqrt(float(max(1, horizon))))
    if daily_delta.empty:
        return float(max(20.0, 10.0 * horizon_scale))
    base = float(daily_delta.tail(12).mean())
    return float(max(base * horizon_scale, 20.0))


def build_cohort_kernel_forecast(
    *,
    drilling_history: pd.DataFrame,
    target_month: object,
    release_history: pd.DataFrame,
    horizons: Sequence[int] = (1, 2),
    kernel_params: Mapping[str, Any] | None = None,
) -> PrototypeKernelForecastResult:
    """Build prototype forecast stream from drilling/completion dynamics."""

    release_prepared = _prepare_release_history(release_history)
    drilling_prepared = _prepare_drilling_history(drilling_history)
    params: Mapping[str, Any]
    if kernel_params is None:
        params = fit_kernel_parameters(
            release_history=release_prepared,
            drilling_history=drilling_prepared,
        )
    else:
        params = dict(kernel_params)

    parsed_horizons = sorted({int(value) for value in horizons})
    if not parsed_horizons or parsed_horizons[0] < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="horizons",
            detail="horizons must include integers >= 1",
        )

    target = pd.Timestamp(target_month).to_period("M").to_timestamp("M")
    latest_release = release_prepared.sort_values("timestamp").iloc[-1]
    base_daily = float(latest_release["daily_target"])
    signal_row = _resolve_latest_signal_row(drilling_prepared, target_month=target)
    signal_values = _compute_signals(signal_row.to_dict())

    z_completion = _zscore_signal(
        "completion_signal",
        signal_values["completion_signal"],
        params=params,
    )
    z_legacy = _zscore_signal(
        "legacy_signal",
        signal_values["legacy_signal"],
        params=params,
    )
    z_duc = _zscore_signal("duc_signal", signal_values["duc_signal"], params=params)

    raw_daily_delta = (
        float(params.get("intercept_daily_delta", 0.0))
        + float(params.get("completion_coef", 0.0)) * z_completion
        + float(params.get("legacy_coef", 0.0)) * z_legacy
        + float(params.get("duc_coef", 0.0)) * z_duc
    )
    max_daily_delta = float(params.get("max_daily_delta", 500.0))
    bounded_daily_delta = bounded_value(
        raw_daily_delta,
        lower=-abs(max_daily_delta),
        upper=abs(max_daily_delta),
        key="daily_delta",
    )
    half_life = bounded_value(
        float(params.get("half_life_months", 2.0)),
        lower=1.0,
        upper=6.0,
        key="half_life_months",
    )

    rows: list[dict[str, Any]] = []
    for horizon in parsed_horizons:
        month_end = (target.to_period("M") + int(horizon) - 1).to_timestamp("M")
        cumulative_impact = cumulative_kernel_weight(
            horizon=int(horizon),
            half_life_months=float(half_life),
        )
        projected_daily = base_daily + bounded_daily_delta * cumulative_impact
        point = daily_average_to_monthly_total(projected_daily, month_end=month_end)
        interval_half_daily = _interval_half_width_daily(
            release_prepared,
            horizon=int(horizon),
        )
        interval_half_monthly = daily_average_to_monthly_total(
            interval_half_daily,
            month_end=month_end,
        )
        rows.append(
            {
                "horizon": int(horizon),
                "prototype_point_forecast": float(point),
                "prototype_lower_95": float(point - 1.96 * interval_half_monthly),
                "prototype_upper_95": float(point + 1.96 * interval_half_monthly),
                "prototype_daily_base": float(base_daily),
                "prototype_daily_delta": float(bounded_daily_delta),
                "prototype_kernel_cumulative_weight": float(cumulative_impact),
            }
        )

    frame = pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)
    diagnostics: dict[str, Any] = {
        "target_month": target.date().isoformat(),
        "n_horizons": int(len(frame)),
        "training_rows": int(params.get("training_rows", 0)),
        "half_life_months": float(half_life),
        "raw_daily_delta": float(raw_daily_delta),
        "bounded_daily_delta": float(bounded_daily_delta),
        "signal_values": signal_values,
        "signal_zscores": {
            "completion_signal": float(z_completion),
            "legacy_signal": float(z_legacy),
            "duc_signal": float(z_duc),
        },
        "kernel_params": dict(params),
    }
    return PrototypeKernelForecastResult(forecast=frame, diagnostics=diagnostics)
