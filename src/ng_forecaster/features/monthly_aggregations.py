"""Lag-safe monthly feature aggregation from daily and weekly signals."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.lag_guard import require_columns


def _linear_slope(values: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = values.to_numpy(dtype=float)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def _normalize_frame(frame: pd.DataFrame, *, label: str) -> pd.DataFrame:
    require_columns(frame, ("timestamp", "value"), key=label)
    normalized = frame.copy()
    normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], errors="coerce")
    if normalized["timestamp"].isna().any():
        raise ContractViolation(
            "invalid_timestamp",
            key=label,
            detail="aggregation input contains invalid timestamps",
        )
    normalized["value"] = pd.to_numeric(normalized["value"], errors="coerce")
    return normalized.sort_values("timestamp").reset_index(drop=True)


def build_monthly_features(
    daily_price: pd.DataFrame,
    weekly_storage: pd.DataFrame,
    *,
    asof: object,
) -> pd.DataFrame:
    """Build deterministic monthly features using only records available at ``asof``."""

    asof_ts = pd.Timestamp(asof)
    if pd.isna(asof_ts):
        raise ContractViolation(
            "invalid_timestamp", key="asof", detail="asof cannot be NaT"
        )

    price = _normalize_frame(daily_price, label="daily_price")
    storage = _normalize_frame(weekly_storage, label="weekly_storage")

    price = price[price["timestamp"] <= asof_ts].copy()
    storage = storage[storage["timestamp"] <= asof_ts].copy()
    if price.empty:
        raise ContractViolation(
            "missing_feature_input",
            asof=asof_ts.to_pydatetime(),
            key="daily_price",
            detail="no daily price observations are available at or before asof",
        )
    if storage.empty:
        raise ContractViolation(
            "missing_feature_input",
            asof=asof_ts.to_pydatetime(),
            key="weekly_storage",
            detail="no weekly storage observations are available at or before asof",
        )

    period = asof_ts.to_period("M")
    price_mtd = price[price["timestamp"].dt.to_period("M") == period]
    if price_mtd.empty:
        raise ContractViolation(
            "missing_feature_input",
            asof=asof_ts.to_pydatetime(),
            key="daily_price",
            detail="no in-month daily records available for asof",
        )

    price_tail = price.tail(7)
    price_last = float(price.iloc[-1]["value"])
    price_prev = float(price.iloc[-2]["value"]) if len(price) > 1 else price_last

    storage_tail = storage.tail(4)
    storage_last = float(storage.iloc[-1]["value"])
    previous_month = period - 1
    previous_month_storage = storage[
        storage["timestamp"].dt.to_period("M") == previous_month
    ]
    prev_month_last = (
        float(previous_month_storage.iloc[-1]["value"])
        if not previous_month_storage.empty
        else storage_last
    )

    rows: list[dict[str, Any]] = [
        {
            "feature_name": "hh_mtd_mean",
            "feature_timestamp": price_mtd.iloc[-1]["timestamp"],
            "available_timestamp": price_mtd.iloc[-1]["timestamp"],
            "block_id": "market_core",
            "value": float(price_mtd["value"].mean()),
            "source_frequency": "daily",
        },
        {
            "feature_name": "hh_last",
            "feature_timestamp": price.iloc[-1]["timestamp"],
            "available_timestamp": price.iloc[-1]["timestamp"],
            "block_id": "market_core",
            "value": price_last,
            "source_frequency": "daily",
        },
        {
            "feature_name": "hh_vol_7d",
            "feature_timestamp": price_tail.iloc[-1]["timestamp"],
            "available_timestamp": price_tail.iloc[-1]["timestamp"],
            "block_id": "market_core",
            "value": float(price_tail["value"].std(ddof=0)),
            "source_frequency": "daily",
        },
        {
            "feature_name": "hh_diff_1d",
            "feature_timestamp": price.iloc[-1]["timestamp"],
            "available_timestamp": price.iloc[-1]["timestamp"],
            "block_id": "market_core",
            "value": float(price_last - price_prev),
            "source_frequency": "daily",
        },
        {
            "feature_name": "stor_last",
            "feature_timestamp": storage.iloc[-1]["timestamp"],
            "available_timestamp": storage.iloc[-1]["timestamp"],
            "block_id": "market_core",
            "value": storage_last,
            "source_frequency": "weekly",
        },
        {
            "feature_name": "stor_mean_4w",
            "feature_timestamp": storage_tail.iloc[-1]["timestamp"],
            "available_timestamp": storage_tail.iloc[-1]["timestamp"],
            "block_id": "market_core",
            "value": float(storage_tail["value"].mean()),
            "source_frequency": "weekly",
        },
        {
            "feature_name": "stor_slope_4w",
            "feature_timestamp": storage_tail.iloc[-1]["timestamp"],
            "available_timestamp": storage_tail.iloc[-1]["timestamp"],
            "block_id": "market_core",
            "value": _linear_slope(storage_tail["value"]),
            "source_frequency": "weekly",
        },
        {
            "feature_name": "stor_mom_change",
            "feature_timestamp": storage.iloc[-1]["timestamp"],
            "available_timestamp": storage.iloc[-1]["timestamp"],
            "block_id": "market_core",
            "value": float(storage_last - prev_month_last),
            "source_frequency": "weekly",
        },
    ]

    features = pd.DataFrame(rows).sort_values("feature_name").reset_index(drop=True)
    return features


def build_weighted_freezeoff_features(
    weather_panel: pd.DataFrame,
    *,
    asof: object,
) -> pd.DataFrame:
    """Aggregate weather freeze-off monthly panel into weighted CP3 runtime features."""

    asof_ts = pd.Timestamp(asof)
    if pd.isna(asof_ts):
        raise ContractViolation(
            "invalid_timestamp",
            key="asof",
            detail="asof cannot be NaT",
        )
    required = {
        "timestamp",
        "available_timestamp",
        "freeze_days",
        "freeze_event_share",
        "freeze_intensity_c",
        "coverage_fraction",
    }
    missing = sorted(required - set(weather_panel.columns))
    if missing:
        raise ContractViolation(
            "missing_column",
            key="weather_panel",
            detail="missing required weather panel columns: " + ", ".join(missing),
        )

    frame = weather_panel.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["available_timestamp"] = pd.to_datetime(
        frame["available_timestamp"], errors="coerce"
    )
    frame = frame[
        (frame["timestamp"] == asof_ts.to_period("M").to_timestamp("M"))
        & (frame["available_timestamp"] <= asof_ts)
    ].copy()
    if frame.empty:
        raise ContractViolation(
            "missing_feature_input",
            key="weather_freezeoff_panel",
            detail="weather panel has no eligible current-month rows at asof",
        )

    weights = pd.to_numeric(frame["coverage_fraction"], errors="coerce").fillna(0.0)
    if float(weights.sum()) <= 0:
        raise ContractViolation(
            "insufficient_source_coverage",
            key="weather_freezeoff_panel",
            detail="weather panel has non-positive coverage weights",
        )
    freeze_days = float(
        (
            pd.to_numeric(frame["freeze_days"], errors="coerce").fillna(0.0) * weights
        ).sum()
        / weights.sum()
    )
    freeze_share = float(
        (
            pd.to_numeric(frame["freeze_event_share"], errors="coerce").fillna(0.0)
            * weights
        ).sum()
        / weights.sum()
    )
    freeze_intensity = float(
        (
            pd.to_numeric(frame["freeze_intensity_c"], errors="coerce").fillna(0.0)
            * weights
        ).sum()
        / weights.sum()
    )
    coverage_fraction = float(
        (
            pd.to_numeric(frame["coverage_fraction"], errors="coerce").fillna(0.0)
            * weights
        ).sum()
        / weights.sum()
    )
    if "extreme_min_c" in frame.columns:
        extreme_min = pd.to_numeric(frame["extreme_min_c"], errors="coerce").dropna()
        extreme_min_mtd = float(extreme_min.min()) if not extreme_min.empty else np.nan
    else:
        extreme_min_mtd = np.nan
    freeze_event_flag = float(freeze_days > 0.0 or freeze_share > 0.0)
    available_ts = pd.Timestamp(frame["available_timestamp"].max())
    feature_ts = asof_ts.normalize()
    rows = [
        {
            "feature_name": "freeze_days_mtd_weighted",
            "feature_timestamp": feature_ts,
            "available_timestamp": available_ts,
            "block_id": "weather_freezeoff",
            "value": freeze_days,
            "source_frequency": "daily",
        },
        {
            "feature_name": "freeze_event_share_mtd_weighted",
            "feature_timestamp": feature_ts,
            "available_timestamp": available_ts,
            "block_id": "weather_freezeoff",
            "value": freeze_share,
            "source_frequency": "daily",
        },
        {
            "feature_name": "freeze_intensity_mtd_weighted",
            "feature_timestamp": feature_ts,
            "available_timestamp": available_ts,
            "block_id": "weather_freezeoff",
            "value": freeze_intensity,
            "source_frequency": "daily",
        },
        {
            "feature_name": "freeze_event_flag",
            "feature_timestamp": feature_ts,
            "available_timestamp": available_ts,
            "block_id": "weather_freezeoff",
            "value": freeze_event_flag,
            "source_frequency": "daily",
        },
        {
            "feature_name": "freeze_event_intensity",
            "feature_timestamp": feature_ts,
            "available_timestamp": available_ts,
            "block_id": "weather_freezeoff",
            "value": freeze_intensity,
            "source_frequency": "daily",
        },
        {
            "feature_name": "freeze_days_mtd",
            "feature_timestamp": feature_ts,
            "available_timestamp": available_ts,
            "block_id": "weather_freezeoff",
            "value": freeze_days,
            "source_frequency": "daily",
        },
        {
            "feature_name": "extreme_min_mtd",
            "feature_timestamp": feature_ts,
            "available_timestamp": available_ts,
            "block_id": "weather_freezeoff",
            "value": extreme_min_mtd,
            "source_frequency": "daily",
        },
        {
            "feature_name": "coverage_fraction_mtd",
            "feature_timestamp": feature_ts,
            "available_timestamp": available_ts,
            "block_id": "weather_freezeoff",
            "value": coverage_fraction,
            "source_frequency": "daily",
        },
    ]
    return pd.DataFrame(rows).sort_values("feature_name").reset_index(drop=True)
