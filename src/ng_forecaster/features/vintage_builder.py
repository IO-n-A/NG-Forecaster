"""Vintage panel construction with as-of contracts, eligibility, and lineage."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Mapping, Sequence

import pandas as pd

from ng_forecaster.data.validators import validate_feature_policy
from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.lag_guard import (
    ensure_available_before_asof,
    ensure_target_lag_eligibility,
    require_columns,
)
from ng_forecaster.features.regime_flags import (
    compute_regime_flags,
    load_regime_thresholds,
)


@dataclass(frozen=True)
class VintageBuildResult:
    """Output for a full as-of build with ``T-1`` and ``T`` slices."""

    asof: pd.Timestamp
    slices: dict[str, pd.DataFrame]
    lineage: dict[str, str]
    target_months: dict[str, pd.Timestamp]


def _normalize_asof(value: object) -> pd.Timestamp:
    asof_ts = pd.Timestamp(value)
    if pd.isna(asof_ts):
        raise ContractViolation(
            "invalid_timestamp", key="asof", detail="asof cannot be NaT"
        )
    return asof_ts


def _resolve_feature_policy(
    feature_policy: Mapping[str, Any] | None,
    feature_names: list[str],
) -> dict[str, Any]:
    if feature_policy is None:
        inferred = {
            "version": 1,
            "default": {"max_age_days": 3650},
            "features": {
                name: {
                    "source_frequency": "daily",
                    "aggregation": "latest",
                    "max_age_days": 3650,
                }
                for name in sorted(set(feature_names))
            },
        }
        return validate_feature_policy(inferred)
    return validate_feature_policy(feature_policy)


def _canonical_lineage_id(
    *,
    asof: pd.Timestamp,
    horizon: str,
    selected_features: pd.DataFrame,
    target_row: pd.Series,
) -> str:
    payload = {
        "asof": asof.isoformat(),
        "horizon": horizon,
        "features": [
            {
                "feature_name": str(row["feature_name"]),
                "feature_timestamp": pd.Timestamp(row["feature_timestamp"]).isoformat(),
                "available_timestamp": pd.Timestamp(
                    row["available_timestamp"]
                ).isoformat(),
                "block_id": str(row["block_id"]),
                "value": float(row["value"]),
            }
            for _, row in selected_features.sort_values(
                ["feature_name", "feature_timestamp", "available_timestamp"]
            ).iterrows()
        ],
        "target_timestamp": pd.Timestamp(target_row["target_timestamp"]).isoformat(),
        "target_value": float(target_row["target_value"]),
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest


def _build_single_horizon(
    features: pd.DataFrame,
    target: pd.DataFrame,
    *,
    horizon: str,
    horizon_asof: pd.Timestamp,
    run_asof: pd.Timestamp,
    feature_policy: Mapping[str, Any],
    min_target_lag_days: int,
    min_target_lag_months: int | None,
    forecast_target_month: pd.Timestamp | None,
    regime_thresholds: Mapping[str, Any],
) -> tuple[pd.DataFrame, str]:
    features_at_asof = features[
        (features["feature_timestamp"] <= horizon_asof)
        & (features["available_timestamp"] <= run_asof)
    ].copy()
    ensure_available_before_asof(features_at_asof, asof=run_asof)
    if features_at_asof.empty:
        raise ContractViolation(
            "feature_missing",
            asof=run_asof.to_pydatetime(),
            key="feature_frame",
            detail="no eligible feature rows at horizon asof",
        )

    selected_rows: list[dict[str, Any]] = []
    row_payload: dict[str, Any] = {
        "asof": horizon_asof,
        "horizon": horizon,
    }
    for feature_name, cfg in feature_policy["features"].items():
        feature_subset = features_at_asof[
            features_at_asof["feature_name"] == feature_name
        ].sort_values("feature_timestamp")
        if feature_subset.empty:
            raise ContractViolation(
                "feature_missing",
                asof=horizon_asof.to_pydatetime(),
                key=feature_name,
                detail="feature has no eligible records at asof",
            )

        latest = feature_subset.iloc[-1]
        latest_ts = pd.Timestamp(latest["feature_timestamp"])
        available_ts = pd.Timestamp(latest["available_timestamp"])
        age_days = int((run_asof - available_ts).days)
        max_age_days = int(cfg["max_age_days"])
        if age_days > max_age_days:
            raise ContractViolation(
                "feature_ineligible",
                asof=run_asof.to_pydatetime(),
                key=feature_name,
                detail=(
                    f"feature age {age_days}d exceeds max_age_days={max_age_days}; "
                    "future fill is not allowed"
                ),
            )

        value = float(latest["value"])
        row_payload[feature_name] = value
        row_payload[f"{feature_name}__ts"] = latest_ts
        row_payload[f"{feature_name}__available_ts"] = available_ts
        row_payload[f"{feature_name}__block_id"] = str(latest.get("block_id", ""))
        selected_rows.append(
            {
                "feature_name": feature_name,
                "feature_timestamp": latest_ts,
                "available_timestamp": available_ts,
                "block_id": str(latest.get("block_id", "unassigned")),
                "value": value,
            }
        )

    if min_target_lag_months is not None:
        if min_target_lag_months < 1:
            raise ContractViolation(
                "invalid_lag_policy",
                asof=horizon_asof.to_pydatetime(),
                key="min_target_lag_months",
                detail="min_target_lag_months must be >= 1 when provided",
            )
        target_cutoff = (
            horizon_asof.to_period("M") - min_target_lag_months
        ).to_timestamp("M")
    else:
        target_cutoff = horizon_asof - timedelta(days=min_target_lag_days)
    target_at_asof = target[target["target_timestamp"] <= target_cutoff].copy()
    if target_at_asof.empty:
        raise ContractViolation(
            "target_missing",
            asof=horizon_asof.to_pydatetime(),
            key="target",
            detail="no target row satisfies lag policy",
        )

    if min_target_lag_months is None:
        ensure_target_lag_eligibility(
            target_at_asof,
            asof=horizon_asof,
            min_lag_days=min_target_lag_days,
        )
    target_latest = target_at_asof.sort_values("target_timestamp").iloc[-1]
    row_payload["target_timestamp"] = pd.Timestamp(target_latest["target_timestamp"])
    row_payload["target_value"] = float(target_latest["target_value"])
    if forecast_target_month is not None:
        row_payload["target_month"] = (
            pd.Timestamp(forecast_target_month).date().isoformat()
        )

    row_payload.update(compute_regime_flags(row_payload, thresholds=regime_thresholds))
    for key, value in sorted(regime_thresholds.items()):
        row_payload[f"regime_threshold__{key}"] = float(value)

    selected_df = pd.DataFrame(selected_rows)
    lineage_id = _canonical_lineage_id(
        asof=horizon_asof,
        horizon=horizon,
        selected_features=selected_df,
        target_row=target_latest,
    )
    row_payload["lineage_id"] = lineage_id

    panel = pd.DataFrame([row_payload])
    return panel, lineage_id


def build_vintage_panel(
    features: pd.DataFrame,
    target: pd.DataFrame,
    *,
    asof: object,
    preprocessing_status: str = "passed",
    min_target_lag_days: int = 1,
    min_target_lag_months: int | None = None,
    feature_policy: Mapping[str, Any] | None = None,
    target_month: object | None = None,
    target_month_sequence: Sequence[object] | None = None,
    regime_thresholds: Mapping[str, Any] | None = None,
    regime_thresholds_path: str = "configs/features.yaml",
) -> VintageBuildResult:
    """Build ``T-1`` and ``T`` vintage slices with strict contracts."""

    if preprocessing_status != "passed":
        raise ContractViolation(
            "preprocess_gate_failed",
            asof=pd.Timestamp(asof).to_pydatetime(),
            key="preprocessing_status",
            detail=f"status={preprocessing_status}",
        )

    require_columns(
        features,
        ("feature_name", "feature_timestamp", "value"),
        key="features",
    )
    require_columns(target, ("target_timestamp", "target_value"), key="target")

    normalized_features = features.copy()
    normalized_features["feature_timestamp"] = pd.to_datetime(
        normalized_features["feature_timestamp"], errors="coerce"
    )
    normalized_features["value"] = pd.to_numeric(
        normalized_features["value"], errors="coerce"
    )
    if "available_timestamp" not in normalized_features.columns:
        normalized_features["available_timestamp"] = normalized_features[
            "feature_timestamp"
        ]
    normalized_features["available_timestamp"] = pd.to_datetime(
        normalized_features["available_timestamp"], errors="coerce"
    )
    if "block_id" not in normalized_features.columns:
        normalized_features["block_id"] = "unassigned"
    normalized_features["block_id"] = (
        normalized_features["block_id"]
        .astype(str)
        .str.strip()
        .replace({"": "unassigned"})
    )
    if normalized_features["feature_timestamp"].isna().any():
        raise ContractViolation(
            "invalid_timestamp",
            key="feature_timestamp",
            detail="features contain invalid timestamps",
        )
    if normalized_features["available_timestamp"].isna().any():
        raise ContractViolation(
            "invalid_timestamp",
            key="available_timestamp",
            detail="features contain invalid availability timestamps",
        )
    normalized_features = normalized_features.sort_values(
        ["feature_name", "feature_timestamp", "available_timestamp"]
    ).reset_index(drop=True)

    normalized_target = target.copy()
    normalized_target["target_timestamp"] = pd.to_datetime(
        normalized_target["target_timestamp"], errors="coerce"
    )
    normalized_target["target_value"] = pd.to_numeric(
        normalized_target["target_value"], errors="coerce"
    )
    if normalized_target["target_timestamp"].isna().any():
        raise ContractViolation(
            "invalid_timestamp",
            key="target_timestamp",
            detail="target contains invalid timestamps",
        )
    normalized_target = normalized_target.sort_values("target_timestamp").reset_index(
        drop=True
    )

    asof_ts = _normalize_asof(asof)
    policy = _resolve_feature_policy(
        feature_policy,
        feature_names=normalized_features["feature_name"].astype(str).tolist(),
    )
    resolved_regime_thresholds = (
        dict(regime_thresholds)
        if regime_thresholds is not None
        else load_regime_thresholds(config_path=regime_thresholds_path)
    )

    if min_target_lag_months is not None and min_target_lag_days != 1:
        raise ContractViolation(
            "invalid_lag_policy",
            asof=asof_ts.to_pydatetime(),
            key="min_target_lag_days",
            detail=(
                "min_target_lag_days cannot be overridden when "
                "min_target_lag_months is provided"
            ),
        )

    slices: dict[str, pd.DataFrame] = {}
    lineage: dict[str, str] = {}
    target_months: dict[str, pd.Timestamp] = {}

    if target_month_sequence is not None:
        if len(target_month_sequence) != 2:
            raise ContractViolation(
                "invalid_target_month_policy",
                asof=asof_ts.to_pydatetime(),
                key="target_month_sequence",
                detail="target_month_sequence must contain exactly two month values",
            )
        target_month_pair = [
            pd.Timestamp(value).to_period("M").to_timestamp("M")
            for value in target_month_sequence
        ]
    elif target_month is not None:
        base_month = pd.Timestamp(target_month).to_period("M").to_timestamp("M")
        target_month_pair = [
            base_month,
            (base_month + pd.DateOffset(months=1)).to_period("M").to_timestamp("M"),
        ]
    else:
        target_month_pair = [
            (asof_ts - timedelta(days=1)).to_period("M").to_timestamp("M"),
            asof_ts.to_period("M").to_timestamp("M"),
        ]

    if target_month is None and target_month_sequence is None:
        horizon_map = {
            "T-1": asof_ts - timedelta(days=1),
            "T": asof_ts,
        }
    else:
        horizon_map = {
            "T-1": asof_ts,
            "T": asof_ts,
        }

    for idx, (horizon, horizon_asof) in enumerate(horizon_map.items()):
        forecast_target_month = target_month_pair[idx]
        panel, lineage_id = _build_single_horizon(
            normalized_features,
            normalized_target,
            horizon=horizon,
            horizon_asof=horizon_asof,
            run_asof=asof_ts,
            feature_policy=policy,
            min_target_lag_days=min_target_lag_days,
            min_target_lag_months=min_target_lag_months,
            forecast_target_month=forecast_target_month,
            regime_thresholds=resolved_regime_thresholds,
        )
        slices[horizon] = panel
        lineage[horizon] = lineage_id
        target_months[horizon] = forecast_target_month

    return VintageBuildResult(
        asof=asof_ts,
        slices=slices,
        lineage=lineage,
        target_months=target_months,
    )
