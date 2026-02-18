"""Data and policy contract validators."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import pandas as pd
import yaml

from ng_forecaster.errors import ContractViolation

ALLOWED_GAP_METHODS = {"ffill", "median"}
ALLOWED_OUTLIER_METHODS = {"winsorize", "clip"}
ALLOWED_FEATURE_FREQUENCIES = {"daily", "weekly", "monthly"}


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML document as a dictionary."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle) or {}
    if not isinstance(payload, dict):
        raise ContractViolation(
            "invalid_yaml_root",
            key=str(path),
            detail="top-level YAML payload must be a mapping",
        )
    return dict(payload)


def validate_preprocessing_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate preprocessing policy and fill deterministic defaults."""

    defaults = {
        "short_gap_limit": 2,
        "short_gap_method": "ffill",
        "long_gap_method": "median",
        "outlier_zscore_threshold": 3.0,
        "outlier_method": "winsorize",
        "min_non_null_ratio": 0.8,
    }
    merged = {**defaults, **dict(policy)}

    short_gap_limit = merged["short_gap_limit"]
    if not isinstance(short_gap_limit, int) or short_gap_limit < 0:
        raise ContractViolation(
            "invalid_preprocess_policy",
            key="short_gap_limit",
            detail="short_gap_limit must be an integer >= 0",
        )

    short_gap_method = merged["short_gap_method"]
    if short_gap_method not in ALLOWED_GAP_METHODS:
        raise ContractViolation(
            "invalid_preprocess_policy",
            key="short_gap_method",
            detail=f"short_gap_method must be one of {sorted(ALLOWED_GAP_METHODS)}",
        )

    long_gap_method = merged["long_gap_method"]
    if long_gap_method not in ALLOWED_GAP_METHODS:
        raise ContractViolation(
            "invalid_preprocess_policy",
            key="long_gap_method",
            detail=f"long_gap_method must be one of {sorted(ALLOWED_GAP_METHODS)}",
        )

    outlier_threshold = merged["outlier_zscore_threshold"]
    if not isinstance(outlier_threshold, (int, float)) or outlier_threshold <= 0:
        raise ContractViolation(
            "invalid_preprocess_policy",
            key="outlier_zscore_threshold",
            detail="outlier_zscore_threshold must be > 0",
        )

    outlier_method = merged["outlier_method"]
    if outlier_method not in ALLOWED_OUTLIER_METHODS:
        raise ContractViolation(
            "invalid_preprocess_policy",
            key="outlier_method",
            detail=f"outlier_method must be one of {sorted(ALLOWED_OUTLIER_METHODS)}",
        )

    ratio = merged["min_non_null_ratio"]
    if not isinstance(ratio, (int, float)) or not 0 <= float(ratio) <= 1:
        raise ContractViolation(
            "invalid_preprocess_policy",
            key="min_non_null_ratio",
            detail="min_non_null_ratio must be between 0 and 1",
        )

    merged["outlier_zscore_threshold"] = float(outlier_threshold)
    merged["min_non_null_ratio"] = float(ratio)
    return merged


def validate_feature_policy(policy: Mapping[str, Any]) -> dict[str, Any]:
    """Validate deterministic feature policy definitions."""

    defaults = {
        "version": 1,
        "default": {"max_age_days": 35},
        "features": {},
    }
    merged: dict[str, Any] = {**defaults, **dict(policy)}

    default_cfg = merged.get("default")
    if not isinstance(default_cfg, Mapping):
        raise ContractViolation(
            "invalid_feature_policy",
            key="default",
            detail="default section must be a mapping",
        )
    default_age = default_cfg.get("max_age_days", 35)
    if not isinstance(default_age, int) or default_age < 0:
        raise ContractViolation(
            "invalid_feature_policy",
            key="default.max_age_days",
            detail="default.max_age_days must be integer >= 0",
        )

    feature_defs = merged.get("features")
    if not isinstance(feature_defs, Mapping) or not feature_defs:
        raise ContractViolation(
            "invalid_feature_policy",
            key="features",
            detail="features mapping is required and cannot be empty",
        )

    normalized_features: dict[str, Any] = {}
    for feature_name, cfg in sorted(feature_defs.items()):
        if not isinstance(cfg, Mapping):
            raise ContractViolation(
                "invalid_feature_policy",
                key=f"features.{feature_name}",
                detail="feature config must be a mapping",
            )

        frequency = cfg.get("source_frequency")
        if frequency not in ALLOWED_FEATURE_FREQUENCIES:
            raise ContractViolation(
                "invalid_feature_policy",
                key=f"features.{feature_name}.source_frequency",
                detail=(
                    "source_frequency must be one of "
                    f"{sorted(ALLOWED_FEATURE_FREQUENCIES)}"
                ),
            )

        aggregation = cfg.get("aggregation")
        if not isinstance(aggregation, str) or not aggregation.strip():
            raise ContractViolation(
                "invalid_feature_policy",
                key=f"features.{feature_name}.aggregation",
                detail="aggregation must be a non-empty string",
            )

        max_age_days = cfg.get("max_age_days", default_age)
        if not isinstance(max_age_days, int) or max_age_days < 0:
            raise ContractViolation(
                "invalid_feature_policy",
                key=f"features.{feature_name}.max_age_days",
                detail="max_age_days must be integer >= 0",
            )

        normalized_features[feature_name] = {
            "source_frequency": frequency,
            "aggregation": aggregation,
            "max_age_days": max_age_days,
        }

    merged["default"] = {"max_age_days": default_age}
    merged["features"] = normalized_features
    return merged


def load_and_validate_preprocessing_policy(path: str | Path) -> dict[str, Any]:
    """Load preprocessing policy from YAML and validate it."""

    return validate_preprocessing_policy(load_yaml(path))


def load_and_validate_feature_policy(path: str | Path) -> dict[str, Any]:
    """Load feature policy from YAML and validate it."""

    return validate_feature_policy(load_yaml(path))


def validate_weather_coverage(
    frame: pd.DataFrame,
    *,
    min_coverage: float = 0.8,
) -> pd.DataFrame:
    """Fail loud when weather freeze-off panel has insufficient coverage."""

    required = {"basin_id", "timestamp", "coverage_fraction"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ContractViolation(
            "missing_column",
            key="weather_freezeoff_panel",
            detail="missing required weather coverage columns: " + ", ".join(missing),
        )
    if frame.empty:
        raise ContractViolation(
            "missing_source_file",
            key="weather_freezeoff_panel",
            detail="weather freeze-off panel is empty",
        )
    if not 0.0 <= float(min_coverage) <= 1.0:
        raise ContractViolation(
            "invalid_preprocess_policy",
            key="min_coverage",
            detail="min_coverage must be between 0 and 1",
        )
    coverage = pd.to_numeric(frame["coverage_fraction"], errors="coerce")
    invalid = frame[coverage.isna() | (coverage < float(min_coverage))]
    if not invalid.empty:
        sample = invalid.iloc[0]
        raise ContractViolation(
            "insufficient_source_coverage",
            key=f"{sample['basin_id']}@{pd.Timestamp(sample['timestamp']).date().isoformat()}",
            detail=(
                f"coverage_fraction={float(sample['coverage_fraction']):.4f} is below "
                f"min_coverage={float(min_coverage):.4f}"
            ),
        )
    return frame


def validate_weather_lineage(frame: pd.DataFrame) -> pd.DataFrame:
    """Fail loud when weather panel lineage columns are missing or empty."""

    required = {"source_id", "lineage_id", "available_timestamp"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ContractViolation(
            "missing_column",
            key="weather_freezeoff_panel",
            detail="missing required weather lineage columns: " + ", ".join(missing),
        )
    if frame.empty:
        raise ContractViolation(
            "missing_source_file",
            key="weather_freezeoff_panel",
            detail="weather freeze-off panel is empty",
        )
    source_ok = frame["source_id"].astype(str).str.strip() != ""
    lineage_ok = frame["lineage_id"].astype(str).str.strip() != ""
    available_ts = pd.to_datetime(frame["available_timestamp"], errors="coerce")
    invalid = frame[~source_ok | ~lineage_ok | available_ts.isna()]
    if not invalid.empty:
        sample = invalid.iloc[0]
        raise ContractViolation(
            "missing_lineage_metadata",
            key=f"{sample.get('basin_id', '<unknown>')}",
            detail=(
                "weather lineage requires non-empty source_id/lineage_id and "
                "valid available_timestamp"
            ),
        )
    return frame
