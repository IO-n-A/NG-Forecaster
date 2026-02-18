"""Deterministic preprocessing runtime for missing and outlier handling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ng_forecaster.data.validators import validate_preprocessing_policy
from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.lag_guard import require_columns


@dataclass(frozen=True)
class PreprocessResult:
    """Structured output for preprocessing stage."""

    data: pd.DataFrame
    missing_flags: pd.DataFrame
    outlier_flags: pd.DataFrame
    summary: dict[str, Any]
    status: str


def _missing_runs(mask: pd.Series) -> list[tuple[int, int]]:
    """Return inclusive index spans for contiguous missing runs."""

    spans: list[tuple[int, int]] = []
    start = -1
    for idx, is_missing in enumerate(mask.tolist()):
        if is_missing and start == -1:
            start = idx
        if not is_missing and start != -1:
            spans.append((start, idx - 1))
            start = -1
    if start != -1:
        spans.append((start, len(mask) - 1))
    return spans


def run_preprocessing(
    frame: pd.DataFrame,
    policy: Mapping[str, Any],
    *,
    value_col: str = "value",
    series_col: str = "series_id",
    timestamp_col: str = "timestamp",
) -> PreprocessResult:
    """Apply deterministic missing-value and outlier policy to an input frame."""

    require_columns(
        frame,
        (series_col, timestamp_col, value_col),
        key="preprocess_input",
    )
    cfg = validate_preprocessing_policy(policy)

    data = frame.copy()
    data[timestamp_col] = pd.to_datetime(data[timestamp_col], errors="coerce")
    if data[timestamp_col].isna().any():
        raise ContractViolation(
            "invalid_timestamp",
            key=timestamp_col,
            detail="preprocess input contains invalid timestamp values",
        )

    data[value_col] = pd.to_numeric(data[value_col], errors="coerce")
    data = data.sort_values([series_col, timestamp_col]).reset_index(drop=True)

    missing_rows: list[dict[str, Any]] = []
    outlier_rows: list[dict[str, Any]] = []
    transformed_groups: list[pd.DataFrame] = []
    low_coverage_series = 0
    unresolved_missing = 0

    for series_id, group in data.groupby(series_col, sort=True):
        group = group.copy().reset_index(drop=True)
        values = group[value_col].astype(float)
        missing_mask = values.isna()

        non_null_ratio = float(1.0 - missing_mask.mean())
        if non_null_ratio < cfg["min_non_null_ratio"]:
            low_coverage_series += 1

        median = float(values.median()) if not np.isnan(values.median()) else 0.0
        filled = values.copy()

        spans = _missing_runs(missing_mask)
        for start, end in spans:
            length = end - start + 1
            method = (
                cfg["short_gap_method"]
                if length <= cfg["short_gap_limit"]
                else cfg["long_gap_method"]
            )
            idx = list(range(start, end + 1))
            if method == "ffill":
                reference = filled.ffill().bfill()
                filled.iloc[idx] = reference.iloc[idx]
            elif method == "median":
                filled.iloc[idx] = median

            for row_idx in idx:
                missing_rows.append(
                    {
                        series_col: series_id,
                        timestamp_col: group.at[row_idx, timestamp_col],
                        "gap_length": length,
                        "treatment": method,
                    }
                )

        still_missing = filled.isna()
        if still_missing.any():
            unresolved_missing += int(still_missing.sum())
            filled[still_missing] = median

        mean = float(filled.mean())
        std = float(filled.std(ddof=0))
        treated = filled.copy()
        if std > 0:
            threshold = cfg["outlier_zscore_threshold"]
            zscore = (filled - mean) / std
            outlier_mask = zscore.abs() > threshold
            if outlier_mask.any():
                lower = mean - threshold * std
                upper = mean + threshold * std
                treated = treated.clip(lower=lower, upper=upper)
                for row_idx in np.where(outlier_mask.to_numpy())[0].tolist():
                    outlier_rows.append(
                        {
                            series_col: series_id,
                            timestamp_col: group.at[row_idx, timestamp_col],
                            "zscore": float(zscore.iloc[row_idx]),
                            "original_value": float(filled.iloc[row_idx]),
                            "treated_value": float(treated.iloc[row_idx]),
                            "treatment": cfg["outlier_method"],
                        }
                    )

        group[value_col] = treated
        transformed_groups.append(group)

    transformed = (
        pd.concat(transformed_groups, ignore_index=True)
        .sort_values([series_col, timestamp_col])
        .reset_index(drop=True)
    )

    missing_flags = pd.DataFrame(missing_rows)
    if missing_flags.empty:
        missing_flags = pd.DataFrame(
            columns=[series_col, timestamp_col, "gap_length", "treatment"]
        )
    else:
        missing_flags = missing_flags.sort_values([series_col, timestamp_col]).reset_index(
            drop=True
        )

    outlier_flags = pd.DataFrame(outlier_rows)
    if outlier_flags.empty:
        outlier_flags = pd.DataFrame(
            columns=[
                series_col,
                timestamp_col,
                "zscore",
                "original_value",
                "treated_value",
                "treatment",
            ]
        )
    else:
        outlier_flags = outlier_flags.sort_values([series_col, timestamp_col]).reset_index(
            drop=True
        )

    status = "passed" if unresolved_missing == 0 and low_coverage_series == 0 else "failed"
    summary = {
        "status": status,
        "row_count": int(len(transformed)),
        "series_count": int(transformed[series_col].nunique()),
        "missing_flag_count": int(len(missing_flags)),
        "outlier_flag_count": int(len(outlier_flags)),
        "unresolved_missing_count": int(unresolved_missing),
        "low_coverage_series_count": int(low_coverage_series),
        "policy": cfg,
    }

    return PreprocessResult(
        data=transformed,
        missing_flags=missing_flags,
        outlier_flags=outlier_flags,
        summary=summary,
        status=status,
    )
