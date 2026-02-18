"""Artifact exporters for preprocessing, lineage, model diagnostics, and policy outputs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import pandas as pd

from ng_forecaster.data.preprocess import PreprocessResult
from ng_forecaster.errors import ContractViolation
from ng_forecaster.evaluation.dm_test import (
    DM_REQUIRED_OUTPUT_COLUMNS,
    ensure_dm_output_schema,
)
from ng_forecaster.features.lag_guard import require_columns

_ABLATION_REQUIRED_COLUMNS = [
    "experiment_id",
    "mae",
    "rmse",
    "mape",
    "dm_vs_baseline_p_value",
    "runtime_seconds",
    "lineage_id",
]


def _ensure_output_dir(path: str | Path) -> Path:
    output_dir = Path(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def export_preprocess_artifacts(
    result: PreprocessResult,
    output_dir: str | Path,
) -> dict[str, Path]:
    """Write required preprocessing artifacts to disk."""

    base = _ensure_output_dir(output_dir)

    summary_path = base / "preprocess_summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(result.summary, handle, sort_keys=True, indent=2, default=str)

    missing_path = base / "missing_value_flags.csv"
    result.missing_flags.to_csv(missing_path, index=False)

    outlier_path = base / "outlier_flags.csv"
    result.outlier_flags.to_csv(outlier_path, index=False)

    return {
        "preprocess_summary": summary_path,
        "missing_value_flags": missing_path,
        "outlier_flags": outlier_path,
    }


def export_feature_lineage(
    lineage: Mapping[str, str] | pd.DataFrame,
    output_dir: str | Path,
    *,
    filename: str = "feature_lineage.csv",
) -> Path:
    """Export deterministic feature lineage IDs."""

    base = _ensure_output_dir(output_dir)
    lineage_path = base / filename

    if isinstance(lineage, pd.DataFrame):
        frame = lineage.copy()
        if "lineage_id" not in frame.columns:
            raise ContractViolation(
                "invalid_lineage_payload",
                key="lineage_id",
                detail="lineage DataFrame must include lineage_id",
            )
    else:
        frame = pd.DataFrame(
            [
                {"horizon": key, "lineage_id": value}
                for key, value in sorted(lineage.items())
            ]
        )

    ordered_columns = sorted(frame.columns.tolist())
    frame = frame[ordered_columns].sort_values(ordered_columns).reset_index(drop=True)
    frame.to_csv(lineage_path, index=False)
    return lineage_path


def export_model_diagnostics(
    diagnostics: Mapping[str, object],
    stability_summary: pd.DataFrame,
    output_dir: str | Path,
    *,
    diagnostics_filename: str = "model_diagnostics.json",
    stability_filename: str = "seed_stability_summary.csv",
) -> dict[str, Path]:
    """Export model diagnostics and seed stability summary artifacts."""

    if "model_family" not in diagnostics:
        raise ContractViolation(
            "invalid_diagnostics_payload",
            key="model_family",
            detail="diagnostics payload must include model_family",
        )

    base = _ensure_output_dir(output_dir)

    diagnostics_path = base / diagnostics_filename
    with diagnostics_path.open("w", encoding="utf-8") as handle:
        json.dump(dict(diagnostics), handle, sort_keys=True, indent=2, default=str)

    stability_path = base / stability_filename
    if stability_summary.empty:
        stability_summary = pd.DataFrame(
            columns=["horizon", "mean", "std", "min", "max", "spread"]
        )
    stability_summary.sort_values(stability_summary.columns.tolist()).to_csv(
        stability_path,
        index=False,
    )

    return {
        "model_diagnostics": diagnostics_path,
        "seed_stability_summary": stability_path,
    }


def _coerce_bool(value: object, *, key: str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ContractViolation(
        "invalid_dm_output",
        key=key,
        detail=f"invalid boolean value: {value}",
    )


def export_dm_results(
    dm_results: pd.DataFrame,
    output_dir: str | Path,
    *,
    filename: str = "dm_results.csv",
) -> Path:
    """Export policy-compliant DM results with schema enforcement."""

    ensure_dm_output_schema(dm_results)

    frame = dm_results.copy()
    for column in ("d_bar", "dm_stat", "p_value", "adjusted_p_value"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
        if frame[column].isna().any():
            raise ContractViolation(
                "invalid_dm_output",
                key=column,
                detail="DM numeric fields must be parseable as floats",
            )

    frame["significant_0_05"] = frame["significant_0_05"].map(
        lambda value: _coerce_bool(value, key="significant_0_05")
    )
    frame["significant_0_01"] = frame["significant_0_01"].map(
        lambda value: _coerce_bool(value, key="significant_0_01")
    )

    expected_005 = frame["adjusted_p_value"] <= 0.05
    expected_001 = frame["adjusted_p_value"] <= 0.01
    if not frame["significant_0_05"].equals(expected_005) or not frame[
        "significant_0_01"
    ].equals(expected_001):
        raise ContractViolation(
            "invalid_dm_output",
            key="significance_flags",
            detail="significance flags must match adjusted p-value thresholds",
        )

    base = _ensure_output_dir(output_dir)
    output_path = base / filename

    ordered_columns = DM_REQUIRED_OUTPUT_COLUMNS + [
        col for col in frame.columns if col not in DM_REQUIRED_OUTPUT_COLUMNS
    ]
    frame = frame[ordered_columns]
    frame = frame.sort_values(
        ["target", "candidate_model", "benchmark_model"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    frame.to_csv(output_path, index=False)
    return output_path


def ensure_ablation_scorecard_schema(scorecard: pd.DataFrame) -> None:
    """Validate ablation scorecard fields required by policy and adoption gates."""

    require_columns(scorecard, _ABLATION_REQUIRED_COLUMNS, key="ablation_scorecard")
    if scorecard.empty:
        raise ContractViolation(
            "empty_ablation_scorecard",
            key="ablation_scorecard",
            detail="ablation scorecard cannot be empty",
        )

    numeric_cols = [
        "mae",
        "rmse",
        "mape",
        "dm_vs_baseline_p_value",
        "runtime_seconds",
    ]
    parsed = scorecard.copy()
    for column in numeric_cols:
        parsed[column] = pd.to_numeric(parsed[column], errors="coerce")
        if parsed[column].isna().any():
            raise ContractViolation(
                "invalid_ablation_scorecard",
                key=column,
                detail="scorecard numeric fields must be parseable as floats",
            )

    if (
        (parsed["dm_vs_baseline_p_value"] < 0) | (parsed["dm_vs_baseline_p_value"] > 1)
    ).any():
        raise ContractViolation(
            "invalid_ablation_scorecard",
            key="dm_vs_baseline_p_value",
            detail="dm_vs_baseline_p_value must be in [0, 1]",
        )

    if (parsed["runtime_seconds"] <= 0).any():
        raise ContractViolation(
            "invalid_ablation_scorecard",
            key="runtime_seconds",
            detail="runtime_seconds must be strictly positive",
        )

    if parsed["lineage_id"].astype(str).str.strip().eq("").any():
        raise ContractViolation(
            "missing_lineage_id",
            key="lineage_id",
            detail="lineage_id must be non-empty",
        )


def export_ablation_scorecard(
    scorecard: pd.DataFrame,
    output_dir: str | Path,
    *,
    filename: str = "ablation_scorecard.csv",
) -> Path:
    """Export ablation scorecard with deterministic ordering and schema checks."""

    ensure_ablation_scorecard_schema(scorecard)

    frame = scorecard.copy()
    base = _ensure_output_dir(output_dir)
    output_path = base / filename

    sort_columns = ["experiment_id"]
    if "target" in frame.columns:
        sort_columns.append("target")

    leading_columns = _ABLATION_REQUIRED_COLUMNS.copy()
    if "target" in frame.columns and "target" not in leading_columns:
        leading_columns.insert(1, "target")

    trailing_columns = [col for col in frame.columns if col not in leading_columns]
    frame = frame[leading_columns + trailing_columns]
    frame = frame.sort_values(sort_columns, ascending=[True] * len(sort_columns))
    frame = frame.reset_index(drop=True)
    frame.to_csv(output_path, index=False)
    return output_path
