"""Realtime rolling-origin replay over explicit as-of checkpoints."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from ng_forecaster.errors import ContractViolation
from ng_forecaster.evaluation.dm_test import (
    DMRunResult,
    run_dm_tests,
    validate_dm_policy,
)
from ng_forecaster.evaluation.metrics import score_point_forecasts
from ng_forecaster.features.lag_guard import require_columns
from ng_forecaster.features.vintage_builder import (
    VintageBuildResult,
    build_vintage_panel,
)
from ng_forecaster.models.selection import select_ablation_candidates


@dataclass(frozen=True)
class ReplayResult:
    """Replay outputs with per-checkpoint traceability."""

    checkpoints: list[pd.Timestamp]
    frame: pd.DataFrame


@dataclass(frozen=True)
class AblationRunResult:
    """Ablation scorecard payload with DM and selection diagnostics."""

    scorecard: pd.DataFrame
    dm_results: pd.DataFrame
    selection_summary: pd.DataFrame
    config: dict[str, Any]


_REQUIRED_ABLATION_IDS = (
    "B0_baseline",
    "B1_plus_preprocessing",
    "B2_plus_feature_expansion",
    "B3_plus_challenger",
    "B4_full_method",
)

DEFAULT_ABLATION_CONFIG: dict[str, Any] = {
    "version": 1,
    "baseline_experiment": "B0_baseline",
    "full_method_experiment": "B4_full_method",
    "selection_metric": "rmse",
    "experiments": [
        {
            "id": "B0_baseline",
            "stage": 0,
            "description": "Lag-safe baseline with no additional enhancements.",
        },
        {
            "id": "B1_plus_preprocessing",
            "stage": 1,
            "description": "Baseline plus deterministic preprocessing policy.",
        },
        {
            "id": "B2_plus_feature_expansion",
            "stage": 2,
            "description": "B1 plus expanded lag-safe feature set.",
        },
        {
            "id": "B3_plus_challenger",
            "stage": 3,
            "description": "B2 plus challenger-assisted decision support.",
        },
        {
            "id": "B4_full_method",
            "stage": 4,
            "description": "Full method stack with all improvements enabled.",
        },
    ],
}


def _merge_dict(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            merged[key] = _merge_dict(base[key], value)
        else:
            merged[key] = value
    return merged


def load_checkpoint_dates(
    path: str | Path, *, column: str = "asof"
) -> list[pd.Timestamp]:
    """Load explicit replay checkpoint dates from a CSV file."""

    frame = pd.read_csv(path)
    if column not in frame.columns:
        raise ContractViolation(
            "missing_column",
            key=column,
            detail="checkpoint file must contain an asof column",
        )
    checkpoints = pd.to_datetime(frame[column], errors="coerce")
    if checkpoints.isna().any():
        raise ContractViolation(
            "invalid_timestamp",
            key=column,
            detail="checkpoint file contains invalid timestamp values",
        )
    return sorted(checkpoints.drop_duplicates().tolist())


def _normalize_checkpoints(checkpoints: Iterable[object]) -> list[pd.Timestamp]:
    normalized = pd.to_datetime(list(checkpoints), errors="coerce")
    if normalized.isna().any():
        raise ContractViolation(
            "invalid_timestamp",
            key="checkpoints",
            detail="replay checkpoints contain invalid timestamp values",
        )
    if len(normalized) == 0:
        raise ContractViolation(
            "empty_checkpoints",
            key="checkpoints",
            detail="at least one replay checkpoint is required",
        )
    return sorted(pd.Series(normalized).drop_duplicates().tolist())


def run_replay(
    features: pd.DataFrame,
    target: pd.DataFrame,
    checkpoints: Iterable[object],
    *,
    preprocessing_status: str = "passed",
    min_target_lag_days: int = 1,
    min_target_lag_months: int | None = None,
    target_month: object | None = None,
    target_month_offset_months: int | None = None,
    feature_policy: Mapping[str, Any] | None = None,
) -> ReplayResult:
    """Build replay outputs for each checkpoint with explicit target-month context."""

    if preprocessing_status != "passed":
        raise ContractViolation(
            "preprocess_gate_failed",
            key="preprocessing_status",
            detail=f"status={preprocessing_status}",
        )

    checkpoint_dates = _normalize_checkpoints(checkpoints)
    rows: list[pd.DataFrame] = []

    for checkpoint in checkpoint_dates:
        checkpoint_target_month: pd.Timestamp | None = None
        if target_month is not None:
            checkpoint_target_month = (
                pd.Timestamp(target_month).to_period("M").to_timestamp("M")
            )
        elif target_month_offset_months is not None:
            checkpoint_target_month = (
                checkpoint.to_period("M") - int(target_month_offset_months)
            ).to_timestamp("M")

        build: VintageBuildResult = build_vintage_panel(
            features,
            target,
            asof=checkpoint,
            preprocessing_status=preprocessing_status,
            min_target_lag_days=min_target_lag_days,
            min_target_lag_months=min_target_lag_months,
            feature_policy=feature_policy,
            target_month=checkpoint_target_month,
        )

        for horizon in ("T-1", "T"):
            panel = build.slices[horizon].copy()
            panel.insert(0, "replay_checkpoint", checkpoint)
            panel.insert(
                1,
                "trace_id",
                (
                    f"{checkpoint.date()}::{horizon}::"
                    f"{build.target_months[horizon].date().isoformat()}::"
                    f"{build.lineage[horizon][:12]}"
                ),
            )
            rows.append(panel)

    output = pd.concat(rows, ignore_index=True)
    output = output.sort_values(["replay_checkpoint", "horizon"]).reset_index(drop=True)
    return ReplayResult(checkpoints=checkpoint_dates, frame=output)


def validate_ablation_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize ablation matrix configuration contract."""

    merged = _merge_dict(DEFAULT_ABLATION_CONFIG, config or {})

    version = int(merged.get("version", 1))
    if version < 1:
        raise ContractViolation(
            "invalid_ablation_config",
            key="version",
            detail="version must be >= 1",
        )

    selection_metric = str(merged.get("selection_metric", "rmse")).strip().lower()
    if selection_metric not in {"mae", "rmse", "mape"}:
        raise ContractViolation(
            "invalid_ablation_config",
            key="selection_metric",
            detail="selection_metric must be one of ['mae', 'rmse', 'mape']",
        )

    experiments_raw = merged.get("experiments")
    if not isinstance(experiments_raw, list) or not experiments_raw:
        raise ContractViolation(
            "invalid_ablation_config",
            key="experiments",
            detail="experiments must be a non-empty list",
        )

    experiments: list[dict[str, Any]] = []
    seen_ids: set[str] = set()
    for position, payload in enumerate(experiments_raw):
        if not isinstance(payload, Mapping):
            raise ContractViolation(
                "invalid_ablation_config",
                key=f"experiments[{position}]",
                detail="experiment entries must be mappings",
            )

        experiment_id = str(payload.get("id", "")).strip()
        if not experiment_id:
            raise ContractViolation(
                "invalid_ablation_config",
                key=f"experiments[{position}].id",
                detail="experiment id must be non-empty",
            )
        if experiment_id in seen_ids:
            raise ContractViolation(
                "invalid_ablation_config",
                key="experiments.id",
                detail=f"duplicate experiment id: {experiment_id}",
            )
        seen_ids.add(experiment_id)

        stage = payload.get("stage", position)
        try:
            stage_value = int(stage)
        except (TypeError, ValueError) as exc:
            raise ContractViolation(
                "invalid_ablation_config",
                key=f"experiments[{position}].stage",
                detail="stage must be an integer",
            ) from exc

        experiments.append(
            {
                "id": experiment_id,
                "stage": stage_value,
                "description": str(payload.get("description", "")).strip(),
            }
        )

    configured_ids = {entry["id"] for entry in experiments}
    missing_required = [
        eid for eid in _REQUIRED_ABLATION_IDS if eid not in configured_ids
    ]
    if missing_required:
        raise ContractViolation(
            "invalid_ablation_config",
            key="experiments",
            detail="missing required experiments: " + ", ".join(missing_required),
        )

    baseline_experiment = str(merged.get("baseline_experiment", "")).strip()
    full_method_experiment = str(merged.get("full_method_experiment", "")).strip()
    if baseline_experiment not in configured_ids:
        raise ContractViolation(
            "invalid_ablation_config",
            key="baseline_experiment",
            detail=f"baseline experiment {baseline_experiment} is not in experiments",
        )
    if full_method_experiment not in configured_ids:
        raise ContractViolation(
            "invalid_ablation_config",
            key="full_method_experiment",
            detail=(
                f"full_method experiment {full_method_experiment} is not in experiments"
            ),
        )

    experiments = sorted(experiments, key=lambda row: (row["stage"], row["id"]))
    return {
        "version": version,
        "baseline_experiment": baseline_experiment,
        "full_method_experiment": full_method_experiment,
        "selection_metric": selection_metric,
        "experiments": experiments,
    }


def _resolve_lineage_id(values: pd.Series) -> str:
    normalized = sorted({str(value).strip() for value in values if str(value).strip()})
    if not normalized:
        raise ContractViolation(
            "missing_lineage_id",
            key="lineage_id",
            detail="lineage_id cannot be empty for ablation scorecard export",
        )
    if len(normalized) == 1:
        return normalized[0]

    digest = hashlib.sha256("|".join(normalized).encode("utf-8")).hexdigest()
    return digest[:16]


def _build_experiment_order(config: Mapping[str, Any]) -> dict[str, int]:
    order: dict[str, int] = {}
    for position, payload in enumerate(config["experiments"]):
        order[str(payload["id"])] = position
    return order


def run_ablation_matrix(
    forecast_frame: pd.DataFrame,
    *,
    config: Mapping[str, Any] | None,
    dm_policy: Mapping[str, Any] | None,
    experiment_col: str = "experiment_id",
    target_col: str = "target",
    candidate_col: str = "candidate_model",
    actual_col: str = "actual",
    forecast_col: str = "forecast",
    runtime_col: str = "runtime_seconds",
    lineage_col: str = "lineage_id",
    asof_col: str = "asof",
    horizon_col: str = "horizon",
) -> AblationRunResult:
    """Run controlled ablation matrix and return adoption-ready scorecard evidence."""

    require_columns(
        forecast_frame,
        (
            experiment_col,
            target_col,
            actual_col,
            forecast_col,
            runtime_col,
            lineage_col,
        ),
        key="ablation_forecasts",
    )

    cfg = validate_ablation_config(config)
    experiment_order = _build_experiment_order(cfg)
    configured_ids = set(experiment_order.keys())

    data = forecast_frame.copy()
    data[experiment_col] = data[experiment_col].astype(str).str.strip()
    data[target_col] = data[target_col].astype(str).str.strip()

    observed_ids = set(data[experiment_col].unique().tolist())
    missing_ids = sorted(configured_ids - observed_ids)
    if missing_ids:
        raise ContractViolation(
            "missing_ablation_experiment",
            key=experiment_col,
            detail="forecast payload is missing configured experiments: "
            + ", ".join(missing_ids),
        )

    data = data[data[experiment_col].isin(configured_ids)].copy()
    data[actual_col] = pd.to_numeric(data[actual_col], errors="coerce")
    data[forecast_col] = pd.to_numeric(data[forecast_col], errors="coerce")
    data[runtime_col] = pd.to_numeric(data[runtime_col], errors="coerce")
    if data[[actual_col, forecast_col, runtime_col]].isna().any().any():
        raise ContractViolation(
            "invalid_metric_payload",
            key="ablation_forecasts",
            detail="actual, forecast, and runtime columns must be numeric",
        )

    if (data[runtime_col] <= 0).any():
        raise ContractViolation(
            "invalid_runtime",
            key=runtime_col,
            detail="runtime_seconds must be strictly positive",
        )

    if asof_col in data.columns:
        data[asof_col] = pd.to_datetime(data[asof_col], errors="coerce")
        if data[asof_col].isna().any():
            raise ContractViolation(
                "invalid_timestamp",
                key=asof_col,
                detail="ablation asof values must be valid timestamps",
            )

    if horizon_col in data.columns:
        data[horizon_col] = pd.to_numeric(data[horizon_col], errors="coerce")
        if data[horizon_col].isna().any():
            raise ContractViolation(
                "invalid_metric_payload",
                key=horizon_col,
                detail="ablation horizon values must be numeric",
            )

    if candidate_col in data.columns:
        selection = select_ablation_candidates(
            data,
            experiment_col=experiment_col,
            target_col=target_col,
            candidate_col=candidate_col,
            actual_col=actual_col,
            forecast_col=forecast_col,
            selection_metric=cfg["selection_metric"],
        )
        selected = selection.selected_forecasts
        selection_summary = selection.selection_summary
    else:
        selected = data.copy()
        selection_summary = (
            selected[[experiment_col, target_col]]
            .drop_duplicates()
            .sort_values([experiment_col, target_col])
            .assign(
                selected_model=lambda frame: frame[experiment_col],
                selected_metric=float("nan"),
                selection_metric=cfg["selection_metric"],
                n_candidates=1,
            )
            .reset_index(drop=True)
        )

    score = score_point_forecasts(
        selected,
        actual_col=actual_col,
        forecast_col=forecast_col,
        group_cols=[experiment_col, target_col],
    )

    runtime_summary = (
        selected.groupby([experiment_col, target_col], sort=True)[runtime_col]
        .max()
        .reset_index()
    )
    lineage_summary = (
        selected.groupby([experiment_col, target_col], sort=True)[lineage_col]
        .apply(_resolve_lineage_id)
        .reset_index(name=lineage_col)
    )

    score = score.merge(runtime_summary, on=[experiment_col, target_col], how="left")
    score = score.merge(lineage_summary, on=[experiment_col, target_col], how="left")
    score = score.merge(selection_summary, on=[experiment_col, target_col], how="left")

    dm_input = selected.rename(
        columns={
            experiment_col: "model",
            target_col: "target",
            actual_col: "actual",
            forecast_col: "forecast",
        }
    )

    dm_columns = ["target", "model", "actual", "forecast"]
    if asof_col in dm_input.columns:
        dm_columns.append(asof_col)
    if horizon_col in dm_input.columns:
        dm_columns.append(horizon_col)

    dm_input = dm_input[dm_columns]
    target_benchmark = {
        target_name: cfg["baseline_experiment"]
        for target_name in sorted(dm_input["target"].astype(str).unique().tolist())
    }

    policy_payload = dict(dm_policy or {})
    policy_payload["benchmark_by_target"] = target_benchmark
    # Ablation uses experiment-vs-baseline DM wiring; disable variant-pair overrides.
    policy_payload["comparison_pairs_by_target"] = {}
    validated_policy = validate_dm_policy(policy_payload)

    dm_run: DMRunResult = run_dm_tests(
        dm_input,
        validated_policy,
        target_col="target",
        model_col="model",
        actual_col="actual",
        forecast_col="forecast",
        asof_col=asof_col,
        horizon_col=horizon_col,
    )

    dm_results = dm_run.results.copy()

    dm_lookup = dm_results.set_index(["target", "candidate_model"])
    baseline_id = cfg["baseline_experiment"]

    scorecard_rows: list[dict[str, Any]] = []
    for _, row in score.iterrows():
        experiment_id = str(row[experiment_col])
        target_name = str(row[target_col])

        dm_p_value = 1.0
        dm_stat = 0.0
        dm_d_bar = 0.0
        if experiment_id != baseline_id:
            key = (target_name, experiment_id)
            if key not in dm_lookup.index:
                raise ContractViolation(
                    "missing_dm_result",
                    key=f"{target_name}:{experiment_id}",
                    detail="missing DM result for ablation candidate vs baseline",
                )
            dm_row = dm_lookup.loc[key]
            dm_p_value = float(dm_row["adjusted_p_value"])
            dm_stat = float(dm_row["dm_stat"])
            dm_d_bar = float(dm_row["d_bar"])

        scorecard_rows.append(
            {
                "experiment_id": experiment_id,
                "target": target_name,
                "mae": float(row["mae"]),
                "rmse": float(row["rmse"]),
                "mape": float(row["mape"]),
                "dm_vs_baseline_p_value": dm_p_value,
                "dm_vs_baseline_stat": dm_stat,
                "dm_vs_baseline_d_bar": dm_d_bar,
                "runtime_seconds": float(row[runtime_col]),
                "lineage_id": str(row[lineage_col]),
                "selected_model": str(row["selected_model"]),
                "selected_metric": (
                    float(row["selected_metric"])
                    if pd.notna(row["selected_metric"])
                    else float("nan")
                ),
                "selection_metric": str(row["selection_metric"]),
                "n_obs": int(row["n_obs"]),
                "n_candidates": int(row["n_candidates"]),
            }
        )

    scorecard = pd.DataFrame(scorecard_rows)
    scorecard["_experiment_order"] = scorecard["experiment_id"].map(experiment_order)
    scorecard = scorecard.sort_values(
        ["_experiment_order", "target"],
        ascending=[True, True],
    ).drop(columns=["_experiment_order"])
    scorecard = scorecard.reset_index(drop=True)

    return AblationRunResult(
        scorecard=scorecard,
        dm_results=dm_results,
        selection_summary=selection_summary,
        config=cfg,
    )
