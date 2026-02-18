"""Model selection and governance decisions for champion/challenger outputs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from ng_forecaster.errors import ContractViolation
from ng_forecaster.evaluation.metrics import score_point_forecasts
from ng_forecaster.features.lag_guard import require_columns


@dataclass(frozen=True)
class SelectionDecision:
    """Governance decision for model selection."""

    selected_model: str
    selected_metric: float
    accepted: bool
    reason: str


@dataclass(frozen=True)
class AblationSelectionResult:
    """Selection payload used by ablation runner wiring."""

    selected_forecasts: pd.DataFrame
    selection_summary: pd.DataFrame


def select_model_by_metric(
    metrics_frame: pd.DataFrame,
    *,
    model_col: str = "candidate_model",
    metric_col: str = "rmse",
    maximize: bool = False,
) -> SelectionDecision:
    """Select a model from metric scoreboard with deterministic tie-breaking."""

    require_columns(metrics_frame, (model_col, metric_col), key="metrics_frame")

    score = metrics_frame[[model_col, metric_col]].copy()
    score[model_col] = score[model_col].astype(str)
    score[metric_col] = pd.to_numeric(score[metric_col], errors="coerce")
    if score[metric_col].isna().any():
        raise ContractViolation(
            "invalid_metric_payload",
            key=metric_col,
            detail="metric values must be numeric",
        )

    score = score.sort_values(
        [metric_col, model_col], ascending=[not maximize, True]
    ).reset_index(drop=True)
    winner = score.iloc[0]
    return SelectionDecision(
        selected_model=str(winner[model_col]),
        selected_metric=float(winner[metric_col]),
        accepted=True,
        reason=f"selected_by_{metric_col}",
    )


def select_ablation_candidates(
    forecast_frame: pd.DataFrame,
    *,
    experiment_col: str = "experiment_id",
    target_col: str = "target",
    candidate_col: str = "candidate_model",
    actual_col: str = "actual",
    forecast_col: str = "forecast",
    selection_metric: str = "rmse",
) -> AblationSelectionResult:
    """Select one candidate forecast per experiment and target for ablation replay."""

    require_columns(
        forecast_frame,
        (experiment_col, target_col, candidate_col, actual_col, forecast_col),
        key="ablation_forecast_frame",
    )

    metric_name = str(selection_metric).strip().lower()
    if metric_name not in {"mae", "rmse", "mape"}:
        raise ContractViolation(
            "invalid_model_policy",
            key="selection_metric",
            detail="selection_metric must be one of ['mae', 'rmse', 'mape']",
        )

    data = forecast_frame.copy()
    data[experiment_col] = data[experiment_col].astype(str).str.strip()
    data[target_col] = data[target_col].astype(str).str.strip()
    data[candidate_col] = data[candidate_col].astype(str).str.strip()
    if data[candidate_col].eq("").any():
        raise ContractViolation(
            "invalid_metric_payload",
            key=candidate_col,
            detail="candidate model names must be non-empty",
        )

    candidate_scores = score_point_forecasts(
        data,
        actual_col=actual_col,
        forecast_col=forecast_col,
        group_cols=[experiment_col, target_col, candidate_col],
    )

    summary_rows: list[dict[str, Any]] = []
    for (experiment_id, target_name), group in candidate_scores.groupby(
        [experiment_col, target_col],
        sort=True,
    ):
        decision = select_model_by_metric(
            group,
            model_col=candidate_col,
            metric_col=metric_name,
            maximize=False,
        )
        summary_rows.append(
            {
                experiment_col: str(experiment_id),
                target_col: str(target_name),
                "selected_model": decision.selected_model,
                "selected_metric": decision.selected_metric,
                "selection_metric": metric_name,
                "n_candidates": int(group[candidate_col].nunique()),
            }
        )

    summary = pd.DataFrame(summary_rows).sort_values(
        [experiment_col, target_col],
        ascending=[True, True],
    )
    summary = summary.reset_index(drop=True)

    selected = data.merge(
        summary[[experiment_col, target_col, "selected_model"]],
        left_on=[experiment_col, target_col, candidate_col],
        right_on=[experiment_col, target_col, "selected_model"],
        how="inner",
    )
    selected = selected.drop(columns=["selected_model"])
    selected = selected.sort_values(
        [experiment_col, target_col, candidate_col],
        ascending=[True, True, True],
    ).reset_index(drop=True)

    if selected.empty:
        raise ContractViolation(
            "invalid_metric_payload",
            key="ablation_selection",
            detail="selection produced no candidate forecasts",
        )

    return AblationSelectionResult(
        selected_forecasts=selected,
        selection_summary=summary,
    )


def enforce_stability_gate(
    stability_summary: pd.DataFrame,
    *,
    max_std: float,
) -> tuple[bool, str]:
    """Enforce seed-stability threshold from summary statistics."""

    require_columns(stability_summary, ("horizon", "std"), key="stability_summary")
    failing = stability_summary[stability_summary["std"] > max_std]
    if failing.empty:
        return True, "stability_gate_passed"

    horizon = int(failing.sort_values("std", ascending=False).iloc[0]["horizon"])
    std_value = float(failing.sort_values("std", ascending=False).iloc[0]["std"])
    return (
        False,
        f"stability_gate_failed:horizon={horizon}:std={std_value:.6f}:max_std={max_std:.6f}",
    )


def enforce_divergence_gate(
    divergence_summary: dict[str, Any],
    *,
    max_mean_abs_divergence: float,
) -> tuple[bool, str]:
    """Enforce divergence threshold between champion and challenger forecasts."""

    if "mean_abs_divergence" not in divergence_summary:
        raise ContractViolation(
            "invalid_metric_payload",
            key="mean_abs_divergence",
            detail="divergence_summary must include mean_abs_divergence",
        )

    value = float(divergence_summary["mean_abs_divergence"])
    if value <= max_mean_abs_divergence:
        return True, "divergence_gate_passed"
    return (
        False,
        (
            "divergence_gate_failed:"
            f"mean_abs_divergence={value:.6f}:"
            f"max_allowed={max_mean_abs_divergence:.6f}"
        ),
    )
