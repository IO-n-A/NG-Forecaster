"""Utilities for multi-trial reproducibility evaluation in rolling validation."""

from __future__ import annotations

from typing import Iterable, Sequence

import pandas as pd

from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.lag_guard import require_columns

_DEFAULT_METRICS: tuple[str, ...] = (
    "mae",
    "rmse",
    "mape_pct",
    "mean_signed_error",
    "mean_abs_error",
    "interval_hit_rate_95",
)


def resolve_seed_schedule(
    *,
    n_trials: int,
    seed_schedule: Sequence[int] | str | None = None,
    default_seed: int = 42,
) -> list[int]:
    """Resolve deterministic per-trial seeds for reproducibility runs."""

    trial_count = int(n_trials)
    if trial_count < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="n_trials",
            detail="n_trials must be >= 1",
        )

    seeds: list[int] = []
    if seed_schedule is None:
        seeds = [int(default_seed) + idx for idx in range(trial_count)]
    elif isinstance(seed_schedule, str):
        tokens = [token.strip() for token in seed_schedule.split(",") if token.strip()]
        if not tokens:
            raise ContractViolation(
                "invalid_model_policy",
                key="seed_schedule",
                detail="seed_schedule string cannot be empty",
            )
        try:
            seeds = [int(token) for token in tokens]
        except ValueError as exc:
            raise ContractViolation(
                "invalid_model_policy",
                key="seed_schedule",
                detail="seed_schedule must contain integer seeds",
            ) from exc
    else:
        try:
            seeds = [int(value) for value in seed_schedule]
        except (TypeError, ValueError) as exc:
            raise ContractViolation(
                "invalid_model_policy",
                key="seed_schedule",
                detail="seed_schedule values must be integers",
            ) from exc

    if not seeds:
        raise ContractViolation(
            "invalid_model_policy",
            key="seed_schedule",
            detail="seed schedule cannot be empty",
        )

    if len(seeds) < trial_count:
        last = int(seeds[-1])
        extension = [last + idx + 1 for idx in range(trial_count - len(seeds))]
        seeds.extend(extension)

    return seeds[:trial_count]


def summarize_trial_scorecard(
    trial_scorecard: pd.DataFrame,
    *,
    metrics: Iterable[str] = _DEFAULT_METRICS,
    std_tolerance: float = 1.0,
) -> pd.DataFrame:
    """Aggregate per-trial scorecards into mean/std stability summaries."""

    require_columns(
        trial_scorecard,
        ("model_variant", "trial_id"),
        key="trial_scorecard",
    )
    metric_list = [str(metric) for metric in metrics]
    require_columns(trial_scorecard, tuple(metric_list), key="trial_scorecard")

    rows: list[dict[str, object]] = []
    tolerance = float(std_tolerance)
    for variant, group in trial_scorecard.groupby("model_variant", sort=True):
        row: dict[str, object] = {
            "model_variant": str(variant),
            "n_trials": int(group["trial_id"].nunique()),
        }
        stability_pass = True
        for metric in metric_list:
            series = pd.to_numeric(group[metric], errors="coerce")
            if series.isna().any():
                raise ContractViolation(
                    "invalid_metric_payload",
                    key=metric,
                    detail="trial metrics must be numeric and non-null",
                )
            mean_value = float(series.mean())
            std_value = float(series.std(ddof=0))
            row[f"{metric}_mean"] = mean_value
            row[f"{metric}_std"] = std_value
            if std_value > tolerance:
                stability_pass = False
        row["stability_flag"] = bool(stability_pass)
        rows.append(row)

    return pd.DataFrame(rows).sort_values("model_variant").reset_index(drop=True)
