"""24-month rolling validation runner for released-only nowcast scoring."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ng_forecaster.data.validators import load_yaml
from ng_forecaster.errors import ContractViolation
from ng_forecaster.evaluation.baselines import (
    BASELINE_VARIANTS,
    build_baseline_point_estimates,
)
from ng_forecaster.evaluation.block_importance import (
    build_block_ablation_forecasts,
    score_block_ablations,
)
from ng_forecaster.evaluation.dm_test import run_dm_tests_by_regime
from ng_forecaster.evaluation.interval_metrics import (
    build_interval_calibration_tables,
    build_interval_scorecard,
    build_interval_scorecard_by_regime,
)
from ng_forecaster.evaluation.month_length_effect import (
    build_calendar_calibration_report,
    build_month_length_by_regime_report,
    build_month_length_effect_report,
)
from ng_forecaster.evaluation.reproducibility import (
    resolve_seed_schedule,
    summarize_trial_scorecard,
)
from ng_forecaster.features.regime_flags import (
    compute_regime_flags,
    load_regime_thresholds,
)
from ng_forecaster.models.fusion import (
    fuse_forecasts,
    load_fusion_policy,
    resolve_calendar_calibration_for_regime,
    resolve_fusion_weights_for_regime_full,
)
from ng_forecaster.models.regime import classify_regime
from ng_forecaster.orchestration.airflow.dags.nowcast_pipeline_weekly import (
    run_nowcast_pipeline_weekly,
)
from ng_forecaster.orchestration.airflow.workflow_support import (
    enforce_release_policy,
    load_market_inputs,
)
from ng_forecaster.reporting.exporters import export_ablation_scorecard
from ng_forecaster.utils.calendar import (
    clamp_day_in_month,
    parse_month_end,
    shift_month_end,
)

DEFAULT_VARIANTS = ("wpd_lstm_one_layer", "wpd_vmd_lstm1", "wpd_vmd_lstm2")
CHALLENGER_VARIANTS = ("challenger_bsts",)
VARIANT_ALIASES: dict[str, tuple[str, ...]] = {
    "full": DEFAULT_VARIANTS,
    "full_plus_regime": (*DEFAULT_VARIANTS, *CHALLENGER_VARIANTS),
    "full_plus_prototype": DEFAULT_VARIANTS,
    "ablations": DEFAULT_VARIANTS,
    "challenger_bsts": CHALLENGER_VARIANTS,
    "baselines": BASELINE_VARIANTS,
}


@dataclass(frozen=True)
class Validation24mResult:
    """Export payload for 24-month validation runs."""

    point_estimates: pd.DataFrame
    scorecard: pd.DataFrame
    summary: dict[str, Any]


def _day_count_class(days: int) -> str:
    return f"{int(days)}d"


def _month_end(value: object) -> pd.Timestamp:
    return parse_month_end(value, key="target_month")


def build_target_month_grid(
    *, end_target_month: object, runs: int
) -> list[pd.Timestamp]:
    """Build deterministic ascending target-month grid ending at end_target_month."""

    if int(runs) < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="runs",
            detail="runs must be >= 1",
        )
    end_month = _month_end(end_target_month)
    months: list[pd.Timestamp] = []
    for offset in reversed(range(int(runs))):
        months.append((end_month.to_period("M") - offset).to_timestamp("M"))
    return months


def derive_policy_admissible_asof(
    *,
    target_month: pd.Timestamp,
    lag_months: int,
    release_day_of_month: int,
    preferred_day: int,
) -> pd.Timestamp:
    """Resolve an admissible as-of day that maps to the requested target month."""

    month_end = shift_month_end(target_month, months=int(lag_months))
    month_anchor = month_end.to_period("M").to_timestamp()
    day_cap = int(month_end.day)
    if int(release_day_of_month) > 1:
        day_cap = min(day_cap, int(release_day_of_month) - 1)
    start_day = clamp_day_in_month(
        year=int(month_anchor.year),
        month=int(month_anchor.month),
        preferred_day=min(max(1, int(preferred_day)), day_cap),
    )

    for day in range(start_day, 0, -1):
        candidate = pd.Timestamp(
            year=int(month_anchor.year),
            month=int(month_anchor.month),
            day=int(day),
        )
        try:
            context = enforce_release_policy(candidate)
        except ContractViolation:
            continue
        mapped_target = _month_end(context["target_month"])
        if mapped_target == target_month:
            return candidate

    raise ContractViolation(
        "lag_policy_violated",
        key=target_month.date().isoformat(),
        detail="unable to derive admissible asof for requested target month",
    )


def _resolve_actual_from_full_history(
    full_history: pd.DataFrame,
    *,
    target_month: pd.Timestamp,
) -> float:
    series = full_history.copy()
    series["timestamp"] = pd.to_datetime(series["timestamp"], errors="coerce")
    series["target_value"] = pd.to_numeric(series["target_value"], errors="coerce")
    series = series[series["timestamp"].notna() & series["target_value"].notna()].copy()
    series["timestamp"] = series["timestamp"].dt.to_period("M").dt.to_timestamp("M")
    row = series[series["timestamp"] == target_month]
    if row.empty:
        raise ContractViolation(
            "missing_target_history_rows",
            key=target_month.date().isoformat(),
            detail="full target history does not include requested target month",
        )
    return float(row.sort_values("timestamp").iloc[-1]["target_value"])


def _extract_target_nowcast(
    nowcast_path: Path,
    *,
    target_month: pd.Timestamp,
) -> dict[str, Any]:
    payload = json.loads(nowcast_path.read_text(encoding="utf-8"))
    rows = payload.get("nowcasts", [])
    if not isinstance(rows, list):
        raise ContractViolation(
            "source_schema_drift",
            key=str(nowcast_path),
            detail="nowcast payload must include nowcasts list",
        )

    target_iso = target_month.date().isoformat()
    for row in rows:
        if isinstance(row, dict) and str(row.get("target_month", "")) == target_iso:
            return dict(row)

    raise ContractViolation(
        "missing_column",
        key=str(nowcast_path),
        detail=f"nowcast payload missing target month {target_iso}",
    )


def _build_variant_override(variant: str, *, seed: int) -> dict[str, Any]:
    return {
        "model": {"variant": variant},
        "strategy": variant,
        "training": {"seed": int(seed)},
    }


def _candidate_horizon_weights(
    *,
    base_horizon_weights: Mapping[int, float],
    base_champion_weight: float,
    candidate_champion_weight: float,
) -> dict[int, float]:
    resolved: dict[int, float] = {}
    for horizon, weight in base_horizon_weights.items():
        delta = float(weight) - float(base_champion_weight)
        candidate = float(candidate_champion_weight) + delta
        resolved[int(horizon)] = min(max(candidate, 0.05), 1.0)
    return resolved


def _load_fusion_inputs(
    fusion_inputs_path: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if not fusion_inputs_path.exists():
        raise ContractViolation(
            "missing_source_file",
            key=str(fusion_inputs_path),
            detail="fusion_inputs.csv is required for weight search and anchor ablations",
        )
    frame = pd.read_csv(fusion_inputs_path)
    required = {
        "horizon",
        "point_forecast",
        "mean_forecast",
        "challenger_lower_95",
        "challenger_upper_95",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ContractViolation(
            "missing_column",
            key=str(fusion_inputs_path),
            detail="fusion inputs missing columns: " + ", ".join(missing),
        )
    champion = frame[["horizon", "point_forecast"]].copy()
    challenger = frame[
        [
            "horizon",
            "mean_forecast",
            "challenger_lower_95",
            "challenger_upper_95",
        ]
    ].rename(
        columns={
            "challenger_lower_95": "lower_95",
            "challenger_upper_95": "upper_95",
        }
    )
    steo_columns = ["horizon", "steo_point_forecast", "steo_lower_95", "steo_upper_95"]
    if "steo_point_forecast" in frame.columns:
        steo = frame[
            [column for column in steo_columns if column in frame.columns]
        ].copy()
        steo = steo[steo["steo_point_forecast"].notna()].copy()
    else:
        steo = pd.DataFrame(
            columns=["horizon", "steo_point_forecast", "steo_lower_95", "steo_upper_95"]
        )

    prototype_columns = [
        "horizon",
        "prototype_point_forecast",
        "prototype_lower_95",
        "prototype_upper_95",
    ]
    if "prototype_point_forecast" in frame.columns:
        prototype = frame[
            [column for column in prototype_columns if column in frame.columns]
        ].copy()
        prototype = prototype[prototype["prototype_point_forecast"].notna()].copy()
    else:
        prototype = pd.DataFrame(
            columns=[
                "horizon",
                "prototype_point_forecast",
                "prototype_lower_95",
                "prototype_upper_95",
            ]
        )
    return champion, challenger, frame, steo, prototype


def _extract_challenger_target_from_fusion_inputs(
    fusion_inputs_path: Path,
) -> tuple[float, float, float]:
    """Extract challenger point/interval for target horizon (h=1)."""

    (
        _champion,
        challenger_frame,
        _raw_frame,
        _steo_frame,
        _prototype_frame,
    ) = _load_fusion_inputs(fusion_inputs_path)
    target = challenger_frame[challenger_frame["horizon"].astype(int) == 1]
    if target.empty:
        target = challenger_frame.sort_values("horizon").head(1)
    if target.empty:
        raise ContractViolation(
            "missing_fusion_overlap",
            key=str(fusion_inputs_path),
            detail="unable to resolve challenger target horizon from fusion inputs",
        )
    row = target.iloc[0]
    return (
        float(row["mean_forecast"]),
        float(row["lower_95"]),
        float(row["upper_95"]),
    )


def _resolve_regime_snapshot(
    feature_rows: pd.DataFrame,
    *,
    asof: pd.Timestamp,
    thresholds: Mapping[str, Any],
) -> dict[str, float | str]:
    required = {
        "feature_name",
        "feature_timestamp",
        "available_timestamp",
        "value",
    }
    missing = sorted(required - set(feature_rows.columns))
    if missing:
        raise ContractViolation(
            "missing_column",
            key="feature_rows",
            detail="feature rows missing columns: " + ", ".join(missing),
        )
    prepared = feature_rows[
        ["feature_name", "feature_timestamp", "available_timestamp", "value"]
    ].copy()
    prepared["feature_timestamp"] = pd.to_datetime(
        prepared["feature_timestamp"],
        errors="coerce",
    )
    prepared["available_timestamp"] = pd.to_datetime(
        prepared["available_timestamp"],
        errors="coerce",
    )
    prepared["value"] = pd.to_numeric(prepared["value"], errors="coerce")
    prepared = prepared[
        prepared["feature_timestamp"].notna()
        & prepared["available_timestamp"].notna()
        & prepared["value"].notna()
    ].copy()
    prepared = prepared[
        (prepared["feature_timestamp"] <= asof)
        & (prepared["available_timestamp"] <= asof)
    ].copy()
    if prepared.empty:
        return {
            "regime_freeze_flag": 0.0,
            "regime_basis_flag": 0.0,
            "regime_transfer_dispersion_flag": 0.0,
            "regime_any_flag": 0.0,
            "regime_score": 0.0,
            "regime_label": "normal",
        }

    prepared = prepared.sort_values(
        ["feature_name", "feature_timestamp", "available_timestamp"]
    )
    latest = prepared.groupby("feature_name", as_index=False).tail(1)
    lookup = {
        str(row["feature_name"]): float(row["value"]) for _, row in latest.iterrows()
    }
    regime_values = compute_regime_flags(lookup, thresholds=thresholds)
    regime_label = classify_regime(regime_values)
    return {
        "regime_freeze_flag": float(regime_values["regime_freeze_flag"]),
        "regime_basis_flag": float(regime_values["regime_basis_flag"]),
        "regime_transfer_dispersion_flag": float(
            regime_values["regime_transfer_dispersion_flag"]
        ),
        "regime_any_flag": float(regime_values["regime_any_flag"]),
        "regime_score": float(regime_values["regime_score"]),
        "regime_label": regime_label,
    }


def _extract_feature_row(
    feature_matrix_path: Path,
    *,
    target_month: pd.Timestamp,
) -> dict[str, Any]:
    if not feature_matrix_path.exists():
        raise ContractViolation(
            "missing_source_file",
            key=str(feature_matrix_path),
            detail=(
                "feature_matrix.csv is required when exporting feature rows or block "
                "ablation artifacts"
            ),
        )
    feature_matrix = pd.read_csv(feature_matrix_path)
    if feature_matrix.empty:
        raise ContractViolation(
            "missing_column",
            key=str(feature_matrix_path),
            detail="feature_matrix.csv exists but contains zero rows",
        )
    if "target_month" in feature_matrix.columns:
        match = feature_matrix[
            feature_matrix["target_month"].astype(str)
            == target_month.date().isoformat()
        ]
        if not match.empty:
            feature_matrix = match
    if "horizon" in feature_matrix.columns:
        preferred = feature_matrix[feature_matrix["horizon"].astype(str) == "T-1"]
        if not preferred.empty:
            row = preferred.iloc[0].to_dict()
            return {str(key): value for key, value in row.items()}
        feature_matrix = feature_matrix.sort_values("horizon")
    row = feature_matrix.iloc[0].to_dict()
    return {str(key): value for key, value in row.items()}


def _load_optional_yaml(path: str | Path | None) -> dict[str, Any]:
    resolved = str(path or "").strip()
    if not resolved:
        return {}
    candidate = Path(resolved)
    if not candidate.exists():
        raise ContractViolation(
            "missing_source_file",
            key=resolved,
            detail="optional YAML path does not exist",
        )
    payload = load_yaml(candidate)
    if not isinstance(payload, Mapping):
        raise ContractViolation(
            "invalid_model_policy",
            key=resolved,
            detail="optional YAML payload must be a mapping",
        )
    return {str(key): value for key, value in dict(payload).items()}


def _resolve_calendar_search_candidate(
    base_calendar: Mapping[str, Any],
    *,
    enabled: bool,
    weight: float,
    max_abs_adjustment: float,
) -> dict[str, Any]:
    day_weights = {
        str(day): float(value) * float(weight)
        for day, value in dict(base_calendar.get("day_weights", {})).items()
    }
    return {
        **dict(base_calendar),
        "enabled": bool(enabled),
        "max_abs_adjustment": float(max_abs_adjustment),
        "day_weights": day_weights,
    }


def _resolve_force_off_champion_weight(
    *,
    steo_weight: float,
    prototype_weight: float,
    key: str,
) -> float:
    candidate = 1.0 - float(steo_weight) - float(prototype_weight)
    if candidate <= 0.0:
        raise ContractViolation(
            "invalid_model_policy",
            key=key,
            detail=(
                "force_bsts_off requires champion_weight > 0 after "
                f"subtracting steo/prototype; resolved={candidate:.6f}"
            ),
        )
    if candidate > 1.0:
        raise ContractViolation(
            "invalid_model_policy",
            key=key,
            detail=(
                "force_bsts_off produced champion_weight > 1; "
                f"resolved={candidate:.6f}"
            ),
        )
    return float(candidate)


def _apply_runtime_fusion_constraints(
    *,
    regime_label: str,
    base_champion_weight: float,
    horizon_weights: Mapping[int, float],
    release_anchor_weight: float,
    steo_weight: float,
    prototype_weight: float,
    month_length_bias_weight: float,
    calendar_calibration_cfg: Mapping[str, Any],
    force_bsts_off: bool,
    multi_shock_disable_calendar: bool,
    multi_shock_disable_prototype: bool,
    multi_shock_max_month_length_bias: float | None,
) -> tuple[dict[int, float], float, float, float, float, dict[str, Any]]:
    """Apply optional runtime constraints while preserving deterministic behavior."""

    resolved_horizon = {
        int(key): float(value) for key, value in horizon_weights.items()
    }
    resolved_anchor = float(release_anchor_weight)
    resolved_steo = float(steo_weight)
    resolved_prototype = float(prototype_weight)
    resolved_month_bias = float(month_length_bias_weight)
    resolved_calendar = {
        str(key): value for key, value in dict(calendar_calibration_cfg).items()
    }
    resolved_champion = float(base_champion_weight)

    if str(regime_label) == "multi_shock":
        if multi_shock_disable_prototype:
            resolved_prototype = 0.0
        if multi_shock_disable_calendar:
            resolved_calendar["enabled"] = False
        if multi_shock_max_month_length_bias is not None:
            resolved_month_bias = min(
                resolved_month_bias,
                float(multi_shock_max_month_length_bias),
            )

    if force_bsts_off:
        resolved_champion = _resolve_force_off_champion_weight(
            steo_weight=resolved_steo,
            prototype_weight=resolved_prototype,
            key=f"force_bsts_off.{regime_label}",
        )
        resolved_horizon = _candidate_horizon_weights(
            base_horizon_weights=resolved_horizon,
            base_champion_weight=float(base_champion_weight),
            candidate_champion_weight=resolved_champion,
        )
    if resolved_champion + resolved_steo + resolved_prototype > 1.0:
        raise ContractViolation(
            "invalid_model_policy",
            key=f"fusion_constraints.{regime_label}",
            detail=(
                "champion_weight + steo_weight + prototype_weight must be <= 1 "
                "after applying constraints"
            ),
        )
    return (
        resolved_horizon,
        resolved_anchor,
        resolved_steo,
        resolved_prototype,
        resolved_month_bias,
        resolved_calendar,
    )


def resolve_validation_variants(variants: Sequence[str]) -> list[str]:
    """Resolve explicit variants and alias groups into deterministic variant list."""

    resolved: list[str] = []
    for value in variants:
        token = str(value).strip()
        if not token:
            continue
        alias_values = VARIANT_ALIASES.get(token)
        if alias_values is None:
            resolved.append(token)
            continue
        resolved.extend(alias_values)
    deduped = list(dict.fromkeys(resolved))
    if not deduped:
        raise ContractViolation(
            "invalid_model_policy",
            key="variants",
            detail="at least one model variant is required after alias resolution",
        )
    return deduped


def _score_variant_group(group: pd.DataFrame) -> dict[str, Any]:
    rmse = float(np.sqrt(np.mean(np.square(group["error"].to_numpy(dtype=float)))))
    february_group = group[group["is_february"].astype(bool)]
    return {
        "n_runs": int(len(group)),
        "mae": float(group["abs_error"].mean()),
        "rmse": rmse,
        "mape_pct": float(group["ape_pct"].mean()),
        "mean_signed_error": float(group["error"].mean()),
        "mean_abs_error": float(group["abs_error"].mean()),
        "interval_hit_rate_95": float(group["interval_hit_95"].astype(float).mean()),
        "february_run_count": int(len(february_group)),
        "february_mape_pct": (
            float(february_group["ape_pct"].mean())
            if len(february_group) > 0
            else np.nan
        ),
        "first_target_month": str(group["target_month"].min()),
        "last_target_month": str(group["target_month"].max()),
    }


def _build_trial_scorecard(point_estimates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (variant, trial_id), group in point_estimates.groupby(
        ["model_variant", "trial_id"],
        sort=True,
    ):
        metrics = _score_variant_group(group)
        rows.append(
            {
                "model_variant": str(variant),
                "trial_id": int(trial_id),
                **metrics,
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["model_variant", "trial_id"])
        .reset_index(drop=True)
    )


def _build_scorecard(
    point_estimates: pd.DataFrame,
    *,
    trial_scorecard: pd.DataFrame,
    n_trials: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    score_rows: list[dict[str, Any]] = []
    for variant, group in point_estimates.groupby("model_variant", sort=True):
        score_rows.append(
            {"model_variant": str(variant), **_score_variant_group(group)}
        )
    scorecard = (
        pd.DataFrame(score_rows).sort_values("model_variant").reset_index(drop=True)
    )

    reproducibility = summarize_trial_scorecard(
        trial_scorecard,
        metrics=(
            "mae",
            "rmse",
            "mape_pct",
            "mean_signed_error",
            "mean_abs_error",
            "interval_hit_rate_95",
        ),
        std_tolerance=1.0,
    )
    reproducibility["n_trials_requested"] = int(n_trials)

    merged = scorecard.merge(reproducibility, on="model_variant", how="left")
    for metric in (
        "mae",
        "rmse",
        "mape_pct",
        "mean_signed_error",
        "mean_abs_error",
        "interval_hit_rate_95",
    ):
        mean_col = f"{metric}_mean"
        if mean_col in merged.columns:
            merged[metric] = merged[mean_col]
    if "n_trials" not in merged.columns:
        merged["n_trials"] = 1
    if "stability_flag" not in merged.columns:
        merged["stability_flag"] = True
    return merged, reproducibility


def _build_month_length_diagnostics(point_estimates: pd.DataFrame) -> pd.DataFrame:
    diagnostics_rows: list[dict[str, Any]] = []
    for (variant, day_class), group in point_estimates.groupby(
        ["model_variant", "day_count_class"],
        sort=True,
    ):
        diagnostics_rows.append(
            {
                "model_variant": str(variant),
                "day_count_class": str(day_class),
                "target_month_days": int(group["target_month_days"].iloc[0]),
                "n_runs": int(len(group)),
                "mean_signed_error": float(group["error"].mean()),
                "mean_abs_error": float(group["abs_error"].mean()),
                "mean_ape_pct": float(group["ape_pct"].mean()),
                "is_february_class": bool(group["is_february"].astype(bool).any()),
            }
        )
    return (
        pd.DataFrame(diagnostics_rows)
        .sort_values(["model_variant", "target_month_days"])
        .reset_index(drop=True)
    )


def _build_scorecard_by_regime(point_estimates: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (regime, variant), group in point_estimates.groupby(
        ["regime_label", "model_variant"],
        sort=True,
    ):
        rmse = float(np.sqrt(np.mean(np.square(group["error"].to_numpy(dtype=float)))))
        rows.append(
            {
                "regime_label": str(regime),
                "model_variant": str(variant),
                "n_runs": int(len(group)),
                "mae": float(group["abs_error"].mean()),
                "rmse": rmse,
                "mape_pct": float(group["ape_pct"].mean()),
                "mean_signed_error": float(group["error"].mean()),
                "mean_abs_error": float(group["abs_error"].mean()),
                "interval_hit_rate_95": float(
                    group["interval_hit_95"].astype(float).mean()
                ),
            }
        )
    if not rows:
        return pd.DataFrame(
            columns=[
                "regime_label",
                "model_variant",
                "n_runs",
                "mae",
                "rmse",
                "mape_pct",
                "mean_signed_error",
                "mean_abs_error",
                "interval_hit_rate_95",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["regime_label", "model_variant"])
        .reset_index(drop=True)
    )


def _enforce_unit_scale_sanity(
    point_estimates: pd.DataFrame,
    *,
    ratio_cap: float = 10.0,
) -> None:
    """Fail loudly when forecast and target scales are implausibly mismatched."""

    for variant, group in point_estimates.groupby("model_variant", sort=True):
        abs_forecast_median = float(
            pd.to_numeric(group["fused_point"], errors="coerce").abs().median()
        )
        abs_actual_median = float(
            pd.to_numeric(group["actual_released"], errors="coerce").abs().median()
        )
        if not np.isfinite(abs_forecast_median) or not np.isfinite(abs_actual_median):
            continue
        if abs_forecast_median <= 0 or abs_actual_median <= 0:
            continue
        ratio = max(
            abs_forecast_median / abs_actual_median,
            abs_actual_median / abs_forecast_median,
        )
        if ratio > float(ratio_cap):
            raise ContractViolation(
                "unit_scale_mismatch",
                key=str(variant),
                detail=(
                    "median absolute forecast/actual scale mismatch exceeds cap: "
                    f"ratio={ratio:.4f}, cap={float(ratio_cap):.4f}, "
                    f"median_abs_forecast={abs_forecast_median:.4f}, "
                    f"median_abs_actual={abs_actual_median:.4f}"
                ),
            )


def run_24_month_validation(
    *,
    end_target_month: object,
    runs: int = 24,
    variants: Sequence[str] = DEFAULT_VARIANTS,
    asof_day: int = 14,
    source_catalog_path: str | Path = "configs/sources.yaml",
    report_root: str | Path = "data/reports",
    dump_feature_row: bool = False,
    n_trials: int = 1,
    seed_schedule: Sequence[int] | str | None = None,
    weight_search: bool = False,
    regime_split: bool = False,
    block_importance: bool = False,
    fusion_config_path: str | Path = "configs/fusion.yaml",
    dm_policy_path: str | Path = "configs/evaluation.yaml",
    feature_blocks_path: str | Path = "configs/feature_blocks.yaml",
    ablation_config_path: str | Path = "configs/experiments/nowcast_ablation.yaml",
    fusion_constraints_path: str | Path | None = None,
    extra_block: str | None = None,
    interval_metrics: bool = False,
) -> Validation24mResult:
    """Execute and score rolling nowcast runs against released monthly values."""

    months = build_target_month_grid(end_target_month=end_target_month, runs=int(runs))
    requested_tokens = {str(token).strip() for token in variants if str(token).strip()}
    run_regime_fusion = "full_plus_regime" in requested_tokens
    run_prototype_fusion = "full_plus_prototype" in requested_tokens
    run_ablation_mode = "ablations" in requested_tokens
    capture_feature_rows = bool(
        dump_feature_row or run_ablation_mode or block_importance
    )

    variant_list = resolve_validation_variants(variants)
    trial_seeds = resolve_seed_schedule(
        n_trials=int(n_trials),
        seed_schedule=seed_schedule,
    )

    source_catalog = load_yaml(source_catalog_path)
    release_cfg = source_catalog["release_calendar"]
    lag_months = int(release_cfg["lag_months"])
    release_day = int(release_cfg["release_day_of_month"])
    run_id = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%SZ")

    fusion_policy = load_fusion_policy(str(fusion_config_path))
    fusion_constraints = _load_optional_yaml(fusion_constraints_path)
    force_bsts_off = bool(fusion_constraints.get("force_bsts_off", False))
    multi_shock_disable_calendar = bool(
        fusion_constraints.get("multi_shock_disable_calendar_calibration", False)
    )
    multi_shock_disable_prototype = bool(
        fusion_constraints.get("multi_shock_disable_prototype", False)
    )
    multi_shock_max_month_length_bias = fusion_constraints.get(
        "multi_shock_max_month_length_bias_weight",
        None,
    )
    if multi_shock_max_month_length_bias is not None:
        multi_shock_max_month_length_bias = float(multi_shock_max_month_length_bias)

    dm_policy = load_yaml(str(dm_policy_path))
    regime_thresholds = load_regime_thresholds(config_path="configs/features.yaml")

    point_rows: list[dict[str, Any]] = []
    manifest_rows: list[dict[str, Any]] = []
    feature_dump_rows: list[dict[str, Any]] = []
    weight_eval_rows: list[dict[str, Any]] = []
    anchor_eval_rows: list[dict[str, Any]] = []
    model_run_cache: dict[tuple[str, str], dict[str, Any]] = {}
    allow_trial_reuse = bool(
        int(n_trials) > 1
        and not bool(weight_search)
        and not capture_feature_rows
        and not run_regime_fusion
        and not run_prototype_fusion
        and not run_ablation_mode
        and not block_importance
    )

    for target_month in months:
        asof = derive_policy_admissible_asof(
            target_month=target_month,
            lag_months=lag_months,
            release_day_of_month=release_day,
            preferred_day=int(asof_day),
        )
        policy_context = enforce_release_policy(asof)
        if _month_end(policy_context["target_month"]) != target_month:
            raise ContractViolation(
                "target_month_mismatch",
                key=asof.date().isoformat(),
                detail=(
                    "derived asof does not map to requested target month: "
                    f"requested={target_month.date().isoformat()} "
                    f"resolved={policy_context['target_month']}"
                ),
            )

        inputs = load_market_inputs(asof)
        steo_gold_meta = dict(inputs.get("steo_gold_meta", {}))
        actual_value = _resolve_actual_from_full_history(
            inputs["target_history_full"], target_month=target_month
        )
        regime_snapshot = _resolve_regime_snapshot(
            inputs["features"],
            asof=asof,
            thresholds=regime_thresholds,
        )

        baseline_lookup = {
            item.model_variant: item
            for item in build_baseline_point_estimates(
                inputs["target_history"],
                target_month=target_month,
                feature_rows=inputs.get("features"),
            )
        }

        for variant in sorted(set(variant_list)):
            is_baseline_variant = variant in BASELINE_VARIANTS
            is_challenger_variant = variant in CHALLENGER_VARIANTS
            for trial_id, seed in enumerate(trial_seeds, start=1):
                artifact_dir = Path("data/artifacts/nowcast") / asof.date().isoformat()
                nowcast_path = artifact_dir / "nowcast.json"
                release_history_path = artifact_dir / "release_history_36m.csv"
                fusion_inputs_path = artifact_dir / "fusion_inputs.csv"

                if is_baseline_variant:
                    estimate = baseline_lookup.get(variant)
                    if estimate is None:
                        raise ContractViolation(
                            "missing_column",
                            key=variant,
                            detail="baseline variant is missing from baseline generator output",
                        )
                    fused_point = float(estimate.fused_point)
                    fused_low = float(fused_point)
                    fused_high = float(fused_point)
                    fused_point_pre_calendar = float(fused_point)
                    calendar_calibration_delta = 0.0
                    calendar_calibration_applied = False
                    manifest_rows.append(
                        {
                            "model_variant": variant,
                            "trial_id": int(trial_id),
                            "seed": int(seed),
                            "asof": asof.date().isoformat(),
                            "target_month": target_month.date().isoformat(),
                            "artifact_dir": "",
                            "nowcast_path": "",
                            "release_history_36m_path": "",
                            "steo_gold_status": str(
                                steo_gold_meta.get("status", "missing_gold_panels")
                            ),
                            "steo_gold_vintage_month": str(
                                steo_gold_meta.get("latest_vintage_month", "")
                            ),
                            "mode": "baseline",
                        }
                    )
                else:
                    fusion_override: dict[str, Any] | None = None
                    if run_regime_fusion or run_prototype_fusion:
                        regime_label = str(regime_snapshot["regime_label"])
                        base_champion_weight = float(
                            fusion_policy.get("base", {}).get("champion_weight", 0.70)
                        )
                        if run_regime_fusion:
                            (
                                horizon_weights,
                                anchor_weight,
                                steo_weight,
                                prototype_weight,
                                month_length_bias_weight,
                            ) = resolve_fusion_weights_for_regime_full(
                                fusion_policy,
                                regime_label=regime_label,
                            )
                        else:
                            base_fusion = dict(fusion_policy.get("base", {}))
                            horizon_weights = {
                                int(k): float(v)
                                for k, v in dict(
                                    base_fusion.get("horizon_weights", {})
                                ).items()
                            }
                            anchor_weight = float(
                                base_fusion.get("release_anchor_weight", 0.15)
                            )
                            steo_weight = float(base_fusion.get("steo_weight", 0.0))
                            prototype_weight = float(
                                base_fusion.get("prototype_weight", 0.0)
                            )
                            month_length_bias_weight = float(
                                base_fusion.get("month_length_bias_weight", 0.0)
                            )
                            if not horizon_weights:
                                horizon_weights = {
                                    1: float(base_champion_weight),
                                    2: float(base_champion_weight),
                                }
                        if run_prototype_fusion:
                            requested_prototype = max(float(prototype_weight), 0.08)
                            residual = 0.98 - base_champion_weight - float(steo_weight)
                            prototype_weight = max(
                                0.0, min(requested_prototype, residual)
                            )
                        calendar_calibration_cfg = (
                            resolve_calendar_calibration_for_regime(
                                fusion_policy,
                                regime_label=regime_label,
                            )
                        )
                        (
                            horizon_weights,
                            anchor_weight,
                            steo_weight,
                            prototype_weight,
                            month_length_bias_weight,
                            calendar_calibration_cfg,
                        ) = _apply_runtime_fusion_constraints(
                            regime_label=regime_label,
                            base_champion_weight=base_champion_weight,
                            horizon_weights=horizon_weights,
                            release_anchor_weight=float(anchor_weight),
                            steo_weight=float(steo_weight),
                            prototype_weight=float(prototype_weight),
                            month_length_bias_weight=float(month_length_bias_weight),
                            calendar_calibration_cfg=calendar_calibration_cfg,
                            force_bsts_off=bool(force_bsts_off),
                            multi_shock_disable_calendar=bool(
                                multi_shock_disable_calendar
                            ),
                            multi_shock_disable_prototype=bool(
                                multi_shock_disable_prototype
                            ),
                            multi_shock_max_month_length_bias=(
                                float(multi_shock_max_month_length_bias)
                                if multi_shock_max_month_length_bias is not None
                                else None
                            ),
                        )
                        champion_weight = (
                            _resolve_force_off_champion_weight(
                                steo_weight=float(steo_weight),
                                prototype_weight=float(prototype_weight),
                                key=f"force_bsts_off.{regime_label}",
                            )
                            if bool(force_bsts_off)
                            else float(base_champion_weight)
                        )
                        fusion_override = {
                            "base": {
                                "champion_weight": float(champion_weight),
                                "horizon_weights": {
                                    str(k): float(v)
                                    for k, v in sorted(horizon_weights.items())
                                },
                                "release_anchor_weight": float(anchor_weight),
                                "steo_weight": float(steo_weight),
                                "prototype_weight": float(prototype_weight),
                                "month_length_bias_weight": float(
                                    month_length_bias_weight
                                ),
                            },
                            "calendar_calibration": dict(calendar_calibration_cfg),
                            "regime_overrides": {
                                str(regime_label): {
                                    "horizon_weights": {
                                        str(k): float(v)
                                        for k, v in sorted(horizon_weights.items())
                                    },
                                    "release_anchor_weight": float(anchor_weight),
                                    "steo_weight": float(steo_weight),
                                    "prototype_weight": float(prototype_weight),
                                    "month_length_bias_weight": float(
                                        month_length_bias_weight
                                    ),
                                    "calendar_calibration": dict(
                                        calendar_calibration_cfg
                                    ),
                                }
                            },
                        }
                    cache_key = (target_month.date().isoformat(), variant)
                    cached_model = model_run_cache.get(cache_key)
                    used_cached_trial = bool(
                        allow_trial_reuse and trial_id > 1 and cached_model is not None
                    )
                    if used_cached_trial and cached_model is not None:
                        target_nowcast = dict(cached_model["target_nowcast"])
                        release_history = pd.DataFrame(
                            cached_model["release_history"]
                        ).copy()
                    else:
                        champion_override = (
                            _build_variant_override(
                                str(DEFAULT_VARIANTS[0]),
                                seed=int(seed),
                            )
                            if is_challenger_variant
                            else _build_variant_override(
                                variant,
                                seed=int(seed),
                            )
                        )
                        run_nowcast_pipeline_weekly(
                            asof=asof.date().isoformat(),
                            champion_config_override=champion_override,
                            fusion_config_override=fusion_override,
                            idempotency_token=(
                                "validation24::"
                                f"{run_id}::"
                                f"{target_month.date().isoformat()}::"
                                f"{variant}::"
                                f"trial{trial_id}"
                            ),
                        )
                        if not nowcast_path.exists():
                            raise ContractViolation(
                                "missing_source_file",
                                key=str(nowcast_path),
                                detail="pipeline run did not emit nowcast.json",
                            )
                        target_nowcast = _extract_target_nowcast(
                            nowcast_path,
                            target_month=target_month,
                        )

                        if not release_history_path.exists():
                            raise ContractViolation(
                                "missing_source_file",
                                key=str(release_history_path),
                                detail=(
                                    "release_history_36m.csv is required for validation "
                                    "runs"
                                ),
                            )
                        release_history = pd.read_csv(release_history_path)
                        if len(release_history) < 36:
                            raise ContractViolation(
                                "insufficient_release_history",
                                key=str(release_history_path),
                                detail=f"expected >=36 rows, received={len(release_history)}",
                            )
                        model_run_cache[cache_key] = {
                            "target_nowcast": dict(target_nowcast),
                            "release_history": release_history.copy(),
                        }

                    if is_challenger_variant:
                        (
                            fused_point,
                            fused_low,
                            fused_high,
                        ) = _extract_challenger_target_from_fusion_inputs(
                            fusion_inputs_path
                        )
                    else:
                        fused_point = float(target_nowcast["fused_point"])
                        fused_low = float(target_nowcast["fused_lower_95"])
                        fused_high = float(target_nowcast["fused_upper_95"])
                    fused_point_pre_calendar = float(
                        target_nowcast.get(
                            "fused_point_pre_calendar_calibration",
                            fused_point,
                        )
                    )
                    calendar_calibration_delta = float(
                        target_nowcast.get("calendar_calibration_delta", 0.0)
                    )
                    calendar_calibration_applied = bool(
                        target_nowcast.get("calendar_calibration_applied", False)
                    )

                    if (weight_search or fusion_policy["anchor_ablation"]) and not bool(
                        is_challenger_variant
                    ):
                        (
                            champion_frame,
                            challenger_frame,
                            fusion_inputs_frame,
                            steo_frame,
                            prototype_frame,
                        ) = _load_fusion_inputs(fusion_inputs_path)
                        regime_label = str(regime_snapshot["regime_label"])
                        (
                            base_horizon,
                            _resolved_anchor_weight,
                            resolved_steo_weight,
                            resolved_prototype_weight,
                            resolved_month_length_bias_weight,
                        ) = resolve_fusion_weights_for_regime_full(
                            fusion_policy,
                            regime_label=regime_label,
                        )
                        base_cw_unconstrained = float(
                            fusion_policy["base"]["champion_weight"]
                        )
                        base_calendar = resolve_calendar_calibration_for_regime(
                            fusion_policy,
                            regime_label=regime_label,
                        )
                        (
                            base_horizon,
                            _resolved_anchor_weight,
                            resolved_steo_weight,
                            resolved_prototype_weight,
                            resolved_month_length_bias_weight,
                            base_calendar,
                        ) = _apply_runtime_fusion_constraints(
                            regime_label=regime_label,
                            base_champion_weight=base_cw_unconstrained,
                            horizon_weights=base_horizon,
                            release_anchor_weight=float(_resolved_anchor_weight),
                            steo_weight=float(resolved_steo_weight),
                            prototype_weight=float(resolved_prototype_weight),
                            month_length_bias_weight=float(
                                resolved_month_length_bias_weight
                            ),
                            calendar_calibration_cfg=base_calendar,
                            force_bsts_off=False,
                            multi_shock_disable_calendar=bool(
                                multi_shock_disable_calendar
                            ),
                            multi_shock_disable_prototype=bool(
                                multi_shock_disable_prototype
                            ),
                            multi_shock_max_month_length_bias=(
                                float(multi_shock_max_month_length_bias)
                                if multi_shock_max_month_length_bias is not None
                                else None
                            ),
                        )
                        base_cw = (
                            _resolve_force_off_champion_weight(
                                steo_weight=float(resolved_steo_weight),
                                prototype_weight=float(resolved_prototype_weight),
                                key=f"force_bsts_off.weight_search.{regime_label}",
                            )
                            if bool(force_bsts_off)
                            else float(base_cw_unconstrained)
                        )
                        steo_payload = (
                            steo_frame
                            if isinstance(steo_frame, pd.DataFrame)
                            and not steo_frame.empty
                            else None
                        )
                        prototype_payload = (
                            prototype_frame
                            if isinstance(prototype_frame, pd.DataFrame)
                            and not prototype_frame.empty
                            else None
                        )

                        if weight_search:
                            champion_grid = fusion_policy["weight_search"].get(
                                "champion_weight_grid",
                                [float(base_cw)],
                            )
                            steo_grid = fusion_policy["weight_search"].get(
                                "steo_weight_grid",
                                [float(resolved_steo_weight)],
                            )
                            prototype_grid = fusion_policy["weight_search"].get(
                                "prototype_weight_grid",
                                [float(resolved_prototype_weight)],
                            )
                            enabled_grid = fusion_policy["weight_search"].get(
                                "calendar_calibration_enabled_grid",
                                [int(bool(base_calendar.get("enabled", False)))],
                            )
                            cal_weight_grid = fusion_policy["weight_search"].get(
                                "calendar_calibration_weight_grid",
                                [1.0],
                            )
                            cal_cap_grid = fusion_policy["weight_search"].get(
                                "calendar_calibration_cap_abs_grid",
                                [float(base_calendar.get("max_abs_adjustment", 0.0))],
                            )
                            if regime_label != "transfer_dispersion":
                                prototype_grid = [0.0]
                            if regime_label == "multi_shock" and bool(
                                multi_shock_disable_prototype
                            ):
                                prototype_grid = [0.0]
                            if regime_label == "multi_shock" and bool(
                                multi_shock_disable_calendar
                            ):
                                enabled_grid = [0]
                            for candidate_sw in steo_grid:
                                if steo_payload is None and float(candidate_sw) > 0:
                                    continue
                                for candidate_pw in prototype_grid:
                                    if (
                                        prototype_payload is None
                                        and float(candidate_pw) > 0
                                    ):
                                        continue
                                    candidate_cw_values = (
                                        [
                                            _resolve_force_off_champion_weight(
                                                steo_weight=float(candidate_sw),
                                                prototype_weight=float(candidate_pw),
                                                key=(
                                                    "force_bsts_off.weight_search"
                                                    f".{regime_label}"
                                                ),
                                            )
                                        ]
                                        if bool(force_bsts_off)
                                        else champion_grid
                                    )
                                    for candidate_cw in candidate_cw_values:
                                        weight_sum = (
                                            float(candidate_cw)
                                            + float(candidate_sw)
                                            + float(candidate_pw)
                                        )
                                        if (
                                            bool(force_bsts_off)
                                            and abs(weight_sum - 1.0) > 1e-9
                                        ):
                                            continue
                                        if (not bool(force_bsts_off)) and (
                                            weight_sum > 1.0
                                        ):
                                            continue
                                        candidate_hw = _candidate_horizon_weights(
                                            base_horizon_weights=base_horizon,
                                            base_champion_weight=base_cw,
                                            candidate_champion_weight=float(
                                                candidate_cw
                                            ),
                                        )
                                        for candidate_aw in fusion_policy[
                                            "weight_search"
                                        ]["release_anchor_weight_grid"]:
                                            for candidate_cal_enabled in enabled_grid:
                                                for (
                                                    candidate_cal_weight
                                                ) in cal_weight_grid:
                                                    for (
                                                        candidate_cal_cap
                                                    ) in cal_cap_grid:
                                                        candidate_calendar = _resolve_calendar_search_candidate(
                                                            base_calendar,
                                                            enabled=bool(
                                                                int(
                                                                    candidate_cal_enabled
                                                                )
                                                            ),
                                                            weight=float(
                                                                candidate_cal_weight
                                                            ),
                                                            max_abs_adjustment=float(
                                                                candidate_cal_cap
                                                            ),
                                                        )
                                                        if (
                                                            regime_label
                                                            == "multi_shock"
                                                            and bool(
                                                                multi_shock_disable_calendar
                                                            )
                                                        ):
                                                            candidate_calendar[
                                                                "enabled"
                                                            ] = False
                                                        candidate_fused = fuse_forecasts(
                                                            champion_frame,
                                                            challenger_frame,
                                                            champion_weight=float(
                                                                candidate_cw
                                                            ),
                                                            release_history=release_history,
                                                            release_anchor_weight=float(
                                                                candidate_aw
                                                            ),
                                                            steo_forecast=steo_payload,
                                                            steo_weight=float(
                                                                candidate_sw
                                                            ),
                                                            prototype_forecast=prototype_payload,
                                                            prototype_weight=float(
                                                                candidate_pw
                                                            ),
                                                            month_length_bias_weight=float(
                                                                resolved_month_length_bias_weight
                                                            ),
                                                            calendar_calibration=candidate_calendar,
                                                            horizon_weights=candidate_hw,
                                                            regime_label=regime_label,
                                                        )
                                                        target_candidate = (
                                                            candidate_fused[
                                                                candidate_fused[
                                                                    "horizon"
                                                                ].astype(int)
                                                                == 1
                                                            ]
                                                        )
                                                        if target_candidate.empty:
                                                            continue
                                                        predicted = float(
                                                            target_candidate.iloc[0][
                                                                "fused_point"
                                                            ]
                                                        )
                                                        error = predicted - actual_value
                                                        ape = (
                                                            abs(error)
                                                            / abs(actual_value)
                                                            * 100.0
                                                            if actual_value != 0
                                                            else np.nan
                                                        )
                                                        weight_eval_rows.append(
                                                            {
                                                                "model_variant": variant,
                                                                "trial_id": int(
                                                                    trial_id
                                                                ),
                                                                "seed": int(seed),
                                                                "asof": asof.date().isoformat(),
                                                                "target_month": target_month.date().isoformat(),
                                                                "regime_label": regime_label,
                                                                "champion_weight": float(
                                                                    candidate_cw
                                                                ),
                                                                "steo_weight": float(
                                                                    candidate_sw
                                                                ),
                                                                "prototype_weight": float(
                                                                    candidate_pw
                                                                ),
                                                                "bsts_weight": float(
                                                                    1.0 - weight_sum
                                                                ),
                                                                "release_anchor_weight": float(
                                                                    candidate_aw
                                                                ),
                                                                "calendar_calibration_enabled": int(
                                                                    bool(
                                                                        candidate_calendar.get(
                                                                            "enabled",
                                                                            False,
                                                                        )
                                                                    )
                                                                ),
                                                                "calendar_calibration_weight": float(
                                                                    candidate_cal_weight
                                                                ),
                                                                "calendar_calibration_cap_abs": float(
                                                                    candidate_cal_cap
                                                                ),
                                                                "month_length_bias_weight": float(
                                                                    resolved_month_length_bias_weight
                                                                ),
                                                                "predicted": predicted,
                                                                "actual_released": float(
                                                                    actual_value
                                                                ),
                                                                "error": float(error),
                                                                "abs_error": float(
                                                                    abs(error)
                                                                ),
                                                                "ape_pct": (
                                                                    float(ape)
                                                                    if np.isfinite(ape)
                                                                    else np.nan
                                                                ),
                                                            }
                                                        )

                        anchor_applied: dict[int, float] = {}
                        if "applied_champion_weight" in fusion_inputs_frame.columns:
                            anchor_applied = {
                                int(row["horizon"]): float(
                                    row["applied_champion_weight"]
                                )
                                for _, row in fusion_inputs_frame.iterrows()
                            }
                        if bool(force_bsts_off):
                            anchor_applied = dict(base_horizon)
                        if not anchor_applied:
                            anchor_applied = base_horizon
                        for candidate_aw in fusion_policy["anchor_ablation"][
                            "release_anchor_weight_grid"
                        ]:
                            candidate_fused = fuse_forecasts(
                                champion_frame,
                                challenger_frame,
                                champion_weight=float(base_cw),
                                release_history=release_history,
                                release_anchor_weight=float(candidate_aw),
                                steo_forecast=steo_payload,
                                steo_weight=float(resolved_steo_weight),
                                prototype_forecast=prototype_payload,
                                prototype_weight=float(resolved_prototype_weight),
                                month_length_bias_weight=float(
                                    resolved_month_length_bias_weight
                                ),
                                calendar_calibration=base_calendar,
                                horizon_weights=anchor_applied,
                                regime_label=regime_label,
                            )
                            target_candidate = candidate_fused[
                                candidate_fused["horizon"].astype(int) == 1
                            ]
                            if target_candidate.empty:
                                continue
                            predicted = float(target_candidate.iloc[0]["fused_point"])
                            error = predicted - actual_value
                            ape = (
                                abs(error) / abs(actual_value) * 100.0
                                if actual_value != 0
                                else np.nan
                            )
                            anchor_eval_rows.append(
                                {
                                    "model_variant": variant,
                                    "trial_id": int(trial_id),
                                    "seed": int(seed),
                                    "asof": asof.date().isoformat(),
                                    "target_month": target_month.date().isoformat(),
                                    "regime_label": regime_label,
                                    "release_anchor_weight": float(candidate_aw),
                                    "predicted": predicted,
                                    "actual_released": float(actual_value),
                                    "error": float(error),
                                    "abs_error": float(abs(error)),
                                    "ape_pct": (
                                        float(ape) if np.isfinite(ape) else np.nan
                                    ),
                                }
                            )

                    manifest_rows.append(
                        {
                            "model_variant": variant,
                            "trial_id": int(trial_id),
                            "seed": int(seed),
                            "asof": asof.date().isoformat(),
                            "target_month": target_month.date().isoformat(),
                            "artifact_dir": str(artifact_dir),
                            "nowcast_path": str(nowcast_path),
                            "release_history_36m_path": str(release_history_path),
                            "steo_gold_status": str(
                                steo_gold_meta.get("status", "missing_gold_panels")
                            ),
                            "steo_gold_vintage_month": str(
                                steo_gold_meta.get("latest_vintage_month", "")
                            ),
                            "mode": (
                                "model_challenger_cached"
                                if bool(is_challenger_variant and used_cached_trial)
                                else (
                                    "model_challenger"
                                    if bool(is_challenger_variant)
                                    else (
                                        "model_cached" if used_cached_trial else "model"
                                    )
                                )
                            ),
                        }
                    )

                    if capture_feature_rows:
                        feature_row = _extract_feature_row(
                            artifact_dir / "feature_matrix.csv",
                            target_month=target_month,
                        )
                        feature_dump_rows.append(
                            {
                                "model_variant": variant,
                                "trial_id": int(trial_id),
                                "seed": int(seed),
                                "asof": asof.date().isoformat(),
                                "target_month": target_month.date().isoformat(),
                                **feature_row,
                            }
                        )

                error = fused_point - actual_value
                error_raw = fused_point_pre_calendar - actual_value
                ape = (
                    (abs(error) / abs(actual_value) * 100.0)
                    if actual_value != 0
                    else np.nan
                )
                ape_raw = (
                    (abs(error_raw) / abs(actual_value) * 100.0)
                    if actual_value != 0
                    else np.nan
                )
                interval_hit = float(fused_low <= actual_value <= fused_high)
                month_days = int(target_month.days_in_month)
                target_month_name = str(target_month.strftime("%B"))
                day_count_class = _day_count_class(month_days)
                is_february = bool(int(target_month.month) == 2)
                monthly_release_history = inputs.get("monthly_release_history")
                if isinstance(monthly_release_history, pd.DataFrame):
                    release_history_rows = int(len(monthly_release_history))
                    if (
                        release_history_rows > 0
                        and "target_value" in monthly_release_history.columns
                    ):
                        release_series = pd.to_numeric(
                            monthly_release_history["target_value"],
                            errors="coerce",
                        ).dropna()
                        if release_series.empty:
                            release_history_last = np.nan
                        else:
                            release_history_last = float(release_series.iloc[-1])
                    else:
                        release_history_last = np.nan
                else:
                    release_history_rows = 0
                    release_history_last = np.nan

                point_rows.append(
                    {
                        "model_variant": variant,
                        "trial_id": int(trial_id),
                        "seed": int(seed),
                        "asof": asof.date().isoformat(),
                        "target_month": target_month.date().isoformat(),
                        "target_month_name": target_month_name,
                        "target_month_days": month_days,
                        "day_count_class": day_count_class,
                        "is_february": is_february,
                        "fused_point": float(fused_point),
                        "fused_point_pre_calendar_calibration": float(
                            fused_point_pre_calendar
                        ),
                        "calendar_calibration_delta": float(calendar_calibration_delta),
                        "calendar_calibration_applied": bool(
                            calendar_calibration_applied
                        ),
                        "fused_lower_95": float(fused_low),
                        "fused_upper_95": float(fused_high),
                        "actual_released": float(actual_value),
                        "error": float(error),
                        "error_raw": float(error_raw),
                        "abs_error": float(abs(error)),
                        "abs_error_raw": float(abs(error_raw)),
                        "ape_pct": float(ape) if np.isfinite(ape) else np.nan,
                        "ape_pct_raw": (
                            float(ape_raw) if np.isfinite(ape_raw) else np.nan
                        ),
                        "interval_hit_95": bool(interval_hit),
                        "release_history_rows": int(release_history_rows),
                        "release_history_last": (
                            float(release_history_last)
                            if np.isfinite(release_history_last)
                            else np.nan
                        ),
                        "steo_gold_status": str(
                            steo_gold_meta.get("status", "missing_gold_panels")
                        ),
                        "steo_gold_vintage_month": str(
                            steo_gold_meta.get("latest_vintage_month", "")
                        ),
                        "steo_gold_feature_rows": int(
                            steo_gold_meta.get("feature_row_count", 0)
                        ),
                        "steo_gold_root": str(steo_gold_meta.get("gold_root", "")),
                        "regime_label": str(regime_snapshot["regime_label"]),
                        "regime_freeze_flag": float(
                            regime_snapshot["regime_freeze_flag"]
                        ),
                        "regime_basis_flag": float(
                            regime_snapshot["regime_basis_flag"]
                        ),
                        "regime_transfer_dispersion_flag": float(
                            regime_snapshot["regime_transfer_dispersion_flag"]
                        ),
                        "regime_any_flag": float(regime_snapshot["regime_any_flag"]),
                        "regime_score": float(regime_snapshot["regime_score"]),
                        "forecast_source": (
                            "challenger"
                            if bool(is_challenger_variant)
                            else ("baseline" if bool(is_baseline_variant) else "fused")
                        ),
                        "ablation_mode": "none",
                        "ablation_block_id": "",
                    }
                )

    point_estimates = (
        pd.DataFrame(point_rows)
        .sort_values(["model_variant", "trial_id", "target_month", "asof"])
        .reset_index(drop=True)
    )
    if point_estimates.empty:
        raise ContractViolation(
            "missing_target_history_rows",
            key="validation_24m_point_estimates",
            detail="validation run produced no scored rows",
        )

    _enforce_unit_scale_sanity(point_estimates, ratio_cap=10.0)

    trial_scorecard = _build_trial_scorecard(point_estimates)
    scorecard, reproducibility = _build_scorecard(
        point_estimates,
        trial_scorecard=trial_scorecard,
        n_trials=int(n_trials),
    )
    month_length_diagnostics = _build_month_length_diagnostics(point_estimates)
    month_length_effect = build_month_length_effect_report(point_estimates)
    month_length_by_regime = build_month_length_by_regime_report(point_estimates)
    calendar_calibration_report = build_calendar_calibration_report(point_estimates)

    report_dir = Path(report_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    regime_counts = (
        point_estimates.groupby(["regime_label"], sort=True)
        .size()
        .rename("n_rows")
        .reset_index()
        .sort_values("regime_label")
        .reset_index(drop=True)
    )
    regime_counts["share_pct"] = (
        regime_counts["n_rows"].astype(float) / float(len(point_estimates)) * 100.0
    )
    regime_counts_path_obj = report_dir / "regime_counts.csv"
    regime_counts.to_csv(regime_counts_path_obj, index=False)
    regime_counts_path = str(regime_counts_path_obj)
    if (regime_split or run_regime_fusion) and int(
        regime_counts["regime_label"].nunique()
    ) < 2:
        raise ContractViolation(
            "regimes_not_informative",
            key="regime_label",
            detail=(
                "expected at least 2 regimes when regime_split/regime-aware mode is enabled; "
                f"observed={int(regime_counts['regime_label'].nunique())}; "
                f"counts_path={regime_counts_path}"
            ),
        )

    point_path = report_dir / "validation_24m_point_estimates.csv"
    scorecard_path = report_dir / "validation_24m_scorecard.csv"
    month_length_diag_path = report_dir / "validation_24m_month_length_diagnostics.csv"
    month_length_effect_path = report_dir / "validation_24m_month_length_effect.csv"
    month_length_by_regime_path = (
        report_dir / "validation_24m_month_length_by_regime.csv"
    )
    calendar_calibration_path = report_dir / "validation_24m_calendar_calibration.csv"
    summary_path = report_dir / "validation_24m_summary.json"
    feature_row_path = report_dir / "validation_24m_feature_rows.csv"
    reproducibility_path = report_dir / "reproducibility_trials_summary.csv"

    point_estimates.to_csv(point_path, index=False)
    scorecard.to_csv(scorecard_path, index=False)
    month_length_diagnostics.to_csv(month_length_diag_path, index=False)
    month_length_effect.to_csv(month_length_effect_path, index=False)
    month_length_by_regime.to_csv(month_length_by_regime_path, index=False)
    calendar_calibration_report.to_csv(calendar_calibration_path, index=False)
    reproducibility.to_csv(reproducibility_path, index=False)

    feature_rows_path_value = ""
    if capture_feature_rows and feature_dump_rows:
        pd.DataFrame(feature_dump_rows).to_csv(feature_row_path, index=False)
        feature_rows_path_value = str(feature_row_path)

    baselines_point_path = ""
    baselines_scorecard_path = ""
    if set(point_estimates["model_variant"].unique().tolist()).issubset(
        set(BASELINE_VARIANTS)
    ):
        baseline_point_path = report_dir / "baselines_24m_point_estimates.csv"
        baseline_scorecard_path = report_dir / "baselines_24m_scorecard.csv"
        point_estimates.to_csv(baseline_point_path, index=False)
        scorecard.to_csv(baseline_scorecard_path, index=False)
        baselines_point_path = str(baseline_point_path)
        baselines_scorecard_path = str(baseline_scorecard_path)

    scorecard_by_regime_path = ""
    dm_by_regime_path = ""
    if regime_split or run_regime_fusion:
        scorecard_by_regime = _build_scorecard_by_regime(point_estimates)
        scorecard_by_regime_path_obj = report_dir / "scorecard_by_regime.csv"
        scorecard_by_regime.to_csv(scorecard_by_regime_path_obj, index=False)
        scorecard_by_regime_path = str(scorecard_by_regime_path_obj)

        dm_by_regime = run_dm_tests_by_regime(
            point_estimates,
            dm_policy,
            regime_col="regime_label",
            model_col="model_variant",
            actual_col="actual_released",
            forecast_col="fused_point",
            asof_col="asof",
            target_col="target_month",
            min_observations=3,
        )
        dm_by_regime_path_obj = report_dir / "dm_by_regime.csv"
        dm_by_regime.to_csv(dm_by_regime_path_obj, index=False)
        dm_by_regime_path = str(dm_by_regime_path_obj)

    interval_scorecard_path = ""
    interval_scorecard_by_regime_path = ""
    interval_calibration_table_path = ""
    interval_calibration_by_regime_path = ""
    if interval_metrics:
        interval_scorecard = build_interval_scorecard(point_estimates)
        interval_scorecard_obj = report_dir / "interval_scorecard.csv"
        interval_scorecard.to_csv(interval_scorecard_obj, index=False)
        interval_scorecard_path = str(interval_scorecard_obj)

        interval_by_regime = build_interval_scorecard_by_regime(point_estimates)
        interval_by_regime_obj = report_dir / "interval_scorecard_by_regime.csv"
        interval_by_regime.to_csv(interval_by_regime_obj, index=False)
        interval_scorecard_by_regime_path = str(interval_by_regime_obj)

        calibration_overall, calibration_by_regime = build_interval_calibration_tables(
            point_estimates
        )
        calibration_table_obj = report_dir / "interval_calibration_table.csv"
        calibration_overall.to_csv(calibration_table_obj, index=False)
        interval_calibration_table_path = str(calibration_table_obj)

        calibration_by_regime_obj = report_dir / "interval_calibration_by_regime.csv"
        calibration_by_regime.to_csv(calibration_by_regime_obj, index=False)
        interval_calibration_by_regime_path = str(calibration_by_regime_obj)

    fusion_weight_search_path = ""
    fusion_weight_selected_path = ""
    fusion_weight_search_v7_path = ""
    fusion_weight_selected_v7_path = ""
    calendar_policy_selected_by_regime_path = ""
    if weight_search and weight_eval_rows:
        weight_eval = pd.DataFrame(weight_eval_rows)
        if "regime_label" not in weight_eval.columns:
            weight_eval["regime_label"] = "normal"
        weight_eval["regime_label"] = weight_eval["regime_label"].astype(str)
        if "calendar_calibration_enabled" not in weight_eval.columns:
            weight_eval["calendar_calibration_enabled"] = 0
        if "calendar_calibration_weight" not in weight_eval.columns:
            weight_eval["calendar_calibration_weight"] = 1.0
        if "calendar_calibration_cap_abs" not in weight_eval.columns:
            weight_eval["calendar_calibration_cap_abs"] = 0.0
        search_summary = (
            weight_eval.groupby(
                [
                    "model_variant",
                    "champion_weight",
                    "steo_weight",
                    "prototype_weight",
                    "bsts_weight",
                    "release_anchor_weight",
                    "calendar_calibration_enabled",
                    "calendar_calibration_weight",
                    "calendar_calibration_cap_abs",
                ],
                sort=True,
            )
            .agg(
                n_runs=("target_month", "count"),
                mae=("abs_error", "mean"),
                rmse=(
                    "error",
                    lambda values: float(np.sqrt(np.mean(np.square(values)))),
                ),
                mape_pct=("ape_pct", "mean"),
            )
            .reset_index()
            .sort_values(
                [
                    "model_variant",
                    "mape_pct",
                    "mae",
                    "champion_weight",
                    "steo_weight",
                    "prototype_weight",
                    "release_anchor_weight",
                    "calendar_calibration_enabled",
                    "calendar_calibration_weight",
                    "calendar_calibration_cap_abs",
                ]
            )
            .reset_index(drop=True)
        )
        fusion_weight_search_path_obj = report_dir / "fusion_weight_search.csv"
        search_summary.to_csv(fusion_weight_search_path_obj, index=False)
        fusion_weight_search_path = str(fusion_weight_search_path_obj)

        selected_payload: dict[str, Any] = {
            "selected": {},
            "selection_metric": "mape_pct",
            "evaluated_rows": int(len(search_summary)),
        }
        for variant, group in search_summary.groupby("model_variant", sort=True):
            best = group.sort_values(["mape_pct", "mae", "champion_weight"]).iloc[0]
            selected_payload["selected"][str(variant)] = {
                "champion_weight": float(best["champion_weight"]),
                "steo_weight": float(best["steo_weight"]),
                "prototype_weight": float(best["prototype_weight"]),
                "bsts_weight": float(best["bsts_weight"]),
                "release_anchor_weight": float(best["release_anchor_weight"]),
                "calendar_calibration_enabled": int(
                    best["calendar_calibration_enabled"]
                ),
                "calendar_calibration_weight": float(
                    best["calendar_calibration_weight"]
                ),
                "calendar_calibration_cap_abs": float(
                    best["calendar_calibration_cap_abs"]
                ),
                "mape_pct": float(best["mape_pct"]),
                "mae": float(best["mae"]),
                "rmse": float(best["rmse"]),
                "n_runs": int(best["n_runs"]),
            }
        fusion_weight_selected_path_obj = report_dir / "fusion_weight_selected.json"
        fusion_weight_selected_path_obj.write_text(
            json.dumps(selected_payload, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        fusion_weight_selected_path = str(fusion_weight_selected_path_obj)

        search_summary_v7 = (
            weight_eval.groupby(
                [
                    "model_variant",
                    "regime_label",
                    "champion_weight",
                    "steo_weight",
                    "prototype_weight",
                    "bsts_weight",
                    "release_anchor_weight",
                    "calendar_calibration_enabled",
                    "calendar_calibration_weight",
                    "calendar_calibration_cap_abs",
                ],
                sort=True,
            )
            .agg(
                n_runs=("target_month", "count"),
                mae=("abs_error", "mean"),
                rmse=(
                    "error",
                    lambda values: float(np.sqrt(np.mean(np.square(values)))),
                ),
                mape_pct=("ape_pct", "mean"),
            )
            .reset_index()
            .sort_values(
                [
                    "model_variant",
                    "regime_label",
                    "mape_pct",
                    "mae",
                    "champion_weight",
                    "steo_weight",
                    "prototype_weight",
                    "calendar_calibration_enabled",
                    "calendar_calibration_weight",
                    "calendar_calibration_cap_abs",
                ]
            )
            .reset_index(drop=True)
        )
        fusion_weight_search_v7_path_obj = report_dir / "fusion_weight_search_v7.csv"
        search_summary_v7.to_csv(fusion_weight_search_v7_path_obj, index=False)
        fusion_weight_search_v7_path = str(fusion_weight_search_v7_path_obj)

        selected_payload_v7: dict[str, Any] = {
            "selected_by_variant_regime": {},
            "selection_metric": "mape_pct",
            "evaluated_rows": int(len(search_summary_v7)),
            "regimes": sorted(
                search_summary_v7["regime_label"].astype(str).dropna().unique().tolist()
            ),
        }
        for (variant, regime_label), group in search_summary_v7.groupby(
            ["model_variant", "regime_label"],
            sort=True,
        ):
            best = group.sort_values(["mape_pct", "mae", "champion_weight"]).iloc[0]
            variant_key = str(variant)
            regime_key = str(regime_label)
            if variant_key not in selected_payload_v7["selected_by_variant_regime"]:
                selected_payload_v7["selected_by_variant_regime"][variant_key] = {}
            selected_payload_v7["selected_by_variant_regime"][variant_key][
                regime_key
            ] = {
                "champion_weight": float(best["champion_weight"]),
                "steo_weight": float(best["steo_weight"]),
                "prototype_weight": float(best["prototype_weight"]),
                "bsts_weight": float(best["bsts_weight"]),
                "release_anchor_weight": float(best["release_anchor_weight"]),
                "calendar_calibration_enabled": int(
                    best["calendar_calibration_enabled"]
                ),
                "calendar_calibration_weight": float(
                    best["calendar_calibration_weight"]
                ),
                "calendar_calibration_cap_abs": float(
                    best["calendar_calibration_cap_abs"]
                ),
                "mape_pct": float(best["mape_pct"]),
                "mae": float(best["mae"]),
                "rmse": float(best["rmse"]),
                "n_runs": int(best["n_runs"]),
            }
        fusion_weight_selected_v7_path_obj = (
            report_dir / "fusion_weight_selected_v7.json"
        )
        fusion_weight_selected_v7_path_obj.write_text(
            json.dumps(selected_payload_v7, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        fusion_weight_selected_v7_path = str(fusion_weight_selected_v7_path_obj)

        calendar_by_regime_rows: list[dict[str, Any]] = []
        for variant, by_regime in selected_payload_v7[
            "selected_by_variant_regime"
        ].items():
            for regime_label, payload in by_regime.items():
                calendar_by_regime_rows.append(
                    {
                        "model_variant": str(variant),
                        "regime_label": str(regime_label),
                        "calendar_calibration_enabled": int(
                            payload.get("calendar_calibration_enabled", 0)
                        ),
                        "calendar_calibration_weight": float(
                            payload.get("calendar_calibration_weight", 1.0)
                        ),
                        "calendar_calibration_cap_abs": float(
                            payload.get("calendar_calibration_cap_abs", 0.0)
                        ),
                    }
                )
        calendar_policy_selected_by_regime = pd.DataFrame(calendar_by_regime_rows)
        if not calendar_policy_selected_by_regime.empty:
            calendar_policy_selected_by_regime = (
                calendar_policy_selected_by_regime.sort_values(
                    ["model_variant", "regime_label"]
                ).reset_index(drop=True)
            )
            calendar_policy_selected_by_regime_obj = (
                report_dir / "calendar_policy_selected_by_regime.csv"
            )
            calendar_policy_selected_by_regime.to_csv(
                calendar_policy_selected_by_regime_obj,
                index=False,
            )
            calendar_policy_selected_by_regime_path = str(
                calendar_policy_selected_by_regime_obj
            )

    anchor_blend_ablation_path = ""
    if anchor_eval_rows:
        anchor_eval = pd.DataFrame(anchor_eval_rows)
        anchor_summary = (
            anchor_eval.groupby(["model_variant", "release_anchor_weight"], sort=True)
            .agg(
                n_runs=("target_month", "count"),
                mae=("abs_error", "mean"),
                rmse=(
                    "error",
                    lambda values: float(np.sqrt(np.mean(np.square(values)))),
                ),
                mape_pct=("ape_pct", "mean"),
            )
            .reset_index()
            .sort_values(["model_variant", "release_anchor_weight"])
            .reset_index(drop=True)
        )
        anchor_path = report_dir / "anchor_blend_ablation.csv"
        anchor_summary.to_csv(anchor_path, index=False)
        anchor_blend_ablation_path = str(anchor_path)

    ablation_scorecard_path = ""
    block_importance_path = ""
    ablation_by_regime_scorecard_path = ""
    multi_shock_attribution_path = ""
    if run_ablation_mode or block_importance:
        if not feature_dump_rows:
            raise ContractViolation(
                "missing_column",
                key="feature_rows",
                detail="ablation scoring requires feature rows; enable dump_feature_row",
            )
        feature_rows = pd.DataFrame(feature_dump_rows)
        ablation_cfg = load_yaml(str(ablation_config_path))
        block_cfg = dict(ablation_cfg.get("block_ablation", {}))
        block_ids = [str(value) for value in block_cfg.get("block_ids", [])]
        extra_block_id = str(extra_block or "").strip()
        if extra_block_id:
            alias_map = {
                "weather_freezeoff_enriched": "weather_freezeoff",
            }
            block_ids = [alias_map.get(extra_block_id, extra_block_id)]
        baseline_variant = str(block_cfg.get("baseline_variant", "")).strip() or None
        block_registry = load_yaml(feature_blocks_path)
        blocks_cfg = block_registry.get("blocks", {})
        if not isinstance(blocks_cfg, Mapping):
            raise ContractViolation(
                "invalid_model_policy",
                key="feature_blocks",
                detail="feature block registry must define a blocks mapping",
            )
        block_feature_map = {
            str(block_id): [str(feature) for feature in cfg.get("features", [])]
            for block_id, cfg in blocks_cfg.items()
            if isinstance(cfg, Mapping) and bool(cfg.get("enabled", True))
        }
        if block_ids:
            block_feature_map = {
                block_id: features
                for block_id, features in block_feature_map.items()
                if block_id in set(block_ids)
            }
        ablation_forecasts = build_block_ablation_forecasts(
            point_estimates,
            feature_rows,
            block_feature_map=block_feature_map,
            baseline_variant=baseline_variant,
        )
        ablation_scorecard, block_importance_frame = score_block_ablations(
            ablation_forecasts,
            dm_policy=dm_policy,
        )
        ablation_scorecard_path_obj = export_ablation_scorecard(
            ablation_scorecard,
            report_dir,
            filename="ablation_scorecard.csv",
        )
        ablation_scorecard_path = str(ablation_scorecard_path_obj)

        block_importance_path_obj = report_dir / "block_importance_24m.csv"
        block_importance_frame.to_csv(block_importance_path_obj, index=False)
        block_importance_path = str(block_importance_path_obj)

        regime_lookup = point_estimates[
            ["asof", "target_month", "regime_label"]
        ].drop_duplicates(subset=["asof", "target_month", "regime_label"])
        merge_keys = (
            ["asof"]
            if "target_month" not in ablation_forecasts.columns
            else ["asof", "target_month"]
        )
        ablation_with_regime = ablation_forecasts.merge(
            regime_lookup,
            on=merge_keys,
            how="left",
        )
        ablation_with_regime["regime_label"] = (
            ablation_with_regime["regime_label"].fillna("unknown").astype(str)
        )
        ablation_with_regime["error"] = pd.to_numeric(
            ablation_with_regime["forecast"], errors="coerce"
        ) - pd.to_numeric(ablation_with_regime["actual"], errors="coerce")
        ablation_with_regime["abs_error"] = ablation_with_regime["error"].abs()
        ablation_with_regime["ape_pct"] = np.where(
            pd.to_numeric(ablation_with_regime["actual"], errors="coerce").abs() > 0,
            ablation_with_regime["abs_error"]
            / pd.to_numeric(ablation_with_regime["actual"], errors="coerce").abs()
            * 100.0,
            np.nan,
        )
        ablation_by_regime = (
            ablation_with_regime.groupby(
                ["regime_label", "experiment_id", "block_id", "ablation_mode"],
                sort=True,
            )
            .agg(
                n_runs=("target_month", "count"),
                mae=("abs_error", "mean"),
                rmse=(
                    "error",
                    lambda values: float(np.sqrt(np.mean(np.square(values)))),
                ),
                mape_pct=("ape_pct", "mean"),
            )
            .reset_index()
            .sort_values(["regime_label", "mape_pct", "mae", "experiment_id"])
            .reset_index(drop=True)
        )
        ablation_by_regime_obj = report_dir / "ablation_by_regime_scorecard.csv"
        ablation_by_regime.to_csv(ablation_by_regime_obj, index=False)
        ablation_by_regime_scorecard_path = str(ablation_by_regime_obj)

        multi_shock_rows: list[dict[str, Any]] = []
        multi_shock_scorecard = ablation_by_regime[
            ablation_by_regime["regime_label"] == "multi_shock"
        ].copy()
        baseline_multi = multi_shock_scorecard[
            multi_shock_scorecard["experiment_id"] == "baseline_full"
        ]
        if not baseline_multi.empty:
            baseline_ref = baseline_multi.iloc[0]
            baseline_mae = float(baseline_ref["mae"])
            baseline_rmse = float(baseline_ref["rmse"])
            baseline_mape = float(baseline_ref["mape_pct"])
            for _, row in multi_shock_scorecard.iterrows():
                if str(row["experiment_id"]) == "baseline_full":
                    continue
                multi_shock_rows.append(
                    {
                        "regime_label": "multi_shock",
                        "experiment_id": str(row["experiment_id"]),
                        "block_id": str(row["block_id"]),
                        "ablation_mode": str(row["ablation_mode"]),
                        "n_runs": int(row["n_runs"]),
                        "mae": float(row["mae"]),
                        "rmse": float(row["rmse"]),
                        "mape_pct": float(row["mape_pct"]),
                        "delta_mae_vs_baseline": float(row["mae"]) - baseline_mae,
                        "delta_rmse_vs_baseline": float(row["rmse"]) - baseline_rmse,
                        "delta_mape_pct_vs_baseline": float(row["mape_pct"])
                        - baseline_mape,
                    }
                )
        multi_shock_attribution = (
            pd.DataFrame(multi_shock_rows)
            if multi_shock_rows
            else pd.DataFrame(
                columns=[
                    "regime_label",
                    "experiment_id",
                    "block_id",
                    "ablation_mode",
                    "n_runs",
                    "mae",
                    "rmse",
                    "mape_pct",
                    "delta_mae_vs_baseline",
                    "delta_rmse_vs_baseline",
                    "delta_mape_pct_vs_baseline",
                ]
            )
        )
        if not multi_shock_attribution.empty:
            multi_shock_attribution = multi_shock_attribution.sort_values(
                "delta_mape_pct_vs_baseline",
                ascending=False,
            ).reset_index(drop=True)
        multi_shock_obj = report_dir / "multi_shock_attribution.csv"
        multi_shock_attribution.to_csv(multi_shock_obj, index=False)
        multi_shock_attribution_path = str(multi_shock_obj)

    summary = {
        "run_id": run_id,
        "target_month_start": months[0].date().isoformat(),
        "target_month_end": months[-1].date().isoformat(),
        "asof_start": point_estimates["asof"].min(),
        "asof_end": point_estimates["asof"].max(),
        "asof_schedule": sorted(point_estimates["asof"].astype(str).unique().tolist()),
        "end_target_month": _month_end(end_target_month).date().isoformat(),
        "runs": int(runs),
        "variants": variant_list,
        "n_trials": int(n_trials),
        "seed_schedule": [int(seed) for seed in trial_seeds],
        "point_estimates_path": str(point_path),
        "scorecard_path": str(scorecard_path),
        "month_length_diagnostics_path": str(month_length_diag_path),
        "month_length_effect_path": str(month_length_effect_path),
        "month_length_by_regime_path": str(month_length_by_regime_path),
        "calendar_calibration_path": str(calendar_calibration_path),
        "feature_rows_path": feature_rows_path_value,
        "dump_feature_row": bool(capture_feature_rows),
        "reproducibility_trials_summary_path": str(reproducibility_path),
        "baselines_point_estimates_path": baselines_point_path,
        "baselines_scorecard_path": baselines_scorecard_path,
        "scorecard_by_regime_path": scorecard_by_regime_path,
        "regime_counts_path": regime_counts_path,
        "dm_by_regime_path": dm_by_regime_path,
        "fusion_weight_search_path": fusion_weight_search_path,
        "fusion_weight_selected_path": fusion_weight_selected_path,
        "fusion_weight_search_v7_path": fusion_weight_search_v7_path,
        "fusion_weight_selected_v7_path": fusion_weight_selected_v7_path,
        "calendar_policy_selected_by_regime_path": calendar_policy_selected_by_regime_path,
        "anchor_blend_ablation_path": anchor_blend_ablation_path,
        "ablation_scorecard_path": ablation_scorecard_path,
        "block_importance_path": block_importance_path,
        "ablation_by_regime_scorecard_path": ablation_by_regime_scorecard_path,
        "multi_shock_attribution_path": multi_shock_attribution_path,
        "interval_scorecard_path": interval_scorecard_path,
        "interval_scorecard_by_regime_path": interval_scorecard_by_regime_path,
        "interval_calibration_table_path": interval_calibration_table_path,
        "interval_calibration_by_regime_path": interval_calibration_by_regime_path,
        "fusion_constraints_path": str(fusion_constraints_path or ""),
        "force_bsts_off": bool(force_bsts_off),
        "multi_shock_disable_calendar_calibration": bool(multi_shock_disable_calendar),
        "multi_shock_disable_prototype": bool(multi_shock_disable_prototype),
        "multi_shock_max_month_length_bias_weight": (
            float(multi_shock_max_month_length_bias)
            if multi_shock_max_month_length_bias is not None
            else None
        ),
        "extra_block": str(extra_block or ""),
        "manifest_rows": manifest_rows,
    }
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8"
    )

    return Validation24mResult(
        point_estimates=point_estimates,
        scorecard=scorecard,
        summary={**summary, "summary_path": str(summary_path)},
    )
