"""Diebold-Mariano policy validation and deterministic runtime execution."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ng_forecaster.data.validators import load_yaml
from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.lag_guard import require_columns

try:
    from scipy.stats import norm
except Exception:  # pragma: no cover - fallback path if scipy is unavailable
    norm = None

DM_REQUIRED_OUTPUT_COLUMNS = [
    "target",
    "candidate_model",
    "benchmark_model",
    "d_bar",
    "dm_stat",
    "p_value",
    "significant_0_05",
    "significant_0_01",
    "adjusted_p_value",
    "hac_lag_used",
]

DM_AUDIT_OUTPUT_COLUMNS = [
    "loss_diff_mean",
    "dm_p_value_one_sided_improve",
    "dm_p_value_two_sided",
]

DEFAULT_DM_POLICY: dict[str, Any] = {
    "version": 1,
    "loss": "mse",
    "sidedness": "two_sided",
    "alpha_levels": [0.05, 0.01],
    "small_sample_adjustment": "harvey",
    "hac_lag_min": 3,
    "hac_lag_max": 6,
    "benchmark_by_target": {},
    "multiple_comparison": "holm",
    "comparison_pairs_by_target": {},
}

_ALLOWED_LOSS = {"mse", "mae"}
_ALLOWED_SIDEDNESS = {
    "two_sided",
    "less",
    "greater",
    "one_sided_less",
    "one_sided_greater",
}
_ALLOWED_ADJUSTMENT = {"none", "harvey"}
_ALLOWED_MULTIPLE_COMPARISON = {"none", "holm", "bonferroni"}


@dataclass(frozen=True)
class DMRunResult:
    """Runtime payload for policy-compliant DM outputs."""

    policy: dict[str, Any]
    results: pd.DataFrame


def _merge_dict(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            merged[key] = _merge_dict(base[key], value)
        else:
            merged[key] = value
    return merged


def _normalize_sidedness(value: str) -> str:
    normalized = value.strip().lower()
    if normalized == "one_sided_less":
        return "less"
    if normalized == "one_sided_greater":
        return "greater"
    return normalized


def _as_numeric_series(values: pd.Series, *, key: str) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    if numeric.isna().any():
        raise ContractViolation(
            "invalid_dm_input",
            key=key,
            detail="values must be numeric and non-null",
        )
    return numeric.astype(float)


def validate_dm_policy(policy: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize DM policy contract."""

    merged = _merge_dict(DEFAULT_DM_POLICY, policy or {})

    version = int(merged["version"])
    if version < 1:
        raise ContractViolation(
            "invalid_dm_policy",
            key="version",
            detail="version must be >= 1",
        )

    loss = str(merged["loss"]).strip().lower()
    if loss not in _ALLOWED_LOSS:
        raise ContractViolation(
            "invalid_dm_policy",
            key="loss",
            detail=f"loss must be one of {sorted(_ALLOWED_LOSS)}",
        )

    sidedness = _normalize_sidedness(str(merged["sidedness"]))
    if sidedness not in _ALLOWED_SIDEDNESS:
        raise ContractViolation(
            "invalid_dm_policy",
            key="sidedness",
            detail=f"sidedness must be one of {sorted(_ALLOWED_SIDEDNESS)}",
        )

    alpha_levels_raw = merged["alpha_levels"]
    if not isinstance(alpha_levels_raw, Sequence) or isinstance(alpha_levels_raw, str):
        raise ContractViolation(
            "invalid_dm_policy",
            key="alpha_levels",
            detail="alpha_levels must be a non-empty list",
        )

    alpha_levels = sorted({float(alpha) for alpha in alpha_levels_raw})
    if not alpha_levels:
        raise ContractViolation(
            "invalid_dm_policy",
            key="alpha_levels",
            detail="alpha_levels must be non-empty",
        )
    for alpha in alpha_levels:
        if alpha <= 0 or alpha >= 1:
            raise ContractViolation(
                "invalid_dm_policy",
                key="alpha_levels",
                detail="alpha levels must be in the open interval (0, 1)",
            )

    for required_alpha in (0.05, 0.01):
        if required_alpha not in alpha_levels:
            raise ContractViolation(
                "invalid_dm_policy",
                key="alpha_levels",
                detail="alpha_levels must include 0.05 and 0.01",
            )

    adjustment = str(merged["small_sample_adjustment"]).strip().lower()
    if adjustment not in _ALLOWED_ADJUSTMENT:
        raise ContractViolation(
            "invalid_dm_policy",
            key="small_sample_adjustment",
            detail=(
                "small_sample_adjustment must be one of "
                f"{sorted(_ALLOWED_ADJUSTMENT)}"
            ),
        )

    try:
        hac_lag_min = int(merged.get("hac_lag_min", 3))
        hac_lag_max = int(merged.get("hac_lag_max", 6))
    except (TypeError, ValueError) as exc:
        raise ContractViolation(
            "invalid_dm_policy",
            key="hac_lag_min/hac_lag_max",
            detail="HAC lag bounds must be integers",
        ) from exc
    if hac_lag_min < 0:
        raise ContractViolation(
            "invalid_dm_policy",
            key="hac_lag_min",
            detail="hac_lag_min must be >= 0",
        )
    if hac_lag_max < hac_lag_min:
        raise ContractViolation(
            "invalid_dm_policy",
            key="hac_lag_max",
            detail="hac_lag_max must be >= hac_lag_min",
        )

    benchmark_mapping_raw = merged["benchmark_by_target"]
    if not isinstance(benchmark_mapping_raw, Mapping) or not benchmark_mapping_raw:
        raise ContractViolation(
            "invalid_dm_policy",
            key="benchmark_by_target",
            detail="benchmark_by_target must be a non-empty mapping",
        )
    benchmark_mapping: dict[str, str] = {}
    for target, model_name in sorted(benchmark_mapping_raw.items()):
        target_name = str(target).strip()
        benchmark_name = str(model_name).strip()
        if not target_name or not benchmark_name:
            raise ContractViolation(
                "invalid_dm_policy",
                key="benchmark_by_target",
                detail="benchmark_by_target keys and values must be non-empty strings",
            )
        benchmark_mapping[target_name] = benchmark_name

    multiple_comparison = str(merged["multiple_comparison"]).strip().lower()
    if multiple_comparison not in _ALLOWED_MULTIPLE_COMPARISON:
        raise ContractViolation(
            "invalid_dm_policy",
            key="multiple_comparison",
            detail=(
                "multiple_comparison must be one of "
                f"{sorted(_ALLOWED_MULTIPLE_COMPARISON)}"
            ),
        )

    comparison_pairs_raw = merged.get("comparison_pairs_by_target", {})
    if not isinstance(comparison_pairs_raw, Mapping):
        raise ContractViolation(
            "invalid_dm_policy",
            key="comparison_pairs_by_target",
            detail="comparison_pairs_by_target must be a mapping",
        )

    comparison_pairs_by_target: dict[str, list[tuple[str, str]]] = {}
    for target, raw_pairs in sorted(comparison_pairs_raw.items()):
        target_name = str(target).strip()
        if not target_name:
            raise ContractViolation(
                "invalid_dm_policy",
                key="comparison_pairs_by_target",
                detail="target keys must be non-empty",
            )
        if not isinstance(raw_pairs, Sequence) or isinstance(raw_pairs, str):
            raise ContractViolation(
                "invalid_dm_policy",
                key=f"comparison_pairs_by_target.{target_name}",
                detail="pair list must be a sequence",
            )
        normalized_pairs: list[tuple[str, str]] = []
        for idx, pair in enumerate(raw_pairs):
            if (
                not isinstance(pair, Sequence)
                or isinstance(pair, str)
                or len(pair) != 2
            ):
                raise ContractViolation(
                    "invalid_dm_policy",
                    key=f"comparison_pairs_by_target.{target_name}[{idx}]",
                    detail="each pair must contain exactly two model labels",
                )
            candidate = str(pair[0]).strip()
            benchmark = str(pair[1]).strip()
            if not candidate or not benchmark:
                raise ContractViolation(
                    "invalid_dm_policy",
                    key=f"comparison_pairs_by_target.{target_name}[{idx}]",
                    detail="pair labels must be non-empty",
                )
            normalized_pairs.append((candidate, benchmark))
        if normalized_pairs:
            comparison_pairs_by_target[target_name] = normalized_pairs

    return {
        "version": version,
        "loss": loss,
        "sidedness": sidedness,
        "alpha_levels": alpha_levels,
        "small_sample_adjustment": adjustment,
        "hac_lag_min": hac_lag_min,
        "hac_lag_max": hac_lag_max,
        "benchmark_by_target": benchmark_mapping,
        "multiple_comparison": multiple_comparison,
        "comparison_pairs_by_target": comparison_pairs_by_target,
    }


def load_and_validate_dm_policy(path: str) -> dict[str, Any]:
    """Load DM policy YAML and validate normalized contract fields."""

    return validate_dm_policy(load_yaml(path))


def _standard_normal_cdf(value: float) -> float:
    if norm is not None:
        return float(norm.cdf(value))
    return 0.5 * (1.0 + math.erf(value / math.sqrt(2.0)))


def _compute_loss(
    actual: pd.Series,
    forecast: pd.Series,
    *,
    loss: str,
) -> pd.Series:
    error = actual - forecast
    if loss == "mse":
        return error**2
    if loss == "mae":
        return error.abs()
    raise ContractViolation(
        "invalid_dm_policy",
        key="loss",
        detail=f"unsupported loss: {loss}",
    )


def _sample_autocovariance(values: np.ndarray, *, lag: int) -> float:
    if lag < 0:
        raise ValueError("lag must be >= 0")
    if lag >= len(values):
        return 0.0
    centered = values - float(np.mean(values))
    if lag == 0:
        return float(np.dot(centered, centered) / len(centered))
    return float(np.dot(centered[lag:], centered[:-lag]) / len(centered))


def _newey_west_long_run_variance(
    values: np.ndarray,
    *,
    lag_used: int,
) -> float:
    gamma0 = _sample_autocovariance(values, lag=0)
    if lag_used <= 0:
        return max(gamma0, 0.0)
    total = gamma0
    for lag in range(1, lag_used + 1):
        gamma = _sample_autocovariance(values, lag=lag)
        weight = 1.0 - (lag / (lag_used + 1.0))
        total += 2.0 * weight * gamma
    return float(max(total, 0.0))


def _compute_dm_statistic(
    differential: pd.Series,
    *,
    sidedness: str,
    small_sample_adjustment: str,
    horizon: int,
    hac_lag_min: int,
    hac_lag_max: int,
) -> tuple[float, float, float, int, float]:
    values = differential.to_numpy(dtype=float)
    n_obs = int(len(values))
    if n_obs < 3:
        raise ContractViolation(
            "insufficient_dm_samples",
            key="n_obs",
            detail=f"DM test requires at least 3 overlapping observations; received={n_obs}",
        )

    d_bar = float(np.mean(values))
    horizon_lag_floor = max(0, int(horizon) - 1)
    lag_lower = max(int(hac_lag_min), horizon_lag_floor)
    lag_upper = max(lag_lower, int(hac_lag_max))
    lag_used = min(max(0, lag_upper), n_obs - 1)
    if lag_used < lag_lower:
        lag_used = min(max(0, lag_lower), n_obs - 1)

    long_run_variance = _newey_west_long_run_variance(values, lag_used=lag_used)
    variance_of_mean = long_run_variance / float(n_obs)

    if variance_of_mean <= 1e-12:
        dm_stat = 0.0
    else:
        dm_stat = d_bar / math.sqrt(variance_of_mean)

    if small_sample_adjustment == "harvey":
        h = max(1, int(horizon))
        adjustment_term = (n_obs + 1 - 2 * h + ((h * (h - 1)) / n_obs)) / n_obs
        if adjustment_term <= 0:
            raise ContractViolation(
                "invalid_dm_adjustment",
                key="small_sample_adjustment",
                detail=(
                    "harvey adjustment is undefined for provided n_obs/horizon: "
                    f"n_obs={n_obs}, horizon={h}"
                ),
            )
        dm_stat = dm_stat * math.sqrt(adjustment_term)

    cdf = _standard_normal_cdf(dm_stat)
    if sidedness == "two_sided":
        p_value = 2.0 * min(cdf, 1.0 - cdf)
    elif sidedness == "less":
        p_value = cdf
    elif sidedness == "greater":
        p_value = 1.0 - cdf
    else:
        raise ContractViolation(
            "invalid_dm_policy",
            key="sidedness",
            detail=f"unsupported sidedness: {sidedness}",
        )

    p_value = float(min(max(p_value, 0.0), 1.0))
    return d_bar, float(dm_stat), p_value, int(lag_used), float(long_run_variance)


def _adjust_p_values(
    frame: pd.DataFrame,
    *,
    method: str,
    target_col: str = "target",
    p_value_col: str = "p_value",
) -> pd.Series:
    if frame.empty:
        return pd.Series(dtype=float)

    adjusted = pd.Series(index=frame.index, dtype=float)
    for target, group in frame.groupby(target_col, sort=True):
        _ = target
        p_values = pd.to_numeric(group[p_value_col], errors="coerce")
        if p_values.isna().any():
            raise ContractViolation(
                "invalid_dm_input",
                key=p_value_col,
                detail="p-values must be numeric",
            )

        m = int(len(group))
        if method == "none" or m == 1:
            adjusted.loc[group.index] = p_values.to_numpy(dtype=float)
            continue

        if method == "bonferroni":
            adjusted.loc[group.index] = np.clip(
                p_values.to_numpy(dtype=float) * m, 0.0, 1.0
            )
            continue

        if method == "holm":
            order = sorted(
                list(group.index),
                key=lambda idx: (
                    float(frame.at[idx, p_value_col]),
                    str(frame.at[idx, "candidate_model"]),
                    str(frame.at[idx, "benchmark_model"]),
                ),
            )
            holm_values: dict[int, float] = {}
            running_max = 0.0
            for rank, index in enumerate(order, start=1):
                p_value = float(frame.at[index, p_value_col])
                candidate_adjusted = min(1.0, p_value * (m - rank + 1))
                running_max = max(running_max, candidate_adjusted)
                holm_values[index] = running_max
            adjusted.loc[group.index] = [holm_values[int(idx)] for idx in group.index]
            continue

        raise ContractViolation(
            "invalid_dm_policy",
            key="multiple_comparison",
            detail=f"unsupported correction method: {method}",
        )

    return adjusted.astype(float)


def run_dm_tests(
    forecasts: pd.DataFrame,
    policy: Mapping[str, Any] | None,
    *,
    target_col: str = "target",
    model_col: str = "model",
    actual_col: str = "actual",
    forecast_col: str = "forecast",
    asof_col: str = "asof",
    horizon_col: str = "horizon",
) -> DMRunResult:
    """Execute policy-compliant DM comparisons against configured benchmarks."""

    require_columns(
        forecasts,
        (target_col, model_col, actual_col, forecast_col),
        key="dm_forecasts",
    )

    cfg = validate_dm_policy(policy)
    data = forecasts.copy()
    data[target_col] = data[target_col].astype(str).str.strip()
    data[model_col] = data[model_col].astype(str).str.strip()

    if data[target_col].eq("").any() or data[model_col].eq("").any():
        raise ContractViolation(
            "invalid_dm_input",
            key="target/model",
            detail="target and model values must be non-empty strings",
        )

    data[actual_col] = _as_numeric_series(data[actual_col], key=actual_col)
    data[forecast_col] = _as_numeric_series(data[forecast_col], key=forecast_col)

    if asof_col in data.columns:
        data[asof_col] = pd.to_datetime(data[asof_col], errors="coerce")
        if data[asof_col].isna().any():
            raise ContractViolation(
                "invalid_timestamp",
                key=asof_col,
                detail="dm input contains invalid asof values",
            )

    if horizon_col in data.columns:
        data[horizon_col] = pd.to_numeric(data[horizon_col], errors="coerce")
        if data[horizon_col].isna().any():
            raise ContractViolation(
                "invalid_dm_input",
                key=horizon_col,
                detail="horizon values must be numeric",
            )

    merge_keys: list[str] = []
    if asof_col in data.columns:
        merge_keys.append(asof_col)
    if horizon_col in data.columns:
        merge_keys.append(horizon_col)

    result_rows: list[dict[str, Any]] = []
    benchmark_by_target: Mapping[str, str] = cfg["benchmark_by_target"]
    pair_mapping: Mapping[str, list[tuple[str, str]]] = cfg[
        "comparison_pairs_by_target"
    ]

    for target in sorted(data[target_col].unique().tolist()):
        target_frame = data[data[target_col] == target].copy()
        configured_pairs = pair_mapping.get(target, [])
        if configured_pairs:
            seen_pairs: set[tuple[str, str]] = set()
            comparison_pairs: list[tuple[str, str]] = []
            for candidate_model, benchmark_model in configured_pairs:
                key = (str(candidate_model), str(benchmark_model))
                if key not in seen_pairs:
                    comparison_pairs.append(key)
                    seen_pairs.add(key)
        else:
            benchmark_model = str(benchmark_by_target.get(target, "")).strip()
            if not benchmark_model:
                raise ContractViolation(
                    "missing_benchmark_policy",
                    key=target,
                    detail="benchmark_by_target must include all observed targets",
                )
            comparison_pairs = [
                (str(candidate), benchmark_model)
                for candidate in sorted(
                    set(target_frame[model_col].unique().tolist()) - {benchmark_model}
                )
            ]

        for candidate_model, benchmark_model in comparison_pairs:
            candidate_frame = target_frame[target_frame[model_col] == candidate_model]
            benchmark_frame = target_frame[target_frame[model_col] == benchmark_model]
            if candidate_frame.empty:
                raise ContractViolation(
                    "missing_candidate_forecast",
                    key=f"{target}:{candidate_model}",
                    detail=f"missing candidate model {candidate_model}",
                )
            if benchmark_frame.empty:
                raise ContractViolation(
                    "missing_benchmark_forecast",
                    key=f"{target}:{benchmark_model}",
                    detail=f"missing benchmark model {benchmark_model}",
                )

            if merge_keys:
                candidate_payload = candidate_frame[
                    merge_keys + [actual_col, forecast_col]
                ].rename(
                    columns={
                        actual_col: "actual_candidate",
                        forecast_col: "forecast_candidate",
                    }
                )
                benchmark_payload = benchmark_frame[
                    merge_keys + [actual_col, forecast_col]
                ].rename(
                    columns={
                        actual_col: "actual_benchmark",
                        forecast_col: "forecast_benchmark",
                    }
                )
                joined = candidate_payload.merge(
                    benchmark_payload,
                    on=merge_keys,
                    how="inner",
                )
                if joined.empty:
                    raise ContractViolation(
                        "missing_dm_overlap",
                        key=f"{target}:{candidate_model}",
                        detail=(
                            "candidate and benchmark forecasts have no shared observation "
                            "keys"
                        ),
                    )

                mismatch = (
                    joined["actual_candidate"] - joined["actual_benchmark"]
                ).abs()
                if (mismatch > 1e-9).any():
                    raise ContractViolation(
                        "inconsistent_actuals",
                        key=f"{target}:{candidate_model}",
                        detail="candidate and benchmark actual values must match",
                    )

                actual = joined["actual_candidate"]
                candidate_forecast = joined["forecast_candidate"]
                benchmark_forecast = joined["forecast_benchmark"]
                horizon_for_adjustment = (
                    int(round(float(joined[horizon_col].mean())))
                    if horizon_col in joined.columns
                    else 1
                )
            else:
                candidate_sorted = candidate_frame[
                    [actual_col, forecast_col]
                ].reset_index(drop=True)
                benchmark_sorted = benchmark_frame[
                    [actual_col, forecast_col]
                ].reset_index(drop=True)
                min_length = min(len(candidate_sorted), len(benchmark_sorted))
                if min_length < 3:
                    raise ContractViolation(
                        "insufficient_dm_samples",
                        key=f"{target}:{candidate_model}",
                        detail="candidate and benchmark overlap has fewer than 3 samples",
                    )
                candidate_sorted = candidate_sorted.iloc[:min_length]
                benchmark_sorted = benchmark_sorted.iloc[:min_length]
                mismatch = (
                    candidate_sorted[actual_col] - benchmark_sorted[actual_col]
                ).abs()
                if (mismatch > 1e-9).any():
                    raise ContractViolation(
                        "inconsistent_actuals",
                        key=f"{target}:{candidate_model}",
                        detail="candidate and benchmark actual values must match",
                    )

                actual = candidate_sorted[actual_col]
                candidate_forecast = candidate_sorted[forecast_col]
                benchmark_forecast = benchmark_sorted[forecast_col]
                horizon_for_adjustment = 1

            candidate_loss = _compute_loss(actual, candidate_forecast, loss=cfg["loss"])
            benchmark_loss = _compute_loss(actual, benchmark_forecast, loss=cfg["loss"])
            differential = candidate_loss - benchmark_loss
            d_bar, dm_stat, p_value, hac_lag_used, hac_long_run_variance = (
                _compute_dm_statistic(
                    differential,
                    sidedness=cfg["sidedness"],
                    small_sample_adjustment=cfg["small_sample_adjustment"],
                    horizon=horizon_for_adjustment,
                    hac_lag_min=int(cfg["hac_lag_min"]),
                    hac_lag_max=int(cfg["hac_lag_max"]),
                )
            )
            cdf = _standard_normal_cdf(dm_stat)
            p_value_one_sided_improve = float(min(max(cdf, 0.0), 1.0))
            p_value_two_sided = float(min(max(2.0 * min(cdf, 1.0 - cdf), 0.0), 1.0))

            result_rows.append(
                {
                    "target": target,
                    "candidate_model": candidate_model,
                    "benchmark_model": benchmark_model,
                    "loss_diff_mean": float(d_bar),
                    "d_bar": float(d_bar),
                    "dm_stat": float(dm_stat),
                    "dm_p_value_one_sided_improve": p_value_one_sided_improve,
                    "dm_p_value_two_sided": p_value_two_sided,
                    "p_value": float(p_value),
                    "hac_lag_used": int(hac_lag_used),
                    "hac_long_run_variance": float(hac_long_run_variance),
                    "n_obs": int(len(differential)),
                }
            )

    if not result_rows:
        raise ContractViolation(
            "missing_dm_candidates",
            key="candidate_model",
            detail="at least one candidate model is required for DM testing",
        )

    result_frame = pd.DataFrame(result_rows).sort_values(
        ["target", "candidate_model", "benchmark_model"],
        ascending=[True, True, True],
    )
    result_frame = result_frame.reset_index(drop=True)

    adjusted = _adjust_p_values(
        result_frame,
        method=cfg["multiple_comparison"],
        target_col="target",
        p_value_col="p_value",
    )
    result_frame["adjusted_p_value"] = adjusted
    result_frame["significant_0_05"] = result_frame["adjusted_p_value"] <= 0.05
    result_frame["significant_0_01"] = result_frame["adjusted_p_value"] <= 0.01

    ordered_columns = (
        DM_REQUIRED_OUTPUT_COLUMNS
        + [
            col
            for col in DM_AUDIT_OUTPUT_COLUMNS
            if col not in DM_REQUIRED_OUTPUT_COLUMNS
        ]
        + ["hac_long_run_variance", "n_obs"]
    )
    result_frame = result_frame[ordered_columns]

    return DMRunResult(policy=cfg, results=result_frame)


def ensure_dm_output_schema(frame: pd.DataFrame) -> None:
    """Validate DM output schema and value bounds before publishing."""

    require_columns(frame, DM_REQUIRED_OUTPUT_COLUMNS, key="dm_results")
    if frame.empty:
        raise ContractViolation(
            "empty_dm_results",
            key="dm_results",
            detail="DM output cannot be empty",
        )

    p_value = pd.to_numeric(frame["p_value"], errors="coerce")
    adjusted = pd.to_numeric(frame["adjusted_p_value"], errors="coerce")
    if p_value.isna().any() or adjusted.isna().any():
        raise ContractViolation(
            "invalid_dm_output",
            key="p_values",
            detail="p_value and adjusted_p_value must be numeric",
        )

    if ((p_value < 0) | (p_value > 1)).any() or ((adjusted < 0) | (adjusted > 1)).any():
        raise ContractViolation(
            "invalid_dm_output",
            key="p_values",
            detail="p_value and adjusted_p_value must be within [0, 1]",
        )

    if (adjusted < p_value).any():
        raise ContractViolation(
            "invalid_dm_output",
            key="adjusted_p_value",
            detail="adjusted_p_value must be greater than or equal to p_value",
        )

    for column in (
        "dm_p_value_one_sided_improve",
        "dm_p_value_two_sided",
    ):
        if column not in frame.columns:
            continue
        values = pd.to_numeric(frame[column], errors="coerce")
        if values.isna().any():
            raise ContractViolation(
                "invalid_dm_output",
                key=column,
                detail=f"{column} must be numeric",
            )
        if ((values < 0) | (values > 1)).any():
            raise ContractViolation(
                "invalid_dm_output",
                key=column,
                detail=f"{column} must be within [0, 1]",
            )

    if "loss_diff_mean" in frame.columns:
        loss_diff_mean = pd.to_numeric(frame["loss_diff_mean"], errors="coerce")
        if loss_diff_mean.isna().any():
            raise ContractViolation(
                "invalid_dm_output",
                key="loss_diff_mean",
                detail="loss_diff_mean must be numeric",
            )


def run_dm_tests_by_regime(
    frame: pd.DataFrame,
    policy: Mapping[str, Any] | None,
    *,
    regime_col: str = "regime_label",
    model_col: str = "model_variant",
    actual_col: str = "actual_released",
    forecast_col: str = "fused_point",
    asof_col: str = "asof",
    target_col: str = "target_month",
    min_observations: int = 3,
) -> pd.DataFrame:
    """Run DM tests split by regime label with deterministic skip diagnostics."""

    require_columns(
        frame,
        (regime_col, model_col, actual_col, forecast_col, target_col),
        key="dm_by_regime_input",
    )
    min_obs = max(1, int(min_observations))
    output_rows: list[dict[str, Any]] = []

    for regime_label, regime_frame in frame.groupby(regime_col, sort=True):
        grouped = regime_frame.copy()
        grouped[target_col] = pd.to_datetime(grouped[target_col], errors="coerce")
        grouped[actual_col] = pd.to_numeric(grouped[actual_col], errors="coerce")
        grouped[forecast_col] = pd.to_numeric(grouped[forecast_col], errors="coerce")
        grouped = grouped[
            grouped[target_col].notna()
            & grouped[actual_col].notna()
            & grouped[forecast_col].notna()
        ].copy()
        if grouped.empty:
            continue

        collapsed = (
            grouped.groupby([model_col, target_col], sort=True)
            .agg(
                actual=(actual_col, "first"),
                forecast=(forecast_col, "mean"),
            )
            .reset_index()
        )
        if (
            collapsed[target_col].nunique() < min_obs
            or collapsed[model_col].nunique() < 2
        ):
            output_rows.append(
                {
                    "regime_label": str(regime_label),
                    "status": "skipped",
                    "skip_reason": "insufficient_observations_or_models",
                    "n_target_months": int(collapsed[target_col].nunique()),
                    "n_models": int(collapsed[model_col].nunique()),
                }
            )
            continue

        dm_input = collapsed.rename(
            columns={
                model_col: "model",
            }
        )
        dm_input["target"] = "ng_prod"
        dm_input["asof"] = dm_input[target_col]
        dm_input["horizon"] = 1
        dm_input = dm_input[
            ["target", "model", "asof", "horizon", "actual", "forecast"]
        ]
        try:
            dm_run = run_dm_tests(
                dm_input,
                policy,
                target_col="target",
                model_col="model",
                actual_col="actual",
                forecast_col="forecast",
                asof_col="asof",
                horizon_col="horizon",
            )
            if dm_run.results.empty:
                output_rows.append(
                    {
                        "regime_label": str(regime_label),
                        "status": "skipped",
                        "skip_reason": "no_comparison_pairs",
                        "n_target_months": int(collapsed[target_col].nunique()),
                        "n_models": int(collapsed[model_col].nunique()),
                    }
                )
                continue
            for _, row in dm_run.results.iterrows():
                payload = row.to_dict()
                payload["regime_label"] = str(regime_label)
                payload["status"] = "ok"
                payload["skip_reason"] = ""
                payload["n_target_months"] = int(collapsed[target_col].nunique())
                payload["n_models"] = int(collapsed[model_col].nunique())
                output_rows.append(payload)
        except ContractViolation as exc:
            output_rows.append(
                {
                    "regime_label": str(regime_label),
                    "status": "skipped",
                    "skip_reason": str(exc.context.reason_code),
                    "n_target_months": int(collapsed[target_col].nunique()),
                    "n_models": int(collapsed[model_col].nunique()),
                }
            )

    if not output_rows:
        return pd.DataFrame(
            columns=[
                "regime_label",
                "status",
                "skip_reason",
                *DM_REQUIRED_OUTPUT_COLUMNS,
                *DM_AUDIT_OUTPUT_COLUMNS,
                "n_obs",
                "n_target_months",
                "n_models",
            ]
        )

    output = pd.DataFrame(output_rows)
    sort_cols = [
        col
        for col in ("regime_label", "status", "candidate_model")
        if col in output.columns
    ]
    if sort_cols:
        output = output.sort_values(sort_cols).reset_index(drop=True)
    return output
