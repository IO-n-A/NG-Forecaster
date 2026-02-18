"""Deterministic challenger with Bayesian state-space style contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from ng_forecaster.data.target_transforms import (
    daily_average_to_monthly_total,
    horizons_to_month_ends,
    monthly_total_to_daily_average,
)
from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.lag_guard import require_columns
from ng_forecaster.models.bsts.bsts_pymc import fit_bsts_with_pymc
from ng_forecaster.models.bsts.regressor_design import build_bsts_regressor_design

try:
    from scipy.stats import t as student_t
except Exception:  # pragma: no cover - fallback path if scipy is unavailable
    student_t = None


DEFAULT_CHALLENGER_CONFIG: dict[str, Any] = {
    "version": 1,
    "model": {
        "family": "bsts",
        "variant": "local_linear_trend_student_t",
        "engine": "deterministic",
    },
    "training": {
        "lookback": 48,
    },
    "state_space": {
        "student_t_df": 6,
        "center_scale": True,
        "pymc_draws": 500,
        "pymc_tune": 500,
        "pymc_chains": 2,
    },
    "target_transform": {
        "normalize_by_days_in_month": True,
        "log": False,
    },
    "forecast": {
        "horizons": [1, 2],
        "alpha": 0.95,
    },
    "exogenous": {
        "transfer_priors": {
            "enabled": False,
            "prior_weight": 0.0,
            "dispersion_weight": 0.0,
            "prior_scale": 1000.0,
            "dispersion_scale": 100.0,
            "max_abs_adjustment": 250000.0,
            "min_confidence_weight": 0.2,
            "confidence_power": 1.0,
        }
    },
}

_ALLOWED_CHALLENGER_ENGINES = {"deterministic", "pymc"}


@dataclass(frozen=True)
class ChallengerRunResult:
    """Output bundle for challenger forecast and interval diagnostics."""

    forecast: pd.DataFrame
    state_summary: dict[str, float]
    diagnostics: dict[str, Any]


def _merge_dict(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            merged[key] = _merge_dict(base[key], value)
        else:
            merged[key] = value
    return merged


def validate_challenger_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize challenger configuration contract."""

    merged = _merge_dict(DEFAULT_CHALLENGER_CONFIG, config or {})

    family = str(merged["model"]["family"])
    if family != "bsts":
        raise ContractViolation(
            "invalid_model_policy",
            key="model.family",
            detail="challenger model family must be bsts",
        )
    engine = str(merged["model"].get("engine", "deterministic")).strip().lower()
    if engine not in _ALLOWED_CHALLENGER_ENGINES:
        raise ContractViolation(
            "invalid_model_policy",
            key="model.engine",
            detail=f"engine must be one of {sorted(_ALLOWED_CHALLENGER_ENGINES)}",
        )

    lookback = int(merged["training"]["lookback"])
    if lookback < 12:
        raise ContractViolation(
            "invalid_model_policy",
            key="training.lookback",
            detail="lookback must be >= 12",
        )

    dof = int(merged["state_space"]["student_t_df"])
    if dof <= 2:
        raise ContractViolation(
            "invalid_model_policy",
            key="state_space.student_t_df",
            detail="student_t_df must be > 2",
        )

    alpha = float(merged["forecast"]["alpha"])
    if alpha <= 0 or alpha >= 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="forecast.alpha",
            detail="alpha must be in the open interval (0, 1)",
        )

    horizons = sorted({int(h) for h in merged["forecast"]["horizons"]})
    if not horizons or horizons[0] < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="forecast.horizons",
            detail="at least one positive horizon is required",
        )

    raw_offsets = merged["forecast"].get("horizon_month_offset", {})
    if not isinstance(raw_offsets, Mapping):
        raise ContractViolation(
            "invalid_model_policy",
            key="forecast.horizon_month_offset",
            detail="horizon_month_offset must be a mapping",
        )
    raw_labels = merged["forecast"].get("horizon_label", {})
    if not isinstance(raw_labels, Mapping):
        raise ContractViolation(
            "invalid_model_policy",
            key="forecast.horizon_label",
            detail="horizon_label must be a mapping",
        )

    horizon_month_offset: dict[str, int] = {}
    horizon_label: dict[str, str] = {}
    for horizon in horizons:
        key = str(horizon)
        offset_value = raw_offsets.get(key, raw_offsets.get(horizon, horizon - 1))
        try:
            parsed_offset = int(offset_value)
        except (TypeError, ValueError) as exc:
            raise ContractViolation(
                "invalid_model_policy",
                key=f"forecast.horizon_month_offset[{key}]",
                detail="offset must be an integer",
            ) from exc
        if parsed_offset < 0:
            raise ContractViolation(
                "invalid_model_policy",
                key=f"forecast.horizon_month_offset[{key}]",
                detail="offset must be >= 0",
            )
        horizon_month_offset[key] = parsed_offset

        label_value = raw_labels.get(key, raw_labels.get(horizon, f"horizon_{horizon}"))
        label = str(label_value).strip()
        if not label:
            raise ContractViolation(
                "invalid_model_policy",
                key=f"forecast.horizon_label[{key}]",
                detail="label must be non-empty",
            )
        horizon_label[key] = label

    merged["training"]["lookback"] = lookback
    merged["state_space"]["student_t_df"] = dof
    merged["state_space"]["center_scale"] = bool(merged["state_space"]["center_scale"])
    pymc_draws = int(merged["state_space"].get("pymc_draws", 500))
    pymc_tune = int(merged["state_space"].get("pymc_tune", 500))
    pymc_chains = int(merged["state_space"].get("pymc_chains", 2))
    if pymc_draws < 100:
        raise ContractViolation(
            "invalid_model_policy",
            key="state_space.pymc_draws",
            detail="pymc_draws must be >= 100",
        )
    if pymc_tune < 100:
        raise ContractViolation(
            "invalid_model_policy",
            key="state_space.pymc_tune",
            detail="pymc_tune must be >= 100",
        )
    if pymc_chains < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="state_space.pymc_chains",
            detail="pymc_chains must be >= 1",
        )
    merged["state_space"]["pymc_draws"] = pymc_draws
    merged["state_space"]["pymc_tune"] = pymc_tune
    merged["state_space"]["pymc_chains"] = pymc_chains
    merged["model"]["engine"] = engine
    merged["forecast"]["alpha"] = alpha
    merged["forecast"]["horizons"] = horizons
    merged["forecast"]["horizon_month_offset"] = horizon_month_offset
    merged["forecast"]["horizon_label"] = horizon_label

    transform_cfg = dict(merged.get("target_transform", {}))
    normalize_by_days = bool(transform_cfg.get("normalize_by_days_in_month", True))
    use_log_transform = bool(transform_cfg.get("log", False))
    if use_log_transform:
        raise ContractViolation(
            "invalid_model_policy",
            key="target_transform.log",
            detail="optional log transform path is disabled by default in Sprint 4A",
        )
    merged["target_transform"] = {
        "normalize_by_days_in_month": normalize_by_days,
        "log": False,
    }

    exogenous_cfg = dict(merged.get("exogenous", {}))
    transfer_cfg = dict(exogenous_cfg.get("transfer_priors", {}))
    enabled = bool(transfer_cfg.get("enabled", False))
    prior_weight = float(transfer_cfg.get("prior_weight", 0.0))
    dispersion_weight = float(transfer_cfg.get("dispersion_weight", 0.0))
    prior_scale = float(transfer_cfg.get("prior_scale", 1000.0))
    dispersion_scale = float(transfer_cfg.get("dispersion_scale", 100.0))
    max_abs_adjustment = float(transfer_cfg.get("max_abs_adjustment", 250000.0))
    min_confidence_weight = float(transfer_cfg.get("min_confidence_weight", 0.2))
    confidence_power = float(transfer_cfg.get("confidence_power", 1.0))
    if prior_scale <= 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="exogenous.transfer_priors.prior_scale",
            detail="prior_scale must be > 0",
        )
    if dispersion_scale <= 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="exogenous.transfer_priors.dispersion_scale",
            detail="dispersion_scale must be > 0",
        )
    if max_abs_adjustment < 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="exogenous.transfer_priors.max_abs_adjustment",
            detail="max_abs_adjustment must be >= 0",
        )
    if min_confidence_weight < 0 or min_confidence_weight > 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="exogenous.transfer_priors.min_confidence_weight",
            detail="min_confidence_weight must be in [0, 1]",
        )
    if confidence_power <= 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="exogenous.transfer_priors.confidence_power",
            detail="confidence_power must be > 0",
        )
    merged["exogenous"] = {
        "transfer_priors": {
            "enabled": enabled,
            "prior_weight": prior_weight,
            "dispersion_weight": dispersion_weight,
            "prior_scale": prior_scale,
            "dispersion_scale": dispersion_scale,
            "max_abs_adjustment": max_abs_adjustment,
            "min_confidence_weight": min_confidence_weight,
            "confidence_power": confidence_power,
        }
    }
    return merged


def _horizon_suffix(horizon: int) -> str:
    if horizon == 1:
        return "t"
    if horizon == 2:
        return "t_plus_1"
    return f"t_plus_{horizon - 1}"


def _transfer_feature_value(
    exogenous_features: Mapping[str, Any],
    *,
    horizon: int,
    metric: str,
) -> float:
    key = f"transfer_prior_{metric}_{_horizon_suffix(horizon)}"
    if key not in exogenous_features:
        raise ContractViolation(
            "missing_column",
            key=key,
            detail="required transfer-prior exogenous feature is missing",
        )
    try:
        value = float(exogenous_features[key])
    except (TypeError, ValueError) as exc:
        raise ContractViolation(
            "invalid_model_policy",
            key=key,
            detail="transfer-prior feature value must be numeric",
        ) from exc
    if np.isnan(value):
        raise ContractViolation(
            "invalid_model_policy",
            key=key,
            detail="transfer-prior feature value cannot be NaN",
        )
    return value


def _apply_transfer_prior_adjustment(
    forecast: pd.DataFrame,
    *,
    exogenous_features: Mapping[str, Any] | None,
    cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, dict[str, Any]]:
    transfer_cfg = dict(cfg.get("exogenous", {}).get("transfer_priors", {}))
    enabled = bool(transfer_cfg.get("enabled", False))
    adjusted = forecast.copy()
    adjusted["transfer_prior_adjustment"] = 0.0
    adjusted["transfer_prior_signal"] = 0.0
    adjusted["transfer_dispersion_signal"] = 0.0
    adjusted["transfer_confidence_weight"] = 1.0

    if not enabled:
        return adjusted, {"enabled": False}
    if exogenous_features is None:
        raise ContractViolation(
            "missing_column",
            key="exogenous_features",
            detail="transfer prior exogenous block is enabled but no features were passed",
        )

    horizons = sorted({int(value) for value in adjusted["horizon"].tolist()})
    required_keys: list[str] = []
    for horizon in horizons:
        required_keys.append(f"transfer_prior_us_bcfd_{_horizon_suffix(horizon)}")
        required_keys.append(f"transfer_prior_dispersion_{_horizon_suffix(horizon)}")
    missing_keys = [key for key in required_keys if key not in exogenous_features]
    if missing_keys:
        return adjusted, {
            "enabled": True,
            "applied": False,
            "status": "missing_features",
            "missing_features": sorted(missing_keys),
        }

    prior_weight = float(transfer_cfg["prior_weight"])
    dispersion_weight = float(transfer_cfg["dispersion_weight"])
    prior_scale = float(transfer_cfg["prior_scale"])
    dispersion_scale = float(transfer_cfg["dispersion_scale"])
    max_abs_adjustment = float(transfer_cfg["max_abs_adjustment"])
    min_confidence_weight = float(transfer_cfg.get("min_confidence_weight", 0.2))
    confidence_power = float(transfer_cfg.get("confidence_power", 1.0))

    for idx, row in adjusted.iterrows():
        horizon = int(row["horizon"])
        prior_value = _transfer_feature_value(
            exogenous_features,
            horizon=horizon,
            metric="us_bcfd",
        )
        dispersion_value = _transfer_feature_value(
            exogenous_features,
            horizon=horizon,
            metric="dispersion",
        )
        prior_signal = float(np.tanh(prior_value / prior_scale))
        dispersion_signal = float(np.tanh(dispersion_value / dispersion_scale))
        confidence = max(min_confidence_weight, 1.0 - abs(dispersion_signal))
        confidence = float(np.power(confidence, confidence_power))
        delta = (
            confidence * prior_weight * prior_signal
            - dispersion_weight * dispersion_signal
        )
        delta = float(np.clip(delta, -max_abs_adjustment, max_abs_adjustment))

        adjusted.at[idx, "mean_forecast"] = float(row["mean_forecast"]) + delta
        adjusted.at[idx, "lower_95"] = float(row["lower_95"]) + delta
        adjusted.at[idx, "upper_95"] = float(row["upper_95"]) + delta
        adjusted.at[idx, "transfer_prior_adjustment"] = delta
        adjusted.at[idx, "transfer_prior_signal"] = prior_signal
        adjusted.at[idx, "transfer_dispersion_signal"] = dispersion_signal
        adjusted.at[idx, "transfer_confidence_weight"] = confidence

    diagnostics = {
        "enabled": True,
        "applied": True,
        "status": "applied",
        "prior_weight": prior_weight,
        "dispersion_weight": dispersion_weight,
        "prior_scale": prior_scale,
        "dispersion_scale": dispersion_scale,
        "max_abs_adjustment": max_abs_adjustment,
        "min_confidence_weight": min_confidence_weight,
        "confidence_power": confidence_power,
        "applied_horizons": adjusted["horizon"].astype(int).tolist(),
    }
    return adjusted, diagnostics


def _count_trailing_nans(series: pd.Series) -> int:
    count = 0
    for value in reversed(series.tolist()):
        if pd.isna(value):
            count += 1
        else:
            break
    return count


def _prepare_observations(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    target_col: str,
    lookback: int,
) -> tuple[pd.DataFrame, pd.Series, int]:
    require_columns(frame, (timestamp_col, target_col), key="challenger_input")

    prepared = frame.copy()
    prepared[timestamp_col] = pd.to_datetime(prepared[timestamp_col], errors="coerce")
    if prepared[timestamp_col].isna().any():
        raise ContractViolation(
            "invalid_timestamp",
            key=timestamp_col,
            detail="challenger input contains invalid timestamps",
        )

    prepared[target_col] = pd.to_numeric(prepared[target_col], errors="coerce")
    prepared = prepared.sort_values(timestamp_col).reset_index(drop=True)
    trailing_missing = _count_trailing_nans(prepared[target_col])

    observed = prepared[prepared[target_col].notna()].copy()
    if len(observed) < lookback:
        raise ContractViolation(
            "insufficient_training_data",
            key=target_col,
            detail=(
                f"challenger requires at least lookback={lookback} non-null observations; "
                f"received={len(observed)}"
            ),
        )
    observed = observed.tail(lookback).reset_index(drop=True)
    return prepared, observed[target_col].astype(float), trailing_missing


def _t_quantile(alpha: float, dof: int) -> float:
    if student_t is None:
        return 1.96 if alpha >= 0.95 else 1.645
    return float(student_t.ppf((1 + alpha) / 2, dof))


def run_challenger_model(
    frame: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
    *,
    timestamp_col: str = "timestamp",
    target_col: str = "target_value",
    exogenous_features: Mapping[str, Any] | None = None,
    artifact_root: str | Path | None = None,
    artifact_tag: str | None = None,
) -> ChallengerRunResult:
    """Run challenger local-trend forecasts with Student-t interval contracts."""

    cfg = validate_challenger_config(config)
    normalize_by_days = bool(
        cfg.get("target_transform", {}).get("normalize_by_days_in_month", False)
    )

    model_frame = frame.copy()
    month_context = None
    if normalize_by_days:
        transformed, month_context = monthly_total_to_daily_average(
            model_frame[model_frame[target_col].notna()].copy(),
            timestamp_col=timestamp_col,
            value_col=target_col,
            out_col="target_value_per_day",
        )
        model_frame[target_col] = pd.NA
        model_frame.loc[transformed.index, target_col] = transformed[
            "target_value_per_day"
        ]

    _, observed, trailing_missing = _prepare_observations(
        model_frame,
        timestamp_col=timestamp_col,
        target_col=target_col,
        lookback=int(cfg["training"]["lookback"]),
    )

    horizons = list(cfg["forecast"]["horizons"])
    while trailing_missing > len(horizons):
        horizons.append(horizons[-1] + 1)

    alpha = float(cfg["forecast"]["alpha"])
    engine = str(cfg["model"]["engine"])
    dof = int(cfg["state_space"]["student_t_df"])
    if engine == "pymc":
        observed_rows = model_frame[model_frame[target_col].notna()].copy()
        observed_rows = observed_rows.sort_values(timestamp_col).tail(len(observed))
        regressors = build_bsts_regressor_design(
            observed,
            timestamps=observed_rows[timestamp_col],
            exogenous_features=exogenous_features,
        )
        steo_observation = None
        if "steo_observation" in observed_rows.columns:
            steo_series = pd.to_numeric(
                observed_rows["steo_observation"],
                errors="coerce",
            )
            if steo_series.notna().all():
                steo_observation = steo_series
        posterior_dir: str | Path | None = None
        if artifact_root is not None and artifact_tag:
            posterior_dir = (
                Path(artifact_root) / str(cfg["model"]["variant"]) / str(artifact_tag)
            )
        pymc_result = fit_bsts_with_pymc(
            observed,
            timestamps=observed_rows[timestamp_col],
            horizons=horizons,
            alpha=alpha,
            draws=int(cfg["state_space"]["pymc_draws"]),
            tune=int(cfg["state_space"]["pymc_tune"]),
            chains=int(cfg["state_space"]["pymc_chains"]),
            regressors=regressors,
            steo_observation=steo_observation,
            artifact_dir=posterior_dir,
        )
        forecast = pymc_result.forecast.copy()
        state_summary = dict(pymc_result.state_summary)
        backend_diagnostics = dict(pymc_result.diagnostics)
    else:
        center_scale = bool(cfg["state_space"]["center_scale"])
        mean = float(observed.mean()) if center_scale else 0.0
        scale = float(observed.std(ddof=0)) if center_scale else 1.0
        if scale <= 0:
            scale = 1.0

        standardized = (observed - mean) / scale
        x = np.arange(len(standardized), dtype=float)
        if len(standardized) >= 2:
            slope, intercept = np.polyfit(x, standardized.to_numpy(dtype=float), 1)
        else:
            slope, intercept = 0.0, float(standardized.iloc[-1])

        fitted = intercept + slope * x
        residuals = standardized.to_numpy(dtype=float) - fitted
        residual_std = float(np.std(residuals, ddof=0))
        if residual_std <= 1e-8:
            residual_std = 1e-8
        quantile = _t_quantile(alpha, dof)

        rows: list[dict[str, Any]] = []
        for horizon in sorted(set(horizons)):
            x_future = float(len(standardized) - 1 + horizon)
            standardized_mean = float(intercept + slope * x_future)
            interval_scale = residual_std * np.sqrt(
                1.0 + horizon / max(1, len(standardized))
            )
            standardized_half_width = quantile * interval_scale

            mean_forecast = standardized_mean * scale + mean
            lower = (standardized_mean - standardized_half_width) * scale + mean
            upper = (standardized_mean + standardized_half_width) * scale + mean

            rows.append(
                {
                    "horizon": int(horizon),
                    "horizon_month_offset": int(
                        cfg["forecast"]["horizon_month_offset"][str(horizon)]
                    ),
                    "horizon_label": str(
                        cfg["forecast"]["horizon_label"][str(horizon)]
                    ),
                    "mean_forecast": float(mean_forecast),
                    "lower_95": float(lower),
                    "upper_95": float(upper),
                    "residual_scale": float(interval_scale * scale),
                }
            )

        forecast = pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)
        state_summary = {
            "center": mean,
            "scale": scale,
            "slope": float(slope),
            "intercept": float(intercept),
            "residual_std": residual_std,
        }
        backend_diagnostics = {"engine": "deterministic"}
    if "horizon_month_offset" not in forecast.columns:
        forecast["horizon_month_offset"] = forecast["horizon"].map(
            lambda value: int(cfg["forecast"]["horizon_month_offset"][str(int(value))])
        )
    if "horizon_label" not in forecast.columns:
        forecast["horizon_label"] = forecast["horizon"].map(
            lambda value: str(cfg["forecast"]["horizon_label"][str(int(value))])
        )
    if normalize_by_days and month_context is not None:
        month_map = horizons_to_month_ends(
            context=month_context,
            horizons=forecast["horizon"].tolist(),
        )
        for column in ("mean_forecast", "lower_95", "upper_95", "residual_scale"):
            forecast[column] = forecast.apply(
                lambda row: daily_average_to_monthly_total(
                    float(row[column]),
                    month_end=month_map[int(row["horizon"])],
                ),
                axis=1,
            )
    forecast, transfer_diag = _apply_transfer_prior_adjustment(
        forecast,
        exogenous_features=exogenous_features,
        cfg=cfg,
    )
    diagnostics = {
        "model_family": cfg["model"]["family"],
        "model_variant": cfg["model"]["variant"],
        "model_engine": engine,
        "lookback": int(cfg["training"]["lookback"]),
        "student_t_df": dof,
        "alpha": alpha,
        "target_transform": {
            "normalize_by_days_in_month": normalize_by_days,
            "log": False,
        },
        "trailing_nan_slots": int(trailing_missing),
        "forecast_rows": int(len(forecast)),
        "horizon_month_offset": dict(cfg["forecast"]["horizon_month_offset"]),
        "horizon_label": dict(cfg["forecast"]["horizon_label"]),
        "backend_diagnostics": backend_diagnostics,
        "exogenous_transfer_priors": transfer_diag,
    }

    return ChallengerRunResult(
        forecast=forecast,
        state_summary=state_summary,
        diagnostics=diagnostics,
    )
