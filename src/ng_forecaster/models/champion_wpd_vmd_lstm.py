"""Champion pipeline with WPD/fuzzy-entropy/VMD-style decomposition contracts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ng_forecaster.data.target_transforms import (
    daily_average_to_monthly_total,
    horizons_to_month_ends,
    monthly_total_to_daily_average,
)
from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.lag_guard import require_columns
from ng_forecaster.models.decomposition import (
    decompose_vmd_signal,
    decompose_wpd_signal,
    rank_components_by_entropy,
    select_vmd_k_by_energy_change,
)
from ng_forecaster.models.corrections.weather_shock import (
    apply_weather_shock_adjustment,
)
from ng_forecaster.models.corrections.weather_shock_state import (
    apply_weather_shock_state_adjustment,
)
from ng_forecaster.models.neural import (
    forecast_component_with_lstm,
    forecast_component_with_pytorch_lstm,
)

DEFAULT_CHAMPION_CONFIG: dict[str, Any] = {
    "version": 1,
    "model": {
        "family": "wpd_vmd_lstm",
        "variant": "wpd_vmd_lstm1",
    },
    "strategy": "wpd_vmd_lstm1",
    "wpd": {
        "wavelet_family": "db",
        "wavelet_order": 5,
        "levels": 3,
    },
    "fuzzy_entropy": {
        "m": 1,
        "n": 2.0,
        "r_policy": "std_0.2",
    },
    "vmd": {
        "k_min": 2,
        "k_max": 6,
        "alpha": 2000.0,
        "tau": 0.0,
        "tol": 1e-7,
        "theta_threshold": 0.05,
    },
    "lstm": {
        "engine": "deterministic",
        "lookback": 36,
        "hidden_units": 32,
        "batch_size": 16,
        "learning_rate": 0.001,
        "repeat_runs": 5,
        "dropout": 0.1,
        "max_epochs": 250,
        "early_stopping_patience": 10,
        "val_fraction": 0.2,
        "min_delta": 1e-4,
    },
    "training": {
        "seed": 42,
        "seed_noise_scale": 0.0,
    },
    "target_transform": {
        "normalize_by_days_in_month": True,
        "log": False,
    },
    "forecast": {
        "horizons": [1, 2],
        "horizon_month_offset": {"1": 0, "2": 1},
        "horizon_label": {"1": "target_month", "2": "target_month_plus_1"},
    },
    "exogenous": {
        "weather_shock": {
            "enabled": False,
            "intercept": 0.0,
            "beta_freeze_intensity": 5000.0,
            "beta_freeze_days": 3000.0,
            "beta_extreme_min": 2000.0,
            "freeze_intensity_scale": 1.0,
            "freeze_days_scale": 5.0,
            "extreme_min_reference_f": 20.0,
            "extreme_min_scale": 15.0,
            "min_coverage_fraction": 0.75,
            "cap_abs_adjustment": 25000.0,
        },
        "weather_shock_state": {
            "enabled": False,
            "persistence": 0.65,
            "impact_weight": 12000.0,
            "recovery_weight": 0.55,
            "cap_abs_adjustment": 20000.0,
        },
        "transfer_priors": {
            "enabled": False,
            "prior_weight": 0.0,
            "dispersion_weight": 0.0,
            "prior_scale": 1000.0,
            "dispersion_scale": 100.0,
            "max_abs_adjustment": 250000.0,
            "min_confidence_weight": 0.2,
            "confidence_power": 1.0,
        },
    },
}

_ALLOWED_VARIANTS = {"wpd_lstm_one_layer", "wpd_vmd_lstm1", "wpd_vmd_lstm2"}
_ALLOWED_LSTM_ENGINES = {"deterministic", "pytorch"}


@dataclass(frozen=True)
class ChampionRunResult:
    """Output bundle for a champion run."""

    point_forecast: pd.DataFrame
    wpd_components: pd.DataFrame
    entropy_scores: pd.DataFrame
    vmd_components: pd.DataFrame
    diagnostics: dict[str, Any]


@dataclass(frozen=True)
class ChampionSeedRunResult:
    """Output for repeated-seed champion runs."""

    forecasts: pd.DataFrame
    diagnostics: dict[str, Any]


def _merge_dict(base: Mapping[str, Any], updates: Mapping[str, Any]) -> dict[str, Any]:
    merged: dict[str, Any] = dict(base)
    for key, value in updates.items():
        if isinstance(value, Mapping) and isinstance(base.get(key), Mapping):
            merged[key] = _merge_dict(base[key], value)
        else:
            merged[key] = value
    return merged


def _migrate_legacy_config(config: dict[str, Any]) -> dict[str, Any]:
    migrated = dict(config)
    if "decomposition" in migrated and isinstance(migrated["decomposition"], Mapping):
        legacy = dict(migrated["decomposition"])
        wpd_cfg = dict(migrated.get("wpd", {}))
        vmd_cfg = dict(migrated.get("vmd", {}))

        if "wpd_components" in legacy and "levels" not in wpd_cfg:
            wpd_cfg["levels"] = int(legacy["wpd_components"])
        if "vmd_modes" in legacy:
            vmd_modes = max(1, int(legacy["vmd_modes"]))
            vmd_cfg.setdefault("k_min", vmd_modes)
            vmd_cfg.setdefault("k_max", max(vmd_modes, vmd_modes + 1))

        migrated["wpd"] = wpd_cfg
        migrated["vmd"] = vmd_cfg

    training = dict(migrated.get("training", {}))
    lstm_cfg = dict(migrated.get("lstm", {}))
    if "lookback" in training and "lookback" not in lstm_cfg:
        lstm_cfg["lookback"] = int(training["lookback"])
    migrated["training"] = training
    migrated["lstm"] = lstm_cfg

    model_cfg = dict(migrated.get("model", {}))
    variant = str(model_cfg.get("variant", "")).strip()
    strategy = variant or str(migrated.get("strategy", "")).strip()
    if strategy:
        model_cfg["variant"] = strategy
        migrated["strategy"] = strategy
    migrated["model"] = model_cfg
    return migrated


def validate_champion_config(config: Mapping[str, Any] | None) -> dict[str, Any]:
    """Validate and normalize champion configuration contract."""

    merged = _migrate_legacy_config(_merge_dict(DEFAULT_CHAMPION_CONFIG, config or {}))

    family = str(merged["model"]["family"]).strip()
    if family != "wpd_vmd_lstm":
        raise ContractViolation(
            "invalid_model_policy",
            key="model.family",
            detail="champion model family must be wpd_vmd_lstm",
        )

    variant = str(merged["model"]["variant"]).strip()
    if variant not in _ALLOWED_VARIANTS:
        raise ContractViolation(
            "invalid_model_policy",
            key="model.variant",
            detail=f"variant must be one of {sorted(_ALLOWED_VARIANTS)}",
        )
    merged["strategy"] = variant

    wpd = dict(merged.get("wpd", {}))
    wavelet_family = str(wpd.get("wavelet_family", "db")).strip().lower()
    wavelet_order = int(wpd.get("wavelet_order", 5))
    levels = int(wpd.get("levels", 3))
    if wavelet_family != "db":
        raise ContractViolation(
            "invalid_model_policy",
            key="wpd.wavelet_family",
            detail="only Daubechies ('db') family is supported",
        )
    if wavelet_order < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="wpd.wavelet_order",
            detail="wavelet_order must be >= 1",
        )
    if levels < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="wpd.levels",
            detail="levels must be >= 1",
        )
    merged["wpd"] = {
        "wavelet_family": wavelet_family,
        "wavelet_order": wavelet_order,
        "levels": levels,
    }

    entropy = dict(merged.get("fuzzy_entropy", {}))
    entropy_m = int(entropy.get("m", 1))
    entropy_n = float(entropy.get("n", 2.0))
    entropy_r_policy = str(entropy.get("r_policy", "std_0.2")).strip()
    if entropy_m < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="fuzzy_entropy.m",
            detail="m must be >= 1",
        )
    if entropy_n <= 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="fuzzy_entropy.n",
            detail="n must be > 0",
        )
    if not entropy_r_policy:
        raise ContractViolation(
            "invalid_model_policy",
            key="fuzzy_entropy.r_policy",
            detail="r_policy must be non-empty",
        )
    merged["fuzzy_entropy"] = {
        "m": entropy_m,
        "n": entropy_n,
        "r_policy": entropy_r_policy,
    }

    vmd = dict(merged.get("vmd", {}))
    k_min = int(vmd.get("k_min", 2))
    k_max = int(vmd.get("k_max", 6))
    alpha = float(vmd.get("alpha", 2000.0))
    tau = float(vmd.get("tau", 0.0))
    tol = float(vmd.get("tol", 1e-7))
    theta_threshold = float(vmd.get("theta_threshold", 0.05))
    if k_min < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="vmd.k_min",
            detail="k_min must be >= 1",
        )
    if k_max < k_min:
        raise ContractViolation(
            "invalid_model_policy",
            key="vmd.k_max",
            detail="k_max must be >= k_min",
        )
    if alpha <= 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="vmd.alpha",
            detail="alpha must be > 0",
        )
    if tau < 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="vmd.tau",
            detail="tau must be >= 0",
        )
    if tol <= 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="vmd.tol",
            detail="tol must be > 0",
        )
    if theta_threshold <= 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="vmd.theta_threshold",
            detail="theta_threshold must be > 0",
        )
    merged["vmd"] = {
        "k_min": k_min,
        "k_max": k_max,
        "alpha": alpha,
        "tau": tau,
        "tol": tol,
        "theta_threshold": theta_threshold,
    }

    lstm = dict(merged.get("lstm", {}))
    engine = str(lstm.get("engine", "deterministic")).strip().lower()
    lookback = int(lstm.get("lookback", 36))
    hidden_units = int(lstm.get("hidden_units", 32))
    batch_size = int(lstm.get("batch_size", 16))
    learning_rate = float(lstm.get("learning_rate", 0.001))
    repeat_runs = int(lstm.get("repeat_runs", 5))
    dropout = float(lstm.get("dropout", 0.1))
    max_epochs = int(lstm.get("max_epochs", 250))
    early_stopping_patience = int(lstm.get("early_stopping_patience", 10))
    val_fraction = float(lstm.get("val_fraction", 0.2))
    min_delta = float(lstm.get("min_delta", 1e-4))
    if engine not in _ALLOWED_LSTM_ENGINES:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm.engine",
            detail=f"engine must be one of {sorted(_ALLOWED_LSTM_ENGINES)}",
        )
    if lookback < 8:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm.lookback",
            detail="lookback must be >= 8",
        )
    if hidden_units < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm.hidden_units",
            detail="hidden_units must be >= 1",
        )
    if batch_size < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm.batch_size",
            detail="batch_size must be >= 1",
        )
    if learning_rate <= 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm.learning_rate",
            detail="learning_rate must be > 0",
        )
    if repeat_runs < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm.repeat_runs",
            detail="repeat_runs must be >= 1",
        )
    if dropout < 0 or dropout >= 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm.dropout",
            detail="dropout must be in [0, 1)",
        )
    if max_epochs < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm.max_epochs",
            detail="max_epochs must be >= 1",
        )
    if early_stopping_patience < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm.early_stopping_patience",
            detail="early_stopping_patience must be >= 1",
        )
    if val_fraction <= 0 or val_fraction >= 0.5:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm.val_fraction",
            detail="val_fraction must be in (0, 0.5)",
        )
    if min_delta < 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm.min_delta",
            detail="min_delta must be >= 0",
        )
    merged["lstm"] = {
        "engine": engine,
        "lookback": lookback,
        "hidden_units": hidden_units,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "repeat_runs": repeat_runs,
        "dropout": dropout,
        "max_epochs": max_epochs,
        "early_stopping_patience": early_stopping_patience,
        "val_fraction": val_fraction,
        "min_delta": min_delta,
    }

    training = dict(merged.get("training", {}))
    seed = int(training.get("seed", 42))
    noise_scale = float(training.get("seed_noise_scale", 0.0))
    if noise_scale < 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="training.seed_noise_scale",
            detail="seed_noise_scale must be >= 0",
        )
    merged["training"] = {"seed": seed, "seed_noise_scale": noise_scale}

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

    forecast = dict(merged.get("forecast", {}))
    horizons = sorted({int(h) for h in forecast.get("horizons", [1, 2])})
    if not horizons or horizons[0] < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="forecast.horizons",
            detail="at least one positive horizon is required",
        )
    raw_offsets = forecast.get("horizon_month_offset", {})
    raw_labels = forecast.get("horizon_label", {})
    if not isinstance(raw_offsets, Mapping):
        raise ContractViolation(
            "invalid_model_policy",
            key="forecast.horizon_month_offset",
            detail="horizon_month_offset must be a mapping",
        )
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
        label_value = raw_labels.get(key, raw_labels.get(horizon, f"horizon_{horizon}"))
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
        label = str(label_value).strip()
        if not label:
            raise ContractViolation(
                "invalid_model_policy",
                key=f"forecast.horizon_label[{key}]",
                detail="label must be non-empty",
            )
        horizon_month_offset[key] = parsed_offset
        horizon_label[key] = label

    merged["forecast"] = {
        "horizons": horizons,
        "horizon_month_offset": horizon_month_offset,
        "horizon_label": horizon_label,
    }

    exogenous_cfg = dict(merged.get("exogenous", {}))
    weather_cfg = dict(exogenous_cfg.get("weather_shock", {}))
    weather_enabled = bool(weather_cfg.get("enabled", False))
    weather_intercept = float(weather_cfg.get("intercept", 0.0))
    weather_beta_intensity = float(weather_cfg.get("beta_freeze_intensity", 5000.0))
    weather_beta_days = float(weather_cfg.get("beta_freeze_days", 3000.0))
    weather_beta_extreme = float(weather_cfg.get("beta_extreme_min", 2000.0))
    weather_intensity_scale = float(weather_cfg.get("freeze_intensity_scale", 1.0))
    weather_days_scale = float(weather_cfg.get("freeze_days_scale", 5.0))
    weather_min_ref = float(weather_cfg.get("extreme_min_reference_f", 20.0))
    weather_min_scale = float(weather_cfg.get("extreme_min_scale", 15.0))
    weather_min_coverage = float(weather_cfg.get("min_coverage_fraction", 0.75))
    weather_cap = float(weather_cfg.get("cap_abs_adjustment", 25000.0))
    if weather_intensity_scale <= 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="exogenous.weather_shock.freeze_intensity_scale",
            detail="freeze_intensity_scale must be > 0",
        )
    if weather_days_scale <= 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="exogenous.weather_shock.freeze_days_scale",
            detail="freeze_days_scale must be > 0",
        )
    if weather_min_scale <= 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="exogenous.weather_shock.extreme_min_scale",
            detail="extreme_min_scale must be > 0",
        )
    if weather_cap < 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="exogenous.weather_shock.cap_abs_adjustment",
            detail="cap_abs_adjustment must be >= 0",
        )
    if weather_min_coverage < 0 or weather_min_coverage > 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="exogenous.weather_shock.min_coverage_fraction",
            detail="min_coverage_fraction must be in [0, 1]",
        )
    weather_state_cfg = dict(exogenous_cfg.get("weather_shock_state", {}))
    weather_state_enabled = bool(weather_state_cfg.get("enabled", False))
    weather_state_persistence = float(weather_state_cfg.get("persistence", 0.65))
    weather_state_impact_weight = float(weather_state_cfg.get("impact_weight", 12000.0))
    weather_state_recovery_weight = float(
        weather_state_cfg.get("recovery_weight", 0.55)
    )
    weather_state_cap = float(weather_state_cfg.get("cap_abs_adjustment", 20000.0))
    if weather_state_persistence < 0 or weather_state_persistence >= 1.0:
        raise ContractViolation(
            "invalid_model_policy",
            key="exogenous.weather_shock_state.persistence",
            detail="persistence must be in [0, 1)",
        )
    if weather_state_impact_weight < 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="exogenous.weather_shock_state.impact_weight",
            detail="impact_weight must be >= 0",
        )
    if weather_state_recovery_weight < 0 or weather_state_recovery_weight > 1.5:
        raise ContractViolation(
            "invalid_model_policy",
            key="exogenous.weather_shock_state.recovery_weight",
            detail="recovery_weight must be in [0, 1.5]",
        )
    if weather_state_cap < 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="exogenous.weather_shock_state.cap_abs_adjustment",
            detail="cap_abs_adjustment must be >= 0",
        )

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
        "weather_shock": {
            "enabled": weather_enabled,
            "intercept": weather_intercept,
            "beta_freeze_intensity": weather_beta_intensity,
            "beta_freeze_days": weather_beta_days,
            "beta_extreme_min": weather_beta_extreme,
            "freeze_intensity_scale": weather_intensity_scale,
            "freeze_days_scale": weather_days_scale,
            "extreme_min_reference_f": weather_min_ref,
            "extreme_min_scale": weather_min_scale,
            "min_coverage_fraction": weather_min_coverage,
            "cap_abs_adjustment": weather_cap,
        },
        "weather_shock_state": {
            "enabled": weather_state_enabled,
            "persistence": weather_state_persistence,
            "impact_weight": weather_state_impact_weight,
            "recovery_weight": weather_state_recovery_weight,
            "cap_abs_adjustment": weather_state_cap,
        },
        "transfer_priors": {
            "enabled": enabled,
            "prior_weight": prior_weight,
            "dispersion_weight": dispersion_weight,
            "prior_scale": prior_scale,
            "dispersion_scale": dispersion_scale,
            "max_abs_adjustment": max_abs_adjustment,
            "min_confidence_weight": min_confidence_weight,
            "confidence_power": confidence_power,
        },
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

        adjusted.at[idx, "point_forecast"] = float(row["point_forecast"]) + delta
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


def _prepare_series(
    frame: pd.DataFrame,
    *,
    timestamp_col: str,
    target_col: str,
    lookback: int,
) -> pd.DataFrame:
    require_columns(frame, (timestamp_col, target_col), key="champion_input")
    prepared = frame.copy()
    prepared[timestamp_col] = pd.to_datetime(prepared[timestamp_col], errors="coerce")
    if prepared[timestamp_col].isna().any():
        raise ContractViolation(
            "invalid_timestamp",
            key=timestamp_col,
            detail="champion input contains invalid timestamps",
        )

    prepared[target_col] = pd.to_numeric(prepared[target_col], errors="coerce")
    prepared = prepared.sort_values(timestamp_col).reset_index(drop=True)
    prepared = prepared[prepared[target_col].notna()].copy()
    minimum_obs = int(lookback) + 2
    if len(prepared) < minimum_obs:
        raise ContractViolation(
            "insufficient_training_data",
            key=target_col,
            detail=(
                "champion requires at least lookback+2 non-null observations; "
                f"lookback={lookback} "
                f"received={len(prepared)}"
            ),
        )
    tail_length = min(len(prepared), max(int(lookback) + 12, minimum_obs))
    return prepared.tail(tail_length).reset_index(drop=True)


def _build_component_mode_rows(
    *,
    timestamps: pd.Series,
    component_name: str,
    component_values: pd.Series,
    variant: str,
    entropy_value: float,
    entropy_threshold: float,
    vmd_cfg: Mapping[str, float | int],
) -> tuple[list[dict[str, Any]], int, pd.DataFrame | None]:
    apply_vmd = variant != "wpd_lstm_one_layer" and entropy_value > entropy_threshold
    if not apply_vmd:
        passthrough_rows = [
            {
                "timestamp": ts,
                "component_name": component_name,
                "mode_name": f"{component_name}::mode_1",
                "value": float(value),
            }
            for ts, value in zip(timestamps, component_values.tolist())
        ]
        return passthrough_rows, 1, None

    try:
        selection = select_vmd_k_by_energy_change(
            component_values,
            k_min=int(vmd_cfg["k_min"]),
            k_max=int(vmd_cfg["k_max"]),
            alpha=float(vmd_cfg["alpha"]),
            tau=float(vmd_cfg["tau"]),
            tol=float(vmd_cfg["tol"]),
            theta_threshold=float(vmd_cfg["theta_threshold"]),
        )
        decomposition = decompose_vmd_signal(
            component_values,
            k=selection.selected_k,
            alpha=float(vmd_cfg["alpha"]),
            tau=float(vmd_cfg["tau"]),
            tol=float(vmd_cfg["tol"]),
        )
    except Exception as exc:
        raise ContractViolation(
            "vmd_component_failed",
            key=component_name,
            detail=f"VMD failed for component/group {component_name}: {exc}",
        ) from exc
    rows: list[dict[str, Any]] = []
    for mode_name in decomposition.modes.columns:
        for ts, value in zip(timestamps, decomposition.modes[mode_name].tolist()):
            rows.append(
                {
                    "timestamp": ts,
                    "component_name": component_name,
                    "mode_name": f"{component_name}::{mode_name}",
                    "value": float(value),
                }
            )
    return rows, int(selection.selected_k), selection.search_table


def run_champion_pipeline(
    frame: pd.DataFrame,
    config: Mapping[str, Any] | None = None,
    *,
    timestamp_col: str = "timestamp",
    target_col: str = "target_value",
    seed: int | None = None,
    exogenous_features: Mapping[str, Any] | None = None,
    artifact_root: str | Path | None = None,
    artifact_tag: str | None = None,
) -> ChampionRunResult:
    """Run champion decomposition and component LSTM point forecasting."""

    cfg = validate_champion_config(config)
    model_seed = int(cfg["training"]["seed"] if seed is None else seed)
    variant = str(cfg["model"]["variant"])

    training = _prepare_series(
        frame,
        timestamp_col=timestamp_col,
        target_col=target_col,
        lookback=int(cfg["lstm"]["lookback"]),
    )
    timestamps = training[timestamp_col]
    normalize_by_days = bool(
        cfg.get("target_transform", {}).get("normalize_by_days_in_month", False)
    )
    month_context = None
    if normalize_by_days:
        transformed, month_context = monthly_total_to_daily_average(
            training,
            timestamp_col=timestamp_col,
            value_col=target_col,
            out_col="target_value_per_day",
        )
        signal = transformed["target_value_per_day"].astype(float)
    else:
        signal = training[target_col].astype(float)

    wpd_result = decompose_wpd_signal(
        signal,
        wavelet_family=str(cfg["wpd"]["wavelet_family"]),
        wavelet_order=int(cfg["wpd"]["wavelet_order"]),
        levels=int(cfg["wpd"]["levels"]),
        component_prefix="PF",
    )
    component_frame = wpd_result.components.copy()
    component_frame.index = training.index
    component_names = sorted(component_frame.columns)

    entropy_df = rank_components_by_entropy(
        component_frame[component_names],
        m=int(cfg["fuzzy_entropy"]["m"]),
        n=float(cfg["fuzzy_entropy"]["n"]),
        r_policy=str(cfg["fuzzy_entropy"]["r_policy"]),
    )
    entropy_map = {
        str(row["component_name"]): float(row["fuzzy_entropy"])
        for _, row in entropy_df.iterrows()
    }
    entropy_threshold = float(entropy_df["fuzzy_entropy"].mean())
    high_complexity_components = sorted(
        [
            str(row["component_name"])
            for _, row in entropy_df.iterrows()
            if float(row["fuzzy_entropy"]) > entropy_threshold
        ]
    )

    wpd_rows: list[dict[str, Any]] = []
    for component_name in component_names:
        for ts, value in zip(timestamps, component_frame[component_name].tolist()):
            wpd_rows.append(
                {
                    "timestamp": ts,
                    "component_name": component_name,
                    "value": float(value),
                }
            )

    vmd_rows: list[dict[str, Any]] = []
    selected_k_by_component: dict[str, int] = {}
    theta_search_tables: dict[str, pd.DataFrame] = {}

    if variant == "wpd_vmd_lstm2":
        ranked = entropy_df.sort_values(["entropy_rank"]).reset_index(drop=True)
        group_labels = ["ultra_high", "high", "mid", "low"]
        groups = np.array_split(ranked["component_name"].tolist(), 4)
        for idx, components in enumerate(groups):
            if len(components) == 0:
                continue
            group_name = f"group_{group_labels[idx]}"
            group_signal = component_frame[list(components)].sum(axis=1)
            rows, selected_k, theta_table = _build_component_mode_rows(
                timestamps=timestamps,
                component_name=group_name,
                component_values=group_signal,
                variant="wpd_vmd_lstm1",
                entropy_value=float(
                    ranked[ranked["component_name"].isin(components)][
                        "fuzzy_entropy"
                    ].mean()
                ),
                entropy_threshold=float("-inf"),
                vmd_cfg=cfg["vmd"],
            )
            vmd_rows.extend(rows)
            selected_k_by_component[group_name] = selected_k
            if theta_table is not None:
                theta_search_tables[group_name] = theta_table
    else:
        for component_name in component_names:
            rows, selected_k, theta_table = _build_component_mode_rows(
                timestamps=timestamps,
                component_name=component_name,
                component_values=component_frame[component_name],
                variant=variant,
                entropy_value=entropy_map[component_name],
                entropy_threshold=entropy_threshold,
                vmd_cfg=cfg["vmd"],
            )
            vmd_rows.extend(rows)
            selected_k_by_component[component_name] = selected_k
            if theta_table is not None:
                theta_search_tables[component_name] = theta_table

    if not vmd_rows:
        raise ContractViolation(
            "invalid_model_policy",
            key="vmd_components",
            detail="no VMD/component modes were generated",
        )

    wpd_df = (
        pd.DataFrame(wpd_rows)
        .sort_values(["component_name", "timestamp"])
        .reset_index(drop=True)
    )
    vmd_df = (
        pd.DataFrame(vmd_rows)
        .sort_values(["component_name", "mode_name", "timestamp"])
        .reset_index(drop=True)
    )

    mode_point_frames: list[pd.DataFrame] = []
    mode_repeat_frames: list[pd.DataFrame] = []
    component_lstm_diagnostics: dict[str, dict[str, Any]] = {}
    lstm_engine = str(cfg["lstm"]["engine"])
    lstm_artifact_root: Path | None = None
    if artifact_root is not None and artifact_tag:
        lstm_artifact_root = Path(artifact_root) / str(variant) / str(artifact_tag)
    for mode_idx, (mode_name, mode_group) in enumerate(
        vmd_df.groupby("mode_name", sort=True)
    ):
        mode_artifact_dir: str | Path | None = None
        if lstm_artifact_root is not None:
            if mode_idx == 0:
                mode_artifact_dir = lstm_artifact_root
            else:
                mode_artifact_dir = lstm_artifact_root / f"mode_{mode_name}"
        if lstm_engine == "pytorch":
            mode_forecast = forecast_component_with_pytorch_lstm(
                mode_group["value"],
                horizons=cfg["forecast"]["horizons"],
                lstm_config=cfg["lstm"],
                component_name=str(mode_name),
                base_seed=model_seed + mode_idx * 97,
                artifact_dir=mode_artifact_dir,
            )
        else:
            mode_forecast = forecast_component_with_lstm(
                mode_group["value"],
                horizons=cfg["forecast"]["horizons"],
                lstm_config=cfg["lstm"],
                component_name=str(mode_name),
                base_seed=model_seed + mode_idx * 97,
            )
        mode_point = mode_forecast.point_forecast.copy()
        mode_point.insert(0, "mode_name", str(mode_name))
        mode_point_frames.append(mode_point)

        mode_repeats = mode_forecast.run_forecasts.copy()
        mode_repeats.insert(0, "mode_name", str(mode_name))
        mode_repeat_frames.append(mode_repeats)
        component_lstm_diagnostics[str(mode_name)] = mode_forecast.diagnostics

    if not mode_point_frames or not mode_repeat_frames:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm_modes",
            detail="no component-level LSTM forecasts were generated",
        )

    mode_points = (
        pd.concat(mode_point_frames, ignore_index=True)
        .sort_values(["horizon", "mode_name"])
        .reset_index(drop=True)
    )
    mode_repeats = (
        pd.concat(mode_repeat_frames, ignore_index=True)
        .sort_values(["horizon", "mode_name", "run_index"])
        .reset_index(drop=True)
    )

    aggregate_repeat = (
        mode_repeats.groupby(["horizon", "run_index"], sort=True)["point_forecast"]
        .sum()
        .reset_index()
    )
    noise_scale = float(cfg["training"]["seed_noise_scale"])
    if noise_scale > 0:
        noise_rng = np.random.default_rng(model_seed + 10_003)
        aggregate_repeat["point_forecast"] = aggregate_repeat["point_forecast"] + (
            noise_rng.normal(0.0, noise_scale, size=len(aggregate_repeat))
        )

    aggregate_point = (
        aggregate_repeat.groupby("horizon", sort=True)["point_forecast"]
        .mean()
        .reset_index()
    )
    repeat_dispersion = (
        aggregate_repeat.groupby("horizon", sort=True)["point_forecast"]
        .agg(["std", "min", "max"])
        .reset_index()
    )
    repeat_dispersion["std"] = repeat_dispersion["std"].fillna(0.0)
    repeat_dispersion["spread"] = repeat_dispersion["max"] - repeat_dispersion["min"]

    point_lookup = {
        int(row["horizon"]): float(row["point_forecast"])
        for _, row in aggregate_point.iterrows()
    }
    if normalize_by_days and month_context is not None:
        month_map = horizons_to_month_ends(
            context=month_context,
            horizons=point_lookup.keys(),
        )
        point_lookup = {
            int(horizon): daily_average_to_monthly_total(
                value,
                month_end=month_map[int(horizon)],
            )
            for horizon, value in point_lookup.items()
        }
    forecast_rows: list[dict[str, Any]] = []
    for horizon in cfg["forecast"]["horizons"]:
        horizon_int = int(horizon)
        if horizon_int not in point_lookup:
            raise ContractViolation(
                "missing_forecast_horizon",
                key=f"horizon={horizon_int}",
                detail="component LSTM aggregation did not return all requested horizons",
            )
        forecast_rows.append(
            {
                "horizon": horizon_int,
                "horizon_month_offset": int(
                    cfg["forecast"]["horizon_month_offset"][str(horizon_int)]
                ),
                "horizon_label": str(
                    cfg["forecast"]["horizon_label"][str(horizon_int)]
                ),
                "point_forecast": point_lookup[horizon_int],
                "model_family": cfg["model"]["family"],
                "model_variant": variant,
                "seed": model_seed,
            }
        )
    point_df = pd.DataFrame(forecast_rows).sort_values("horizon").reset_index(drop=True)
    point_df, transfer_diag = _apply_transfer_prior_adjustment(
        point_df,
        exogenous_features=exogenous_features,
        cfg=cfg,
    )
    point_df, weather_diag = apply_weather_shock_adjustment(
        point_df,
        exogenous_features=exogenous_features,
        cfg=cfg,
    )
    point_df, weather_state_diag = apply_weather_shock_state_adjustment(
        point_df,
        exogenous_features=exogenous_features,
        cfg=cfg,
    )

    effective_vmd_modes = (
        1
        if variant == "wpd_lstm_one_layer"
        else int(np.mean(list(selected_k_by_component.values())))
    )
    component_artifacts = {
        mode_name: {
            key: str(payload[key])
            for key in ("model_path", "manifest_path")
            if key in payload
        }
        for mode_name, payload in component_lstm_diagnostics.items()
        if isinstance(payload, Mapping)
    }
    diagnostics = {
        "model_family": cfg["model"]["family"],
        "model_variant": variant,
        "lstm_engine": str(lstm_engine),
        "decomposition_depth": (
            "one_layer" if variant == "wpd_lstm_one_layer" else "two_layer"
        ),
        "vmd_applied": variant != "wpd_lstm_one_layer",
        "seed": model_seed,
        "lookback": int(cfg["lstm"]["lookback"]),
        "target_transform": {
            "normalize_by_days_in_month": normalize_by_days,
            "log": False,
        },
        "wpd_component_count": int(len(component_names)),
        "wpd_wavelet": f"{cfg['wpd']['wavelet_family']}{cfg['wpd']['wavelet_order']}",
        "wpd_levels": int(cfg["wpd"]["levels"]),
        "wpd_node_paths": list(wpd_result.node_paths),
        "wpd_reconstruction_rmse": float(wpd_result.reconstruction_rmse),
        "wpd_reconstruction_relative_l2": float(wpd_result.reconstruction_relative_l2),
        "entropy_threshold_mean": entropy_threshold,
        "high_complexity_components": high_complexity_components,
        "vmd_modes": int(effective_vmd_modes),
        "vmd_selected_k_by_component": {
            key: int(value) for key, value in sorted(selected_k_by_component.items())
        },
        "theta_search_rows": {
            key: int(len(value)) for key, value in sorted(theta_search_tables.items())
        },
        "vmd_strategy": variant,
        "lstm_component_count": int(mode_points["mode_name"].nunique()),
        "lstm_repeat_runs": int(cfg["lstm"]["repeat_runs"]),
        "component_lstm_input_shape": [int(cfg["lstm"]["lookback"]), 1],
        "component_readout_rmse_mean": float(
            np.mean(
                [
                    float(payload.get("readout_rmse_mean", 0.0))
                    for payload in component_lstm_diagnostics.values()
                ]
            )
        ),
        "seed_repeat_stats": {
            str(int(row["horizon"])): {
                "std": float(row["std"]),
                "spread": float(row["spread"]),
            }
            for _, row in repeat_dispersion.iterrows()
        },
        "mode_forecast_rows": int(len(mode_points)),
        "mode_repeat_rows": int(len(mode_repeats)),
        "training_obs": int(len(signal)),
        "last_observation": float(signal.iloc[-1]),
        "horizon_month_offset": dict(cfg["forecast"]["horizon_month_offset"]),
        "horizon_label": dict(cfg["forecast"]["horizon_label"]),
        "component_lstm_diagnostics": component_lstm_diagnostics,
        "component_lstm_artifacts": component_artifacts,
        "lstm_artifact_root": (
            str(lstm_artifact_root) if lstm_artifact_root is not None else ""
        ),
        "exogenous_weather_shock": weather_diag,
        "exogenous_weather_shock_state": weather_state_diag,
        "exogenous_transfer_priors": transfer_diag,
    }

    return ChampionRunResult(
        point_forecast=point_df,
        wpd_components=wpd_df,
        entropy_scores=entropy_df,
        vmd_components=vmd_df,
        diagnostics=diagnostics,
    )


def run_champion_seed_repeats(
    frame: pd.DataFrame,
    config: Mapping[str, Any] | None,
    seeds: Sequence[int],
    *,
    timestamp_col: str = "timestamp",
    target_col: str = "target_value",
    exogenous_features: Mapping[str, Any] | None = None,
) -> ChampionSeedRunResult:
    """Execute repeated-seed champion runs for stability diagnostics."""

    if not seeds:
        raise ContractViolation(
            "invalid_model_policy",
            key="training.seed_repeats",
            detail="at least one seed is required for repeated runs",
        )

    cfg = validate_champion_config(config)
    rows: list[pd.DataFrame] = []
    for model_seed in seeds:
        run = run_champion_pipeline(
            frame,
            cfg,
            timestamp_col=timestamp_col,
            target_col=target_col,
            seed=int(model_seed),
            exogenous_features=exogenous_features,
        )
        rows.append(run.point_forecast)

    forecasts = (
        pd.concat(rows, ignore_index=True)
        .sort_values(["seed", "horizon"])
        .reset_index(drop=True)
    )
    diagnostics = {
        "seed_count": len(seeds),
        "seeds": [int(model_seed) for model_seed in seeds],
    }
    return ChampionSeedRunResult(forecasts=forecasts, diagnostics=diagnostics)
