"""Regime-flag engineering for freeze-off, basis stress, and transfer dispersion."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ng_forecaster.data.validators import load_yaml
from ng_forecaster.errors import ContractViolation

DEFAULT_REGIME_THRESHOLDS: dict[str, float] = {
    "freeze_days_high": 5.0,
    "freeze_intensity_high": 1.5,
    "freeze_event_share_high": 0.20,
    "basis_spread_high": 1.00,
    "transfer_dispersion_high": 2.50,
}

_BASIS_SPREAD_KEYS = (
    "steo_regional_residential_spread_usd_mcf_t",
    "steo_regional_commercial_spread_usd_mcf_t",
    "steo_regional_industrial_spread_usd_mcf_t",
    "steo_regional_residential_spread_usd_mcf_t_plus_1",
    "steo_regional_commercial_spread_usd_mcf_t_plus_1",
    "steo_regional_industrial_spread_usd_mcf_t_plus_1",
)
_TRANSFER_DISPERSION_KEYS = (
    "transfer_prior_dispersion_t",
    "transfer_prior_dispersion_t_plus_1",
)


def _as_float(value: object) -> float:
    if not isinstance(value, (int, float, str)):
        return 0.0
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return 0.0
    if pd.isna(numeric):
        return 0.0
    return numeric


def _positive_thresholds(thresholds: Mapping[str, Any]) -> dict[str, float]:
    normalized: dict[str, float] = {}
    for key, default in DEFAULT_REGIME_THRESHOLDS.items():
        value = _as_float(thresholds.get(key, default))
        if value <= 0:
            raise ContractViolation(
                "invalid_model_policy",
                key=f"regime_thresholds.{key}",
                detail="threshold must be > 0",
            )
        normalized[key] = value
    return normalized


def load_regime_thresholds(
    *,
    config_path: str = "configs/features.yaml",
) -> dict[str, float]:
    """Load and validate regime thresholds from features config."""

    payload = load_yaml(config_path)
    thresholds = payload.get("regime_thresholds", {})
    if thresholds is None:
        thresholds = {}
    if not isinstance(thresholds, Mapping):
        raise ContractViolation(
            "invalid_feature_policy",
            key="regime_thresholds",
            detail="regime_thresholds must be a mapping when provided",
        )
    return _positive_thresholds(thresholds)


def compute_regime_flags(
    values: Mapping[str, Any],
    *,
    thresholds: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    """Compute regime flags from a feature-value mapping."""

    limits = _positive_thresholds(thresholds or DEFAULT_REGIME_THRESHOLDS)

    freeze_days = max(
        _as_float(values.get("freeze_days_mtd_weighted")),
        _as_float(values.get("freeze_days_mtd")),
    )
    freeze_intensity = max(
        _as_float(values.get("freeze_intensity_mtd_weighted")),
        _as_float(values.get("freeze_event_intensity")),
    )
    freeze_share = max(
        _as_float(values.get("freeze_event_share_mtd_weighted")),
        _as_float(values.get("freeze_event_flag")),
    )
    freeze_score = max(
        freeze_days / limits["freeze_days_high"],
        freeze_intensity / limits["freeze_intensity_high"],
        freeze_share / limits["freeze_event_share_high"],
    )
    freeze_flag = int(freeze_score >= 1.0)

    basis_proxy = max(abs(_as_float(values.get(key))) for key in _BASIS_SPREAD_KEYS)
    basis_score = basis_proxy / limits["basis_spread_high"]
    basis_flag = int(basis_score >= 1.0)

    transfer_proxy = max(
        abs(_as_float(values.get(key))) for key in _TRANSFER_DISPERSION_KEYS
    )
    transfer_score = transfer_proxy / limits["transfer_dispersion_high"]
    transfer_flag = int(transfer_score >= 1.0)

    composite_score = float((freeze_score + basis_score + transfer_score) / 3.0)
    any_flag = int(any((freeze_flag, basis_flag, transfer_flag)))

    return {
        "regime_freeze_flag": float(freeze_flag),
        "regime_basis_flag": float(basis_flag),
        "regime_transfer_dispersion_flag": float(transfer_flag),
        "regime_any_flag": float(any_flag),
        "regime_freeze_score": float(freeze_score),
        "regime_basis_score": float(basis_score),
        "regime_transfer_dispersion_score": float(transfer_score),
        "regime_score": composite_score,
        "regime_basis_spread_proxy": float(basis_proxy),
        "regime_transfer_dispersion_proxy": float(transfer_proxy),
    }


def append_regime_flags(
    panel: pd.DataFrame,
    *,
    thresholds: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Append regime flags to a vintage panel row frame."""

    if panel.empty:
        return panel.copy()
    rows = []
    for _, row in panel.iterrows():
        rows.append(compute_regime_flags(row.to_dict(), thresholds=thresholds))
    flags = pd.DataFrame(rows, index=panel.index)
    return pd.concat(
        [panel.reset_index(drop=True), flags.reset_index(drop=True)], axis=1
    )
