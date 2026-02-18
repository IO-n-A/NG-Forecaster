"""Regime bucketing utilities for CP4 regime-aware fusion policies."""

from __future__ import annotations

from typing import Any, Mapping

import pandas as pd

from ng_forecaster.features.lag_guard import require_columns

REGIME_NORMAL = "normal"
REGIME_FREEZE_OFF = "freeze_off"
REGIME_BASIS_BLOWOUT = "basis_blowout"
REGIME_TRANSFER_DISPERSION = "transfer_dispersion"
REGIME_MULTI_SHOCK = "multi_shock"

REGIME_ORDER: tuple[str, ...] = (
    REGIME_NORMAL,
    REGIME_FREEZE_OFF,
    REGIME_BASIS_BLOWOUT,
    REGIME_TRANSFER_DISPERSION,
    REGIME_MULTI_SHOCK,
)


def _flag(values: Mapping[str, Any], *keys: str) -> int:
    for key in keys:
        if key not in values:
            continue
        raw = values.get(key)
        if raw is None:
            continue
        try:
            numeric = float(raw)
        except (TypeError, ValueError):
            continue
        if pd.isna(numeric):
            continue
        return int(numeric >= 0.5)
    return 0


def classify_regime(values: Mapping[str, Any]) -> str:
    """Classify one row of regime features into a deterministic regime label."""

    freeze = _flag(values, "regime_freeze_flag", "freeze_event_flag")
    basis = _flag(values, "regime_basis_flag", "basis_blowout_flag")
    transfer = _flag(
        values,
        "regime_transfer_dispersion_flag",
        "transfer_dispersion_flag",
    )
    active = int(bool(freeze) + bool(basis) + bool(transfer))
    if active == 0:
        return REGIME_NORMAL
    if active > 1:
        return REGIME_MULTI_SHOCK
    if freeze == 1:
        return REGIME_FREEZE_OFF
    if basis == 1:
        return REGIME_BASIS_BLOWOUT
    return REGIME_TRANSFER_DISPERSION


def classify_regime_frame(
    frame: pd.DataFrame,
    *,
    freeze_col: str = "regime_freeze_flag",
    basis_col: str = "regime_basis_flag",
    transfer_col: str = "regime_transfer_dispersion_flag",
    output_col: str = "regime_label",
) -> pd.DataFrame:
    """Append a regime label column to a frame of scored runs."""

    require_columns(frame, (freeze_col, basis_col, transfer_col), key="regime_frame")
    output = frame.copy()
    output[output_col] = output.apply(
        lambda row: classify_regime(
            {
                "regime_freeze_flag": row[freeze_col],
                "regime_basis_flag": row[basis_col],
                "regime_transfer_dispersion_flag": row[transfer_col],
            }
        ),
        axis=1,
    )
    return output
