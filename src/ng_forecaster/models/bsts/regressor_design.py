"""Regressor-design helpers for BSTS challenger models."""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np
import pandas as pd


def _safe_zscore(values: pd.Series) -> pd.Series:
    mean = float(values.mean())
    std = float(values.std(ddof=0))
    if std <= 1e-12:
        return pd.Series(0.0, index=values.index)
    return (values - mean) / std


def build_bsts_regressor_design(
    observed: pd.Series,
    *,
    timestamps: pd.Series,
    exogenous_features: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build compact shrinkage-ready regressor summaries for BSTS."""

    y = pd.to_numeric(observed, errors="coerce").astype(float).reset_index(drop=True)
    ts = pd.to_datetime(timestamps, errors="coerce").reset_index(drop=True)
    design = pd.DataFrame(index=y.index)
    design["level_z"] = _safe_zscore(y)
    design["delta_1"] = _safe_zscore(y.diff().fillna(0.0))
    design["momentum_3"] = _safe_zscore((y - y.shift(3)).fillna(0.0))
    rolling = y.rolling(window=6, min_periods=3).mean().fillna(y.expanding().mean())
    design["roll6_z"] = _safe_zscore(rolling)
    month_angle = 2.0 * np.pi * (ts.dt.month.astype(float) / 12.0)
    design["month_sin"] = np.sin(month_angle)
    design["month_cos"] = np.cos(month_angle)

    if exogenous_features:
        candidate_keys = [
            "freeze_intensity_mtd_weighted",
            "freeze_days_mtd_weighted",
            "regime_basis_spread_proxy",
            "transfer_prior_us_bcfd_t",
            "transfer_prior_dispersion_t",
            "steo_dry_prod_bcfd_t",
            "steo_henry_hub_price_t",
            "wti_spot_price_usd_t",
        ]
        for key in candidate_keys:
            if key not in exogenous_features:
                continue
            try:
                value = float(exogenous_features[key])
            except (TypeError, ValueError):
                continue
            if np.isnan(value):
                continue
            design[f"exo_{key}"] = value

    design = design.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    for column in list(design.columns):
        design[column] = pd.to_numeric(design[column], errors="coerce").fillna(0.0)
    return design
