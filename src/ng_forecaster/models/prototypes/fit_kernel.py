"""Fitting utilities for the prototype cohort kernel forecaster."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.lag_guard import require_columns
from ng_forecaster.models.prototypes.kernels import bounded_value

_DRILLING_REQUIRED = (
    "timestamp",
    "available_timestamp",
    "new_wells_completed",
    "new_well_gas_production",
    "existing_gas_production_change",
    "duc_inventory",
    "completion_to_duc_ratio",
)


def load_drilling_metrics_history(
    *,
    asof: object,
    panel_path: str = "data/gold/steo_drilling_metrics_panel.parquet",
    lookback_months: int = 36,
) -> pd.DataFrame:
    """Load latest-asof drilling metrics panel for kernel fitting/forecasting."""

    panel = pd.read_parquet(panel_path)
    require_columns(panel, _DRILLING_REQUIRED, key="steo_drilling_metrics_panel")

    frame = panel[list(_DRILLING_REQUIRED)].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["available_timestamp"] = pd.to_datetime(
        frame["available_timestamp"],
        errors="coerce",
    )
    for column in _DRILLING_REQUIRED:
        if column in {"timestamp", "available_timestamp"}:
            continue
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame[
        frame["timestamp"].notna() & frame["available_timestamp"].notna()
    ].copy()
    if frame.empty:
        raise ContractViolation(
            "missing_feature_input",
            key=panel_path,
            detail="drilling metrics panel is empty after timestamp normalization",
        )

    asof_ts = pd.Timestamp(asof)
    frame = frame[frame["available_timestamp"] <= asof_ts].copy()
    if frame.empty:
        raise ContractViolation(
            "lag_policy_violated",
            key=panel_path,
            detail="no drilling rows are available at requested asof",
        )

    frame = frame.sort_values(["timestamp", "available_timestamp"])
    frame = frame.groupby("timestamp", as_index=False).tail(1)
    frame = frame.sort_values("timestamp").reset_index(drop=True)

    parsed_lookback = int(lookback_months)
    if parsed_lookback < 6:
        raise ContractViolation(
            "invalid_model_policy",
            key="lookback_months",
            detail="lookback_months must be >= 6",
        )
    frame = frame.tail(parsed_lookback).reset_index(drop=True)
    if len(frame) < 12:
        raise ContractViolation(
            "insufficient_release_history",
            key=panel_path,
            detail=f"need >=12 drilling rows after asof filter; received={len(frame)}",
        )
    return frame


def _prepare_release_history(release_history: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        release_history, ("timestamp", "target_value"), key="release_history"
    )
    frame = release_history[["timestamp", "target_value"]].copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["target_value"] = pd.to_numeric(frame["target_value"], errors="coerce")
    frame = frame[frame["timestamp"].notna() & frame["target_value"].notna()].copy()
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    frame["days_in_month"] = frame["timestamp"].dt.days_in_month.astype(float)
    frame["daily_target"] = frame["target_value"] / frame["days_in_month"]
    if len(frame) < 12:
        raise ContractViolation(
            "insufficient_release_history",
            key="release_history",
            detail=f"need >=12 release rows for kernel fitting; received={len(frame)}",
        )
    return frame.reset_index(drop=True)


def _build_training_frame(
    *,
    release_history: pd.DataFrame,
    drilling_history: pd.DataFrame,
) -> pd.DataFrame:
    frame = drilling_history.copy()
    frame["completion_signal"] = (
        pd.to_numeric(frame["new_wells_completed"], errors="coerce")
        * pd.to_numeric(frame["new_well_gas_production"], errors="coerce")
        / pd.to_numeric(frame["duc_inventory"], errors="coerce").clip(lower=1.0)
    )
    frame["legacy_signal"] = pd.to_numeric(
        frame["existing_gas_production_change"],
        errors="coerce",
    )
    frame["duc_signal"] = pd.to_numeric(
        frame["completion_to_duc_ratio"], errors="coerce"
    )
    frame = frame[
        ["timestamp", "completion_signal", "legacy_signal", "duc_signal"]
    ].copy()
    frame = frame.merge(
        release_history[["timestamp", "daily_target"]],
        on="timestamp",
        how="inner",
    )
    frame = frame.sort_values("timestamp").reset_index(drop=True)
    frame["daily_target_next"] = frame["daily_target"].shift(-1)
    frame["daily_delta_next"] = frame["daily_target_next"] - frame["daily_target"]
    frame = frame.dropna(
        subset=[
            "completion_signal",
            "legacy_signal",
            "duc_signal",
            "daily_target",
            "daily_delta_next",
        ]
    ).reset_index(drop=True)
    return frame


def fit_kernel_parameters(
    *,
    release_history: pd.DataFrame,
    drilling_history: pd.DataFrame,
    ridge_alpha: float = 0.25,
    min_rows: int = 12,
) -> dict[str, Any]:
    """Fit constrained kernel sensitivities from aligned release/drilling history."""

    release_prepared = _prepare_release_history(release_history)
    drilling_prepared = drilling_history.copy()
    require_columns(drilling_prepared, _DRILLING_REQUIRED, key="drilling_history")
    for column in _DRILLING_REQUIRED:
        if column in {"timestamp", "available_timestamp"}:
            drilling_prepared[column] = pd.to_datetime(
                drilling_prepared[column],
                errors="coerce",
            )
        else:
            drilling_prepared[column] = pd.to_numeric(
                drilling_prepared[column],
                errors="coerce",
            )
    drilling_prepared = drilling_prepared[drilling_prepared["timestamp"].notna()].copy()

    training = _build_training_frame(
        release_history=release_prepared,
        drilling_history=drilling_prepared,
    )
    if len(training) < int(min_rows):
        raise ContractViolation(
            "insufficient_release_history",
            key="kernel_training_rows",
            detail=f"need >= {int(min_rows)} aligned rows; received={len(training)}",
        )

    signal_cols = ("completion_signal", "legacy_signal", "duc_signal")
    signal_mean: dict[str, float] = {}
    signal_std: dict[str, float] = {}
    z_frame = pd.DataFrame(index=training.index)
    for column in signal_cols:
        values = pd.to_numeric(training[column], errors="coerce")
        mean_value = float(values.mean())
        std_value = float(values.std(ddof=0))
        if not np.isfinite(std_value) or std_value < 1e-9:
            std_value = 1.0
        z_frame[column] = (values - mean_value) / std_value
        signal_mean[column] = mean_value
        signal_std[column] = std_value

    y = pd.to_numeric(training["daily_delta_next"], errors="coerce").to_numpy(
        dtype=float
    )
    x = z_frame[list(signal_cols)].to_numpy(dtype=float)
    x_aug = np.column_stack([np.ones(len(x)), x])

    alpha = float(ridge_alpha)
    if alpha < 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="ridge_alpha",
            detail="ridge_alpha must be >= 0",
        )
    ridge = np.diag([0.0, alpha, alpha, alpha])
    xtx = x_aug.T @ x_aug + ridge
    xty = x_aug.T @ y
    try:
        beta = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        beta = np.linalg.lstsq(xtx, xty, rcond=None)[0]

    intercept = bounded_value(beta[0], lower=-5000.0, upper=5000.0, key="intercept")
    completion_coef = bounded_value(
        beta[1],
        lower=0.0,
        upper=8000.0,
        key="completion_coef",
    )
    legacy_coef = bounded_value(
        beta[2],
        lower=-8000.0,
        upper=8000.0,
        key="legacy_coef",
    )
    duc_coef = bounded_value(beta[3], lower=0.0, upper=8000.0, key="duc_coef")

    y_series = pd.Series(y)
    if len(y_series) >= 3 and y_series.std(ddof=0) > 0:
        autocorr = float(y_series.autocorr(lag=1))
    else:
        autocorr = 0.0
    if not np.isfinite(autocorr) or autocorr <= 0:
        half_life = 2.0
    elif autocorr >= 0.999:
        half_life = 6.0
    else:
        half_life = float(np.log(0.5) / np.log(autocorr))
    half_life = bounded_value(
        half_life,
        lower=1.0,
        upper=6.0,
        key="half_life_months",
    )

    max_daily_delta = float(pd.Series(y).abs().quantile(0.90))
    if not np.isfinite(max_daily_delta) or max_daily_delta <= 0:
        max_daily_delta = float(pd.Series(y).abs().mean())
    if not np.isfinite(max_daily_delta) or max_daily_delta <= 0:
        max_daily_delta = 500.0
    max_daily_delta = max(max_daily_delta, 25.0)

    return {
        "intercept_daily_delta": float(intercept),
        "completion_coef": float(completion_coef),
        "legacy_coef": float(legacy_coef),
        "duc_coef": float(duc_coef),
        "signal_mean": signal_mean,
        "signal_std": signal_std,
        "half_life_months": float(half_life),
        "max_daily_delta": float(max_daily_delta),
        "training_rows": int(len(training)),
        "ridge_alpha": float(alpha),
    }
