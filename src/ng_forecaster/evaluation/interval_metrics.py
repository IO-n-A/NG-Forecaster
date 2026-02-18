"""Interval-quality metrics and calibration exports for 24m validation."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.lag_guard import require_columns

_ALPHAS = (0.5, 0.2, 0.05, 0.01)  # nominal coverages: 50/80/95/99
_PINBALL_QUANTILES = (0.1, 0.5, 0.9)
_BASE_LOWER_Q = 0.025
_BASE_UPPER_Q = 0.975


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    require_columns(
        frame,
        ("model_variant", "actual_released", "fused_point", "fused_lower_95", "fused_upper_95"),
        key="point_estimates",
    )
    data = frame.copy()
    for column in ("actual_released", "fused_point", "fused_lower_95", "fused_upper_95"):
        data[column] = pd.to_numeric(data[column], errors="coerce")
    data = data.dropna(
        subset=["actual_released", "fused_point", "fused_lower_95", "fused_upper_95"]
    ).copy()
    if data.empty:
        raise ContractViolation(
            "invalid_metric_payload",
            key="point_estimates",
            detail="interval metrics require non-empty numeric rows",
        )
    data["fused_lower_95"] = data[["fused_lower_95", "fused_upper_95"]].min(axis=1)
    data["fused_upper_95"] = data[["fused_lower_95", "fused_upper_95"]].max(axis=1)
    data["interval_width_95"] = data["fused_upper_95"] - data["fused_lower_95"]
    return data


def _pinball(y: pd.Series, qhat: pd.Series, q: float) -> pd.Series:
    diff = y - qhat
    return pd.Series(np.where(diff >= 0, q * diff, (q - 1.0) * diff), index=y.index).abs()


def _interval_score(y: pd.Series, lower: pd.Series, upper: pd.Series, alpha: float) -> pd.Series:
    below = (lower - y).clip(lower=0.0)
    above = (y - upper).clip(lower=0.0)
    return (upper - lower) + (2.0 / float(alpha)) * (below + above)


def _coverage_from_center(
    center: pd.Series,
    half_95: pd.Series,
    y: pd.Series,
    *,
    alpha: float,
) -> pd.Series:
    if float(alpha) == 0.05:
        scale = 1.0
    else:
        # Deterministic width scaling from the available 95% interval.
        scale = float((1.0 - alpha) / 0.95)
    half = half_95 * scale
    lower = center - half
    upper = center + half
    return ((y >= lower) & (y <= upper)).astype(float)


def _quantile_from_95(
    lower_95: pd.Series,
    upper_95: pd.Series,
    *,
    q: float,
) -> pd.Series:
    weight = float((q - _BASE_LOWER_Q) / (_BASE_UPPER_Q - _BASE_LOWER_Q))
    weight = min(max(weight, 0.0), 1.0)
    return lower_95 + weight * (upper_95 - lower_95)


def _score_group(group: pd.DataFrame) -> dict[str, Any]:
    y = group["actual_released"].astype(float)
    median = group["fused_point"].astype(float)
    lower_95 = group["fused_lower_95"].astype(float)
    upper_95 = group["fused_upper_95"].astype(float)
    half_95 = (upper_95 - lower_95) / 2.0
    center = median

    row: dict[str, Any] = {
        "n_obs": int(len(group)),
        "mean_interval_width_95": float((upper_95 - lower_95).mean()),
        "wis": float(
            (
                0.5 * (y - median).abs()
                + 0.5 * _interval_score(y, lower_95, upper_95, alpha=0.05)
            ).mean()
        ),
    }
    for alpha in _ALPHAS:
        nominal = int(round((1.0 - float(alpha)) * 100))
        row[f"coverage_{nominal}"] = float(
            _coverage_from_center(center, half_95, y, alpha=float(alpha)).mean()
        )

    for q in _PINBALL_QUANTILES:
        qhat = median if float(q) == 0.5 else _quantile_from_95(lower_95, upper_95, q=float(q))
        row[f"pinball_q{int(round(q * 100)):02d}"] = float(_pinball(y, qhat, float(q)).mean())
    return row


def build_interval_scorecard(point_estimates: pd.DataFrame) -> pd.DataFrame:
    data = _prepare(point_estimates)
    rows: list[dict[str, Any]] = []
    for variant, group in data.groupby("model_variant", sort=True):
        rows.append({"model_variant": str(variant), **_score_group(group)})
    return pd.DataFrame(rows).sort_values("model_variant").reset_index(drop=True)


def build_interval_scorecard_by_regime(point_estimates: pd.DataFrame) -> pd.DataFrame:
    data = _prepare(point_estimates)
    if "regime_label" not in data.columns:
        data["regime_label"] = "unknown"
    rows: list[dict[str, Any]] = []
    for (regime_label, variant), group in data.groupby(
        ["regime_label", "model_variant"],
        sort=True,
    ):
        rows.append(
            {
                "regime_label": str(regime_label),
                "model_variant": str(variant),
                **_score_group(group),
            }
        )
    if not rows:
        return pd.DataFrame(columns=["regime_label", "model_variant"])
    return pd.DataFrame(rows).sort_values(["regime_label", "model_variant"]).reset_index(drop=True)


def build_interval_calibration_tables(
    point_estimates: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    data = _prepare(point_estimates)
    if "regime_label" not in data.columns:
        data["regime_label"] = "unknown"

    overall_rows: list[dict[str, Any]] = []
    by_regime_rows: list[dict[str, Any]] = []

    def _append_rows(prefix: dict[str, Any], group: pd.DataFrame, dest: list[dict[str, Any]]) -> None:
        y = group["actual_released"].astype(float)
        center = group["fused_point"].astype(float)
        half_95 = (group["fused_upper_95"].astype(float) - group["fused_lower_95"].astype(float)) / 2.0
        for alpha in _ALPHAS:
            nominal = float(1.0 - float(alpha))
            empirical = float(
                _coverage_from_center(center, half_95, y, alpha=float(alpha)).mean()
            )
            dest.append(
                {
                    **prefix,
                    "nominal_coverage": nominal,
                    "empirical_coverage": empirical,
                    "coverage_gap": empirical - nominal,
                    "n_obs": int(len(group)),
                    "mean_interval_width_95": float(
                        (
                            group["fused_upper_95"].astype(float)
                            - group["fused_lower_95"].astype(float)
                        ).mean()
                    ),
                }
            )

    for variant, group in data.groupby("model_variant", sort=True):
        _append_rows({"model_variant": str(variant)}, group, overall_rows)

    for (regime_label, variant), group in data.groupby(
        ["regime_label", "model_variant"],
        sort=True,
    ):
        _append_rows(
            {"regime_label": str(regime_label), "model_variant": str(variant)},
            group,
            by_regime_rows,
        )

    overall = (
        pd.DataFrame(overall_rows)
        .sort_values(["model_variant", "nominal_coverage"])
        .reset_index(drop=True)
    )
    by_regime = (
        pd.DataFrame(by_regime_rows)
        .sort_values(["regime_label", "model_variant", "nominal_coverage"])
        .reset_index(drop=True)
    )
    return overall, by_regime
