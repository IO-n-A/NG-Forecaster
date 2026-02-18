"""Controlled month-length effect diagnostics for validation exports."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.lag_guard import require_columns


def _normal_cdf(value: float) -> float:
    return 0.5 * (1.0 + math.erf(float(value) / math.sqrt(2.0)))


def _safe_numeric(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    return values.replace([np.inf, -np.inf], np.nan)


def _regression_day31_effect(frame: pd.DataFrame) -> tuple[float, float, float, float]:
    """Estimate day31 effect with month/regime fixed effects via OLS."""

    required = {"ape_pct", "target_month", "target_month_days", "regime_label"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ContractViolation(
            "missing_column",
            key="month_length_effect",
            detail="missing columns for regression: " + ", ".join(missing),
        )

    reg = frame[
        ["ape_pct", "target_month", "target_month_days", "regime_label"]
    ].copy()
    reg["ape_pct"] = _safe_numeric(reg["ape_pct"])
    reg["target_month"] = pd.to_datetime(reg["target_month"], errors="coerce")
    reg["target_month_days"] = pd.to_numeric(reg["target_month_days"], errors="coerce")
    reg = reg.dropna()
    if len(reg) < 8:
        return np.nan, np.nan, np.nan, np.nan

    reg["day31"] = (reg["target_month_days"].astype(int) == 31).astype(float)
    reg["month_num"] = reg["target_month"].dt.month.astype(int)

    month_dummies = pd.get_dummies(reg["month_num"], prefix="m", drop_first=True)
    regime_dummies = pd.get_dummies(
        reg["regime_label"].astype(str), prefix="r", drop_first=True
    )
    x = pd.concat(
        [
            pd.Series(1.0, index=reg.index, name="intercept"),
            reg["day31"],
            month_dummies,
            regime_dummies,
        ],
        axis=1,
    )
    y = reg["ape_pct"].astype(float).to_numpy()
    x_matrix = x.to_numpy(dtype=float)

    beta, _, _, _ = np.linalg.lstsq(x_matrix, y, rcond=None)
    fitted = x_matrix @ beta
    resid = y - fitted
    n_obs, n_params = x_matrix.shape
    dof = max(1, n_obs - n_params)
    rss = float(np.sum(np.square(resid)))
    sigma2 = rss / float(dof)
    xtx_inv = np.linalg.pinv(x_matrix.T @ x_matrix)
    var_beta = sigma2 * xtx_inv

    day31_idx = list(x.columns).index("day31")
    coef = float(beta[day31_idx])
    stderr = float(max(var_beta[day31_idx, day31_idx], 0.0) ** 0.5)
    if stderr <= 1e-12:
        return coef, 0.0, np.nan, np.nan
    t_stat = coef / stderr
    p_value_two_sided = 2.0 * min(_normal_cdf(t_stat), 1.0 - _normal_cdf(t_stat))
    return coef, stderr, float(t_stat), float(p_value_two_sided)


def build_month_length_effect_report(point_estimates: pd.DataFrame) -> pd.DataFrame:
    """Build per-variant raw and controlled month-length effect metrics."""

    require_columns(
        point_estimates,
        ("model_variant", "target_month_days", "ape_pct", "target_month", "regime_label"),
        key="point_estimates",
    )
    rows: list[dict[str, float | int | str]] = []
    for variant, group in point_estimates.groupby("model_variant", sort=True):
        g = group.copy()
        g["target_month_days"] = pd.to_numeric(g["target_month_days"], errors="coerce")
        g["ape_pct"] = _safe_numeric(g["ape_pct"])
        g = g.dropna(subset=["target_month_days", "ape_pct"])
        day30 = g[g["target_month_days"].astype(int) == 30]["ape_pct"]
        day31 = g[g["target_month_days"].astype(int) == 31]["ape_pct"]
        raw_gap = np.nan
        if not day30.empty and not day31.empty:
            raw_gap = float(day31.mean() - day30.mean())

        coef, stderr, t_stat, p_value = _regression_day31_effect(g)
        rows.append(
            {
                "model_variant": str(variant),
                "n_obs": int(len(g)),
                "n_30d": int(len(day30)),
                "n_31d": int(len(day31)),
                "mean_ape_30d": float(day30.mean()) if not day30.empty else np.nan,
                "mean_ape_31d": float(day31.mean()) if not day31.empty else np.nan,
                "raw_gap_31d_minus_30d_ape_pct": float(raw_gap),
                "controlled_day31_coef_ape_pct": float(coef),
                "controlled_day31_stderr": float(stderr),
                "controlled_day31_t_stat": float(t_stat),
                "controlled_day31_p_value_two_sided": float(p_value),
            }
        )

    return pd.DataFrame(rows).sort_values("model_variant").reset_index(drop=True)


def build_month_length_by_regime_report(point_estimates: pd.DataFrame) -> pd.DataFrame:
    """Build within-regime month-length diagnostics."""

    require_columns(
        point_estimates,
        ("model_variant", "regime_label", "target_month_days", "ape_pct"),
        key="point_estimates",
    )
    rows: list[dict[str, float | int | str]] = []
    for (variant, regime), group in point_estimates.groupby(
        ["model_variant", "regime_label"],
        sort=True,
    ):
        g = group.copy()
        g["target_month_days"] = pd.to_numeric(g["target_month_days"], errors="coerce")
        g["ape_pct"] = _safe_numeric(g["ape_pct"])
        g = g.dropna(subset=["target_month_days", "ape_pct"])
        day30 = g[g["target_month_days"].astype(int) == 30]["ape_pct"]
        day31 = g[g["target_month_days"].astype(int) == 31]["ape_pct"]
        gap = np.nan
        if not day30.empty and not day31.empty:
            gap = float(day31.mean() - day30.mean())
        rows.append(
            {
                "model_variant": str(variant),
                "regime_label": str(regime),
                "n_obs": int(len(g)),
                "n_30d": int(len(day30)),
                "n_31d": int(len(day31)),
                "mean_ape_30d": float(day30.mean()) if not day30.empty else np.nan,
                "mean_ape_31d": float(day31.mean()) if not day31.empty else np.nan,
                "gap_31d_minus_30d_ape_pct": float(gap),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["model_variant", "regime_label"])
        .reset_index(drop=True)
    )


def build_calendar_calibration_report(point_estimates: pd.DataFrame) -> pd.DataFrame:
    """Summarize raw vs calibrated error metrics by variant/day class."""

    require_columns(
        point_estimates,
        (
            "model_variant",
            "day_count_class",
            "ape_pct",
            "ape_pct_raw",
            "abs_error",
            "abs_error_raw",
            "calendar_calibration_applied",
        ),
        key="point_estimates",
    )
    rows: list[dict[str, float | int | str | bool]] = []
    for (variant, day_class), group in point_estimates.groupby(
        ["model_variant", "day_count_class"],
        sort=True,
    ):
        g = group.copy()
        g["ape_pct"] = _safe_numeric(g["ape_pct"])
        g["ape_pct_raw"] = _safe_numeric(g["ape_pct_raw"])
        g["abs_error"] = _safe_numeric(g["abs_error"])
        g["abs_error_raw"] = _safe_numeric(g["abs_error_raw"])
        rows.append(
            {
                "model_variant": str(variant),
                "day_count_class": str(day_class),
                "n_obs": int(len(g)),
                "calibration_applied_rate": float(
                    pd.to_numeric(
                        g["calendar_calibration_applied"].astype(int),
                        errors="coerce",
                    ).mean()
                ),
                "mean_abs_error_raw": float(g["abs_error_raw"].mean()),
                "mean_abs_error_calibrated": float(g["abs_error"].mean()),
                "delta_abs_error_calibrated_minus_raw": float(
                    g["abs_error"].mean() - g["abs_error_raw"].mean()
                ),
                "mean_ape_pct_raw": float(g["ape_pct_raw"].mean()),
                "mean_ape_pct_calibrated": float(g["ape_pct"].mean()),
                "delta_ape_pct_calibrated_minus_raw": float(
                    g["ape_pct"].mean() - g["ape_pct_raw"].mean()
                ),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["model_variant", "day_count_class"])
        .reset_index(drop=True)
    )
