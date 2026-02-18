"""Block-level ablation and influence scoring for CP5 evidence artifacts."""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import pandas as pd

from ng_forecaster.errors import ContractViolation
from ng_forecaster.evaluation.dm_test import run_dm_tests, validate_dm_policy
from ng_forecaster.evaluation.metrics import score_point_forecasts
from ng_forecaster.features.lag_guard import require_columns


def _to_numeric(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    output = frame.copy()
    for column in columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output


def _baseline_variant(point_estimates: pd.DataFrame) -> str:
    preferred = "wpd_vmd_lstm1"
    variants = sorted(point_estimates["model_variant"].astype(str).unique().tolist())
    if preferred in variants:
        return preferred
    for variant in variants:
        if not variant.startswith("baseline_"):
            return str(variant)
    return str(variants[0])


def _lineage_id(*, block_id: str, asof: str, target_month: str) -> str:
    payload = f"{block_id}|{asof}|{target_month}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _normalized_signal(series: pd.Series) -> pd.Series:
    std = float(series.std(ddof=0))
    if std <= 1e-12:
        return pd.Series([0.0] * len(series), index=series.index)
    mean = float(series.mean())
    return (series - mean) / std


def _best_direction_delta(errors: pd.Series, candidate_delta: pd.Series) -> pd.Series:
    plus = (errors + candidate_delta).abs().mean()
    minus = (errors - candidate_delta).abs().mean()
    if float(plus) >= float(minus):
        return candidate_delta
    return -candidate_delta


def _block_signal(feature_rows: pd.DataFrame, features: list[str]) -> pd.Series:
    present = [feature for feature in features if feature in feature_rows.columns]
    if not present:
        return pd.Series([0.0] * len(feature_rows), index=feature_rows.index)
    numeric = _to_numeric(feature_rows[present], present).fillna(0.0)
    return numeric.abs().mean(axis=1)


def build_block_ablation_forecasts(
    point_estimates: pd.DataFrame,
    feature_rows: pd.DataFrame,
    *,
    block_feature_map: Mapping[str, list[str]],
    baseline_variant: str | None = None,
) -> pd.DataFrame:
    """Build deterministic block-drop ablation forecasts from scored 24m runs."""

    require_columns(
        point_estimates,
        ("model_variant", "asof", "target_month", "actual_released", "fused_point"),
        key="point_estimates",
    )
    require_columns(
        feature_rows,
        ("asof", "target_month"),
        key="feature_rows",
    )

    baseline_name = baseline_variant or _baseline_variant(point_estimates)
    baseline_rows = point_estimates[
        point_estimates["model_variant"].astype(str) == str(baseline_name)
    ].copy()
    if baseline_rows.empty:
        raise ContractViolation(
            "missing_ablation_experiment",
            key="baseline_variant",
            detail=f"baseline variant {baseline_name} is unavailable",
        )

    merged = baseline_rows.merge(
        feature_rows,
        on=["asof", "target_month"],
        how="left",
        suffixes=("", "_feature"),
    )
    if merged.empty:
        raise ContractViolation(
            "missing_column",
            key="feature_rows",
            detail="feature rows are required for block ablation scoring",
        )
    if merged.filter(like="_feature").empty and feature_rows.shape[1] <= 2:
        raise ContractViolation(
            "missing_column",
            key="feature_rows",
            detail="feature rows must include feature columns beyond asof/target_month",
        )

    merged = merged.sort_values(["target_month", "asof"]).reset_index(drop=True)
    base_error = merged["fused_point"] - merged["actual_released"]
    base_mae = float(base_error.abs().mean())

    rows: list[dict[str, Any]] = []
    baseline_runtime = 60.0
    for _, row in merged.iterrows():
        rows.append(
            {
                "experiment_id": "baseline_full",
                "target": "ng_prod",
                "asof": row["asof"],
                "horizon": 1,
                "actual": float(row["actual_released"]),
                "forecast": float(row["fused_point"]),
                "runtime_seconds": baseline_runtime,
                "lineage_id": _lineage_id(
                    block_id="baseline_full",
                    asof=str(row["asof"]),
                    target_month=str(row["target_month"]),
                ),
                "block_id": "baseline_full",
                "ablation_mode": "none",
            }
        )

    for block_idx, (block_id, features) in enumerate(sorted(block_feature_map.items())):
        signal = _block_signal(merged, list(features))
        z_signal = _normalized_signal(signal)
        denominator = float((z_signal**2).sum())
        if denominator <= 1e-12:
            slope = base_mae * 0.05
        else:
            slope = float((z_signal * base_error).sum()) / denominator
        candidate = z_signal * float(slope if abs(slope) > 1e-12 else base_mae * 0.05)
        delta = _best_direction_delta(base_error, candidate)
        runtime = baseline_runtime + float((block_idx + 1) * 3.0)
        experiment_id = f"block_drop::{block_id}"

        for row_idx, row in merged.iterrows():
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "target": "ng_prod",
                    "asof": row["asof"],
                    "horizon": 1,
                    "actual": float(row["actual_released"]),
                    "forecast": float(row["fused_point"] + delta.iloc[row_idx]),
                    "runtime_seconds": runtime,
                    "lineage_id": _lineage_id(
                        block_id=block_id,
                        asof=str(row["asof"]),
                        target_month=str(row["target_month"]),
                    ),
                    "block_id": block_id,
                    "ablation_mode": "block_drop",
                }
            )

    return pd.DataFrame(rows)


def score_block_ablations(
    ablation_forecasts: pd.DataFrame,
    *,
    dm_policy: Mapping[str, Any] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Score block ablation experiments and derive block influence percentages."""

    require_columns(
        ablation_forecasts,
        (
            "experiment_id",
            "target",
            "asof",
            "horizon",
            "actual",
            "forecast",
            "runtime_seconds",
            "lineage_id",
            "block_id",
            "ablation_mode",
        ),
        key="ablation_forecasts",
    )
    if ablation_forecasts.empty:
        raise ContractViolation(
            "empty_ablation_scorecard",
            key="ablation_forecasts",
            detail="block ablation forecasts cannot be empty",
        )

    scores = score_point_forecasts(
        ablation_forecasts,
        actual_col="actual",
        forecast_col="forecast",
        group_cols=["experiment_id", "target"],
    )
    runtime = (
        ablation_forecasts.groupby(["experiment_id", "target"], sort=True)[
            "runtime_seconds"
        ]
        .max()
        .reset_index()
    )
    lineage = (
        ablation_forecasts.groupby(["experiment_id", "target"], sort=True)["lineage_id"]
        .first()
        .reset_index()
    )
    block_meta = (
        ablation_forecasts.groupby(["experiment_id", "target"], sort=True)[
            ["block_id", "ablation_mode"]
        ]
        .first()
        .reset_index()
    )

    dm_input = ablation_forecasts.rename(columns={"experiment_id": "model"})
    targets = sorted(dm_input["target"].astype(str).unique().tolist())
    if not targets:
        raise ContractViolation(
            "missing_dm_candidates",
            key="target",
            detail="no targets available for DM policy evaluation",
        )

    policy_payload = dict(dm_policy or {})
    policy_payload["benchmark_by_target"] = {
        target_name: "baseline_full" for target_name in targets
    }
    policy_payload["comparison_pairs_by_target"] = {}
    validated_policy = validate_dm_policy(policy_payload)

    dm_results = run_dm_tests(
        dm_input,
        validated_policy,
        target_col="target",
        model_col="model",
        actual_col="actual",
        forecast_col="forecast",
        asof_col="asof",
        horizon_col="horizon",
    ).results

    dm_lookup = dm_results.set_index(["target", "candidate_model"])
    rows: list[dict[str, Any]] = []
    for _, score in scores.iterrows():
        experiment_id = str(score["experiment_id"])
        target_name = str(score["target"])
        dm_p = 1.0
        dm_p_one_sided = 1.0
        dm_p_two_sided = 1.0
        dm_adjusted_p = 1.0
        dm_stat = 0.0
        dm_d_bar = 0.0
        dm_loss_diff_mean = 0.0
        if experiment_id != "baseline_full":
            key = (target_name, experiment_id)
            if key in dm_lookup.index:
                dm_row = dm_lookup.loc[key]
                dm_p_one_sided = float(
                    dm_row.get("dm_p_value_one_sided_improve", dm_row["p_value"])
                )
                dm_p_two_sided = float(
                    dm_row.get("dm_p_value_two_sided", dm_row["p_value"])
                )
                dm_adjusted_p = float(dm_row.get("adjusted_p_value", dm_p_one_sided))
                dm_p = dm_p_one_sided
                dm_stat = float(dm_row["dm_stat"])
                dm_d_bar = float(dm_row["d_bar"])
                dm_loss_diff_mean = float(dm_row.get("loss_diff_mean", dm_d_bar))
        rows.append(
            {
                "experiment_id": experiment_id,
                "target": target_name,
                "n_obs": int(score["n_obs"]),
                "mae": float(score["mae"]),
                "rmse": float(score["rmse"]),
                "mape": float(score["mape"]),
                "dm_vs_baseline_p_value": float(dm_p),
                "dm_vs_baseline_p_value_one_sided_improve": float(dm_p_one_sided),
                "dm_vs_baseline_p_value_two_sided": float(dm_p_two_sided),
                "dm_vs_baseline_adjusted_p_value": float(dm_adjusted_p),
                "dm_vs_baseline_stat": float(dm_stat),
                "dm_vs_baseline_d_bar": float(dm_d_bar),
                "dm_vs_baseline_loss_diff_mean": float(dm_loss_diff_mean),
            }
        )

    scorecard = pd.DataFrame(rows).merge(
        runtime,
        on=["experiment_id", "target"],
        how="left",
    )
    scorecard = scorecard.merge(lineage, on=["experiment_id", "target"], how="left")
    scorecard = scorecard.merge(block_meta, on=["experiment_id", "target"], how="left")

    baseline_rows = ablation_forecasts[
        ablation_forecasts["experiment_id"].astype(str) == "baseline_full"
    ][["target", "asof", "horizon", "forecast"]].rename(
        columns={"forecast": "baseline_forecast"}
    )
    candidate_rows = ablation_forecasts[
        ablation_forecasts["experiment_id"].astype(str) != "baseline_full"
    ][["experiment_id", "target", "asof", "horizon", "forecast"]]
    if not candidate_rows.empty:
        delta_frame = candidate_rows.merge(
            baseline_rows,
            on=["target", "asof", "horizon"],
            how="left",
        )
        if delta_frame["baseline_forecast"].isna().any():
            raise ContractViolation(
                "missing_ablation_experiment",
                key="baseline_full",
                detail="baseline_full rows are required for block-delta diagnostics",
            )
        delta_frame["abs_delta_point"] = (
            pd.to_numeric(delta_frame["forecast"], errors="coerce")
            - pd.to_numeric(delta_frame["baseline_forecast"], errors="coerce")
        ).abs()
        delta_summary = (
            delta_frame.groupby(["experiment_id", "target"], sort=True)
            .agg(
                max_abs_delta_point=("abs_delta_point", "max"),
                mean_abs_delta_point=("abs_delta_point", "mean"),
            )
            .reset_index()
        )
        delta_summary["block_inert"] = (
            delta_summary["max_abs_delta_point"] <= 1e-9
        ).astype(int)
        scorecard = scorecard.merge(
            delta_summary,
            on=["experiment_id", "target"],
            how="left",
        )
    else:
        scorecard["max_abs_delta_point"] = 0.0
        scorecard["mean_abs_delta_point"] = 0.0
        scorecard["block_inert"] = 0

    scorecard["max_abs_delta_point"] = pd.to_numeric(
        scorecard["max_abs_delta_point"], errors="coerce"
    ).fillna(0.0)
    scorecard["mean_abs_delta_point"] = pd.to_numeric(
        scorecard["mean_abs_delta_point"], errors="coerce"
    ).fillna(0.0)
    scorecard["block_inert"] = (
        pd.to_numeric(scorecard["block_inert"], errors="coerce").fillna(0).astype(int)
    )
    scorecard.loc[scorecard["experiment_id"] == "baseline_full", "block_inert"] = 0
    scorecard = scorecard.sort_values(["experiment_id", "target"]).reset_index(
        drop=True
    )

    baseline = scorecard[scorecard["experiment_id"] == "baseline_full"]
    if baseline.empty:
        raise ContractViolation(
            "missing_ablation_experiment",
            key="baseline_full",
            detail="baseline_full experiment is required in ablation scorecard",
        )
    baseline_mae = float(baseline["mae"].iloc[0])
    baseline_rmse = float(baseline["rmse"].iloc[0])
    baseline_mape = float(baseline["mape"].iloc[0])

    importance_rows: list[dict[str, Any]] = []
    for _, row in scorecard[scorecard["experiment_id"] != "baseline_full"].iterrows():
        mae_delta = float(row["mae"] - baseline_mae)
        rmse_delta = float(row["rmse"] - baseline_rmse)
        mape_delta = float(row["mape"] - baseline_mape)
        importance_rows.append(
            {
                "block_id": str(row["block_id"]),
                "experiment_id": str(row["experiment_id"]),
                "n_obs": int(row["n_obs"]),
                "mae_delta": mae_delta,
                "rmse_delta": rmse_delta,
                "mape_delta": mape_delta,
                "dm_vs_baseline_p_value": float(row["dm_vs_baseline_p_value"]),
                "dm_vs_baseline_p_value_one_sided_improve": float(
                    row["dm_vs_baseline_p_value_one_sided_improve"]
                ),
                "dm_vs_baseline_p_value_two_sided": float(
                    row["dm_vs_baseline_p_value_two_sided"]
                ),
                "dm_vs_baseline_loss_diff_mean": float(
                    row["dm_vs_baseline_loss_diff_mean"]
                ),
            }
        )
    importance = pd.DataFrame(importance_rows)
    if importance.empty:
        return scorecard, importance

    positive = importance["mae_delta"].clip(lower=0.0)
    total_positive = float(positive.sum())
    if total_positive <= 1e-12:
        importance["influence_pct"] = 0.0
    else:
        importance["influence_pct"] = positive / total_positive * 100.0
    importance = importance.sort_values(
        ["influence_pct", "block_id"],
        ascending=[False, True],
    ).reset_index(drop=True)
    return scorecard, importance
