"""Lookback sweep runtime with dependence-aware DM scoring and winner selection."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd
import yaml

from ng_forecaster.data.validators import load_yaml
from ng_forecaster.errors import ContractViolation
from ng_forecaster.evaluation.dm_test import run_dm_tests
from ng_forecaster.evaluation.validation_24m import (
    build_target_month_grid,
    derive_policy_admissible_asof,
)
from ng_forecaster.orchestration.airflow.workflow_support import load_market_inputs
from ng_forecaster.orchestration.airflow.dags.nowcast_pipeline_weekly import (
    run_nowcast_pipeline_weekly,
)


@dataclass(frozen=True)
class LookbackSweepResult:
    """Output payload for lookback sweep runs."""

    scorecard: pd.DataFrame
    errors: pd.DataFrame
    dm_results: pd.DataFrame
    winner: dict[str, Any]


def _month_end(value: object) -> pd.Timestamp:
    ts = pd.Timestamp(value)
    if pd.isna(ts):
        raise ContractViolation(
            "invalid_timestamp",
            key="month",
            detail="failed to parse month timestamp",
        )
    return ts.to_period("M").to_timestamp("M")


def _parse_horizons(values: Iterable[object]) -> list[int]:
    parsed = sorted({int(str(value).strip()) for value in values})
    if not parsed or parsed[0] < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="horizons",
            detail="horizons must contain at least one positive integer",
        )
    return parsed


def _resolve_actual(history_full: pd.DataFrame, *, month: pd.Timestamp) -> float:
    frame = history_full.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["target_value"] = pd.to_numeric(frame["target_value"], errors="coerce")
    frame = frame[frame["timestamp"].notna() & frame["target_value"].notna()].copy()
    frame["timestamp"] = frame["timestamp"].dt.to_period("M").dt.to_timestamp("M")
    row = frame[frame["timestamp"] == month]
    if row.empty:
        raise ContractViolation(
            "missing_target_history_rows",
            key=month.date().isoformat(),
            detail="full history does not include requested month",
        )
    return float(row.sort_values("timestamp").iloc[-1]["target_value"])


def _prepare_errors(errors: pd.DataFrame) -> pd.DataFrame:
    frame = errors.copy()
    frame["ape_pct"] = pd.to_numeric(frame["ape_pct"], errors="coerce")
    frame["abs_error"] = pd.to_numeric(frame["abs_error"], errors="coerce")
    frame["alr"] = pd.to_numeric(frame["alr"], errors="coerce")
    frame = frame.sort_values(
        ["lookback", "trial_id", "target_month", "horizon"]
    ).reset_index(drop=True)
    return frame


def _build_scorecard(errors: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    trial_level = (
        errors.groupby(["lookback", "trial_id"], sort=True)["ape_pct"]
        .mean()
        .reset_index(name="trial_mape_pct")
    )
    trial_stats = (
        trial_level.groupby("lookback", sort=True)["trial_mape_pct"]
        .agg(["mean", "std", "count"])
        .reset_index()
        .rename(columns={"mean": "trial_mape_mean", "std": "trial_mape_std"})
    )

    for lookback, group in errors.groupby("lookback", sort=True):
        horizon_metrics: dict[int, dict[str, float]] = {}
        for horizon, horizon_group in group.groupby("horizon", sort=True):
            horizon_metrics[int(horizon)] = {
                "mape_pct": float(horizon_group["ape_pct"].mean()),
                "mae": float(horizon_group["abs_error"].mean()),
                "alr": float(horizon_group["alr"].mean(skipna=True)),
            }

        trial_row = trial_stats[trial_stats["lookback"] == lookback]
        if trial_row.empty:
            trial_mape_std = 0.0
            trial_count = int(group["trial_id"].nunique())
        else:
            trial_mape_std = float(
                pd.to_numeric(trial_row.iloc[0]["trial_mape_std"], errors="coerce")
                if "trial_mape_std" in trial_row.columns
                else 0.0
            )
            if np.isnan(trial_mape_std):
                trial_mape_std = 0.0
            trial_count = int(trial_row.iloc[0]["count"])

        se = trial_mape_std / float(np.sqrt(trial_count)) if trial_count > 1 else 0.0

        row: dict[str, Any] = {
            "lookback": int(lookback),
            "n_rows": int(len(group)),
            "combined_mape_pct": float(group["ape_pct"].mean()),
            "combined_mae": float(group["abs_error"].mean()),
            "combined_alr": float(group["alr"].mean(skipna=True)),
            "trial_mape_std": float(trial_mape_std),
            "trial_mape_se": float(se),
            "n_trials": int(trial_count),
        }
        for horizon in sorted(horizon_metrics):
            suffix = "t_minus_1" if int(horizon) == 1 else f"t_plus_{int(horizon) - 1}"
            metrics = horizon_metrics[horizon]
            row[f"mape_pct_{suffix}"] = float(metrics["mape_pct"])
            row[f"mae_{suffix}"] = float(metrics["mae"])
            row[f"alr_{suffix}"] = float(metrics["alr"])
        rows.append(row)

    return pd.DataFrame(rows).sort_values("lookback").reset_index(drop=True)


def _build_pair_dm_frame(
    candidate: pd.DataFrame,
    baseline: pd.DataFrame,
    *,
    horizon: int,
) -> pd.DataFrame:
    merge_keys = ["trial_id", "target_month", "asof"]
    left = candidate[merge_keys + ["actual_released", "fused_point"]].rename(
        columns={
            "actual_released": "actual_candidate",
            "fused_point": "forecast_candidate",
        }
    )
    right = baseline[merge_keys + ["actual_released", "fused_point"]].rename(
        columns={
            "actual_released": "actual_baseline",
            "fused_point": "forecast_baseline",
        }
    )
    joined = left.merge(right, on=merge_keys, how="inner")
    if joined.empty:
        raise ContractViolation(
            "missing_dm_overlap",
            key=f"horizon={horizon}",
            detail="candidate and baseline lookbacks share no overlap keys",
        )
    if (
        (joined["actual_candidate"] - joined["actual_baseline"]).abs() > 1e-9
    ).any():
        raise ContractViolation(
            "inconsistent_actuals",
            key=f"horizon={horizon}",
            detail="candidate and baseline actual series must match",
        )

    rows: list[dict[str, Any]] = []
    for _, row in joined.iterrows():
        payload = {
            "target": f"horizon_{horizon}",
            "asof": pd.Timestamp(row["asof"]),
            "horizon": int(horizon),
            "actual": float(row["actual_candidate"]),
        }
        rows.append(
            {
                **payload,
                "model": "candidate",
                "forecast": float(row["forecast_candidate"]),
            }
        )
        rows.append(
            {
                **payload,
                "model": "baseline",
                "forecast": float(row["forecast_baseline"]),
            }
        )
    return pd.DataFrame(rows)


def _run_dm_by_lookback(
    errors: pd.DataFrame,
    *,
    baseline_lookback: int,
    horizons: Sequence[int],
    dm_policy: Mapping[str, Any],
) -> pd.DataFrame:
    baseline_all = errors[errors["lookback"] == int(baseline_lookback)].copy()
    if baseline_all.empty:
        raise ContractViolation(
            "invalid_model_policy",
            key="dm_baseline_lookback",
            detail=f"baseline lookback {baseline_lookback} does not exist in sweep",
        )

    rows: list[dict[str, Any]] = []
    for lookback in sorted(errors["lookback"].unique().tolist()):
        if int(lookback) == int(baseline_lookback):
            continue
        for horizon in sorted({int(item) for item in horizons}):
            candidate = errors[
                (errors["lookback"] == int(lookback))
                & (errors["horizon"] == int(horizon))
            ].copy()
            baseline = baseline_all[baseline_all["horizon"] == int(horizon)].copy()
            if candidate.empty or baseline.empty:
                continue
            dm_frame = _build_pair_dm_frame(candidate, baseline, horizon=int(horizon))
            policy = {
                **dict(dm_policy),
                "benchmark_by_target": {f"horizon_{int(horizon)}": "baseline"},
                "comparison_pairs_by_target": {
                    f"horizon_{int(horizon)}": [["candidate", "baseline"]]
                },
            }
            try:
                result = run_dm_tests(
                    dm_frame,
                    policy,
                    target_col="target",
                    model_col="model",
                    actual_col="actual",
                    forecast_col="forecast",
                    asof_col="asof",
                    horizon_col="horizon",
                ).results
            except ContractViolation as exc:
                if exc.context.reason_code != "insufficient_dm_samples":
                    raise
                rows.append(
                    {
                        "lookback": int(lookback),
                        "baseline_lookback": int(baseline_lookback),
                        "horizon": int(horizon),
                        "target": f"horizon_{int(horizon)}",
                        "candidate_model": "candidate",
                        "benchmark_model": "baseline",
                        "d_bar": np.nan,
                        "dm_stat": np.nan,
                        "p_value": np.nan,
                        "adjusted_p_value": np.nan,
                        "significant_0_05": False,
                        "significant_0_01": False,
                        "hac_lag_used": np.nan,
                        "hac_long_run_variance": np.nan,
                        "n_obs": int(dm_frame["asof"].nunique()),
                        "status": "skipped_insufficient_dm_samples",
                    }
                )
                continue
            if result.empty:
                continue
            dm_row = result.iloc[0].to_dict()
            dm_row["lookback"] = int(lookback)
            dm_row["baseline_lookback"] = int(baseline_lookback)
            dm_row["horizon"] = int(horizon)
            dm_row["status"] = "ok"
            rows.append(dm_row)

    if not rows:
        return pd.DataFrame(
            columns=[
                "lookback",
                "baseline_lookback",
                "horizon",
                "d_bar",
                "dm_stat",
                "p_value",
                "adjusted_p_value",
                "significant_0_05",
                "significant_0_01",
                "hac_lag_used",
                "hac_long_run_variance",
                "n_obs",
            ]
        )
    return (
        pd.DataFrame(rows)
        .sort_values(["lookback", "horizon"])
        .reset_index(drop=True)
    )


def _select_winner(
    scorecard: pd.DataFrame,
    dm_results: pd.DataFrame,
    *,
    baseline_lookback: int,
) -> dict[str, Any]:
    frame = scorecard.copy()
    frame["combined_mape_pct"] = pd.to_numeric(frame["combined_mape_pct"], errors="coerce")
    frame["trial_mape_std"] = pd.to_numeric(frame["trial_mape_std"], errors="coerce").fillna(0.0)
    frame["trial_mape_se"] = pd.to_numeric(frame["trial_mape_se"], errors="coerce").fillna(0.0)
    frame["dm_confirmed"] = True

    if not dm_results.empty:
        for lookback, group in dm_results.groupby("lookback", sort=True):
            # Candidate must beat baseline on mean loss differential and be at least
            # directionally significant at 10%.
            confirmed = bool(
                (pd.to_numeric(group["d_bar"], errors="coerce") < 0).all()
                and (pd.to_numeric(group["p_value"], errors="coerce") <= 0.10).all()
            )
            frame.loc[frame["lookback"] == int(lookback), "dm_confirmed"] = confirmed

    frame["stability_penalty"] = frame["trial_mape_std"]
    eligible = frame[(frame["dm_confirmed"]) | (frame["lookback"] == int(baseline_lookback))]
    if eligible.empty:
        eligible = frame.copy()

    eligible = eligible.sort_values(
        ["combined_mape_pct", "stability_penalty", "lookback"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    best = eligible.iloc[0]
    threshold = float(best["combined_mape_pct"]) + float(best["trial_mape_se"])
    within_one_se = eligible[
        pd.to_numeric(eligible["combined_mape_pct"], errors="coerce") <= threshold
    ].copy()
    within_one_se = within_one_se.sort_values("lookback").reset_index(drop=True)
    winner = within_one_se.iloc[0]

    return {
        "selected_lookback": int(winner["lookback"]),
        "selection_policy": {
            "primary_metric": "combined_mape_pct",
            "dm_requirement": "d_bar<0 and p_value<=0.10 vs baseline across horizons",
            "stability_penalty": "trial_mape_std",
            "tie_break": "smallest_lookback_within_1se",
        },
        "baseline_lookback": int(baseline_lookback),
        "best_combined_mape_pct": float(best["combined_mape_pct"]),
        "winner_combined_mape_pct": float(winner["combined_mape_pct"]),
        "winner_trial_mape_std": float(winner["trial_mape_std"]),
        "winner_trial_mape_se": float(winner["trial_mape_se"]),
        "winner_dm_confirmed": bool(winner["dm_confirmed"]),
        "one_se_threshold": float(threshold),
    }


def _write_promoted_config(*, path: Path, winner_lookback: int) -> None:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(config, dict):
        raise ContractViolation(
            "invalid_model_policy",
            key=str(path),
            detail="champion config YAML must be a mapping",
        )
    lstm_cfg = config.get("lstm")
    if not isinstance(lstm_cfg, dict):
        lstm_cfg = {}
        config["lstm"] = lstm_cfg
    lstm_cfg["lookback"] = int(winner_lookback)
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")


def run_lookback_sweep(
    *,
    end_target_month: object,
    lookbacks: Sequence[int],
    replay_months: int = 24,
    horizons: Sequence[int] = (1, 2),
    n_trials: int = 1,
    variant: str = "wpd_vmd_lstm1",
    asof_day: int = 14,
    dm_baseline_lookback: int = 36,
    dm_policy_path: str | Path = "configs/evaluation.yaml",
    source_catalog_path: str | Path = "configs/sources.yaml",
    report_root: str | Path = "data/reports",
    promote_winner: bool = True,
    champion_config_path: str | Path = "configs/model_champion.yaml",
) -> LookbackSweepResult:
    """Execute lookback sweep and export scorecard/error/DM/winner artifacts."""

    lookback_grid = sorted({int(value) for value in lookbacks})
    if not lookback_grid:
        raise ContractViolation(
            "invalid_model_policy",
            key="lookbacks",
            detail="lookback sweep requires at least one lookback value",
        )
    if int(n_trials) < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="n_trials",
            detail="n_trials must be >= 1",
        )
    horizon_values = _parse_horizons(horizons)

    months = build_target_month_grid(
        end_target_month=end_target_month,
        runs=int(replay_months),
    )
    source_catalog = load_yaml(source_catalog_path)
    release_cfg = dict(source_catalog["release_calendar"])
    lag_months = int(release_cfg["lag_months"])
    release_day = int(release_cfg["release_day_of_month"])
    dm_policy = load_yaml(dm_policy_path)

    errors_rows: list[dict[str, Any]] = []
    for lookback in lookback_grid:
        for target_month in months:
            asof = derive_policy_admissible_asof(
                target_month=target_month,
                lag_months=lag_months,
                release_day_of_month=release_day,
                preferred_day=int(asof_day),
            )
            inputs = load_market_inputs(asof)
            target_history_full = inputs["target_history_full"]
            for trial_id in range(1, int(n_trials) + 1):
                seed = int(10_000 + lookback * 31 + trial_id * 17)
                run_nowcast_pipeline_weekly(
                    asof=asof.date().isoformat(),
                    champion_config_override={
                        "model": {"variant": str(variant)},
                        "strategy": str(variant),
                        "lstm": {"lookback": int(lookback)},
                        "training": {"seed": int(seed)},
                    },
                    idempotency_token=(
                        "validation24::lookback_sweep::"
                        f"{target_month.date().isoformat()}::"
                        f"lb{lookback}::trial{trial_id}"
                    ),
                )
                artifact_dir = Path("data/artifacts/nowcast") / asof.date().isoformat()
                fusion_inputs_path = artifact_dir / "fusion_inputs.csv"
                if not fusion_inputs_path.exists():
                    raise ContractViolation(
                        "missing_source_file",
                        key=str(fusion_inputs_path),
                        detail="nowcast run did not export fusion_inputs.csv",
                    )
                fusion_inputs = pd.read_csv(fusion_inputs_path)
                if "horizon" not in fusion_inputs.columns or "fused_point" not in fusion_inputs.columns:
                    raise ContractViolation(
                        "missing_column",
                        key=str(fusion_inputs_path),
                        detail="fusion inputs must include horizon and fused_point",
                    )
                for horizon in horizon_values:
                    target_for_horizon = (
                        target_month.to_period("M") + int(horizon) - 1
                    ).to_timestamp("M")
                    predicted_row = fusion_inputs[
                        pd.to_numeric(fusion_inputs["horizon"], errors="coerce")
                        == int(horizon)
                    ]
                    if predicted_row.empty:
                        raise ContractViolation(
                            "missing_forecast_horizon",
                            key=f"{fusion_inputs_path}:h{horizon}",
                            detail="fusion output is missing requested horizon",
                        )
                    predicted = float(predicted_row.iloc[0]["fused_point"])
                    try:
                        actual = _resolve_actual(
                            target_history_full,
                            month=target_for_horizon,
                        )
                    except ContractViolation as exc:
                        if exc.context.reason_code == "missing_target_history_rows":
                            continue
                        raise
                    error = float(predicted - actual)
                    ape = abs(error) / abs(actual) * 100.0 if actual != 0 else np.nan
                    alr = (
                        float(abs(np.log(actual) - np.log(predicted)))
                        if actual > 0 and predicted > 0
                        else np.nan
                    )
                    errors_rows.append(
                        {
                            "lookback": int(lookback),
                            "trial_id": int(trial_id),
                            "seed": int(seed),
                            "asof": asof.date().isoformat(),
                            "target_month": target_month.date().isoformat(),
                            "horizon": int(horizon),
                            "target_month_for_horizon": target_for_horizon.date().isoformat(),
                            "model_variant": str(variant),
                            "fused_point": float(predicted),
                            "actual_released": float(actual),
                            "error": float(error),
                            "abs_error": float(abs(error)),
                            "ape_pct": float(ape) if np.isfinite(ape) else np.nan,
                            "alr": float(alr) if np.isfinite(alr) else np.nan,
                        }
                    )

    errors = _prepare_errors(pd.DataFrame(errors_rows))
    scorecard = _build_scorecard(errors)
    dm_results = _run_dm_by_lookback(
        errors,
        baseline_lookback=int(dm_baseline_lookback),
        horizons=horizon_values,
        dm_policy=dm_policy,
    )
    winner = _select_winner(
        scorecard,
        dm_results,
        baseline_lookback=int(dm_baseline_lookback),
    )

    report_dir = Path(report_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    scorecard_path = report_dir / "lookback_sweep_scorecard.csv"
    errors_path = report_dir / "lookback_sweep_errors.parquet"
    dm_path = report_dir / "lookback_sweep_dm.csv"
    winner_path = report_dir / "lookback_winner.json"

    scorecard.to_csv(scorecard_path, index=False)
    errors.to_parquet(errors_path, index=False)
    dm_results.to_csv(dm_path, index=False)
    winner_path.write_text(json.dumps(winner, sort_keys=True, indent=2), encoding="utf-8")

    if promote_winner:
        _write_promoted_config(
            path=Path(champion_config_path),
            winner_lookback=int(winner["selected_lookback"]),
        )

    return LookbackSweepResult(
        scorecard=scorecard,
        errors=errors,
        dm_results=dm_results,
        winner={**winner, "winner_path": str(winner_path)},
    )
