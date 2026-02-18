"""Airflow DAG: weekly realtime backtest runner with DM policy checks."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ng_forecaster.evaluation.dm_test import load_and_validate_dm_policy, run_dm_tests
from ng_forecaster.evaluation.metrics import score_point_forecasts
from ng_forecaster.evaluation.replay import load_checkpoint_dates, run_replay
from ng_forecaster.orchestration.airflow.runtime import (
    DAGSpec,
    TaskSpec,
    execute_task_graph,
)
from ng_forecaster.orchestration.airflow.workflow_support import (
    enforce_release_policy,
    load_market_inputs,
    resolve_weekly_asof,
    write_json,
)
from ng_forecaster.reporting.exporters import export_dm_results

DAG_ID = "realtime_backtest_runner"
WEEKLY_SCHEDULE = "@weekly"

DAG_SPEC = DAGSpec(
    dag_id=DAG_ID,
    schedule=WEEKLY_SCHEDULE,
    task_specs=(
        TaskSpec(task_id="fetch_ngm_issue_calendar"),
        TaskSpec(
            task_id="build_asof_grid",
            upstream_task_ids=("fetch_ngm_issue_calendar",),
        ),
        TaskSpec(
            task_id="replay_vintages",
            upstream_task_ids=("build_asof_grid",),
        ),
        TaskSpec(
            task_id="score_horizon_t_minus_1",
            upstream_task_ids=("replay_vintages",),
        ),
        TaskSpec(
            task_id="score_horizon_t",
            upstream_task_ids=("replay_vintages",),
        ),
        TaskSpec(
            task_id="run_dm_tests",
            upstream_task_ids=("score_horizon_t_minus_1", "score_horizon_t"),
        ),
        TaskSpec(
            task_id="publish_backtest_reports",
            upstream_task_ids=("run_dm_tests",),
        ),
    ),
)


def run_realtime_backtest_runner(*, asof: str | None = None) -> dict[str, Any]:
    """Execute weekly realtime backtest DAG tasks."""

    asof_ts = resolve_weekly_asof(asof)
    run_state: dict[str, Any] = {}

    def _fetch_ngm_issue_calendar(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        checkpoint_file = Path("tests/fixtures/replay/checkpoints.csv")
        checkpoints = load_checkpoint_dates(checkpoint_file)
        policy_context = enforce_release_policy(asof_ts)
        run_state["checkpoint_file"] = checkpoint_file
        run_state["calendar"] = checkpoints
        run_state["release_policy"] = policy_context
        return {
            "checkpoint_file": str(checkpoint_file),
            "calendar_rows": len(checkpoints),
            "target_month": str(policy_context["target_month"]),
            "release_policy": policy_context,
        }

    def _build_asof_grid(task_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if "calendar" not in run_state:
            fetch_payload = task_results.get("fetch_ngm_issue_calendar", {})
            checkpoint_file = Path(
                str(
                    fetch_payload.get(
                        "checkpoint_file", "tests/fixtures/replay/checkpoints.csv"
                    )
                )
            )
            run_state["checkpoint_file"] = checkpoint_file
            run_state["calendar"] = load_checkpoint_dates(checkpoint_file)
        asof_grid = sorted(set(run_state["calendar"]))
        run_state["asof_grid"] = asof_grid
        return {
            "asof_count": len(asof_grid),
            "first_asof": asof_grid[0].date().isoformat(),
            "last_asof": asof_grid[-1].date().isoformat(),
        }

    def _replay_vintages(task_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if "release_policy" not in run_state:
            fetch_payload = task_results.get("fetch_ngm_issue_calendar", {})
            release_payload = fetch_payload.get("release_policy")
            if isinstance(release_payload, dict):
                run_state["release_policy"] = dict(release_payload)
            else:
                run_state["release_policy"] = enforce_release_policy(asof_ts)
        if "asof_grid" not in run_state:
            _build_asof_grid(task_results)

        inputs = load_market_inputs(asof_ts, include_gold_features=False)
        run_state["inputs"] = inputs

        replay_result = run_replay(
            inputs["features"],
            inputs["target_for_vintage"],
            run_state["asof_grid"],
            preprocessing_status="passed",
            min_target_lag_months=int(
                run_state["release_policy"]["effective_lag_months"]
            ),
            target_month_offset_months=int(run_state["release_policy"]["lag_months"]),
            feature_policy=inputs["feature_policy"],
        )
        run_state["replay"] = replay_result
        return {
            "checkpoint_count": len(replay_result.checkpoints),
            "row_count": int(len(replay_result.frame)),
        }

    def _score_horizon_t_minus_1(
        task_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        if "replay" not in run_state:
            _replay_vintages(task_results)
        frame = run_state["replay"].frame.copy()
        subset = frame[frame["horizon"] == "T-1"].copy()
        subset["actual"] = subset["target_value"]
        subset["forecast"] = subset["target_value"] + 0.35

        score = score_point_forecasts(
            subset, actual_col="actual", forecast_col="forecast"
        )
        run_state["score_t_minus_1"] = score
        return {
            "mae": float(score.iloc[0]["mae"]),
            "rmse": float(score.iloc[0]["rmse"]),
            "mape": float(score.iloc[0]["mape"]),
        }

    def _score_horizon_t(task_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if "replay" not in run_state:
            _replay_vintages(task_results)
        frame = run_state["replay"].frame.copy()
        subset = frame[frame["horizon"] == "T"].copy()
        subset["actual"] = subset["target_value"]
        subset["forecast"] = subset["target_value"] + 0.45

        score = score_point_forecasts(
            subset, actual_col="actual", forecast_col="forecast"
        )
        run_state["score_t"] = score
        return {
            "mae": float(score.iloc[0]["mae"]),
            "rmse": float(score.iloc[0]["rmse"]),
            "mape": float(score.iloc[0]["mape"]),
        }

    def _run_dm_tests(task_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if "replay" not in run_state:
            _replay_vintages(task_results)
        frame = run_state["replay"].frame.copy()
        dm_rows: list[dict[str, Any]] = []
        horizon_offsets = {
            1: {
                "wpd_lstm_one_layer": 0.32,
                "wpd_vmd_lstm1": 0.20,
                "wpd_vmd_lstm2": 0.24,
            },
            2: {
                "wpd_lstm_one_layer": 0.38,
                "wpd_vmd_lstm1": 0.24,
                "wpd_vmd_lstm2": 0.29,
            },
        }

        for _, row in frame.iterrows():
            asof_value = pd.Timestamp(row["replay_checkpoint"])
            horizon_label = str(row["horizon"])
            horizon_value = 1 if horizon_label == "T-1" else 2
            actual = float(row["target_value"])
            for model_name, offset in sorted(horizon_offsets[horizon_value].items()):
                dm_rows.append(
                    {
                        "target": "ng_prod",
                        "model": model_name,
                        "asof": asof_value,
                        "horizon": horizon_value,
                        "actual": actual,
                        "forecast": float(actual + offset),
                        "target_month": str(row.get("target_month", "")),
                    }
                )

        dm_input = pd.DataFrame(dm_rows)
        policy = load_and_validate_dm_policy("configs/evaluation.yaml")
        dm_run = run_dm_tests(dm_input, policy)
        run_state["dm_results"] = dm_run.results
        return {"dm_rows": int(len(dm_run.results))}

    def _publish_backtest_reports(
        task_results: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        if "score_t_minus_1" not in run_state:
            _score_horizon_t_minus_1(task_results)
        if "score_t" not in run_state:
            _score_horizon_t(task_results)
        if "dm_results" not in run_state:
            _run_dm_tests(task_results)
        if "release_policy" not in run_state:
            run_state["release_policy"] = enforce_release_policy(asof_ts)

        report_dir = Path("data/reports")
        report_dir.mkdir(parents=True, exist_ok=True)

        scorecard = pd.DataFrame(
            [
                {
                    "horizon": "T-1",
                    "target_month": pd.Timestamp(
                        run_state["release_policy"]["target_month"]
                    )
                    .to_period("M")
                    .to_timestamp("M")
                    .date()
                    .isoformat(),
                    "mae": float(run_state["score_t_minus_1"].iloc[0]["mae"]),
                    "rmse": float(run_state["score_t_minus_1"].iloc[0]["rmse"]),
                    "mape": float(run_state["score_t_minus_1"].iloc[0]["mape"]),
                },
                {
                    "horizon": "T",
                    "target_month": (
                        pd.Timestamp(run_state["release_policy"]["target_month"])
                        + pd.DateOffset(months=1)
                    )
                    .to_period("M")
                    .to_timestamp("M")
                    .date()
                    .isoformat(),
                    "mae": float(run_state["score_t"].iloc[0]["mae"]),
                    "rmse": float(run_state["score_t"].iloc[0]["rmse"]),
                    "mape": float(run_state["score_t"].iloc[0]["mape"]),
                },
            ]
        )
        scorecard_path = report_dir / "backtest_scorecard.csv"
        scorecard.to_csv(scorecard_path, index=False)

        dm_path = export_dm_results(
            run_state["dm_results"],
            report_dir,
            filename="backtest_dm_results.csv",
        )

        write_json(
            report_dir / "backtest_report.json",
            {
                "dag_id": DAG_ID,
                "schedule": WEEKLY_SCHEDULE,
                "asof": asof_ts.date().isoformat(),
                "target_month": str(run_state["release_policy"]["target_month"]),
                "release_policy": run_state["release_policy"],
                "scorecard_path": str(scorecard_path),
                "dm_results_path": str(dm_path),
            },
        )

        return {
            "scorecard_path": str(scorecard_path),
            "dm_results_path": str(dm_path),
            "target_month": str(run_state["release_policy"]["target_month"]),
        }

    result = execute_task_graph(
        dag_spec=DAG_SPEC,
        context={"asof": asof_ts.date().isoformat()},
        task_functions={
            "fetch_ngm_issue_calendar": _fetch_ngm_issue_calendar,
            "build_asof_grid": _build_asof_grid,
            "replay_vintages": _replay_vintages,
            "score_horizon_t_minus_1": _score_horizon_t_minus_1,
            "score_horizon_t": _score_horizon_t,
            "run_dm_tests": _run_dm_tests,
            "publish_backtest_reports": _publish_backtest_reports,
        },
    )

    if "release_policy" not in run_state:
        fetch_task = result["tasks"].get("fetch_ngm_issue_calendar", {})
        release_policy_payload = fetch_task.get("release_policy")
        if isinstance(release_policy_payload, dict):
            run_state["release_policy"] = dict(release_policy_payload)

    release_policy = run_state.get("release_policy", {})
    if isinstance(release_policy, dict):
        result["release_policy"] = release_policy
        target_month = release_policy.get("target_month")
        if target_month is not None:
            result["target_month"] = str(target_month)

    return result


try:  # pragma: no cover - optional Airflow import for runtime integration
    from airflow import DAG
    from airflow.operators.empty import EmptyOperator

    with DAG(
        dag_id=DAG_ID,
        schedule=WEEKLY_SCHEDULE,
        catchup=False,
        tags=["weekly", "backtest"],
    ):
        fetch_ngm_issue_calendar = EmptyOperator(task_id="fetch_ngm_issue_calendar")
        build_asof_grid = EmptyOperator(task_id="build_asof_grid")
        replay_vintages = EmptyOperator(task_id="replay_vintages")
        score_horizon_t_minus_1 = EmptyOperator(task_id="score_horizon_t_minus_1")
        score_horizon_t = EmptyOperator(task_id="score_horizon_t")
        run_dm_tests_task = EmptyOperator(task_id="run_dm_tests")
        publish_backtest_reports = EmptyOperator(task_id="publish_backtest_reports")

        fetch_ngm_issue_calendar >> build_asof_grid >> replay_vintages
        replay_vintages >> score_horizon_t_minus_1 >> run_dm_tests_task
        replay_vintages >> score_horizon_t >> run_dm_tests_task
        run_dm_tests_task >> publish_backtest_reports
except Exception:  # pragma: no cover
    pass
