"""Airflow DAG: weekly nowcast ablation runner with DM-vs-baseline evidence publishing."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from ng_forecaster.data.validators import load_yaml
from ng_forecaster.evaluation.replay import (
    run_ablation_matrix,
    validate_ablation_config,
)
from ng_forecaster.orchestration.airflow.runtime import (
    DAGSpec,
    TaskSpec,
    execute_task_graph,
)
from ng_forecaster.orchestration.airflow.workflow_support import (
    build_ablation_runtime_frame,
    enforce_release_policy,
    load_market_inputs,
    resolve_weekly_asof,
    write_json,
)
from ng_forecaster.reporting.exporters import (
    export_ablation_scorecard,
    export_dm_results,
)

DAG_ID = "nowcast_ablation_runner"
WEEKLY_SCHEDULE = "@weekly"

DAG_SPEC = DAGSpec(
    dag_id=DAG_ID,
    schedule=WEEKLY_SCHEDULE,
    task_specs=(
        TaskSpec(task_id="load_ablation_matrix"),
        TaskSpec(
            task_id="execute_ablation_experiments",
            upstream_task_ids=("load_ablation_matrix",),
        ),
        TaskSpec(
            task_id="score_ablation_experiments",
            upstream_task_ids=("execute_ablation_experiments",),
        ),
        TaskSpec(
            task_id="run_dm_vs_baseline",
            upstream_task_ids=("score_ablation_experiments",),
        ),
        TaskSpec(
            task_id="publish_ablation_scorecard",
            upstream_task_ids=("run_dm_vs_baseline",),
        ),
    ),
)


def run_nowcast_ablation_runner(*, asof: str | None = None) -> dict[str, Any]:
    """Execute weekly ablation DAG tasks with N6 attribution contracts."""

    asof_ts = resolve_weekly_asof(asof)
    run_state: dict[str, Any] = {}

    def _load_ablation_matrix(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        config = validate_ablation_config(
            load_yaml("configs/experiments/nowcast_ablation.yaml")
        )
        run_state["ablation_config"] = config
        return {
            "experiment_count": len(config["experiments"]),
            "baseline_experiment": config["baseline_experiment"],
            "full_method_experiment": config["full_method_experiment"],
        }

    def _execute_ablation_experiments(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        policy_context = enforce_release_policy(asof_ts)
        run_state["release_policy"] = policy_context
        inputs = load_market_inputs(asof_ts)
        runtime_frame = build_ablation_runtime_frame(
            inputs["target_history"],
            target_month=pd.Timestamp(policy_context["target_month"]),
        )
        run_state["runtime_frame"] = runtime_frame
        return {
            "runtime_rows": int(len(runtime_frame)),
            "target_count": int(runtime_frame["target"].nunique()),
            "target_month": str(policy_context["target_month"]),
        }

    def _score_ablation_experiments(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        policy = load_yaml("configs/evaluation.yaml")
        ablation_result = run_ablation_matrix(
            run_state["runtime_frame"],
            config=run_state["ablation_config"],
            dm_policy=policy,
        )
        run_state["ablation_result"] = ablation_result
        return {
            "scorecard_rows": int(len(ablation_result.scorecard)),
            "dm_rows": int(len(ablation_result.dm_results)),
        }

    def _run_dm_vs_baseline(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        scorecard = run_state["ablation_result"].scorecard
        baseline = run_state["ablation_config"]["baseline_experiment"]
        non_baseline = scorecard[scorecard["experiment_id"] != baseline]
        return {
            "baseline_experiment": baseline,
            "max_adjusted_p_value": float(non_baseline["dm_vs_baseline_p_value"].max()),
            "improving_experiments": int(
                (non_baseline["dm_vs_baseline_d_bar"] < 0).sum()
            ),
        }

    def _publish_ablation_scorecard(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        scorecard = run_state["ablation_result"].scorecard
        dm_results = run_state["ablation_result"].dm_results

        report_dir = Path("data/reports")
        report_dir.mkdir(parents=True, exist_ok=True)

        scorecard_path = export_ablation_scorecard(scorecard, report_dir)
        dm_path = export_dm_results(
            dm_results,
            report_dir,
            filename="ablation_dm_results.csv",
        )

        min_asof = run_state["runtime_frame"]["asof"].min()
        max_asof = run_state["runtime_frame"]["asof"].max()
        range_label = f"{min_asof.date().isoformat()}_{max_asof.date().isoformat()}"
        artifact_dir = Path("data/artifacts/ablation") / range_label
        artifact_dir.mkdir(parents=True, exist_ok=True)

        export_ablation_scorecard(
            scorecard,
            artifact_dir,
            filename="ablation_scorecard.csv",
        )
        export_dm_results(
            dm_results,
            artifact_dir,
            filename="dm_results.csv",
        )

        write_json(
            report_dir / "ablation_report.json",
            {
                "dag_id": DAG_ID,
                "schedule": WEEKLY_SCHEDULE,
                "asof": asof_ts.date().isoformat(),
                "target_month": str(run_state["release_policy"]["target_month"]),
                "release_policy": run_state["release_policy"],
                "scorecard_path": str(scorecard_path),
                "dm_results_path": str(dm_path),
                "artifact_dir": str(artifact_dir),
            },
        )

        return {
            "scorecard_path": str(scorecard_path),
            "dm_results_path": str(dm_path),
            "artifact_dir": str(artifact_dir),
        }

    return execute_task_graph(
        dag_spec=DAG_SPEC,
        context={"asof": asof_ts.date().isoformat()},
        task_functions={
            "load_ablation_matrix": _load_ablation_matrix,
            "execute_ablation_experiments": _execute_ablation_experiments,
            "score_ablation_experiments": _score_ablation_experiments,
            "run_dm_vs_baseline": _run_dm_vs_baseline,
            "publish_ablation_scorecard": _publish_ablation_scorecard,
        },
    )


try:  # pragma: no cover - optional Airflow import for runtime integration
    from airflow import DAG
    from airflow.operators.empty import EmptyOperator

    with DAG(
        dag_id=DAG_ID,
        schedule=WEEKLY_SCHEDULE,
        catchup=False,
        tags=["weekly", "ablation"],
    ):
        load_ablation_matrix = EmptyOperator(task_id="load_ablation_matrix")
        execute_ablation_experiments = EmptyOperator(
            task_id="execute_ablation_experiments"
        )
        score_ablation_experiments = EmptyOperator(task_id="score_ablation_experiments")
        run_dm_vs_baseline = EmptyOperator(task_id="run_dm_vs_baseline")
        publish_ablation_scorecard = EmptyOperator(task_id="publish_ablation_scorecard")

        (
            load_ablation_matrix
            >> execute_ablation_experiments
            >> score_ablation_experiments
            >> run_dm_vs_baseline
            >> publish_ablation_scorecard
        )
except Exception:  # pragma: no cover
    pass
