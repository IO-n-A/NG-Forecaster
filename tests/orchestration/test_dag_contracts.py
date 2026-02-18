from __future__ import annotations

from ng_forecaster.orchestration.airflow.dags.eia_api_ingest import DAG_SPEC as API_DAG
from ng_forecaster.orchestration.airflow.dags.eia_bulk_ingest import (
    DAG_SPEC as BULK_DAG,
)
from ng_forecaster.orchestration.airflow.dags.eia_metadata_refresh import (
    DAG_SPEC as META_DAG,
)
from ng_forecaster.orchestration.airflow.dags.nowcast_ablation_runner import (
    DAG_SPEC as ABLATION_DAG,
)
from ng_forecaster.orchestration.airflow.dags.nowcast_pipeline_weekly import (
    DAG_SPEC as PIPELINE_DAG,
)
from ng_forecaster.orchestration.airflow.dags.realtime_backtest_runner import (
    DAG_SPEC as BACKTEST_DAG,
)
from ng_forecaster.orchestration.airflow.dags.weather_nasa_power_daily import (
    DAG_SPEC as WEATHER_DAG,
)


def _task_ids(dag_spec) -> list[str]:
    return [task.task_id for task in dag_spec.task_specs]


def test_weekly_schedule_contract_is_enforced_for_all_dags() -> None:
    for dag_spec in (
        META_DAG,
        API_DAG,
        BULK_DAG,
        PIPELINE_DAG,
        BACKTEST_DAG,
        ABLATION_DAG,
    ):
        assert dag_spec.schedule == "@weekly"
    assert WEATHER_DAG.schedule == "@daily"


def test_nowcast_pipeline_declares_required_task_graph() -> None:
    assert _task_ids(PIPELINE_DAG) == [
        "resolve_asof",
        "resolve_release_calendar_window",
        "ensure_weather_gold_ready",
        "build_vintage_panel",
        "run_preprocessing_gate",
        "build_feature_matrix",
        "train_or_load_champion",
        "run_champion_seed_repeats",
        "train_or_load_challenger",
        "run_challenger_intervals",
        "run_calibration_checks",
        "apply_dm_policy",
        "publish_nowcast_artifacts",
    ]


def test_backtest_and_ablation_task_graphs_match_architecture_contracts() -> None:
    assert _task_ids(BACKTEST_DAG) == [
        "fetch_ngm_issue_calendar",
        "build_asof_grid",
        "replay_vintages",
        "score_horizon_t_minus_1",
        "score_horizon_t",
        "run_dm_tests",
        "publish_backtest_reports",
    ]
    assert _task_ids(ABLATION_DAG) == [
        "load_ablation_matrix",
        "execute_ablation_experiments",
        "score_ablation_experiments",
        "run_dm_vs_baseline",
        "publish_ablation_scorecard",
    ]
