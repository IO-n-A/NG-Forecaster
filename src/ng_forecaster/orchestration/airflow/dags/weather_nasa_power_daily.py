"""Airflow DAG: daily NASA POWER ingest and weather freeze-off gold publication."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from ng_forecaster.data.gold_publish import publish_weather_freezeoff_panel
from ng_forecaster.errors import ContractViolation
from ng_forecaster.ingest.adapters.nasa_power import retrieve_nasa_power_t2m_min
from ng_forecaster.orchestration.airflow.runtime import (
    DAGSpec,
    TaskSpec,
    execute_task_graph,
)
from ng_forecaster.orchestration.airflow.workflow_support import write_json

DAG_ID = "weather_nasa_power_daily"
DAILY_SCHEDULE = "@daily"

DAG_SPEC = DAGSpec(
    dag_id=DAG_ID,
    schedule=DAILY_SCHEDULE,
    task_specs=(
        TaskSpec(task_id="retrieve_nasa_power_daily"),
        TaskSpec(
            task_id="publish_weather_freezeoff_gold",
            upstream_task_ids=("retrieve_nasa_power_daily",),
        ),
    ),
)


def _resolve_daily_asof(asof: str | None) -> pd.Timestamp:
    if asof is None:
        return pd.Timestamp.utcnow().normalize()
    parsed = pd.Timestamp(asof)
    if pd.isna(parsed):
        raise ContractViolation(
            "invalid_timestamp",
            key="asof",
            detail="asof could not be parsed",
        )
    return parsed.normalize()


def run_weather_nasa_power_daily(*, asof: str | None = None) -> dict[str, Any]:
    """Execute daily weather ingest and gold publication pipeline."""

    asof_ts = _resolve_daily_asof(asof)
    asof_label = asof_ts.date().isoformat()
    run_state: dict[str, Any] = {}

    def _retrieve_nasa_power_daily(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        token = os.getenv("NASA_EARTHDATA_TOKEN", "")
        frame = retrieve_nasa_power_t2m_min(
            asof=asof_ts,
            start_date=(asof_ts.to_period("M") - 35).to_timestamp(),
            basin_geojson_path="data/reference/dpr_basin_polygons.geojson",
            token=token,
        )
        output_dir = (
            Path("data/bronze/weather/nasa_power_t2m_min") / f"asof={asof_label}"
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / "nasa_power_t2m_min_daily.parquet"
        frame.to_parquet(output_path, index=False)
        run_state["bronze_output_path"] = output_path
        return {
            "output_path": str(output_path),
            "row_count": int(len(frame)),
            "basin_count": int(frame["basin_id"].nunique()),
        }

    def _publish_weather_freezeoff_gold(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        payload = publish_weather_freezeoff_panel(
            asof=asof_ts,
            bronze_root="data/bronze/weather/nasa_power_t2m_min",
            gold_root="data/gold",
            report_root="data/reports",
        )
        run_state["weather_payload"] = payload
        return payload

    result = execute_task_graph(
        dag_spec=DAG_SPEC,
        context={"asof": asof_label},
        task_functions={
            "retrieve_nasa_power_daily": _retrieve_nasa_power_daily,
            "publish_weather_freezeoff_gold": _publish_weather_freezeoff_gold,
        },
    )
    manifest = {
        "asof": asof_label,
        "dag_id": DAG_ID,
        "schedule": DAILY_SCHEDULE,
        "task_results": result["task_results"],
    }
    write_json("data/reports/weather_nasa_power_daily_manifest.json", manifest)
    return result


try:  # pragma: no cover - optional Airflow import for runtime integration
    from airflow import DAG
    from airflow.operators.empty import EmptyOperator

    with DAG(
        dag_id=DAG_ID,
        schedule=DAILY_SCHEDULE,
        catchup=False,
        tags=["daily", "weather"],
    ):
        retrieve_nasa_power_daily = EmptyOperator(task_id="retrieve_nasa_power_daily")
        publish_weather_freezeoff_gold = EmptyOperator(
            task_id="publish_weather_freezeoff_gold"
        )
        retrieve_nasa_power_daily >> publish_weather_freezeoff_gold
except Exception:  # pragma: no cover
    pass
