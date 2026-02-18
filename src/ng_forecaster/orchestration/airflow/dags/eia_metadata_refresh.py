"""Airflow DAG: weekly metadata refresh with idempotency and governance wiring."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from ng_forecaster.ingest.catalog import load_source_catalog
from ng_forecaster.orchestration.airflow.runtime import (
    DAGSpec,
    TaskSpec,
    execute_task_graph,
)
from ng_forecaster.orchestration.airflow.workflow_support import (
    bootstrap_status,
    resolve_weekly_asof,
    write_json,
)

DAG_ID = "eia_metadata_refresh"
WEEKLY_SCHEDULE = "@weekly"

DAG_SPEC = DAGSpec(
    dag_id=DAG_ID,
    schedule=WEEKLY_SCHEDULE,
    task_specs=(
        TaskSpec(task_id="load_source_catalog"),
        TaskSpec(
            task_id="scan_bootstrap_inventory",
            upstream_task_ids=("load_source_catalog",),
        ),
        TaskSpec(
            task_id="publish_metadata_snapshot",
            upstream_task_ids=("scan_bootstrap_inventory",),
        ),
    ),
)


def run_eia_metadata_refresh(*, asof: str | None = None) -> dict[str, Any]:
    """Execute weekly metadata refresh tasks in dependency order."""

    asof_ts = resolve_weekly_asof(asof)
    run_state: dict[str, Any] = {}

    def _load_source_catalog(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        catalog = load_source_catalog("configs/sources.yaml")
        config = {
            "version": catalog.version,
            "defaults": catalog.defaults,
            "release_calendar": catalog.release_calendar,
            "sources": [
                {
                    "id": source.source_id,
                    "role": source.role,
                    "ingest_stream": source.ingest_stream,
                    "retrieval_mode": source.retrieval_mode,
                    "filename": source.filename,
                    "required": source.required,
                    "freshness_max_age_days": source.freshness_max_age_days,
                    "parse": source.parse,
                }
                for source in catalog.sources
            ],
        }
        run_state["sources"] = config
        return {
            "source_count": len(catalog.sources),
            "config_version": int(catalog.version),
        }

    def _scan_bootstrap_inventory(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        status = bootstrap_status()
        run_state["bootstrap_status"] = status
        return {
            "bootstrap_available": bool(status["available"]),
            "raw_file_count": int(status["raw_file_count"]),
            "report_file_count": int(status["report_file_count"]),
        }

    def _publish_metadata_snapshot(
        _: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        snapshot_path = Path("data/reports/source_catalog_snapshot.json")
        payload = {
            "asof": asof_ts.date().isoformat(),
            "dag_id": DAG_ID,
            "schedule": WEEKLY_SCHEDULE,
            "sources": run_state["sources"],
            "bootstrap": run_state["bootstrap_status"],
        }
        write_json(snapshot_path, payload)
        return {"snapshot_path": str(snapshot_path)}

    return execute_task_graph(
        dag_spec=DAG_SPEC,
        context={"asof": asof_ts.date().isoformat()},
        task_functions={
            "load_source_catalog": _load_source_catalog,
            "scan_bootstrap_inventory": _scan_bootstrap_inventory,
            "publish_metadata_snapshot": _publish_metadata_snapshot,
        },
    )


try:  # pragma: no cover - optional Airflow import for runtime integration
    from airflow import DAG
    from airflow.operators.empty import EmptyOperator

    with DAG(
        dag_id=DAG_ID,
        schedule=WEEKLY_SCHEDULE,
        catchup=False,
        tags=["weekly", "metadata"],
    ):
        load_source_catalog = EmptyOperator(task_id="load_source_catalog")
        scan_bootstrap_inventory = EmptyOperator(task_id="scan_bootstrap_inventory")
        publish_metadata_snapshot = EmptyOperator(task_id="publish_metadata_snapshot")

        load_source_catalog >> scan_bootstrap_inventory >> publish_metadata_snapshot
except Exception:  # pragma: no cover - Airflow may be unavailable in unit-test env
    pass
