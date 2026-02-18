"""Airflow DAG: weekly bulk ingestion with idempotent manifest and dead-letter routing."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any

import pandas as pd

from ng_forecaster.errors import ContractViolation
from ng_forecaster.ingest.catalog import build_ingestion_plan, load_source_catalog
from ng_forecaster.orchestration.airflow.runtime import (
    DAGSpec,
    TaskSpec,
    execute_task_graph,
)
from ng_forecaster.orchestration.airflow.workflow_support import (
    bootstrap_status,
    resolve_weekly_asof,
    sha256_file,
    write_json,
)

DAG_ID = "eia_bulk_ingest"
WEEKLY_SCHEDULE = "@weekly"

DAG_SPEC = DAGSpec(
    dag_id=DAG_ID,
    schedule=WEEKLY_SCHEDULE,
    task_specs=(
        TaskSpec(task_id="load_bulk_manifest"),
        TaskSpec(
            task_id="ingest_bulk_payloads",
            upstream_task_ids=("load_bulk_manifest",),
        ),
        TaskSpec(
            task_id="publish_bulk_manifest",
            upstream_task_ids=("ingest_bulk_payloads",),
        ),
    ),
)


def run_eia_bulk_ingest(*, asof: str | None = None) -> dict[str, Any]:
    """Execute weekly bulk ingestion DAG tasks."""

    asof_ts = resolve_weekly_asof(asof)
    asof_label = asof_ts.date().isoformat()
    run_state: dict[str, Any] = {}

    def _load_bulk_manifest(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        catalog = load_source_catalog("configs/sources.yaml")
        run_state["catalog"] = catalog
        status = bootstrap_status()
        try:
            plan = build_ingestion_plan(
                catalog=catalog,
                ingest_stream="bulk",
                asof=pd.Timestamp(asof_ts),
                bootstrap_root=status["raw_dir"],
                stream_roots={
                    "api": os.getenv("NGF_API_SOURCE_ROOT", "data/bronze/eia_api"),
                    "bulk": os.getenv("NGF_BULK_SOURCE_ROOT", "data/bronze/eia_bulk"),
                },
                fixture_root=os.getenv(
                    "NGF_FIXTURE_SOURCE_ROOT",
                    "tests/fixtures/orchestration/bootstrap_raw",
                ),
            )
        except ContractViolation as exc:
            if (
                status["available"] is False
                and exc.context.reason_code == "missing_source_file"
            ):
                plan = tuple()
            else:
                raise
        run_state["source_plan"] = plan
        run_state["bootstrap_available"] = bool(status["available"])
        return {
            "manifest_file_count": len(plan),
            "source_root": str(status["raw_dir"]),
            "bootstrap_available": bool(status["available"]),
            "catalog_version": int(catalog.version),
        }

    def _ingest_bulk_payloads(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        target_root = Path("data/bronze/eia_bulk") / f"asof={asof_label}"
        dead_letter_root = Path("data/bronze/dead_letter") / f"asof={asof_label}"
        target_root.mkdir(parents=True, exist_ok=True)
        dead_letter_root.mkdir(parents=True, exist_ok=True)

        copied: list[dict[str, Any]] = []
        dead_lettered: list[dict[str, Any]] = []
        for source in run_state["source_plan"]:
            source_file = Path(source.path)
            suffix = source_file.suffix.lower()
            if suffix not in {".xls", ".xlsx", ".csv", ".json"}:
                destination = dead_letter_root / source_file.name
                if source_file.resolve() != destination.resolve():
                    shutil.copy2(source_file, destination)
                dead_lettered.append(
                    {
                        "source_id": source.source_id,
                        "filename": source_file.name,
                        "source_path": str(source_file),
                        "origin": source.origin,
                    }
                )
                continue

            destination = target_root / source_file.name
            if source_file.resolve() != destination.resolve():
                shutil.copy2(source_file, destination)
            copied.append(
                {
                    "source_id": source.source_id,
                    "filename": source_file.name,
                    "source_path": str(source_file),
                    "origin": source.origin,
                    "retrieval_mode": source.retrieval_mode,
                    "sha256": sha256_file(destination),
                    "size_bytes": int(destination.stat().st_size),
                }
            )

        run_state["target_root"] = target_root
        run_state["dead_letter_root"] = dead_letter_root
        run_state["copied_files"] = copied
        run_state["dead_lettered"] = dead_lettered
        return {
            "target_root": str(target_root),
            "copied_file_count": len(copied),
            "dead_letter_count": len(dead_lettered),
        }

    def _publish_bulk_manifest(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        manifest_path = Path("data/reports/eia_bulk_ingest_manifest.json")
        payload = {
            "asof": asof_label,
            "dag_id": DAG_ID,
            "schedule": WEEKLY_SCHEDULE,
            "target_root": str(run_state["target_root"]),
            "dead_letter_root": str(run_state["dead_letter_root"]),
            "bootstrap_available": bool(run_state["bootstrap_available"]),
            "catalog_version": int(run_state["catalog"].version),
            "files": run_state["copied_files"],
            "dead_lettered": run_state["dead_lettered"],
        }
        write_json(manifest_path, payload)
        return {"manifest_path": str(manifest_path)}

    return execute_task_graph(
        dag_spec=DAG_SPEC,
        context={"asof": asof_label},
        task_functions={
            "load_bulk_manifest": _load_bulk_manifest,
            "ingest_bulk_payloads": _ingest_bulk_payloads,
            "publish_bulk_manifest": _publish_bulk_manifest,
        },
    )


try:  # pragma: no cover - optional Airflow import for runtime integration
    from airflow import DAG
    from airflow.operators.empty import EmptyOperator

    with DAG(
        dag_id=DAG_ID,
        schedule=WEEKLY_SCHEDULE,
        catchup=False,
        tags=["weekly", "ingest", "bulk"],
    ):
        load_bulk_manifest = EmptyOperator(task_id="load_bulk_manifest")
        ingest_bulk_payloads = EmptyOperator(task_id="ingest_bulk_payloads")
        publish_bulk_manifest = EmptyOperator(task_id="publish_bulk_manifest")

        load_bulk_manifest >> ingest_bulk_payloads >> publish_bulk_manifest
except Exception:  # pragma: no cover
    pass
