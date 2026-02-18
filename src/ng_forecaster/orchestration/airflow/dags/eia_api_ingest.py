"""Airflow DAG: weekly API ingestion with retries, idempotency keys, and secret redaction."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pandas as pd

from ng_forecaster.ingest.api_client import stage_sources_for_stream
from ng_forecaster.ingest.catalog import load_source_catalog
from ng_forecaster.orchestration.airflow.runtime import (
    DAGSpec,
    TaskSpec,
    TransientHTTPError,
    execute_task_graph,
    resolve_secret,
)
from ng_forecaster.orchestration.airflow.workflow_support import (
    bootstrap_status,
    resolve_weekly_asof,
    write_json,
)

DAG_ID = "eia_api_ingest"
WEEKLY_SCHEDULE = "@weekly"

DAG_SPEC = DAGSpec(
    dag_id=DAG_ID,
    schedule=WEEKLY_SCHEDULE,
    task_specs=(
        TaskSpec(task_id="resolve_api_secret"),
        TaskSpec(
            task_id="ingest_bootstrap_payloads",
            upstream_task_ids=("resolve_api_secret",),
        ),
        TaskSpec(
            task_id="publish_ingest_manifest",
            upstream_task_ids=("ingest_bootstrap_payloads",),
        ),
    ),
)


def run_eia_api_ingest(*, asof: str | None = None) -> dict[str, Any]:
    """Execute weekly API ingestion DAG tasks."""

    asof_ts = resolve_weekly_asof(asof)
    asof_label = asof_ts.date().isoformat()
    run_state: dict[str, Any] = {}

    def _resolve_api_secret(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        secret = resolve_secret(
            name="eia_api_key", env_var="EIA_API_KEY", required=False
        )
        run_state["api_secret"] = secret
        return {
            "secret_name": secret.name,
            "secret_redacted": secret.redacted,
            "has_api_key": bool(secret.value),
        }

    def _ingest_bootstrap_payloads(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if os.getenv("NGF_SIMULATE_API_429", "0") == "1":
            raise TransientHTTPError(
                status_code=429,
                detail="simulated API throttling for retry/backoff validation",
            )

        catalog = load_source_catalog("configs/sources.yaml")
        run_state["catalog"] = catalog

        status = bootstrap_status()
        target_root = Path("data/bronze/eia_api") / f"asof={asof_label}"
        target_root.mkdir(parents=True, exist_ok=True)
        staged_files = stage_sources_for_stream(
            asof=pd.Timestamp(asof_ts),
            ingest_stream="api",
            target_root=target_root,
            catalog=catalog,
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

        copied: list[dict[str, Any]] = []
        for staged in staged_files:
            copied.append(
                {
                    "source_id": staged.source_id,
                    "filename": staged.filename,
                    "source_path": str(staged.source_path),
                    "origin": staged.origin,
                    "retrieval_mode": staged.retrieval_mode,
                    "sha256": staged.sha256,
                    "size_bytes": int(staged.size_bytes),
                }
            )

        run_state["copied_files"] = copied
        run_state["target_root"] = target_root
        run_state["bootstrap_available"] = bool(status["available"])
        return {
            "target_root": str(target_root),
            "copied_file_count": len(copied),
            "bootstrap_available": bool(status["available"]),
            "catalog_version": int(catalog.version),
        }

    def _publish_ingest_manifest(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        manifest_path = Path("data/reports/eia_api_ingest_manifest.json")
        payload = {
            "asof": asof_label,
            "dag_id": DAG_ID,
            "schedule": WEEKLY_SCHEDULE,
            "target_root": str(run_state["target_root"]),
            "bootstrap_available": bool(run_state["bootstrap_available"]),
            "catalog_version": int(run_state["catalog"].version),
            "api_secret_redacted": run_state["api_secret"].redacted,
            "files": run_state["copied_files"],
        }
        write_json(manifest_path, payload)
        return {"manifest_path": str(manifest_path)}

    return execute_task_graph(
        dag_spec=DAG_SPEC,
        context={"asof": asof_label},
        task_functions={
            "resolve_api_secret": _resolve_api_secret,
            "ingest_bootstrap_payloads": _ingest_bootstrap_payloads,
            "publish_ingest_manifest": _publish_ingest_manifest,
        },
    )


try:  # pragma: no cover - optional Airflow import for runtime integration
    from airflow import DAG
    from airflow.operators.empty import EmptyOperator

    with DAG(
        dag_id=DAG_ID,
        schedule=WEEKLY_SCHEDULE,
        catchup=False,
        tags=["weekly", "ingest"],
    ):
        resolve_api_secret = EmptyOperator(task_id="resolve_api_secret")
        ingest_bootstrap_payloads = EmptyOperator(task_id="ingest_bootstrap_payloads")
        publish_ingest_manifest = EmptyOperator(task_id="publish_ingest_manifest")

        resolve_api_secret >> ingest_bootstrap_payloads >> publish_ingest_manifest
except Exception:  # pragma: no cover
    pass
