from __future__ import annotations

import json
from pathlib import Path

import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.qa.ops_readiness import check_ops_readiness


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _prepare_valid_ops_bundle(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    dag_dir = tmp_path / "dags"
    dag_dir.mkdir(parents=True, exist_ok=True)
    for name in (
        "eia_metadata_refresh.py",
        "eia_api_ingest.py",
        "eia_bulk_ingest.py",
        "nowcast_pipeline_weekly.py",
        "realtime_backtest_runner.py",
        "nowcast_ablation_runner.py",
    ):
        _write(dag_dir / name, '"""Airflow DAG: weekly schedule."""\n')
    _write(
        dag_dir / "weather_nasa_power_daily.py", '"""Airflow DAG: daily schedule."""\n'
    )

    runbook = tmp_path / "runbook_weekly.md"
    _write(runbook, "# Weekly Runbook\n\n- incident handling\n")

    incident_log = tmp_path / "incidents_log.jsonl"
    incident_payload = {
        "timestamp": "2026-02-14T12:00:00Z",
        "severity": "info",
        "status": "closed",
        "summary": "no active incidents",
    }
    _write(incident_log, json.dumps(incident_payload) + "\n")

    release_approval = tmp_path / "release_approval.json"
    _write(
        release_approval,
        json.dumps(
            {
                "status": "approved",
                "approver": "QA_Lead",
                "approved_at": "2026-02-14T12:10:00Z",
                "scope": "weekly_nowcast",
            }
        ),
    )

    return dag_dir, runbook, incident_log, release_approval


def _write_feature_blocks(path: Path) -> None:
    _write(
        path,
        "\n".join(
            [
                "version: 1",
                "defaults:",
                "  asof_rule: available_timestamp_lte_asof",
                "  max_staleness_days: 30",
                "blocks:",
                "  market_core:",
                "    enabled: true",
                "    asof_rule: available_timestamp_lte_asof",
                "    max_staleness_days: 30",
                "    features:",
                "      - hh_last",
            ]
        )
        + "\n",
    )


def _write_source_catalog(path: Path) -> None:
    _write(
        path,
        "\n".join(
            [
                "version: 1",
                "defaults:",
                "  freshness_max_age_days: 120",
                "  bootstrap_root: data/bootstrap/raw",
                "release_calendar:",
                "  lag_months: 2",
                "  release_day_of_month: 30",
                "  admissible_day_window:",
                "    start_day: 1",
                "    end_day: 31",
                "sources:",
                "  - id: ng_prod_target",
                "    role: target_history",
                "    ingest_stream: bulk",
                "    retrieval_mode: bulk_snapshot",
                "    filename: target.xls",
                "    required: true",
                "    freshness_max_age_days: 180",
                "  - id: steo_latest_workbook",
                "    role: driver",
                "    ingest_stream: bulk",
                "    retrieval_mode: archive_glob",
                "    filename: steo_m.xlsx",
                "    required: false",
                "    freshness_max_age_days: 3650",
                "  - id: nasa_power_t2m_min",
                "    role: driver",
                "    ingest_stream: api",
                "    retrieval_mode: api_snapshot",
                "    filename: nasa_power_t2m_min_daily.parquet",
                "    required: false",
                "    freshness_max_age_days: 3650",
            ]
        )
        + "\n",
    )


def test_ops_readiness_passes_for_complete_governance_bundle(tmp_path: Path) -> None:
    dag_dir, runbook, incident_log, release_approval = _prepare_valid_ops_bundle(
        tmp_path
    )
    feature_blocks = tmp_path / "feature_blocks.yaml"
    source_catalog = tmp_path / "sources.yaml"
    gold_root = tmp_path / "gold"
    gold_root.mkdir(parents=True, exist_ok=True)
    _write_feature_blocks(feature_blocks)
    _write_source_catalog(source_catalog)
    result = check_ops_readiness(
        dag_dir=dag_dir,
        runbook_path=runbook,
        incident_log_path=incident_log,
        release_approval_path=release_approval,
        feature_blocks_path=feature_blocks,
        source_catalog_path=source_catalog,
        gold_root=gold_root,
        forbidden_paths=(),
    )
    assert result.passed


def test_ops_readiness_fails_when_release_is_not_approved(tmp_path: Path) -> None:
    dag_dir, runbook, incident_log, release_approval = _prepare_valid_ops_bundle(
        tmp_path
    )
    feature_blocks = tmp_path / "feature_blocks.yaml"
    source_catalog = tmp_path / "sources.yaml"
    gold_root = tmp_path / "gold"
    gold_root.mkdir(parents=True, exist_ok=True)
    _write_feature_blocks(feature_blocks)
    _write_source_catalog(source_catalog)
    _write(
        release_approval,
        json.dumps(
            {
                "status": "pending",
                "approver": "QA_Lead",
                "approved_at": "2026-02-14T12:10:00Z",
                "scope": "weekly_nowcast",
            }
        ),
    )

    with pytest.raises(ContractViolation, match="reason_code=release_not_approved"):
        check_ops_readiness(
            dag_dir=dag_dir,
            runbook_path=runbook,
            incident_log_path=incident_log,
            release_approval_path=release_approval,
            feature_blocks_path=feature_blocks,
            source_catalog_path=source_catalog,
            gold_root=gold_root,
            forbidden_paths=(),
        )


def test_ops_readiness_fails_when_forbidden_path_exists(tmp_path: Path) -> None:
    dag_dir, runbook, incident_log, release_approval = _prepare_valid_ops_bundle(
        tmp_path
    )
    feature_blocks = tmp_path / "feature_blocks.yaml"
    source_catalog = tmp_path / "sources.yaml"
    gold_root = tmp_path / "gold"
    gold_root.mkdir(parents=True, exist_ok=True)
    _write_feature_blocks(feature_blocks)
    _write_source_catalog(source_catalog)
    forbidden = tmp_path / "data_new"
    forbidden.mkdir(parents=True, exist_ok=True)

    with pytest.raises(ContractViolation, match="reason_code=forbidden_path_present"):
        check_ops_readiness(
            dag_dir=dag_dir,
            runbook_path=runbook,
            incident_log_path=incident_log,
            release_approval_path=release_approval,
            feature_blocks_path=feature_blocks,
            source_catalog_path=source_catalog,
            gold_root=gold_root,
            forbidden_paths=(forbidden,),
        )
