"""Operations readiness gate for weekly orchestration governance."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path

from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.block_registry import (
    enabled_feature_policies,
    load_feature_block_registry,
    required_block_metadata_keys,
)
from ng_forecaster.ingest.catalog import load_source_catalog

_REQUIRED_DAG_FILES = (
    "eia_metadata_refresh.py",
    "eia_api_ingest.py",
    "eia_bulk_ingest.py",
    "weather_nasa_power_daily.py",
    "nowcast_pipeline_weekly.py",
    "realtime_backtest_runner.py",
    "nowcast_ablation_runner.py",
)
_WEEKLY_DAG_FILES = (
    "eia_metadata_refresh.py",
    "eia_api_ingest.py",
    "eia_bulk_ingest.py",
    "nowcast_pipeline_weekly.py",
    "realtime_backtest_runner.py",
    "nowcast_ablation_runner.py",
)
_REQUIRED_RELEASE_FIELDS = {"status", "approver", "approved_at", "scope"}


@dataclass(frozen=True)
class OpsReadinessResult:
    """Result payload for operations readiness checks."""

    passed: bool
    dag_dir: Path
    runbook_path: Path
    incident_log_path: Path
    release_approval_path: Path

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "dag_dir": str(self.dag_dir),
            "runbook_path": str(self.runbook_path),
            "incident_log_path": str(self.incident_log_path),
            "release_approval_path": str(self.release_approval_path),
        }


def check_ops_readiness(
    *,
    dag_dir: str | Path = "src/ng_forecaster/orchestration/airflow/dags",
    runbook_path: str | Path = "docs/operations/runbook_weekly.md",
    incident_log_path: str | Path = "data/reports/incidents_log.jsonl",
    release_approval_path: str | Path = "data/reports/release_approval.json",
    feature_blocks_path: str | Path = "configs/feature_blocks.yaml",
    source_catalog_path: str | Path = "configs/sources.yaml",
    gold_root: str | Path = "data/gold",
    forbidden_paths: tuple[str | Path, ...] = ("data/new",),
) -> OpsReadinessResult:
    """Validate governance prerequisites for weekly operations readiness."""

    dags = Path(dag_dir)
    runbook = Path(runbook_path)
    incident_log = Path(incident_log_path)
    approval = Path(release_approval_path)

    missing_dags = [name for name in _REQUIRED_DAG_FILES if not (dags / name).exists()]
    if missing_dags:
        raise ContractViolation(
            "missing_dag",
            key="airflow_dags",
            detail="missing DAG files: " + ", ".join(sorted(missing_dags)),
        )

    for dag_name in _WEEKLY_DAG_FILES:
        dag_text = (dags / dag_name).read_text(encoding="utf-8")
        if "weekly" not in dag_text.lower():
            raise ContractViolation(
                "invalid_dag_schedule",
                key=dag_name,
                detail="DAG file must explicitly reference weekly cadence",
            )
    weather_dag_text = (dags / "weather_nasa_power_daily.py").read_text(encoding="utf-8")
    if "daily" not in weather_dag_text.lower():
        raise ContractViolation(
            "invalid_dag_schedule",
            key="weather_nasa_power_daily.py",
            detail="weather DAG must explicitly reference daily cadence",
        )

    if not runbook.exists() or not runbook.read_text(encoding="utf-8").strip():
        raise ContractViolation(
            "missing_runbook",
            key="runbook",
            detail=f"missing or empty runbook: {runbook}",
        )

    if not incident_log.exists():
        raise ContractViolation(
            "missing_incident_log",
            key="incident_log",
            detail=f"missing incident log: {incident_log}",
        )
    log_lines = [
        line.strip() for line in incident_log.read_text(encoding="utf-8").splitlines()
    ]
    valid_lines = [line for line in log_lines if line]
    if not valid_lines:
        raise ContractViolation(
            "empty_incident_log",
            key="incident_log",
            detail="incident log must contain at least one entry",
        )

    for line in valid_lines:
        try:
            payload = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ContractViolation(
                "invalid_incident_log",
                key="incident_log",
                detail="incident log contains non-JSON line",
            ) from exc
        if not {"timestamp", "severity", "status", "summary"} <= set(payload.keys()):
            raise ContractViolation(
                "invalid_incident_log",
                key="incident_log",
                detail="incident entries must include timestamp,severity,status,summary",
            )

    if not approval.exists():
        raise ContractViolation(
            "missing_release_approval",
            key="release_approval",
            detail=f"missing approval file: {approval}",
        )

    approval_payload = json.loads(approval.read_text(encoding="utf-8"))
    missing_fields = sorted(_REQUIRED_RELEASE_FIELDS - set(approval_payload.keys()))
    if missing_fields:
        raise ContractViolation(
            "invalid_release_approval",
            key="release_approval",
            detail="missing fields: " + ", ".join(missing_fields),
        )

    if str(approval_payload["status"]).strip().lower() != "approved":
        raise ContractViolation(
            "release_not_approved",
            key="release_approval",
            detail=f"status={approval_payload['status']}",
        )

    registry = load_feature_block_registry(feature_blocks_path)
    policies = enabled_feature_policies(registry)
    if not policies:
        raise ContractViolation(
            "invalid_feature_block_registry",
            key=str(feature_blocks_path),
            detail="feature block registry has no enabled feature policies",
        )
    required_policy_fields = set(required_block_metadata_keys()) | {"block_id"}
    for feature_name, policy in sorted(policies.items()):
        missing_fields = [
            key for key in sorted(required_policy_fields) if key not in policy
        ]
        if missing_fields:
            raise ContractViolation(
                "missing_feature_block_metadata",
                key=feature_name,
                detail="missing block metadata: " + ", ".join(missing_fields),
            )
        if not str(policy["block_id"]).strip():
            raise ContractViolation(
                "missing_feature_block_metadata",
                key=feature_name,
                detail="block_id cannot be empty for enabled feature",
            )

    source_catalog = load_source_catalog(source_catalog_path)
    reference_ts = datetime.now(tz=timezone.utc)
    gold_path = Path(gold_root)
    staleness_targets = (
        ("steo_latest_workbook", gold_path / "steo_observation_panel.parquet"),
        ("nasa_power_t2m_min", gold_path / "weather_freezeoff_panel.parquet"),
    )
    freshness_lookup = {
        source.source_id: int(source.freshness_max_age_days)
        for source in source_catalog.sources
    }
    for source_id, artifact in staleness_targets:
        if not artifact.exists():
            continue
        max_age_days = freshness_lookup.get(source_id)
        if max_age_days is None:
            raise ContractViolation(
                "invalid_source_catalog",
                key=source_id,
                detail="required source freshness policy missing from source catalog",
            )
        age_days = int(
            (reference_ts - datetime.fromtimestamp(artifact.stat().st_mtime, tz=timezone.utc)).days
        )
        if age_days > max_age_days:
            raise ContractViolation(
                "stale_source",
                key=source_id,
                detail=(
                    f"artifact {artifact} is stale by {age_days} days "
                    f"(max_age_days={max_age_days})"
                ),
            )

    for forbidden in forbidden_paths:
        candidate = Path(forbidden)
        if candidate.exists():
            raise ContractViolation(
                "forbidden_path_present",
                key=str(candidate),
                detail="forbidden path exists and must be retired before release",
            )

    return OpsReadinessResult(
        passed=True,
        dag_dir=dags,
        runbook_path=runbook,
        incident_log_path=incident_log,
        release_approval_path=approval,
    )
