"""End-to-end Sprint 4B data refresh cycle orchestration."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd

from ng_forecaster.data.gold_publish import publish_steo_gold_marts
from ng_forecaster.data.migration import migrate_data_new_inventory, retire_data_new
from ng_forecaster.data.silver_normalize import normalize_steo_vintages
from ng_forecaster.errors import ContractViolation


@dataclass(frozen=True)
class DataRefreshResult:
    """Summary payload for data refresh cycle."""

    asof: str
    migration_report: str | None
    silver_report: str
    gold_report: str
    retrieval_reports: tuple[str, ...]
    retired_data_new: bool
    report_path: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "asof": self.asof,
            "migration_report": self.migration_report,
            "silver_report": self.silver_report,
            "gold_report": self.gold_report,
            "retrieval_reports": list(self.retrieval_reports),
            "retired_data_new": bool(self.retired_data_new),
            "report_path": self.report_path,
        }


def _run_script(command: list[str]) -> dict[str, Any]:
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    payload_text = completed.stdout.strip().splitlines()[-1]
    parsed = json.loads(payload_text)
    if not isinstance(parsed, dict):
        raise ContractViolation(
            "source_schema_drift",
            key="data_refresh_script_output",
            detail="script output payload must be a JSON object",
        )
    return dict(parsed)


def run_data_refresh_cycle(
    *,
    asof: object,
    inventory_path: str | Path = "data/reports/sprint4b_data_new_inventory.csv",
    source_root: str | Path = "data/new",
    bronze_root: str | Path = "data/bronze",
    run_steo_retrieval: bool = False,
    run_context_retrieval: bool = False,
    retire_source_dir: bool = False,
) -> DataRefreshResult:
    """Run retrieval (optional) + migration + silver normalize + gold publish."""

    asof_ts = pd.Timestamp(asof)
    asof_label = asof_ts.date().isoformat()

    retrieval_reports: list[str] = []
    repo_root = Path(__file__).resolve().parents[3]
    python_bin = sys.executable

    if run_steo_retrieval:
        steo_payload = _run_script(
            [
                python_bin,
                str(repo_root / "scripts/data/retrieve_steo_archives.py"),
                "--start-month",
                "2020-01-31",
                "--end-month",
                asof_label,
                "--output-root",
                "data/bronze/eia_bulk/steo_vintages",
            ]
        )
        retrieval_reports.append(str(steo_payload.get("report_path", "")))

    if run_context_retrieval:
        context_payload = _run_script(
            [
                python_bin,
                str(repo_root / "scripts/data/retrieve_context_reports.py"),
                "--output-root",
                "data/bronze/context/external",
            ]
        )
        retrieval_reports.append(str(context_payload.get("report_path", "")))

    migration_report: str | None = None
    source_path = Path(source_root)
    if source_path.exists() and source_path.is_dir() and any(source_path.iterdir()):
        migration = migrate_data_new_inventory(
            inventory_path=inventory_path,
            source_root=source_root,
            bronze_root=bronze_root,
        )
        migration_report = str(migration.report_path)

    silver = normalize_steo_vintages(
        bronze_root=Path(bronze_root) / "eia_bulk" / "steo_vintages",
        silver_root="data/silver/steo_vintages",
    )
    gold = publish_steo_gold_marts(
        silver_root="data/silver/steo_vintages",
        gold_root="data/gold",
    )

    retired = False
    if retire_source_dir:
        retire_data_new(source_root)
        retired = True

    payload = {
        "asof": asof_label,
        "migration_report": migration_report,
        "silver_report": str(silver.get("report_path", "")),
        "gold_report": str(gold.get("report_path", "")),
        "retrieval_reports": retrieval_reports,
        "retired_data_new": retired,
    }
    report_path = Path("data/reports/data_refresh_cycle_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8"
    )

    return DataRefreshResult(
        asof=asof_label,
        migration_report=migration_report,
        silver_report=str(silver.get("report_path", "")),
        gold_report=str(gold.get("report_path", "")),
        retrieval_reports=tuple(retrieval_reports),
        retired_data_new=retired,
        report_path=str(report_path),
    )
