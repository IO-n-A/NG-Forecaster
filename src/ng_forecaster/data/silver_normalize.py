"""Silver-layer normalization for STEO vintage workbooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ng_forecaster.data.bronze_writer import sha256_file
from ng_forecaster.errors import ContractViolation
from ng_forecaster.ingest.adapters.steo_vintages import parse_steo_vintage_workbook


def _discover_vintage_workbooks(bronze_root: Path) -> list[Path]:
    direct = sorted(
        [
            path / "steo_m.xlsx"
            for path in bronze_root.glob("vintage_month=*")
            if (path / "steo_m.xlsx").exists()
        ],
        key=lambda item: item.parent.name,
    )
    if direct:
        return direct

    fallback = sorted(bronze_root.rglob("*_base.xlsx"))
    if fallback:
        return fallback

    return []


def normalize_steo_vintages(
    *,
    bronze_root: str | Path = "data/bronze/eia_bulk/steo_vintages",
    silver_root: str | Path = "data/silver/steo_vintages",
    report_root: str | Path = "data/reports",
) -> dict[str, Any]:
    """Parse STEO workbook vintages into deterministic silver parquet tables."""

    source_root = Path(bronze_root)
    if not source_root.exists():
        raise ContractViolation(
            "missing_source_file",
            key=str(source_root),
            detail="bronze STEO vintage root does not exist",
        )

    workbooks = _discover_vintage_workbooks(source_root)
    if not workbooks:
        raise ContractViolation(
            "missing_source_file",
            key=str(source_root),
            detail="no STEO workbooks discovered for silver normalization",
        )

    target_root = Path(silver_root)
    target_root.mkdir(parents=True, exist_ok=True)

    summary_rows: list[dict[str, Any]] = []
    for workbook in sorted(workbooks, key=lambda item: str(item)):
        parsed = parse_steo_vintage_workbook(workbook)
        vintage_dir = target_root / f"vintage_month={parsed.vintage_month}"
        vintage_dir.mkdir(parents=True, exist_ok=True)

        table_paths: dict[str, str] = {}
        for table_id, frame in sorted(parsed.tables.items()):
            table_meta = dict(parsed.table_metadata.get(table_id, {}))
            if frame.empty and bool(table_meta.get("available", True)):
                raise ContractViolation(
                    "source_schema_drift",
                    key=f"{workbook}:{table_id}",
                    detail="parsed STEO table is empty",
                )
            available_timestamp = pd.to_datetime(
                parsed.forecast_completed_on
                or pd.Timestamp(f"{parsed.vintage_month}-01")
                .to_period("M")
                .to_timestamp("M"),
                errors="coerce",
            )
            if pd.isna(available_timestamp):
                raise ContractViolation(
                    "invalid_timestamp",
                    key=f"{workbook}:{table_id}",
                    detail="unable to resolve available_timestamp from forecast metadata",
                )
            frame = frame.copy()
            frame["available_timestamp"] = pd.Timestamp(available_timestamp)
            destination = vintage_dir / f"{table_id}.parquet"
            frame.to_parquet(destination, index=False)
            table_paths[table_id] = str(destination)

        metadata = {
            "vintage_month": parsed.vintage_month,
            "workbook_path": str(parsed.workbook_path),
            "workbook_sha256": sha256_file(parsed.workbook_path),
            "forecast_completed_on": parsed.forecast_completed_on,
            "table_metadata": parsed.table_metadata,
            "table_paths": table_paths,
        }
        metadata_path = vintage_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(metadata, sort_keys=True, indent=2),
            encoding="utf-8",
        )

        summary_entry: dict[str, Any] = {
            "vintage_month": parsed.vintage_month,
            "workbook_path": str(parsed.workbook_path),
            "forecast_completed_on": parsed.forecast_completed_on,
            "metadata_path": str(metadata_path),
        }
        for table_id, frame in sorted(parsed.tables.items()):
            summary_entry[f"{table_id}_rows"] = int(len(frame))
        summary_rows.append(summary_entry)

    summary = (
        pd.DataFrame(summary_rows).sort_values("vintage_month").reset_index(drop=True)
    )
    report_dir = Path(report_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    summary_path = report_dir / "sprint4b_silver_normalize_summary.csv"
    summary.to_csv(summary_path, index=False)

    payload = {
        "bronze_root": str(source_root),
        "silver_root": str(target_root),
        "vintage_count": int(len(summary)),
        "summary_path": str(summary_path),
    }
    report_path = report_dir / "sprint4b_silver_normalize_report.json"
    report_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    payload["report_path"] = str(report_path)
    return payload
