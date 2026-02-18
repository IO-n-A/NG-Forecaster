"""Sprint 4B migration utilities for retiring `data/new` into bronze/gold storage."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import shutil
from typing import Any

import pandas as pd

from ng_forecaster.data.bronze_writer import sha256_file
from ng_forecaster.errors import ContractViolation
from ng_forecaster.ingest.adapters.steo_vintages import (
    parse_vintage_month_from_filename,
)

_REQUIRED_INVENTORY_COLUMNS = {
    "relative_path",
    "integration_layer_target",
    "integration_action",
    "priority",
    "is_duplicate",
    "duplicate_of",
}


@dataclass(frozen=True)
class MigrationResult:
    """Summary payload for `data/new` migration execution."""

    migrated_count: int
    skipped_count: int
    duplicate_count: int
    unresolved_count: int
    report_path: Path
    index_path: Path

    def as_dict(self) -> dict[str, Any]:
        return {
            "migrated_count": int(self.migrated_count),
            "skipped_count": int(self.skipped_count),
            "duplicate_count": int(self.duplicate_count),
            "unresolved_count": int(self.unresolved_count),
            "report_path": str(self.report_path),
            "index_path": str(self.index_path),
        }


def _copy_file(source: Path, destination: Path) -> dict[str, Any]:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if source.resolve() != destination.resolve():
        shutil.copy2(source, destination)
    return {
        "source_path": str(source),
        "destination_path": str(destination),
        "bytes": int(destination.stat().st_size),
        "sha256": sha256_file(destination),
    }


def _resolve_destination(
    *, row: pd.Series, source_path: Path, bronze_root: Path
) -> Path:
    relative = str(row["relative_path"])
    layer = str(row["integration_layer_target"]).strip().lower()

    if relative == "steo_archives/manifest.csv":
        return bronze_root / "eia_bulk" / "steo_vintages" / "manifest.csv"

    if (
        relative.startswith("steo_archives/xlsx/")
        and source_path.suffix.lower() == ".xlsx"
    ):
        vintage = parse_vintage_month_from_filename(source_path)
        return (
            bronze_root
            / "eia_bulk"
            / "steo_vintages"
            / f"vintage_month={vintage}"
            / "steo_m.xlsx"
        )

    if (
        relative.startswith("steo_archives/pdf/")
        and source_path.suffix.lower() == ".pdf"
    ):
        vintage = parse_vintage_month_from_filename(source_path)
        return (
            bronze_root
            / "eia_bulk"
            / "steo_vintages"
            / f"vintage_month={vintage}"
            / "steo_full.pdf"
        )

    if layer == "bronze":
        return bronze_root / "eia_bulk" / "lineage" / relative

    return bronze_root / "context" / relative


def migrate_data_new_inventory(
    *,
    inventory_path: str | Path,
    source_root: str | Path = "data/new",
    bronze_root: str | Path = "data/bronze",
    include_priorities: tuple[str, ...] = ("P0", "P1"),
    report_root: str | Path = "data/reports",
) -> MigrationResult:
    """Migrate evaluated `data/new` inventory rows into bronze/context storage."""

    inventory_file = Path(inventory_path)
    if not inventory_file.exists():
        raise ContractViolation(
            "missing_source_file",
            key=str(inventory_file),
            detail="inventory CSV does not exist",
        )

    inventory = pd.read_csv(inventory_file)
    missing_columns = sorted(_REQUIRED_INVENTORY_COLUMNS - set(inventory.columns))
    if missing_columns:
        raise ContractViolation(
            "source_schema_drift",
            key=str(inventory_file),
            detail="inventory missing required columns: " + ", ".join(missing_columns),
        )

    if inventory["integration_action"].astype(str).str.strip().eq("").any():
        raise ContractViolation(
            "source_schema_drift",
            key="integration_action",
            detail="inventory contains empty integration_action rows",
        )

    selected = inventory[
        inventory["priority"].astype(str).str.upper().isin(include_priorities)
    ].copy()
    selected = selected.sort_values("relative_path").reset_index(drop=True)

    src_root = Path(source_root)
    dst_root = Path(bronze_root)
    records: list[dict[str, Any]] = []
    unresolved: list[str] = []

    for _, row in selected.iterrows():
        relative = str(row["relative_path"]).strip()
        if not relative:
            continue

        is_duplicate = str(row["is_duplicate"]).strip().lower() == "true"
        if is_duplicate:
            records.append(
                {
                    "relative_path": relative,
                    "status": "duplicate_skipped",
                    "duplicate_of": str(row.get("duplicate_of", "")),
                }
            )
            continue

        source_path = src_root / relative
        if not source_path.exists() or not source_path.is_file():
            unresolved.append(relative)
            records.append(
                {
                    "relative_path": relative,
                    "status": "missing_source",
                    "source_path": str(source_path),
                }
            )
            continue

        destination = _resolve_destination(
            row=row,
            source_path=source_path,
            bronze_root=dst_root,
        )
        copy_meta = _copy_file(source_path, destination)
        records.append(
            {
                "relative_path": relative,
                "status": "migrated",
                "priority": str(row["priority"]),
                "integration_layer_target": str(row["integration_layer_target"]),
                "integration_action": str(row["integration_action"]),
                **copy_meta,
            }
        )

    index = pd.DataFrame(records)
    report_dir = Path(report_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    index_path = report_dir / "sprint4b_migration_index.csv"
    index.to_csv(index_path, index=False)

    migrated_count = int(index["status"].eq("migrated").sum())
    duplicate_count = int(index["status"].eq("duplicate_skipped").sum())
    missing_count = int(index["status"].eq("missing_source").sum())
    skipped_count = int(len(index) - migrated_count - duplicate_count - missing_count)

    payload = {
        "inventory_path": str(inventory_file),
        "source_root": str(src_root),
        "bronze_root": str(dst_root),
        "include_priorities": list(include_priorities),
        "row_count": int(len(selected)),
        "migrated_count": migrated_count,
        "duplicate_count": duplicate_count,
        "missing_source_count": missing_count,
        "skipped_count": skipped_count,
        "unresolved_relative_paths": unresolved,
        "index_path": str(index_path),
    }
    report_path = report_dir / "sprint4b_migration_report.json"
    report_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    return MigrationResult(
        migrated_count=migrated_count,
        skipped_count=skipped_count,
        duplicate_count=duplicate_count,
        unresolved_count=len(unresolved),
        report_path=report_path,
        index_path=index_path,
    )


def retire_data_new(source_root: str | Path = "data/new") -> None:
    """Delete `data/new` after migration acceptance."""

    target = Path(source_root)
    if not target.exists():
        return
    if not target.is_dir():
        raise ContractViolation(
            "invalid_source_catalog",
            key=str(target),
            detail="expected data/new retirement target to be a directory",
        )
    shutil.rmtree(target)
