"""Deterministic bronze-layer snapshot writer with hash/index manifests."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Iterable

import pandas as pd

from ng_forecaster.errors import ContractViolation


@dataclass(frozen=True)
class BronzeFileRecord:
    """Single copied bronze file with deterministic metadata."""

    relative_path: str
    source_path: str
    destination_path: str
    size_bytes: int
    sha256: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "relative_path": self.relative_path,
            "source_path": self.source_path,
            "destination_path": self.destination_path,
            "size_bytes": int(self.size_bytes),
            "sha256": self.sha256,
        }


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 hash for a file path."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _resolve_relative_path(path: Path, *, source_root: Path | None) -> Path:
    if source_root is None:
        return Path(path.name)
    try:
        relative = path.relative_to(source_root)
    except ValueError:
        return Path(path.name)
    return relative


def write_bronze_snapshot(
    *,
    files: Iterable[str | Path],
    ingest_stream: str,
    asof: object,
    bronze_root: str | Path = "data/bronze",
    source_root: str | Path | None = None,
    manifest_path: str | Path | None = None,
    index_path: str | Path | None = None,
) -> dict[str, Any]:
    """Copy files into bronze snapshot path and write deterministic manifests."""

    stream = str(ingest_stream).strip().lower()
    if not stream:
        raise ContractViolation(
            "invalid_source_catalog",
            key="ingest_stream",
            detail="ingest_stream must be non-empty",
        )

    asof_ts = pd.Timestamp(asof)
    if pd.isna(asof_ts):
        raise ContractViolation(
            "invalid_timestamp",
            key="asof",
            detail="asof could not be parsed",
        )

    source_base = Path(source_root) if source_root is not None else None
    destination_root = Path(bronze_root) / stream / f"asof={asof_ts.date().isoformat()}"
    destination_root.mkdir(parents=True, exist_ok=True)

    materialized_files: list[Path] = []
    for item in files:
        path = Path(item)
        if not path.exists() or not path.is_file():
            raise ContractViolation(
                "missing_source_file",
                key=str(path),
                detail="bronze writer input file does not exist",
            )
        materialized_files.append(path)

    records: list[BronzeFileRecord] = []
    for source in sorted(materialized_files, key=lambda item: str(item)):
        relative = _resolve_relative_path(source, source_root=source_base)
        destination = destination_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        if source.resolve() != destination.resolve():
            shutil.copy2(source, destination)

        records.append(
            BronzeFileRecord(
                relative_path=relative.as_posix(),
                source_path=str(source),
                destination_path=str(destination),
                size_bytes=int(destination.stat().st_size),
                sha256=sha256_file(destination),
            )
        )

    payload = {
        "asof": asof_ts.date().isoformat(),
        "ingest_stream": stream,
        "destination_root": str(destination_root),
        "file_count": int(len(records)),
        "files": [record.as_dict() for record in records],
    }

    resolved_manifest_path = (
        Path(manifest_path)
        if manifest_path is not None
        else Path("data/reports")
        / f"bronze_{stream}_manifest_{asof_ts.date().isoformat()}.json"
    )
    resolved_manifest_path.parent.mkdir(parents=True, exist_ok=True)
    resolved_manifest_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    resolved_index_path = (
        Path(index_path)
        if index_path is not None
        else Path("data/reports")
        / f"bronze_{stream}_index_{asof_ts.date().isoformat()}.csv"
    )
    resolved_index_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([record.as_dict() for record in records]).to_csv(
        resolved_index_path,
        index=False,
    )

    payload["manifest_path"] = str(resolved_manifest_path)
    payload["index_path"] = str(resolved_index_path)
    return payload
