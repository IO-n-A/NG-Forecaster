"""Deterministic source staging, API retrieval, and local parsing helpers."""

from __future__ import annotations

import hashlib
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ng_forecaster.errors import ContractViolation
from ng_forecaster.ingest.catalog import (
    SourceCatalog,
    SourceDefinition,
    build_ingestion_plan,
    load_source_catalog,
    resolve_source_path,
)

try:  # pragma: no cover - optional dependency in constrained environments
    import requests  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    requests = None


@dataclass(frozen=True)
class StagedSourceFile:
    """Single copied source file with deterministic checksum metadata."""

    source_id: str
    filename: str
    source_path: Path
    destination_path: Path
    retrieval_mode: str
    origin: str
    sha256: str
    size_bytes: int

    def as_dict(self) -> dict[str, Any]:
        return {
            "source_id": self.source_id,
            "filename": self.filename,
            "source_path": str(self.source_path),
            "destination_path": str(self.destination_path),
            "retrieval_mode": self.retrieval_mode,
            "origin": self.origin,
            "sha256": self.sha256,
            "size_bytes": self.size_bytes,
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def stage_sources_for_stream(
    *,
    asof: pd.Timestamp,
    ingest_stream: str,
    target_root: str | Path,
    catalog: SourceCatalog | None = None,
    catalog_path: str | Path = "configs/sources.yaml",
    bootstrap_root: str | Path | None = None,
    stream_roots: Mapping[str, str | Path] | None = None,
    fixture_root: str | Path = "tests/fixtures/orchestration/bootstrap_raw",
) -> tuple[StagedSourceFile, ...]:
    """Resolve and copy stream sources into bronze destination directory."""

    resolved_catalog = catalog or load_source_catalog(catalog_path)
    plan = build_ingestion_plan(
        catalog=resolved_catalog,
        ingest_stream=ingest_stream,
        asof=asof,
        bootstrap_root=bootstrap_root,
        stream_roots=stream_roots,
        fixture_root=fixture_root,
    )

    destination_root = Path(target_root)
    destination_root.mkdir(parents=True, exist_ok=True)

    staged: list[StagedSourceFile] = []
    for item in plan:
        destination = destination_root / item.filename
        source_path = item.path.resolve()
        destination_path = destination.resolve()
        if source_path != destination_path:
            shutil.copy2(item.path, destination)
        staged.append(
            StagedSourceFile(
                source_id=item.source_id,
                filename=item.filename,
                source_path=item.path,
                destination_path=destination,
                retrieval_mode=item.retrieval_mode,
                origin=item.origin,
                sha256=_sha256_file(destination),
                size_bytes=int(destination.stat().st_size),
            )
        )

    return tuple(
        sorted(
            staged,
            key=lambda row: (row.source_id, row.filename, row.destination_path.name),
        )
    )


def _normalize_month_end(values: pd.Series, *, key: str) -> pd.Series:
    timestamps = pd.to_datetime(values, errors="coerce")
    if timestamps.isna().any():
        raise ContractViolation(
            "invalid_timestamp",
            key=key,
            detail="contains unparseable timestamps",
        )
    return timestamps.dt.to_period("M").dt.to_timestamp("M")


def _select_target_columns(
    frame: pd.DataFrame,
    *,
    timestamp_hint: str | None,
    value_hint: str | None,
) -> tuple[str, str]:
    columns = [str(col) for col in frame.columns]

    def _match_hint(hint: str | None) -> str | None:
        if hint is None:
            return None
        hint_norm = hint.strip().lower()
        for col in columns:
            if col.strip().lower() == hint_norm:
                return col
        for col in columns:
            if hint_norm in col.strip().lower():
                return col
        return None

    ts_col = _match_hint(timestamp_hint or "date")
    value_col = _match_hint(value_hint or "production")

    if ts_col is None:
        for col in columns:
            parsed = pd.to_datetime(frame[col], errors="coerce")
            if parsed.notna().sum() >= max(1, len(frame) // 2):
                ts_col = col
                break

    if value_col is None:
        for col in columns:
            if col == ts_col:
                continue
            parsed = pd.to_numeric(frame[col], errors="coerce")
            if parsed.notna().sum() >= max(1, len(frame) // 2):
                value_col = col
                break

    if ts_col is None or value_col is None:
        raise ContractViolation(
            "source_schema_drift",
            key="target_history_columns",
            detail="unable to resolve timestamp/value columns in target history source",
        )
    return ts_col, value_col


def parse_eia_target_history(
    source_path: str | Path,
    *,
    parse_config: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Parse monthly target history from EIA workbook into canonical schema."""

    path = Path(source_path)
    cfg = dict(parse_config or {})
    sheet = str(cfg.get("sheet", "Data 1"))
    header_row = int(cfg.get("header_row", 2))

    try:
        raw = pd.read_excel(path, sheet_name=sheet, header=header_row)
    except Exception as exc:
        raise ContractViolation(
            "target_history_load_failed",
            key=str(path),
            detail=f"unable to read target history workbook: {exc}",
        ) from exc

    if raw.empty:
        raise ContractViolation(
            "missing_target_history_rows",
            key=str(path),
            detail="target history source has no data rows",
        )

    raw = raw.dropna(axis=1, how="all").dropna(axis=0, how="all").copy()
    if raw.empty:
        raise ContractViolation(
            "missing_target_history_rows",
            key=str(path),
            detail="target history source has only empty rows/columns",
        )

    ts_hint = cfg.get("timestamp_col")
    value_hint = cfg.get("value_col")
    ts_col, value_col = _select_target_columns(
        raw,
        timestamp_hint=str(ts_hint) if ts_hint is not None else None,
        value_hint=str(value_hint) if value_hint is not None else None,
    )

    target = pd.DataFrame(
        {
            "timestamp": _normalize_month_end(raw[ts_col], key=f"{path.name}:{ts_col}"),
            "target_value": pd.to_numeric(raw[value_col], errors="coerce"),
        }
    )
    target = target[target["target_value"].notna()].copy()
    if target.empty:
        raise ContractViolation(
            "missing_target_history_rows",
            key=str(path),
            detail="target history source has no numeric target rows",
        )

    target = target.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    target = target.reset_index(drop=True)
    return target


def resolve_target_history_source(
    *,
    asof: pd.Timestamp,
    catalog: SourceCatalog | None = None,
    catalog_path: str | Path = "configs/sources.yaml",
    bootstrap_root: str | Path | None = None,
    stream_roots: Mapping[str, str | Path] | None = None,
    fixture_root: str | Path = "tests/fixtures/orchestration/bootstrap_raw",
) -> tuple[SourceDefinition, Path, str]:
    """Resolve required target-history source file and return its origin."""

    resolved_catalog = catalog or load_source_catalog(catalog_path)
    target_sources = [
        source for source in resolved_catalog.sources if source.role == "target_history"
    ]
    if not target_sources:
        raise ContractViolation(
            "invalid_source_catalog",
            key="sources.target_history",
            detail="target_history source is required",
        )

    for source in target_sources:
        path, origin = resolve_source_path(
            source,
            asof=asof,
            bootstrap_root=bootstrap_root
            or resolved_catalog.defaults["bootstrap_root"],
            stream_roots=stream_roots,
            fixture_root=fixture_root,
        )
        if path is not None:
            return source, path, origin

    raise ContractViolation(
        "missing_source_file",
        key="target_history",
        detail="unable to resolve any target_history source file",
    )


def fetch_eia_monthly_series_history(
    *,
    series_id: str,
    api_key: str | None,
    asof: pd.Timestamp | None = None,
    lookback_months: int = 36,
    timeout_seconds: int = 10,
) -> pd.DataFrame:
    """Fetch monthly EIA series history from API v2 and return canonical schema."""

    token = str(api_key or "").strip()
    if not token:
        return pd.DataFrame(
            columns=["timestamp", "target_value", "series_id", "source"]
        )

    if requests is None:
        raise ContractViolation(
            "missing_dependency",
            key="requests",
            detail="requests package is required for EIA API retrieval",
        )

    series = str(series_id).strip()
    if not series:
        raise ContractViolation(
            "invalid_source_catalog",
            key="series_id",
            detail="series_id must be a non-empty string",
        )
    if lookback_months < 1:
        raise ContractViolation(
            "invalid_source_catalog",
            key="lookback_months",
            detail="lookback_months must be >= 1",
        )

    url = f"https://api.eia.gov/v2/seriesid/{series}"
    response = requests.get(
        url,
        params={"api_key": token},
        timeout=float(timeout_seconds),
    )
    response.raise_for_status()
    payload = response.json()

    data_rows = payload.get("response", {}).get("data", [])
    if not isinstance(data_rows, list):
        raise ContractViolation(
            "source_schema_drift",
            key="response.data",
            detail="EIA payload response.data must be a list",
        )

    rows: list[dict[str, Any]] = []
    for item in data_rows:
        if not isinstance(item, Mapping):
            continue
        period = item.get("period")
        value = item.get("value")
        timestamp = pd.to_datetime(period, errors="coerce")
        numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
        if pd.isna(timestamp) or pd.isna(numeric):
            continue
        rows.append(
            {
                "timestamp": pd.Timestamp(timestamp).to_period("M").to_timestamp("M"),
                "target_value": float(numeric),
                "series_id": series,
                "source": "eia_api_v2",
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=["timestamp", "target_value", "series_id", "source"]
        )

    frame = pd.DataFrame(rows)
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")

    if asof is not None:
        asof_ts = pd.Timestamp(asof).to_period("M").to_timestamp("M")
        frame = frame[frame["timestamp"] <= asof_ts].copy()

    frame = frame.tail(int(lookback_months)).reset_index(drop=True)
    return frame
