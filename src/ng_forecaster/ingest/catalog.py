"""Source catalog contracts and deterministic source resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ng_forecaster.data.validators import load_yaml
from ng_forecaster.errors import ContractViolation

_ALLOWED_STREAMS = {"api", "bulk"}
_ALLOWED_RETRIEVAL_MODES = {
    "api_snapshot",
    "bulk_snapshot",
    "bootstrap_file",
    "fixture_csv",
    "archive_manifest",
    "archive_glob",
}
_ALLOWED_ROLES = {"target_history", "driver", "metadata"}


@dataclass(frozen=True)
class SourceDefinition:
    """Single source entry from ``configs/sources.yaml``."""

    source_id: str
    role: str
    ingest_stream: str
    retrieval_mode: str
    filename: str
    required: bool
    freshness_max_age_days: int
    parse: dict[str, Any]


@dataclass(frozen=True)
class SourceCatalog:
    """Validated source catalog payload."""

    version: int
    defaults: dict[str, Any]
    release_calendar: dict[str, Any]
    sources: tuple[SourceDefinition, ...]

    def by_stream(self, ingest_stream: str) -> tuple[SourceDefinition, ...]:
        """Return stream-specific sources in deterministic ID order."""

        stream = str(ingest_stream).strip().lower()
        return tuple(
            source for source in self.sources if source.ingest_stream == stream
        )

    def target_sources(self) -> tuple[SourceDefinition, ...]:
        """Return target-history sources ordered by ID."""

        return tuple(
            source for source in self.sources if source.role == "target_history"
        )


@dataclass(frozen=True)
class ResolvedSource:
    """Resolved input file for a source definition."""

    source_id: str
    ingest_stream: str
    retrieval_mode: str
    filename: str
    path: Path
    origin: str


def _as_positive_int(value: object, *, key: str, allow_zero: bool = False) -> int:
    if isinstance(value, bool):
        parsed = int(value)
    elif isinstance(value, int):
        parsed = value
    elif isinstance(value, str):
        token = value.strip()
        try:
            parsed = int(token)
        except ValueError as exc:
            raise ContractViolation(
                "invalid_source_catalog",
                key=key,
                detail="value must be an integer",
            ) from exc
    else:
        raise ContractViolation(
            "invalid_source_catalog",
            key=key,
            detail="value must be an integer",
        )

    min_allowed = 0 if allow_zero else 1
    if parsed < min_allowed:
        comparator = ">=" if allow_zero else ">="
        raise ContractViolation(
            "invalid_source_catalog",
            key=key,
            detail=f"value must be {comparator} {min_allowed}",
        )
    return parsed


def _as_non_empty_string(value: object, *, key: str) -> str:
    parsed = str(value).strip()
    if not parsed:
        raise ContractViolation(
            "invalid_source_catalog",
            key=key,
            detail="value must be a non-empty string",
        )
    return parsed


def _normalize_release_calendar(payload: Mapping[str, Any]) -> dict[str, Any]:
    lag_months = _as_positive_int(
        payload.get("lag_months", 2), key="release_calendar.lag_months"
    )
    release_day = _as_positive_int(
        payload.get("release_day_of_month", 30),
        key="release_calendar.release_day_of_month",
    )
    if release_day > 31:
        raise ContractViolation(
            "invalid_source_catalog",
            key="release_calendar.release_day_of_month",
            detail="release_day_of_month must be <= 31",
        )

    raw_window = payload.get("admissible_day_window", {})
    if not isinstance(raw_window, Mapping):
        raise ContractViolation(
            "invalid_source_catalog",
            key="release_calendar.admissible_day_window",
            detail="admissible_day_window must be a mapping",
        )
    start_day = _as_positive_int(
        raw_window.get("start_day", 1),
        key="release_calendar.admissible_day_window.start_day",
    )
    end_day = _as_positive_int(
        raw_window.get("end_day", 31),
        key="release_calendar.admissible_day_window.end_day",
    )
    if start_day > end_day or end_day > 31:
        raise ContractViolation(
            "invalid_source_catalog",
            key="release_calendar.admissible_day_window",
            detail="window must satisfy 1 <= start_day <= end_day <= 31",
        )

    return {
        "lag_months": lag_months,
        "release_day_of_month": release_day,
        "admissible_day_window": {
            "start_day": start_day,
            "end_day": end_day,
        },
    }


def validate_source_catalog(payload: Mapping[str, Any] | None) -> SourceCatalog:
    """Validate source catalog shape and return normalized dataclasses."""

    raw = dict(payload or {})
    version = _as_positive_int(raw.get("version", 1), key="version")

    defaults_raw = raw.get("defaults", {})
    if not isinstance(defaults_raw, Mapping):
        raise ContractViolation(
            "invalid_source_catalog",
            key="defaults",
            detail="defaults must be a mapping",
        )
    defaults = dict(defaults_raw)
    defaults_freshness = _as_positive_int(
        defaults.get("freshness_max_age_days", 120),
        key="defaults.freshness_max_age_days",
    )
    bootstrap_root = _as_non_empty_string(
        defaults.get("bootstrap_root", "data/bootstrap/raw"),
        key="defaults.bootstrap_root",
    )
    defaults["freshness_max_age_days"] = defaults_freshness
    defaults["bootstrap_root"] = bootstrap_root

    release_raw = raw.get("release_calendar", {})
    if not isinstance(release_raw, Mapping):
        raise ContractViolation(
            "invalid_source_catalog",
            key="release_calendar",
            detail="release_calendar must be a mapping",
        )
    release_calendar = _normalize_release_calendar(release_raw)

    sources_raw = raw.get("sources", [])
    if not isinstance(sources_raw, list) or not sources_raw:
        raise ContractViolation(
            "invalid_source_catalog",
            key="sources",
            detail="sources must be a non-empty list",
        )

    source_defs: list[SourceDefinition] = []
    seen_ids: set[str] = set()
    for idx, item in enumerate(sources_raw):
        if not isinstance(item, Mapping):
            raise ContractViolation(
                "invalid_source_catalog",
                key=f"sources[{idx}]",
                detail="source entries must be mappings",
            )

        source_id = _as_non_empty_string(item.get("id", ""), key=f"sources[{idx}].id")
        if source_id in seen_ids:
            raise ContractViolation(
                "invalid_source_catalog",
                key="sources.id",
                detail=f"duplicate source id: {source_id}",
            )
        seen_ids.add(source_id)

        role = _as_non_empty_string(
            item.get("role", ""), key=f"sources[{idx}].role"
        ).lower()
        if role not in _ALLOWED_ROLES:
            raise ContractViolation(
                "invalid_source_catalog",
                key=f"sources[{idx}].role",
                detail=f"role must be one of {sorted(_ALLOWED_ROLES)}",
            )

        ingest_stream = _as_non_empty_string(
            item.get("ingest_stream", ""),
            key=f"sources[{idx}].ingest_stream",
        ).lower()
        if ingest_stream not in _ALLOWED_STREAMS:
            raise ContractViolation(
                "invalid_source_catalog",
                key=f"sources[{idx}].ingest_stream",
                detail=f"ingest_stream must be one of {sorted(_ALLOWED_STREAMS)}",
            )

        retrieval_mode = _as_non_empty_string(
            item.get("retrieval_mode", ""),
            key=f"sources[{idx}].retrieval_mode",
        ).lower()
        if retrieval_mode not in _ALLOWED_RETRIEVAL_MODES:
            raise ContractViolation(
                "invalid_source_catalog",
                key=f"sources[{idx}].retrieval_mode",
                detail=(
                    "retrieval_mode must be one of "
                    f"{sorted(_ALLOWED_RETRIEVAL_MODES)}"
                ),
            )

        filename = _as_non_empty_string(
            item.get("filename", ""),
            key=f"sources[{idx}].filename",
        )

        freshness = _as_positive_int(
            item.get("freshness_max_age_days", defaults_freshness),
            key=f"sources[{idx}].freshness_max_age_days",
        )

        parse = item.get("parse", {})
        if parse is None:
            parse = {}
        if not isinstance(parse, Mapping):
            raise ContractViolation(
                "invalid_source_catalog",
                key=f"sources[{idx}].parse",
                detail="parse must be a mapping",
            )

        source_defs.append(
            SourceDefinition(
                source_id=source_id,
                role=role,
                ingest_stream=ingest_stream,
                retrieval_mode=retrieval_mode,
                filename=filename,
                required=bool(item.get("required", True)),
                freshness_max_age_days=freshness,
                parse=dict(parse),
            )
        )

    source_defs = sorted(source_defs, key=lambda source: source.source_id)
    if not any(
        source.role == "target_history" and source.required for source in source_defs
    ):
        raise ContractViolation(
            "invalid_source_catalog",
            key="sources",
            detail="at least one required target_history source is mandatory",
        )

    return SourceCatalog(
        version=version,
        defaults=defaults,
        release_calendar=release_calendar,
        sources=tuple(source_defs),
    )


def load_source_catalog(path: str | Path = "configs/sources.yaml") -> SourceCatalog:
    """Load and validate source catalog YAML."""

    payload = load_yaml(path)
    return validate_source_catalog(payload)


def _parse_snapshot_asof_dir(path: Path) -> pd.Timestamp | None:
    name = path.name
    if not name.startswith("asof="):
        return None
    token = name[len("asof=") :].strip()
    parsed = pd.to_datetime(token, errors="coerce")
    if pd.isna(parsed):
        return None
    return pd.Timestamp(parsed)


def _resolve_snapshot_file(
    *,
    root: Path,
    filename: str,
    asof: pd.Timestamp,
) -> Path | None:
    if not root.exists():
        return None

    candidates: list[tuple[pd.Timestamp, Path]] = []
    for path in sorted(root.iterdir()):
        if not path.is_dir():
            continue
        asof_ts = _parse_snapshot_asof_dir(path)
        if asof_ts is None:
            continue
        if asof_ts > asof:
            continue
        source_path = path / filename
        if source_path.exists():
            candidates.append((asof_ts, source_path))

    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda item: (item[0], str(item[1])))
    return candidates[-1][1]


def _resolve_archive_glob_file(
    *,
    root: Path,
    pattern: str,
    asof: pd.Timestamp,
) -> Path | None:
    if not root.exists():
        return None

    candidates: list[tuple[pd.Timestamp, Path]] = []
    for path in sorted(root.glob(pattern)):
        if not path.is_file():
            continue
        vintage_token = None
        for part in path.parts:
            if part.startswith("vintage_month="):
                vintage_token = part.split("=", 1)[1]
                break
        if vintage_token is None:
            candidates.append((pd.Timestamp.min, path))
            continue
        parsed = pd.to_datetime(vintage_token, errors="coerce")
        if pd.isna(parsed):
            continue
        vintage_month = pd.Timestamp(parsed).to_period("M").to_timestamp("M")
        if vintage_month <= asof.to_period("M").to_timestamp("M"):
            candidates.append((vintage_month, path))

    if not candidates:
        return None
    candidates = sorted(candidates, key=lambda item: (item[0], str(item[1])))
    return candidates[-1][1]


def resolve_source_path(
    source: SourceDefinition,
    *,
    asof: pd.Timestamp,
    bootstrap_root: str | Path,
    stream_roots: Mapping[str, str | Path] | None = None,
    fixture_root: str | Path = "tests/fixtures/orchestration/bootstrap_raw",
) -> tuple[Path | None, str]:
    """Resolve a source file path by retrieval mode and as-of priority."""

    roots = {
        "api": Path("data/bronze/eia_api"),
        "bulk": Path("data/bronze/eia_bulk"),
    }
    for key, value in dict(stream_roots or {}).items():
        roots[str(key)] = Path(value)

    bootstrap_path = Path(bootstrap_root) / source.filename
    fixture_path = Path(fixture_root) / source.filename
    stream_root = roots.get(source.ingest_stream, Path())

    mode = source.retrieval_mode
    if mode in {"api_snapshot", "bulk_snapshot"}:
        snapshot = _resolve_snapshot_file(
            root=stream_root, filename=source.filename, asof=asof
        )
        if snapshot is not None:
            return snapshot, f"{source.ingest_stream}_snapshot"
        if bootstrap_path.exists():
            return bootstrap_path, "bootstrap_raw"
        if fixture_path.exists():
            return fixture_path, "fixture_fallback"
        return None, "missing"

    if mode == "bootstrap_file":
        if bootstrap_path.exists():
            return bootstrap_path, "bootstrap_raw"
        if fixture_path.exists():
            return fixture_path, "fixture_fallback"
        snapshot = _resolve_snapshot_file(
            root=stream_root, filename=source.filename, asof=asof
        )
        if snapshot is not None:
            return snapshot, f"{source.ingest_stream}_snapshot"
        return None, "missing"

    if mode == "fixture_csv":
        if fixture_path.exists():
            return fixture_path, "fixture_root"
        snapshot = _resolve_snapshot_file(
            root=stream_root, filename=source.filename, asof=asof
        )
        if snapshot is not None:
            return snapshot, f"{source.ingest_stream}_snapshot"
        if bootstrap_path.exists():
            return bootstrap_path, "bootstrap_raw"
        return None, "missing"

    if mode == "archive_manifest":
        archive_manifest = str(source.parse.get("archive_manifest", "")).strip()
        if archive_manifest:
            manifest_path = stream_root / archive_manifest
        else:
            manifest_path = stream_root / source.filename
        if manifest_path.exists():
            return manifest_path, f"{source.ingest_stream}_archive_manifest"
        if bootstrap_path.exists():
            return bootstrap_path, "bootstrap_raw"
        if fixture_path.exists():
            return fixture_path, "fixture_fallback"
        return None, "missing"

    if mode == "archive_glob":
        pattern = str(source.parse.get("glob", "")).strip()
        if not pattern:
            raise ContractViolation(
                "invalid_source_catalog",
                key=f"parse.glob:{source.source_id}",
                detail="archive_glob retrieval_mode requires parse.glob",
            )
        archive_match = _resolve_archive_glob_file(
            root=stream_root,
            pattern=pattern,
            asof=asof,
        )
        if archive_match is not None:
            return archive_match, f"{source.ingest_stream}_archive_glob"
        if bootstrap_path.exists():
            return bootstrap_path, "bootstrap_raw"
        if fixture_path.exists():
            return fixture_path, "fixture_fallback"
        return None, "missing"

    raise ContractViolation(
        "invalid_source_catalog",
        key=f"retrieval_mode:{source.source_id}",
        detail=f"unsupported retrieval mode: {mode}",
    )


def build_ingestion_plan(
    *,
    catalog: SourceCatalog,
    ingest_stream: str,
    asof: pd.Timestamp,
    bootstrap_root: str | Path | None = None,
    stream_roots: Mapping[str, str | Path] | None = None,
    fixture_root: str | Path = "tests/fixtures/orchestration/bootstrap_raw",
) -> tuple[ResolvedSource, ...]:
    """Resolve deterministic source file plan for API or bulk ingestion."""

    stream = str(ingest_stream).strip().lower()
    if stream not in _ALLOWED_STREAMS:
        raise ContractViolation(
            "invalid_source_catalog",
            key="ingest_stream",
            detail=f"ingest_stream must be one of {sorted(_ALLOWED_STREAMS)}",
        )

    resolved_bootstrap_root = Path(
        bootstrap_root or catalog.defaults.get("bootstrap_root", "data/bootstrap/raw")
    )

    plan: list[ResolvedSource] = []
    for source in catalog.by_stream(stream):
        source_path, origin = resolve_source_path(
            source,
            asof=asof,
            bootstrap_root=resolved_bootstrap_root,
            stream_roots=stream_roots,
            fixture_root=fixture_root,
        )
        if source_path is None:
            if source.required:
                raise ContractViolation(
                    "missing_source_file",
                    key=source.source_id,
                    detail=(
                        f"required source file is missing: {source.filename}; "
                        f"mode={source.retrieval_mode}"
                    ),
                )
            continue

        plan.append(
            ResolvedSource(
                source_id=source.source_id,
                ingest_stream=source.ingest_stream,
                retrieval_mode=source.retrieval_mode,
                filename=source.filename,
                path=source_path,
                origin=origin,
            )
        )

    return tuple(sorted(plan, key=lambda item: (item.source_id, item.filename)))
