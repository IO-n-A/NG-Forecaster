"""Parser utilities for EIA STEO vintage workbooks."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Mapping, cast

import pandas as pd

from ng_forecaster.errors import ContractViolation

_MONTH_TO_NUMBER = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_VINTAGE_TOKEN_RE = re.compile(r"(?P<month>[a-z]{3})(?P<year>\d{2})", re.IGNORECASE)
_NOTE_PREFIXES = ("notes", "(a)", "(b)", "(c)", "(d)", "(e)", "eia completed")


@dataclass(frozen=True)
class ParsedSteoVintage:
    """Parsed STEO vintage payload and metadata."""

    vintage_month: str
    workbook_path: Path
    forecast_completed_on: str | None
    tables: dict[str, pd.DataFrame]
    table_metadata: dict[str, dict[str, Any]]


DEFAULT_TABLE_SHEETS = {
    "table_2": "2tab",
    "table_4a": "4atab",
    "table_5a": "5atab",
    "table_5b": "5btab",
    "table_10a": "10atab",
    "table_10b": "10btab",
}
_OPTIONAL_TABLE_INTRO_MONTH = {
    "table_10a": pd.Timestamp("2024-06-30"),
    "table_10b": pd.Timestamp("2024-06-30"),
}


def parse_vintage_month_from_filename(path: str | Path) -> str:
    """Resolve vintage month (`YYYY-MM`) from STEO workbook filename."""

    resolved_path = Path(path)
    for part in resolved_path.parts:
        if part.startswith("vintage_month="):
            token = part.split("=", 1)[1].strip()
            parsed = pd.to_datetime(token, errors="coerce")
            if not pd.isna(parsed):
                month = pd.Timestamp(parsed).to_period("M").to_timestamp("M")
                return cast(str, month.strftime("%Y-%m"))

    stem = resolved_path.stem.lower()
    match = _VINTAGE_TOKEN_RE.search(stem)
    if match is None:
        raise ContractViolation(
            "source_schema_drift",
            key=str(path),
            detail="unable to infer vintage month token from filename",
        )

    month_token = match.group("month").lower()
    year_token = int(match.group("year"))
    month = _MONTH_TO_NUMBER.get(month_token)
    if month is None:
        raise ContractViolation(
            "source_schema_drift",
            key=str(path),
            detail=f"invalid month token in vintage filename: {month_token}",
        )

    year = 2000 + year_token
    return f"{year:04d}-{month:02d}"


def _as_clean_text(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and pd.isna(value):
        return ""
    return str(value).strip()


def _parse_forecast_completed_on(raw: pd.DataFrame) -> str | None:
    if raw.empty or raw.shape[0] < 4:
        return None
    candidate = _as_clean_text(raw.iat[3, 0])
    if not candidate:
        return None
    timestamp = pd.to_datetime(candidate, errors="coerce")
    if pd.isna(timestamp):
        return None
    parsed: pd.Timestamp = pd.Timestamp(timestamp)
    return cast(str, parsed.date().isoformat())


def _resolve_month_columns(raw: pd.DataFrame, *, key: str) -> dict[int, pd.Timestamp]:
    if raw.shape[0] < 4 or raw.shape[1] < 6:
        raise ContractViolation(
            "source_schema_drift",
            key=key,
            detail="sheet is too small to contain STEO month headers",
        )

    month_columns: dict[int, pd.Timestamp] = {}
    active_year: int | None = None

    for col in range(2, raw.shape[1]):
        month_token = _as_clean_text(raw.iat[3, col]).lower()[:3]
        month = _MONTH_TO_NUMBER.get(month_token)
        if month is None:
            continue

        year_cell = raw.iat[2, col]
        if not pd.isna(year_cell):
            year_candidate = pd.to_numeric(
                pd.Series([year_cell]), errors="coerce"
            ).iloc[0]
            if not pd.isna(year_candidate):
                active_year = int(year_candidate)

        if active_year is None:
            raise ContractViolation(
                "source_schema_drift",
                key=key,
                detail="unable to infer year for month header columns",
            )

        timestamp = pd.Timestamp(year=active_year, month=month, day=1)
        month_columns[col] = timestamp.to_period("M").to_timestamp("M")

    if len(month_columns) < 24:
        raise ContractViolation(
            "source_schema_drift",
            key=key,
            detail=(
                "insufficient monthly columns detected in STEO sheet; "
                f"expected at least 24, found {len(month_columns)}"
            ),
        )

    return month_columns


def _coerce_numeric(value: object) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(numeric):
        return None
    return float(numeric)


def _is_section_row(
    series_id: str, description: str, values: list[float | None]
) -> bool:
    if series_id:
        return False
    if not description:
        return False
    if any(item is not None for item in values):
        return False
    lowered = description.lower()
    return not lowered.startswith(_NOTE_PREFIXES)


def parse_steo_table(
    *,
    workbook_path: str | Path,
    sheet_name: str,
    table_id: str,
    vintage_month: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse one STEO table sheet into deterministic long format rows."""

    source = Path(workbook_path)
    try:
        raw = pd.read_excel(source, sheet_name=sheet_name, header=None)
    except Exception as exc:  # pragma: no cover - depends on local IO state
        raise ContractViolation(
            "source_schema_drift",
            key=f"{source}:{sheet_name}",
            detail=f"unable to read STEO sheet: {exc}",
        ) from exc

    month_columns = _resolve_month_columns(raw, key=f"{source}:{sheet_name}")
    vintage_end = pd.Timestamp(f"{vintage_month}-01").to_period("M").to_timestamp("M")

    section = ""
    rows: list[dict[str, Any]] = []
    for row_idx in range(4, raw.shape[0]):
        series_id = _as_clean_text(raw.iat[row_idx, 0])
        description = _as_clean_text(raw.iat[row_idx, 1])

        values = [_coerce_numeric(raw.iat[row_idx, col]) for col in month_columns]
        if _is_section_row(series_id, description, values):
            section = description
            continue

        if not series_id:
            continue
        if series_id.lower() in {"table of contents", "forecast date:"}:
            continue

        non_null = 0
        for col, timestamp in month_columns.items():
            numeric = _coerce_numeric(raw.iat[row_idx, col])
            if numeric is None:
                continue
            non_null += 1
            rows.append(
                {
                    "vintage_month": vintage_month,
                    "table_id": table_id,
                    "sheet_name": sheet_name,
                    "series_id": series_id,
                    "description": description,
                    "section": section,
                    "timestamp": timestamp,
                    "value": numeric,
                    "is_forecast": bool(timestamp > vintage_end),
                }
            )

        if (
            non_null == 0
            and description
            and not description.lower().startswith(_NOTE_PREFIXES)
        ):
            section = description if not section else section

    if not rows:
        raise ContractViolation(
            "source_schema_drift",
            key=f"{source}:{sheet_name}",
            detail="no numeric table rows parsed from STEO sheet",
        )

    frame = pd.DataFrame(rows)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
    frame = frame[frame["timestamp"].notna() & frame["value"].notna()].copy()
    frame = frame.sort_values(["series_id", "timestamp"]).drop_duplicates(
        ["series_id", "timestamp"], keep="last"
    )
    frame = frame.reset_index(drop=True)

    metadata = {
        "table_id": table_id,
        "sheet_name": sheet_name,
        "available": True,
        "row_count": int(len(frame)),
        "series_count": int(frame["series_id"].nunique()),
        "month_count": int(frame["timestamp"].nunique()),
        "min_timestamp": frame["timestamp"].min().date().isoformat(),
        "max_timestamp": frame["timestamp"].max().date().isoformat(),
    }
    return frame, metadata


def parse_steo_vintage_workbook(
    workbook_path: str | Path,
    *,
    vintage_month: str | None = None,
    table_sheets: Mapping[str, str] | None = None,
) -> ParsedSteoVintage:
    """Parse required STEO tables from a single workbook."""

    source = Path(workbook_path)
    if not source.exists():
        raise ContractViolation(
            "missing_source_file",
            key=str(source),
            detail="STEO workbook does not exist",
        )

    resolved_vintage = vintage_month or parse_vintage_month_from_filename(source)
    vintage_month_end = (
        pd.Timestamp(f"{resolved_vintage}-01").to_period("M").to_timestamp("M")
    )
    sheets = dict(DEFAULT_TABLE_SHEETS)
    if table_sheets is not None:
        sheets.update({str(key): str(value) for key, value in table_sheets.items()})

    workbook = pd.ExcelFile(source)
    available_sheet_names = set(workbook.sheet_names)

    tables: dict[str, pd.DataFrame] = {}
    table_metadata: dict[str, dict[str, Any]] = {}
    for table_id, sheet_name in sorted(sheets.items()):
        if sheet_name not in available_sheet_names:
            intro_month = _OPTIONAL_TABLE_INTRO_MONTH.get(table_id)
            if intro_month is not None and vintage_month_end < intro_month:
                tables[table_id] = pd.DataFrame(
                    columns=[
                        "vintage_month",
                        "table_id",
                        "sheet_name",
                        "series_id",
                        "description",
                        "section",
                        "timestamp",
                        "value",
                        "is_forecast",
                    ]
                )
                table_metadata[table_id] = {
                    "table_id": table_id,
                    "sheet_name": sheet_name,
                    "available": False,
                    "row_count": 0,
                    "series_count": 0,
                    "month_count": 0,
                    "missing_reason": "table_not_published_for_vintage",
                }
                continue
            raise ContractViolation(
                "source_schema_drift",
                key=f"{source}:{sheet_name}",
                detail="required STEO sheet is missing from workbook",
            )

        frame, metadata = parse_steo_table(
            workbook_path=source,
            sheet_name=sheet_name,
            table_id=table_id,
            vintage_month=resolved_vintage,
        )
        tables[table_id] = frame
        table_metadata[table_id] = metadata

    forecast_completed_on = _parse_forecast_completed_on(
        pd.read_excel(source, sheet_name=sheets["table_5a"], header=None)
    )

    return ParsedSteoVintage(
        vintage_month=resolved_vintage,
        workbook_path=source,
        forecast_completed_on=forecast_completed_on,
        tables=tables,
        table_metadata=table_metadata,
    )
