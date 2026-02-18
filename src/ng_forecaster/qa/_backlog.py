"""Backlog markdown table parser shared by QA validation utilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
import re

_TABLE_HEADER = "Timestamp (YYYY-MM-DD HH:MM TZ)"


@dataclass(frozen=True)
class BacklogEntry:
    """Parsed live sync row from docs/coding/backlog.md."""

    row_number: int
    timestamp: datetime
    timestamp_label: str
    engineer: str
    track: str
    task_id: str
    status: str
    blocker: str
    next_action: str
    dependency_handoff: str


def _split_markdown_row(line: str) -> list[str]:
    stripped = line.strip()
    if not stripped.startswith("|") or not stripped.endswith("|"):
        return []
    return [cell.strip() for cell in stripped.strip("|").split("|")]


def _parse_timestamp(raw: str) -> tuple[datetime, str]:
    value = raw.strip()
    match = re.match(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2})\s+(.+)$", value)
    if match is None:
        raise ValueError(f"invalid backlog timestamp: {raw}")
    dt = datetime.strptime(match.group(1), "%Y-%m-%d %H:%M")
    return dt, match.group(2).strip()


def parse_live_sync_entries(path: str | Path) -> list[BacklogEntry]:
    """Parse Live Sync Log rows into structured entries."""

    backlog_path = Path(path)
    lines = backlog_path.read_text(encoding="utf-8").splitlines()

    header_idx = -1
    for idx, line in enumerate(lines):
        cells = _split_markdown_row(line)
        if _TABLE_HEADER in cells and "Engineer" in cells and "Task ID" in cells:
            header_idx = idx
            break
    if header_idx == -1:
        raise ValueError(f"could not find live sync log table in {backlog_path}")

    header_cells = _split_markdown_row(lines[header_idx])
    if len(lines) <= header_idx + 2:
        return []

    entries: list[BacklogEntry] = []
    for row_idx in range(header_idx + 2, len(lines)):
        cells = _split_markdown_row(lines[row_idx])
        if not cells:
            if entries:
                break
            continue
        if all(set(cell) <= {"-", ":"} for cell in cells):
            continue
        if len(cells) != len(header_cells):
            break

        record = dict(zip(header_cells, cells))
        raw_ts = record[_TABLE_HEADER]
        parsed_ts, tz_label = _parse_timestamp(raw_ts)
        entries.append(
            BacklogEntry(
                row_number=row_idx + 1,
                timestamp=parsed_ts,
                timestamp_label=tz_label,
                engineer=record["Engineer"].strip(),
                track=record["Track"].strip(),
                task_id=record["Task ID"].strip(),
                status=record["Status"].strip(),
                blocker=record["Blocker"].strip(),
                next_action=record["Next Action"].strip(),
                dependency_handoff=record["Dependency/Handoff"].strip(),
            )
        )

    return sorted(entries, key=lambda entry: (entry.timestamp, entry.row_number))


def entry_mentions_handoff(entry: BacklogEntry, handoff_id: str) -> bool:
    """Return True when entry text references a given handoff id."""

    key = handoff_id.strip().lower()
    haystack = " ".join(
        [entry.task_id, entry.blocker, entry.next_action, entry.dependency_handoff]
    ).lower()
    return key in haystack


def parse_inline_fields(text: str) -> dict[str, list[str]]:
    """Extract semicolon-delimited key/value list fields from free text."""

    pattern = re.compile(
        r"(changed_files|tests(?:_run)?|caveats)\s*[:=]\s*([^;]+)", re.IGNORECASE
    )
    parsed: dict[str, list[str]] = {
        "changed_files": [],
        "tests_run": [],
        "caveats": [],
    }
    for key, raw_values in pattern.findall(text):
        canonical_key = "tests_run" if key.lower().startswith("tests") else key.lower()
        values = [item.strip() for item in raw_values.split(",") if item.strip()]
        parsed[canonical_key] = values
    return parsed


def parse_inline_reason_code(text: str) -> str | None:
    """Extract reason_code token from a semicolon-delimited field."""

    match = re.search(r"reason_code\s*[:=]\s*([a-zA-Z0-9_:-]+)", text)
    if match is None:
        return None
    return match.group(1).strip().lower()


def has_owner_tag(text: str) -> bool:
    """Check whether blocker text includes an explicit owner tag."""

    return bool(re.search(r"\bowner\s*[:=]\s*[A-Za-z0-9_-]+", text))


def has_timestamp_tag(text: str) -> bool:
    """Check whether blocker text includes an explicit timestamp tag."""

    if re.search(r"\btimestamp\s*[:=]\s*\d{4}-\d{2}-\d{2}", text):
        return True
    return bool(re.search(r"\bts\s*[:=]\s*\d{4}-\d{2}-\d{2}", text))
