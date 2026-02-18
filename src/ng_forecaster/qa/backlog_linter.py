"""Lint checks for backlog cadence and blocked-entry completeness."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta
import json
from pathlib import Path

from ng_forecaster.qa._backlog import (
    BacklogEntry,
    has_owner_tag,
    has_timestamp_tag,
    parse_live_sync_entries,
)


@dataclass(frozen=True)
class LintViolation:
    """Single backlog lint violation."""

    kind: str
    reason_code: str
    row_number: int
    engineer: str
    task_id: str
    detail: str

    def as_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "reason_code": self.reason_code,
            "row_number": self.row_number,
            "engineer": self.engineer,
            "task_id": self.task_id,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class BacklogLintReport:
    """Machine-readable lint output."""

    checked_rows: int
    cadence_violations: tuple[LintViolation, ...]
    blocker_violations: tuple[LintViolation, ...]

    @property
    def passed(self) -> bool:
        return not self.cadence_violations and not self.blocker_violations

    def as_dict(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "checked_rows": self.checked_rows,
            "cadence_violations": [item.as_dict() for item in self.cadence_violations],
            "blocker_violations": [item.as_dict() for item in self.blocker_violations],
            "violation_count": len(self.cadence_violations)
            + len(self.blocker_violations),
        }

    def to_json(self) -> str:
        return json.dumps(self.as_dict(), sort_keys=True, indent=2)

    def to_text(self) -> str:
        if self.passed:
            return f"PASS: checked {self.checked_rows} backlog rows with no cadence/blocker violations"

        lines = [
            (
                f"FAIL: {len(self.cadence_violations)} cadence violation(s), "
                f"{len(self.blocker_violations)} blocker violation(s)"
            )
        ]
        for violation in list(self.cadence_violations) + list(self.blocker_violations):
            lines.append(
                (
                    f"- row={violation.row_number} task={violation.task_id} "
                    f"reason_code={violation.reason_code} detail={violation.detail}"
                )
            )
        return "\n".join(lines)


def _lint_cadence(
    entries: list[BacklogEntry], max_gap_minutes: int
) -> list[LintViolation]:
    violations: list[LintViolation] = []
    by_engineer: dict[str, list[BacklogEntry]] = {}
    for entry in entries:
        by_engineer.setdefault(entry.engineer, []).append(entry)

    max_gap = timedelta(minutes=max_gap_minutes)
    for engineer, engineer_entries in by_engineer.items():
        ordered = sorted(
            engineer_entries, key=lambda entry: (entry.timestamp, entry.row_number)
        )
        for previous, current in zip(ordered, ordered[1:]):
            gap = current.timestamp - previous.timestamp
            if gap > max_gap:
                violations.append(
                    LintViolation(
                        kind="cadence",
                        reason_code="cadence_gap_exceeded",
                        row_number=current.row_number,
                        engineer=engineer,
                        task_id=current.task_id,
                        detail=f"gap_minutes={int(gap.total_seconds() // 60)} > {max_gap_minutes}",
                    )
                )
    return sorted(violations, key=lambda item: item.row_number)


def _is_empty(value: str) -> bool:
    normalized = value.strip().lower()
    return normalized in {"", "none", "<pending>", "n/a"}


def _lint_blockers(entries: list[BacklogEntry]) -> list[LintViolation]:
    violations: list[LintViolation] = []
    latest_by_task: dict[str, BacklogEntry] = {}
    for entry in sorted(entries, key=lambda item: (item.timestamp, item.row_number)):
        latest_by_task[entry.task_id] = entry

    for entry in latest_by_task.values():
        if entry.status.strip().lower() != "blocked":
            continue

        blocker = entry.blocker
        if _is_empty(blocker):
            violations.append(
                LintViolation(
                    kind="blocker",
                    reason_code="missing_blocker_text",
                    row_number=entry.row_number,
                    engineer=entry.engineer,
                    task_id=entry.task_id,
                    detail="blocked entries must include blocker detail",
                )
            )
        else:
            if not has_owner_tag(blocker):
                violations.append(
                    LintViolation(
                        kind="blocker",
                        reason_code="missing_owner",
                        row_number=entry.row_number,
                        engineer=entry.engineer,
                        task_id=entry.task_id,
                        detail="blocker must include owner=<name>",
                    )
                )
            if not has_timestamp_tag(blocker):
                violations.append(
                    LintViolation(
                        kind="blocker",
                        reason_code="missing_timestamp",
                        row_number=entry.row_number,
                        engineer=entry.engineer,
                        task_id=entry.task_id,
                        detail="blocker must include timestamp=<YYYY-MM-DD>",
                    )
                )

        if _is_empty(entry.next_action):
            violations.append(
                LintViolation(
                    kind="blocker",
                    reason_code="missing_next_action",
                    row_number=entry.row_number,
                    engineer=entry.engineer,
                    task_id=entry.task_id,
                    detail="blocked entries must include next action",
                )
            )

    return sorted(violations, key=lambda item: item.row_number)


def lint_backlog(
    backlog_path: str | Path,
    *,
    max_gap_minutes: int = 90,
) -> BacklogLintReport:
    """Run cadence and blocker lint checks against backlog sync entries."""

    entries = parse_live_sync_entries(backlog_path)
    cadence_violations = tuple(_lint_cadence(entries, max_gap_minutes=max_gap_minutes))
    blocker_violations = tuple(_lint_blockers(entries))
    return BacklogLintReport(
        checked_rows=len(entries),
        cadence_violations=cadence_violations,
        blocker_violations=blocker_violations,
    )
