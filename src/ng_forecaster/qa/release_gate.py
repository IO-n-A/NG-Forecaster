"""Release gate builder and policy checks for Sprint 1 signoff."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import re

from ng_forecaster.qa._backlog import (
    BacklogEntry,
    has_owner_tag,
    has_timestamp_tag,
    parse_live_sync_entries,
)
from ng_forecaster.qa.handoff_validator import (
    HandoffValidationResult,
    validate_handoff_evidence,
)

_REQUIRED_HANDOFFS = ("handoff-n1", "handoff-n3", "handoff-n2")
_REQUIRED_TEST_GROUPS = (
    "pytest tests/qa -q",
    "pytest tests/integration/test_n3_gate_contract.py -q",
    "pytest tests/integration/test_cross_track_contracts.py -q",
    "pytest tests/integration/test_n1_n3_n2_handoff_matrix.py -q",
)


@dataclass(frozen=True)
class BlockedIssue:
    """Unresolved blocked task summary."""

    task_id: str
    row_number: int
    reason_codes: tuple[str, ...]
    blocker: str
    next_action: str

    def as_dict(self) -> dict[str, object]:
        return {
            "task_id": self.task_id,
            "row_number": self.row_number,
            "reason_codes": list(self.reason_codes),
            "blocker": self.blocker,
            "next_action": self.next_action,
        }


@dataclass(frozen=True)
class ReleaseGateResult:
    """Release gate outcome and generated packet payload."""

    passed: bool
    handoffs: dict[str, HandoffValidationResult]
    missing_test_groups: tuple[str, ...]
    unresolved_blockers: tuple[BlockedIssue, ...]
    packet: dict[str, object]

    def to_text(self) -> str:
        if self.passed:
            return "PASS: release gate checks passed and packet generated"
        return (
            "FAIL: release gate blocked "
            f"(missing_test_groups={len(self.missing_test_groups)}, "
            f"unresolved_blockers={len(self.unresolved_blockers)})"
        )


def _extract_logged_commands(entries: list[BacklogEntry]) -> set[str]:
    commands: set[str] = set()
    command_pattern = re.compile(
        r"(pytest\s+[^`;,]+|ruff\s+check\s+[^`;,]+|black\s+--check\s+[^`;,]+|python\s+[^`;,]+)"
    )

    for entry in entries:
        text = " ; ".join([entry.next_action, entry.dependency_handoff])
        for token in re.findall(r"`([^`]+)`", text):
            commands.add(token.strip())
        for match in command_pattern.findall(text):
            commands.add(match.strip())
    return commands


def _latest_entries_by_task(entries: list[BacklogEntry]) -> dict[str, BacklogEntry]:
    latest: dict[str, BacklogEntry] = {}
    for entry in sorted(entries, key=lambda item: (item.timestamp, item.row_number)):
        latest[entry.task_id] = entry
    return latest


def _resolve_unresolved_blockers(
    entries: list[BacklogEntry],
) -> tuple[BlockedIssue, ...]:
    unresolved: list[BlockedIssue] = []
    latest_by_task = _latest_entries_by_task(entries)

    for task_id, entry in sorted(latest_by_task.items()):
        if entry.status.strip().lower() != "blocked":
            continue

        reason_codes: list[str] = []
        if not has_owner_tag(entry.blocker):
            reason_codes.append("missing_owner")
        if not has_timestamp_tag(entry.blocker):
            reason_codes.append("missing_timestamp")
        if entry.next_action.strip().lower() in {"", "none", "n/a"}:
            reason_codes.append("missing_next_action")

        if reason_codes:
            unresolved.append(
                BlockedIssue(
                    task_id=task_id,
                    row_number=entry.row_number,
                    reason_codes=tuple(sorted(reason_codes)),
                    blocker=entry.blocker,
                    next_action=entry.next_action,
                )
            )

    return tuple(sorted(unresolved, key=lambda item: item.row_number))


def build_release_packet(
    *,
    backlog_path: str | Path = "docs/coding/backlog.md",
    output_path: str | Path = "data/reports/sprint1_release_packet.json",
    required_handoffs: tuple[str, ...] = _REQUIRED_HANDOFFS,
    required_test_groups: tuple[str, ...] = _REQUIRED_TEST_GROUPS,
) -> ReleaseGateResult:
    """Build release packet and enforce triad/test/blocker policy checks."""

    entries = parse_live_sync_entries(backlog_path)

    handoff_results = {
        handoff_id: validate_handoff_evidence(backlog_path, handoff_id=handoff_id)
        for handoff_id in required_handoffs
    }

    logged_commands = _extract_logged_commands(entries)
    missing_test_groups = tuple(
        sorted(
            command
            for command in required_test_groups
            if command not in logged_commands
        )
    )

    unresolved_blockers = _resolve_unresolved_blockers(entries)

    packet: dict[str, object] = {
        "handoffs": {
            handoff_id: result.as_dict()
            for handoff_id, result in sorted(
                handoff_results.items(), key=lambda item: item[0]
            )
        },
        "required_test_groups": list(required_test_groups),
        "logged_commands": sorted(logged_commands),
        "missing_test_groups": list(missing_test_groups),
        "unresolved_blockers": [item.as_dict() for item in unresolved_blockers],
        "release_status": "passed",
    }

    triad_failures = [
        handoff_id
        for handoff_id, result in handoff_results.items()
        if not result.complete
    ]

    if triad_failures:
        packet["release_status"] = "failed"
        packet["release_reason_code"] = "triad_incomplete"
        packet["triad_failures"] = sorted(triad_failures)

    if missing_test_groups:
        packet["release_status"] = "failed"
        packet["release_reason_code"] = "missing_test_evidence"

    if unresolved_blockers:
        packet["release_status"] = "failed"
        packet["release_reason_code"] = "unresolved_blocked_item"

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(packet, sort_keys=True, indent=2), encoding="utf-8")

    passed = (
        packet["release_status"] == "passed"
        and not missing_test_groups
        and not unresolved_blockers
        and all(result.complete for result in handoff_results.values())
    )

    return ReleaseGateResult(
        passed=passed,
        handoffs=handoff_results,
        missing_test_groups=missing_test_groups,
        unresolved_blockers=unresolved_blockers,
        packet=packet,
    )
