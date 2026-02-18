"""Typed QA contracts used by handoff and release gate validators."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

HandoffState = Literal[
    "handoff_ready", "handoff_received", "in_progress", "done", "blocked", "failed"
]

BLOCKED_REASON_CODES = frozenset(
    {
        "missing_owner",
        "missing_next_action",
        "missing_timestamp",
        "unresolved_dependency",
    }
)
FAILED_REASON_CODES = frozenset(
    {
        "triad_incomplete",
        "invalid_payload",
        "cadence_gap_exceeded",
        "schema_validation_failed",
        "determinism_mismatch",
        "unresolved_blocked_item",
        "missing_test_evidence",
    }
)


@dataclass(frozen=True)
class EvidencePayload:
    """Structured handoff evidence payload."""

    changed_files: tuple[str, ...]
    tests_run: tuple[str, ...]
    caveats: tuple[str, ...]

    def is_complete(self) -> bool:
        return bool(self.changed_files and self.tests_run and self.caveats)

    def as_dict(self) -> dict[str, list[str]]:
        return {
            "changed_files": list(self.changed_files),
            "tests_run": list(self.tests_run),
            "caveats": list(self.caveats),
        }


@dataclass(frozen=True)
class HandoffReadyStatus:
    """Producer handoff status entry."""

    handoff_id: str
    engineer: str
    task_id: str
    timestamp: datetime
    payload: EvidencePayload
    reason_code: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "handoff_id": self.handoff_id,
            "engineer": self.engineer,
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(timespec="minutes"),
            "status": "handoff_ready",
            "payload": self.payload.as_dict(),
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class HandoffReceivedStatus:
    """Consumer handoff receipt status entry."""

    handoff_id: str
    engineer: str
    task_id: str
    timestamp: datetime
    validation_note: str
    reason_code: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "handoff_id": self.handoff_id,
            "engineer": self.engineer,
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(timespec="minutes"),
            "status": "handoff_received",
            "validation_note": self.validation_note,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class QAAcknowledgmentStatus:
    """QA_Lead QA acknowledgment entry."""

    handoff_id: str
    engineer: str
    task_id: str
    timestamp: datetime
    qa_state: Literal["in_progress", "done", "blocked", "failed"]
    note: str
    reason_code: str | None = None

    def as_dict(self) -> dict[str, object]:
        return {
            "handoff_id": self.handoff_id,
            "engineer": self.engineer,
            "task_id": self.task_id,
            "timestamp": self.timestamp.isoformat(timespec="minutes"),
            "status": self.qa_state,
            "note": self.note,
            "reason_code": self.reason_code,
        }


@dataclass(frozen=True)
class GateStatus:
    """Machine-readable gate outcome."""

    gate_id: str
    outcome: Literal["passed", "failed", "blocked"]
    reason_code: str | None = None
    detail: str = ""

    def as_dict(self) -> dict[str, object]:
        return {
            "gate_id": self.gate_id,
            "outcome": self.outcome,
            "reason_code": self.reason_code,
            "detail": self.detail,
        }


def normalize_reason_code(status: str, reason_code: str | None) -> str | None:
    """Normalize and validate reason codes for blocked/failed states."""

    normalized_status = status.strip().lower()
    normalized_reason = (reason_code or "").strip().lower() or None

    if normalized_status == "blocked":
        if normalized_reason is None:
            return "missing_reason_code"
        if normalized_reason not in BLOCKED_REASON_CODES:
            return f"invalid_blocked_reason:{normalized_reason}"
        return normalized_reason

    if normalized_status == "failed":
        if normalized_reason is None:
            return "missing_reason_code"
        if normalized_reason not in FAILED_REASON_CODES:
            return f"invalid_failed_reason:{normalized_reason}"
        return normalized_reason

    return normalized_reason
