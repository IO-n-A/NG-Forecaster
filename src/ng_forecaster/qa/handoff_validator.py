"""Handoff evidence validator for N1/N3/N2 triad completeness."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal, cast

from ng_forecaster.qa._backlog import (
    BacklogEntry,
    entry_mentions_handoff,
    parse_inline_fields,
    parse_inline_reason_code,
    parse_live_sync_entries,
)
from ng_forecaster.qa.contracts import (
    EvidencePayload,
    HandoffReadyStatus,
    HandoffReceivedStatus,
    QAAcknowledgmentStatus,
    normalize_reason_code,
)
from ng_forecaster.qa.status_matrix import build_triad_status

QAAckState = Literal["in_progress", "done", "blocked", "failed"]


@dataclass(frozen=True)
class HandoffValidationResult:
    """Outcome for one handoff triad/evidence validation."""

    handoff_id: str
    complete: bool
    missing_roles: tuple[str, ...]
    reason_codes: tuple[str, ...]
    payload: EvidencePayload | None
    row_numbers: tuple[int, ...]

    def as_dict(self) -> dict[str, object]:
        return {
            "handoff_id": self.handoff_id,
            "complete": self.complete,
            "missing_roles": list(self.missing_roles),
            "reason_codes": list(self.reason_codes),
            "payload": None if self.payload is None else self.payload.as_dict(),
            "row_numbers": list(self.row_numbers),
        }

    def to_text(self) -> str:
        if self.complete:
            return f"{self.handoff_id}: PASS (triad complete; rows={list(self.row_numbers)})"
        missing = ", ".join(self.missing_roles) or "none"
        reasons = ", ".join(self.reason_codes) or "none"
        return (
            f"{self.handoff_id}: FAIL (missing_roles={missing}; "
            f"reason_codes={reasons}; rows={list(self.row_numbers)})"
        )


def _latest(entries: list[BacklogEntry]) -> BacklogEntry | None:
    if not entries:
        return None
    return sorted(entries, key=lambda entry: (entry.timestamp, entry.row_number))[-1]


def _entry_text(entry: BacklogEntry) -> str:
    return "; ".join([entry.blocker, entry.next_action, entry.dependency_handoff])


def _build_payload(entry: BacklogEntry) -> tuple[EvidencePayload, tuple[str, ...]]:
    fields = parse_inline_fields(_entry_text(entry))

    changed_files = tuple(sorted(set(fields["changed_files"])))
    tests_run = tuple(sorted(set(fields["tests_run"])))
    caveats = tuple(sorted(set(fields["caveats"])))

    reason_codes: list[str] = []
    if not changed_files:
        reason_codes.append("missing_changed_files")
    if not tests_run:
        reason_codes.append("missing_tests_run")
    if not caveats:
        reason_codes.append("missing_caveats")

    return (
        EvidencePayload(
            changed_files=changed_files,
            tests_run=tests_run,
            caveats=caveats,
        ),
        tuple(sorted(reason_codes)),
    )


def _resolve_ready_status(
    entry: BacklogEntry, handoff_id: str
) -> tuple[HandoffReadyStatus, tuple[str, ...]]:
    payload, payload_reasons = _build_payload(entry)
    reason_code = parse_inline_reason_code(_entry_text(entry))
    normalized_reason = normalize_reason_code(entry.status, reason_code)

    reason_codes = list(payload_reasons)
    if normalized_reason:
        reason_codes.append(normalized_reason)

    status = HandoffReadyStatus(
        handoff_id=handoff_id,
        engineer=entry.engineer,
        task_id=entry.task_id,
        timestamp=entry.timestamp,
        payload=payload,
        reason_code=normalized_reason,
    )
    return status, tuple(sorted(set(reason_codes)))


def _resolve_received_status(
    entry: BacklogEntry, handoff_id: str
) -> tuple[HandoffReceivedStatus, tuple[str, ...]]:
    reason_code = parse_inline_reason_code(_entry_text(entry))
    normalized_reason = normalize_reason_code(entry.status, reason_code)
    status = HandoffReceivedStatus(
        handoff_id=handoff_id,
        engineer=entry.engineer,
        task_id=entry.task_id,
        timestamp=entry.timestamp,
        validation_note=entry.next_action,
        reason_code=normalized_reason,
    )
    return status, tuple(code for code in [normalized_reason] if code)


def _resolve_qa_status(
    entry: BacklogEntry, handoff_id: str
) -> tuple[QAAcknowledgmentStatus, tuple[str, ...]]:
    reason_code = parse_inline_reason_code(_entry_text(entry))
    normalized_reason = normalize_reason_code(entry.status, reason_code)
    qa_state_raw = entry.status.strip().lower()
    if qa_state_raw not in {"in_progress", "done", "blocked", "failed"}:
        qa_state: QAAckState = "in_progress"
        normalized_reason = "invalid_qa_state"
    else:
        qa_state = cast(QAAckState, qa_state_raw)

    status = QAAcknowledgmentStatus(
        handoff_id=handoff_id,
        engineer=entry.engineer,
        task_id=entry.task_id,
        timestamp=entry.timestamp,
        qa_state=qa_state,
        note=entry.next_action,
        reason_code=normalized_reason,
    )
    return status, tuple(code for code in [normalized_reason] if code)


def _rows_for_handoff(
    entries: list[BacklogEntry], handoff_id: str
) -> list[BacklogEntry]:
    return [entry for entry in entries if entry_mentions_handoff(entry, handoff_id)]


def validate_handoff_evidence(
    backlog_path: str | Path,
    *,
    handoff_id: str = "handoff-n1",
) -> HandoffValidationResult:
    """Validate triad completeness and handoff evidence payload fields."""

    entries = parse_live_sync_entries(backlog_path)
    scoped = _rows_for_handoff(entries, handoff_id)

    ready_entry = _latest(
        [entry for entry in scoped if entry.status.strip().lower() == "handoff_ready"]
    )
    received_entry = _latest(
        [
            entry
            for entry in scoped
            if entry.status.strip().lower() == "handoff_received"
        ]
    )
    qa_entry = _latest(
        [
            entry
            for entry in scoped
            if entry.engineer == "QA_Lead"
            and entry.status.strip().lower()
            in {"in_progress", "done", "blocked", "failed"}
        ]
    )

    reason_codes: list[str] = []
    ready_status: HandoffReadyStatus | None = None
    received_status: HandoffReceivedStatus | None = None
    qa_status: QAAcknowledgmentStatus | None = None
    payload: EvidencePayload | None = None

    if ready_entry is not None:
        ready_status, ready_reasons = _resolve_ready_status(ready_entry, handoff_id)
        payload = ready_status.payload
        reason_codes.extend(ready_reasons)

    if received_entry is not None:
        received_status, received_reasons = _resolve_received_status(
            received_entry, handoff_id
        )
        reason_codes.extend(received_reasons)

    if qa_entry is not None:
        qa_status, qa_reasons = _resolve_qa_status(qa_entry, handoff_id)
        reason_codes.extend(qa_reasons)

    triad = build_triad_status(
        handoff_id=handoff_id,
        ready=ready_status,
        received=received_status,
        qa_ack=qa_status,
    )

    reason_codes.extend(triad.reason_codes)
    if triad.missing_roles:
        reason_codes.append("triad_incomplete")

    row_numbers = [
        entry.row_number
        for entry in [ready_entry, received_entry, qa_entry]
        if entry is not None
    ]
    dedup_reasons = tuple(sorted(set(code for code in reason_codes if code)))

    return HandoffValidationResult(
        handoff_id=handoff_id,
        complete=(len(triad.missing_roles) == 0 and len(dedup_reasons) == 0),
        missing_roles=triad.missing_roles,
        reason_codes=dedup_reasons,
        payload=payload,
        row_numbers=tuple(sorted(row_numbers)),
    )
