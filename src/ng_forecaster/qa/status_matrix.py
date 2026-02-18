"""Triad status matrix helpers for handoff completeness checks."""

from __future__ import annotations

from dataclasses import dataclass

from ng_forecaster.qa.contracts import (
    HandoffReadyStatus,
    HandoffReceivedStatus,
    QAAcknowledgmentStatus,
)


@dataclass(frozen=True)
class TriadStatus:
    """Resolved handoff triad state for one handoff id."""

    handoff_id: str
    ready: HandoffReadyStatus | None
    received: HandoffReceivedStatus | None
    qa_ack: QAAcknowledgmentStatus | None
    missing_roles: tuple[str, ...]
    reason_codes: tuple[str, ...]

    @property
    def complete(self) -> bool:
        return len(self.missing_roles) == 0 and len(self.reason_codes) == 0

    def as_dict(self) -> dict[str, object]:
        return {
            "handoff_id": self.handoff_id,
            "complete": self.complete,
            "missing_roles": list(self.missing_roles),
            "reason_codes": list(self.reason_codes),
            "ready": None if self.ready is None else self.ready.as_dict(),
            "received": None if self.received is None else self.received.as_dict(),
            "qa_ack": None if self.qa_ack is None else self.qa_ack.as_dict(),
        }


def build_triad_status(
    *,
    handoff_id: str,
    ready: HandoffReadyStatus | None,
    received: HandoffReceivedStatus | None,
    qa_ack: QAAcknowledgmentStatus | None,
) -> TriadStatus:
    """Build a single triad state object with deterministic missing/reason lists."""

    missing_roles: list[str] = []
    reason_codes: list[str] = []

    if ready is None:
        missing_roles.append("handoff_ready")
    elif ready.reason_code:
        reason_codes.append(ready.reason_code)

    if received is None:
        missing_roles.append("handoff_received")
    elif received.reason_code:
        reason_codes.append(received.reason_code)

    if qa_ack is None:
        missing_roles.append("qa_ack")
    elif qa_ack.reason_code:
        reason_codes.append(qa_ack.reason_code)

    return TriadStatus(
        handoff_id=handoff_id,
        ready=ready,
        received=received,
        qa_ack=qa_ack,
        missing_roles=tuple(sorted(missing_roles)),
        reason_codes=tuple(sorted(code for code in reason_codes if code)),
    )


def build_status_matrix(
    *,
    handoff_ids: list[str],
    ready_statuses: list[HandoffReadyStatus],
    received_statuses: list[HandoffReceivedStatus],
    qa_statuses: list[QAAcknowledgmentStatus],
) -> dict[str, TriadStatus]:
    """Build triad status matrix for multiple handoffs."""

    ready_map = {entry.handoff_id: entry for entry in ready_statuses}
    received_map = {entry.handoff_id: entry for entry in received_statuses}
    qa_map = {entry.handoff_id: entry for entry in qa_statuses}

    matrix: dict[str, TriadStatus] = {}
    for handoff_id in sorted(set(handoff_ids)):
        matrix[handoff_id] = build_triad_status(
            handoff_id=handoff_id,
            ready=ready_map.get(handoff_id),
            received=received_map.get(handoff_id),
            qa_ack=qa_map.get(handoff_id),
        )
    return matrix
