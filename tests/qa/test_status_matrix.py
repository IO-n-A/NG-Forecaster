from __future__ import annotations

from datetime import datetime

from ng_forecaster.qa.contracts import (
    EvidencePayload,
    HandoffReadyStatus,
    HandoffReceivedStatus,
    QAAcknowledgmentStatus,
)
from ng_forecaster.qa.status_matrix import build_status_matrix, build_triad_status

_TS = datetime(2026, 2, 14, 12, 0)


def _ready(handoff_id: str) -> HandoffReadyStatus:
    return HandoffReadyStatus(
        handoff_id=handoff_id,
        engineer="Parallel_Lead",
        task_id="S1A",
        timestamp=_TS,
        payload=EvidencePayload(
            changed_files=("a.py",),
            tests_run=("pytest tests/qa -q",),
            caveats=("none",),
        ),
    )


def _received(handoff_id: str) -> HandoffReceivedStatus:
    return HandoffReceivedStatus(
        handoff_id=handoff_id,
        engineer="Core_Lead",
        task_id="S1B",
        timestamp=_TS,
        validation_note="ok",
    )


def _qa(handoff_id: str) -> QAAcknowledgmentStatus:
    return QAAcknowledgmentStatus(
        handoff_id=handoff_id,
        engineer="QA_Lead",
        task_id="S1C",
        timestamp=_TS,
        qa_state="done",
        note="ack",
    )


def test_triad_status_complete_when_all_roles_present() -> None:
    triad = build_triad_status(
        handoff_id="handoff-n1",
        ready=_ready("handoff-n1"),
        received=_received("handoff-n1"),
        qa_ack=_qa("handoff-n1"),
    )

    assert triad.complete
    assert triad.missing_roles == ()
    assert triad.reason_codes == ()


def test_triad_status_reports_missing_roles_and_reason_codes() -> None:
    broken_ready = HandoffReadyStatus(
        handoff_id="handoff-n3",
        engineer="Parallel_Lead",
        task_id="S1A",
        timestamp=_TS,
        payload=EvidencePayload(
            changed_files=("a.py",),
            tests_run=("pytest tests/qa -q",),
            caveats=("none",),
        ),
        reason_code="invalid_payload",
    )
    triad = build_triad_status(
        handoff_id="handoff-n3",
        ready=broken_ready,
        received=None,
        qa_ack=None,
    )

    assert not triad.complete
    assert triad.missing_roles == ("handoff_received", "qa_ack")
    assert triad.reason_codes == ("invalid_payload",)


def test_status_matrix_is_keyed_by_handoff_id() -> None:
    matrix = build_status_matrix(
        handoff_ids=["handoff-n3", "handoff-n1"],
        ready_statuses=[_ready("handoff-n1")],
        received_statuses=[_received("handoff-n1")],
        qa_statuses=[_qa("handoff-n1")],
    )

    assert sorted(matrix.keys()) == ["handoff-n1", "handoff-n3"]
    assert matrix["handoff-n1"].complete
    assert not matrix["handoff-n3"].complete
