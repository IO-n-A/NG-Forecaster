from __future__ import annotations

from datetime import datetime

from ng_forecaster.qa.contracts import (
    EvidencePayload,
    HandoffReadyStatus,
    HandoffReceivedStatus,
    QAAcknowledgmentStatus,
    normalize_reason_code,
)


def test_evidence_payload_completeness_and_serialization() -> None:
    payload = EvidencePayload(
        changed_files=("a.py", "b.py"),
        tests_run=("pytest tests/qa -q",),
        caveats=("none",),
    )

    assert payload.is_complete()
    assert payload.as_dict() == {
        "changed_files": ["a.py", "b.py"],
        "tests_run": ["pytest tests/qa -q"],
        "caveats": ["none"],
    }


def test_status_objects_are_deterministic() -> None:
    ts = datetime(2026, 2, 14, 12, 0)
    payload = EvidencePayload(
        changed_files=("x.py",),
        tests_run=("pytest tests/x -q",),
        caveats=("none",),
    )
    ready = HandoffReadyStatus(
        handoff_id="handoff-n1",
        engineer="Parallel_Lead",
        task_id="S1A-A3",
        timestamp=ts,
        payload=payload,
    )
    received = HandoffReceivedStatus(
        handoff_id="handoff-n1",
        engineer="Core_Lead",
        task_id="S1B-A4",
        timestamp=ts,
        validation_note="validated",
    )
    qa = QAAcknowledgmentStatus(
        handoff_id="handoff-n1",
        engineer="QA_Lead",
        task_id="S1C-N1-01",
        timestamp=ts,
        qa_state="done",
        note="qa ack",
    )

    assert ready.as_dict()["status"] == "handoff_ready"
    assert received.as_dict()["status"] == "handoff_received"
    assert qa.as_dict()["status"] == "done"


def test_normalize_reason_code_for_blocked_and_failed_states() -> None:
    assert normalize_reason_code("blocked", "missing_owner") == "missing_owner"
    assert normalize_reason_code("failed", "triad_incomplete") == "triad_incomplete"
    assert normalize_reason_code("blocked", None) == "missing_reason_code"
    assert normalize_reason_code("done", None) is None
