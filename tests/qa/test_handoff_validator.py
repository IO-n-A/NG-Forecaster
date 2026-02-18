from __future__ import annotations

from pathlib import Path

from ng_forecaster.qa.handoff_validator import validate_handoff_evidence

_HEADER = """# Backlog

## 2) Live Sync Log

| Timestamp (YYYY-MM-DD HH:MM TZ) | Engineer | Track | Task ID | Status | Blocker | Next Action | Dependency/Handoff |
|---|---|---|---|---|---|---|---|
"""


def _write_backlog(path: Path, body_rows: list[str]) -> Path:
    text = _HEADER + "\n".join(body_rows) + "\n"
    path.write_text(text, encoding="utf-8")
    return path


def test_handoff_validator_passes_with_complete_triad_and_payload(
    tmp_path: Path,
) -> None:
    backlog_path = _write_backlog(
        tmp_path / "backlog.md",
        [
            "| 2026-02-14 10:00 UTC | Parallel_Lead | parallel | S1A-A3 | handoff_ready | none | changed_files: tests/fixtures/leakage/asof_valid.csv, tests/helpers/replay_harness.py; tests_run: pytest tests/evaluation/test_replay_harness.py -q; caveats: none | handoff-n1 |",
            "| 2026-02-14 10:10 UTC | Core_Lead | core | S1B-A4 | handoff_received | none | received and validated handoff payload | handoff-n1 |",
            "| 2026-02-14 10:20 UTC | QA_Lead | integration | S1C-N1-01 | done | none | qa acknowledgement complete | handoff-n1 |",
        ],
    )

    result = validate_handoff_evidence(backlog_path, handoff_id="handoff-n1")
    assert result.complete
    assert result.reason_codes == ()
    assert result.missing_roles == ()


def test_handoff_validator_fails_when_payload_and_qa_are_missing(
    tmp_path: Path,
) -> None:
    backlog_path = _write_backlog(
        tmp_path / "backlog.md",
        [
            "| 2026-02-14 10:00 UTC | Parallel_Lead | parallel | S1A-A3 | handoff_ready | none | changed_files: tests/fixtures/leakage/asof_valid.csv; tests_run: pytest tests/evaluation/test_replay_harness.py -q | handoff-n1 |",
            "| 2026-02-14 10:10 UTC | Core_Lead | core | S1B-A4 | handoff_received | none | received and validated handoff payload | handoff-n1 |",
        ],
    )

    result = validate_handoff_evidence(backlog_path, handoff_id="handoff-n1")
    assert not result.complete
    assert "triad_incomplete" in result.reason_codes
    assert "missing_caveats" in result.reason_codes
    assert result.missing_roles == ("qa_ack",)
