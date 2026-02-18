"""Integration coverage for N1/N3/N2 handoff triad matrix and release packet readiness."""

from __future__ import annotations

from pathlib import Path

from ng_forecaster.qa.handoff_validator import validate_handoff_evidence
from ng_forecaster.qa.release_gate import build_release_packet

_HEADER = """# Backlog

## 2) Live Sync Log

| Timestamp (YYYY-MM-DD HH:MM TZ) | Engineer | Track | Task ID | Status | Blocker | Next Action | Dependency/Handoff |
|---|---|---|---|---|---|---|---|
"""


REQUIRED_TESTS = (
    "pytest tests/qa -q",
    "pytest tests/integration/test_n3_gate_contract.py -q",
    "pytest tests/integration/test_cross_track_contracts.py -q",
    "pytest tests/integration/test_n1_n3_n2_handoff_matrix.py -q",
)


def _row(
    ts: str,
    engineer: str,
    track: str,
    task: str,
    status: str,
    blocker: str,
    next_action: str,
    dep: str,
) -> str:
    return f"| {ts} | {engineer} | {track} | {task} | {status} | {blocker} | {next_action} | {dep} |"


def _write_backlog(path: Path, rows: list[str]) -> Path:
    path.write_text(_HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _complete_handoff_rows(
    handoff_id: str, ts_prefix: str, include_test_log: bool = False
) -> list[str]:
    tests_segment = "tests_run: pytest tests/integration -q; "
    if include_test_log:
        tests_segment = "tests_run: " + ", ".join(REQUIRED_TESTS) + "; "

    return [
        _row(
            f"2026-02-14 {ts_prefix} UTC",
            "Parallel_Lead",
            "parallel",
            "S1A",
            "handoff_ready",
            "none",
            (
                "changed_files: tests/a.py, tests/b.py; "
                f"{tests_segment}"
                "caveats: none"
            ),
            handoff_id,
        ),
        _row(
            "2026-02-14 10:10 UTC",
            "Core_Lead",
            "core",
            "S1B",
            "handoff_received",
            "none",
            "validated consumer contract",
            handoff_id,
        ),
        _row(
            "2026-02-14 10:20 UTC",
            "QA_Lead",
            "integration",
            "S1C",
            "done",
            "none",
            "qa acknowledgment complete",
            handoff_id,
        ),
    ]


def test_handoff_matrix_and_release_gate_pass_when_triads_complete(
    tmp_path: Path,
) -> None:
    rows = _complete_handoff_rows("handoff-n1", "10:00", include_test_log=True)
    rows += _complete_handoff_rows("handoff-n3", "11:00")
    rows += _complete_handoff_rows("handoff-n2", "12:00")

    backlog = _write_backlog(tmp_path / "backlog.md", rows)

    for handoff_id in ("handoff-n1", "handoff-n3", "handoff-n2"):
        result = validate_handoff_evidence(backlog, handoff_id=handoff_id)
        assert result.complete

    release = build_release_packet(
        backlog_path=backlog,
        output_path=tmp_path / "release_packet.json",
    )
    assert release.passed


def test_handoff_matrix_fails_when_consumer_entry_is_missing(tmp_path: Path) -> None:
    rows = _complete_handoff_rows("handoff-n1", "10:00", include_test_log=True)
    rows += [
        row
        for row in _complete_handoff_rows("handoff-n3", "11:00")
        if "handoff_received" not in row
    ]
    rows += _complete_handoff_rows("handoff-n2", "12:00")

    backlog = _write_backlog(tmp_path / "backlog.md", rows)

    n3 = validate_handoff_evidence(backlog, handoff_id="handoff-n3")
    assert not n3.complete
    assert "handoff_received" in n3.missing_roles

    release = build_release_packet(
        backlog_path=backlog,
        output_path=tmp_path / "release_packet.json",
    )
    assert not release.passed
