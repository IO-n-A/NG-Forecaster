from __future__ import annotations

import json
from pathlib import Path

from ng_forecaster.qa.release_gate import build_release_packet

_HEADER = """# Backlog

## 2) Live Sync Log

| Timestamp (YYYY-MM-DD HH:MM TZ) | Engineer | Track | Task ID | Status | Blocker | Next Action | Dependency/Handoff |
|---|---|---|---|---|---|---|---|
"""


def _row(
    ts: str,
    engineer: str,
    track: str,
    task_id: str,
    status: str,
    blocker: str,
    next_action: str,
    dep: str,
) -> str:
    return f"| {ts} | {engineer} | {track} | {task_id} | {status} | {blocker} | {next_action} | {dep} |"


def _write_backlog(path: Path, rows: list[str]) -> Path:
    path.write_text(_HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def _triad_rows(handoff_id: str, base_ts: str) -> list[str]:
    return [
        _row(
            base_ts,
            "Parallel_Lead",
            "parallel",
            "S1A",
            "handoff_ready",
            "none",
            (
                "changed_files: tests/a.py, tests/b.py; "
                "tests_run: pytest tests/qa -q, pytest tests/integration/test_n3_gate_contract.py -q, "
                "pytest tests/integration/test_cross_track_contracts.py -q, "
                "pytest tests/integration/test_n1_n3_n2_handoff_matrix.py -q; "
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
            "received and validated",
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


def test_release_gate_builds_packet_when_requirements_are_met(tmp_path: Path) -> None:
    backlog = _write_backlog(
        tmp_path / "backlog.md",
        _triad_rows("handoff-n1", "2026-02-14 10:00 UTC")
        + _triad_rows("handoff-n3", "2026-02-14 11:00 UTC")
        + _triad_rows("handoff-n2", "2026-02-14 12:00 UTC"),
    )
    output = tmp_path / "release_packet.json"

    result = build_release_packet(backlog_path=backlog, output_path=output)

    assert result.passed
    assert output.exists()
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["release_status"] == "passed"


def test_release_gate_fails_when_required_tests_not_logged(tmp_path: Path) -> None:
    rows = _triad_rows("handoff-n1", "2026-02-14 10:00 UTC")
    rows += _triad_rows("handoff-n3", "2026-02-14 11:00 UTC")
    rows += _triad_rows("handoff-n2", "2026-02-14 12:00 UTC")
    rows = [
        row.replace("pytest tests/integration/test_cross_track_contracts.py -q, ", "")
        for row in rows
    ]

    backlog = _write_backlog(tmp_path / "backlog.md", rows)
    result = build_release_packet(
        backlog_path=backlog,
        output_path=tmp_path / "release_packet.json",
    )

    assert not result.passed
    assert (
        "pytest tests/integration/test_cross_track_contracts.py -q"
        in result.missing_test_groups
    )


def test_release_gate_fails_on_unresolved_blocker_without_owner_timestamp(
    tmp_path: Path,
) -> None:
    rows = _triad_rows("handoff-n1", "2026-02-14 10:00 UTC")
    rows += _triad_rows("handoff-n3", "2026-02-14 11:00 UTC")
    rows += _triad_rows("handoff-n2", "2026-02-14 12:00 UTC")
    rows.append(
        _row(
            "2026-02-14 12:30 UTC",
            "Core_Lead",
            "core",
            "S1B-INT-02",
            "blocked",
            "waiting on qa",
            "none",
            "handoff-n2",
        )
    )

    backlog = _write_backlog(tmp_path / "backlog.md", rows)
    result = build_release_packet(
        backlog_path=backlog,
        output_path=tmp_path / "release_packet.json",
    )

    assert not result.passed
    assert result.unresolved_blockers
    assert result.unresolved_blockers[0].task_id == "S1B-INT-02"
