from __future__ import annotations

from pathlib import Path

from ng_forecaster.qa.backlog_linter import lint_backlog

_HEADER = """# Backlog

## 2) Live Sync Log

| Timestamp (YYYY-MM-DD HH:MM TZ) | Engineer | Track | Task ID | Status | Blocker | Next Action | Dependency/Handoff |
|---|---|---|---|---|---|---|---|
"""


def _write(path: Path, rows: list[str]) -> Path:
    path.write_text(_HEADER + "\n".join(rows) + "\n", encoding="utf-8")
    return path


def test_backlog_linter_detects_cadence_gap_and_blocker_completeness(
    tmp_path: Path,
) -> None:
    backlog = _write(
        tmp_path / "backlog.md",
        [
            "| 2026-02-14 09:00 UTC | QA_Lead | integration | S1C-G0 | in_progress | none | begin work | handoff-n1 |",
            "| 2026-02-14 11:45 UTC | QA_Lead | integration | S1C-N1-02 | in_progress | none | continue work | handoff-n1 |",
            "| 2026-02-14 12:00 UTC | Core_Lead | core | S1B-INT-02 | blocked | waiting on triad | none | handoff-n1 |",
        ],
    )

    report = lint_backlog(backlog, max_gap_minutes=90)
    assert not report.passed
    cadence_codes = [item.reason_code for item in report.cadence_violations]
    blocker_codes = [item.reason_code for item in report.blocker_violations]
    assert "cadence_gap_exceeded" in cadence_codes
    assert "missing_owner" in blocker_codes
    assert "missing_timestamp" in blocker_codes
    assert "missing_next_action" in blocker_codes


def test_backlog_linter_passes_on_compliant_entries(tmp_path: Path) -> None:
    backlog = _write(
        tmp_path / "backlog.md",
        [
            "| 2026-02-14 09:00 UTC | QA_Lead | integration | S1C-G0 | in_progress | none | begin work | handoff-n1 |",
            "| 2026-02-14 09:50 UTC | QA_Lead | integration | S1C-N1-02 | in_progress | none | continue work | handoff-n1 |",
            "| 2026-02-14 10:00 UTC | Core_Lead | core | S1B-INT-02 | blocked | owner=Parallel_Lead; timestamp=2026-02-14; detail=awaiting handoff | follow up at 10:30 UTC | handoff-n1 |",
        ],
    )

    report = lint_backlog(backlog, max_gap_minutes=90)
    assert report.passed
