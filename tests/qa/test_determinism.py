from __future__ import annotations

import json
from pathlib import Path

from ng_forecaster.qa.determinism import compare_rerun_snapshots


def _write_snapshot(
    path: Path, *, value: str = "1.0", generated_at: str = "2026-02-14T10:00:00Z"
) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "summary.json").write_text(
        json.dumps({"status": "passed", "generated_at": generated_at, "metric": 1}),
        encoding="utf-8",
    )
    (path / "output.csv").write_text(
        "\n".join(
            [
                "id,value,generated_at",
                f"A,{value},{generated_at}",
            ]
        ),
        encoding="utf-8",
    )


def test_determinism_allows_metadata_timestamp_differences(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_snapshot(left, generated_at="2026-02-14T10:00:00Z")
    _write_snapshot(right, generated_at="2026-02-14T12:00:00Z")

    result = compare_rerun_snapshots(left, right)
    assert result.matches


def test_determinism_detects_csv_value_mismatch(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_snapshot(left, value="1.0")
    _write_snapshot(right, value="9.0")

    result = compare_rerun_snapshots(left, right)
    assert not result.matches
    assert result.reason_code == "csv_value_mismatch"
    assert "column=value" in (result.first_diff or "")


def test_determinism_detects_file_set_mismatch(tmp_path: Path) -> None:
    left = tmp_path / "left"
    right = tmp_path / "right"
    _write_snapshot(left)
    _write_snapshot(right)
    (right / "extra.txt").write_text("unexpected", encoding="utf-8")

    result = compare_rerun_snapshots(left, right)
    assert not result.matches
    assert result.reason_code == "file_set_mismatch"
