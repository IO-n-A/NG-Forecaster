"""Fixture helpers for Parallel_Lead Phase G orchestration reliability tests."""

from __future__ import annotations

import csv
import shutil
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RetryBackoffCase:
    """Retry/backoff case loaded from fixture matrix."""

    case_id: str
    status_sequence: tuple[int, ...]
    max_attempts: int
    initial_delay_seconds: float
    backoff_multiplier: float
    expected_status: str
    expected_attempts: int
    expected_sleep_sequence: tuple[float, ...]


def load_retry_backoff_cases(path: Path) -> list[RetryBackoffCase]:
    """Load retry/backoff fixture matrix rows."""
    cases: list[RetryBackoffCase] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            status_sequence = tuple(
                int(item.strip())
                for item in row["status_sequence"].split("|")
                if item.strip()
            )
            expected_sleep = tuple(
                float(item.strip())
                for item in row["expected_sleep_sequence"].split("|")
                if item.strip()
            )
            cases.append(
                RetryBackoffCase(
                    case_id=row["case_id"].strip(),
                    status_sequence=status_sequence,
                    max_attempts=int(row["max_attempts"]),
                    initial_delay_seconds=float(row["initial_delay_seconds"]),
                    backoff_multiplier=float(row["backoff_multiplier"]),
                    expected_status=row["expected_status"].strip(),
                    expected_attempts=int(row["expected_attempts"]),
                    expected_sleep_sequence=expected_sleep,
                )
            )

    return sorted(cases, key=lambda item: item.case_id)


def stage_bootstrap_raw_fixture(src_dir: Path, dest_raw_dir: Path) -> list[str]:
    """Copy bootstrap raw fixture files into a destination raw directory."""
    dest_raw_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    for source_file in sorted(src_dir.glob("*")):
        if not source_file.is_file():
            continue
        destination = dest_raw_dir / source_file.name
        shutil.copy2(source_file, destination)
        copied.append(source_file.name)
    return copied


def build_bootstrap_status_payload(raw_dir: Path, report_dir: Path) -> dict[str, object]:
    """Build deterministic bootstrap status payload for DAG monkeypatching."""
    raw_files = sorted(path.name for path in raw_dir.glob("*") if path.is_file())
    report_files = sorted(path.name for path in report_dir.glob("*") if path.is_file())
    return {
        "bootstrap_root": str(raw_dir.parent),
        "raw_dir": str(raw_dir),
        "report_dir": str(report_dir),
        "raw_file_count": len(raw_files),
        "report_file_count": len(report_files),
        "raw_files": raw_files,
        "report_files": report_files,
        "available": bool(raw_files),
    }
