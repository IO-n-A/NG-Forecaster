"""Deterministic reference preprocessing used by Sprint 1A tests."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Sequence


@dataclass
class PreprocessOutput:
    """Container for deterministic preprocess outputs and audit artifacts."""

    cleaned_rows: list[dict[str, object]]
    missing_flags: list[dict[str, str]]
    outlier_flags: list[dict[str, str]]
    summary: dict[str, object]


def load_series_fixture(path: Path) -> list[dict[str, object]]:
    """Load a preprocess fixture with optional missing numeric values."""
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            raw_value = (row.get("value") or "").strip()
            value = None if raw_value == "" else float(raw_value)
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "series_id": row["series_id"],
                    "value": value,
                }
            )

    return sorted(rows, key=lambda row: str(row["timestamp"]))


def _nearest_fill_value(values: Sequence[float | None], start: int, end: int) -> float:
    for idx in range(start - 1, -1, -1):
        candidate = values[idx]
        if candidate is not None:
            return float(candidate)

    for idx in range(end + 1, len(values)):
        candidate = values[idx]
        if candidate is not None:
            return float(candidate)

    return 0.0


def _find_missing_segments(values: Sequence[float | None]) -> list[tuple[int, int, int]]:
    segments: list[tuple[int, int, int]] = []
    idx = 0
    while idx < len(values):
        if values[idx] is not None:
            idx += 1
            continue

        start = idx
        while idx < len(values) and values[idx] is None:
            idx += 1
        end = idx - 1
        segments.append((start, end, end - start + 1))

    return segments


def run_reference_preprocess(
    rows: Sequence[dict[str, object]],
    *,
    asof: str | None = None,
    short_gap_max: int = 2,
    outlier_z_threshold: float = 3.5,
) -> PreprocessOutput:
    """Apply deterministic missing-value and outlier policies."""
    if not rows:
        raise ValueError("rows must be non-empty")

    ordered_rows = sorted(rows, key=lambda row: str(row["timestamp"]))
    values: list[float | None] = [row["value"] if isinstance(row["value"], float) else None for row in ordered_rows]
    asof_value = asof or str(ordered_rows[-1]["timestamp"])

    missing_flags: list[dict[str, str]] = []
    outlier_flags: list[dict[str, str]] = []

    segments = _find_missing_segments(values)
    for start, end, gap_length in segments:
        gap_type = "short" if gap_length <= short_gap_max else "long"
        policy = "forward_fill" if gap_type == "short" else "leave_missing"
        if gap_type == "short":
            fill_value = _nearest_fill_value(values, start, end)
            for idx in range(start, end + 1):
                values[idx] = fill_value

        for idx in range(start, end + 1):
            missing_flags.append(
                {
                    "asof": asof_value,
                    "series_id": str(ordered_rows[idx]["series_id"]),
                    "timestamp": str(ordered_rows[idx]["timestamp"]),
                    "gap_type": gap_type,
                    "gap_length": str(gap_length),
                    "imputation_policy": policy,
                }
            )

    non_missing = [value for value in values if value is not None]
    if non_missing:
        location = float(median(non_missing))
        abs_devs = [abs(value - location) for value in non_missing]
        spread = float(median(abs_devs))

        if spread > 0:
            for idx, value in enumerate(values):
                if value is None:
                    continue
                robust_z = 0.6745 * (value - location) / spread
                if abs(robust_z) > outlier_z_threshold:
                    outlier_flags.append(
                        {
                            "asof": asof_value,
                            "series_id": str(ordered_rows[idx]["series_id"]),
                            "timestamp": str(ordered_rows[idx]["timestamp"]),
                            "method": "robust_zscore_mad",
                            "z_score": f"{robust_z:.6f}",
                            "action": "winsorize_to_median",
                        }
                    )
                    values[idx] = location

    cleaned_rows: list[dict[str, object]] = []
    for idx, row in enumerate(ordered_rows):
        cleaned_value = values[idx]
        cleaned_rows.append(
            {
                "timestamp": str(row["timestamp"]),
                "series_id": str(row["series_id"]),
                "value": None if cleaned_value is None else round(float(cleaned_value), 6),
            }
        )

    missing_short_gap_count = sum(1 for row in missing_flags if row["gap_type"] == "short")
    missing_long_gap_count = sum(1 for row in missing_flags if row["gap_type"] == "long")
    status = "failed" if missing_long_gap_count > 0 else "passed"

    summary: dict[str, object] = {
        "asof": asof_value,
        "status": status,
        "row_count": len(cleaned_rows),
        "missing_short_gap_count": missing_short_gap_count,
        "missing_long_gap_count": missing_long_gap_count,
        "outlier_count": len(outlier_flags),
    }

    return PreprocessOutput(
        cleaned_rows=cleaned_rows,
        missing_flags=missing_flags,
        outlier_flags=outlier_flags,
        summary=summary,
    )
