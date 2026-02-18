"""Fixture loaders for Parallel_Lead Phase E DM policy matrix tests."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DMPolicyCase:
    """Single DM policy contract case loaded from fixture matrix."""

    case_id: str
    data_id: str
    sidedness: str
    multiple_comparison: str
    benchmark_by_target: dict[str, str]
    expected_status: str
    expected_reason_code: str | None

    def to_policy(self) -> dict[str, Any]:
        """Build a policy payload suitable for run_dm_tests."""
        return {
            "version": 1,
            "loss": "mse",
            "sidedness": self.sidedness,
            "alpha_levels": [0.05, 0.01],
            "small_sample_adjustment": "harvey",
            "benchmark_by_target": dict(self.benchmark_by_target),
            "multiple_comparison": self.multiple_comparison,
        }


def load_dm_policy_cases(path: Path) -> list[DMPolicyCase]:
    """Load policy matrix contract cases sorted by case identifier."""
    cases: list[DMPolicyCase] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            benchmark_mapping = json.loads(row["benchmark_by_target"])
            if not isinstance(benchmark_mapping, dict):
                raise ValueError("benchmark_by_target must deserialize to a mapping")

            reason_code = row["expected_reason_code"].strip() or None
            cases.append(
                DMPolicyCase(
                    case_id=row["case_id"].strip(),
                    data_id=row["data_id"].strip(),
                    sidedness=row["sidedness"].strip(),
                    multiple_comparison=row["multiple_comparison"].strip(),
                    benchmark_by_target={
                        str(key): str(value) for key, value in benchmark_mapping.items()
                    },
                    expected_status=row["expected_status"].strip(),
                    expected_reason_code=reason_code,
                )
            )

    return sorted(cases, key=lambda item: item.case_id)


def load_dm_forecast_matrix(path: Path) -> pd.DataFrame:
    """Load deterministic DM forecast matrix fixture for all cases."""
    frame = pd.read_csv(path)
    ordered = frame.sort_values(["data_id", "target", "model", "asof"]).reset_index(
        drop=True
    )
    return ordered


def frame_for_data_id(frame: pd.DataFrame, data_id: str) -> pd.DataFrame:
    """Filter fixture matrix for a specific scenario data_id."""
    selected = frame[frame["data_id"] == data_id].copy()
    selected = selected.drop(columns=["data_id"])
    selected["asof"] = pd.to_datetime(selected["asof"], errors="raise")
    selected["horizon"] = pd.to_numeric(selected["horizon"], errors="raise")
    selected["actual"] = pd.to_numeric(selected["actual"], errors="raise")
    selected["forecast"] = pd.to_numeric(selected["forecast"], errors="raise")
    return selected.reset_index(drop=True)
