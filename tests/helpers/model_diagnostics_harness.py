"""Reference diagnostics harness for Parallel_Lead N4 stability/calibration checks."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
from typing import Mapping, Sequence


class ModelDiagnosticsError(ValueError):
    """Contract violation for model diagnostics fixture inputs."""

    def __init__(self, *, artifact: str, key: str, reason_code: str, detail: str = "") -> None:
        message = f"artifact={artifact} | key={key} | reason_code={reason_code}"
        if detail:
            message = f"{message} | detail={detail}"
        super().__init__(message)


@dataclass(frozen=True)
class StabilitySummary:
    """Seed-repeat stability summary for one model and as-of group."""

    asof: str
    model: str
    horizon: str
    seed_count: int
    mean_prediction: float
    std_prediction: float
    stable: bool


@dataclass(frozen=True)
class CalibrationSummary:
    """Interval calibration summary for one model and horizon group."""

    model: str
    horizon: str
    observations: int
    nominal_coverage: float
    empirical_coverage: float
    passed: bool


@dataclass(frozen=True)
class DivergenceAlert:
    """Champion/challenger spread diagnostic for one as-of row."""

    asof: str
    horizon: str
    champion_point: float
    challenger_point: float
    abs_spread: float
    alert: bool


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        return [dict(row) for row in reader]


def _as_float(row: Mapping[str, str], key: str, *, artifact: str) -> float:
    try:
        return float(row[key])
    except (ValueError, KeyError) as exc:
        raise ModelDiagnosticsError(
            artifact=artifact,
            key=key,
            reason_code="invalid_numeric",
            detail=str(row.get(key, "<missing>")),
        ) from exc


def load_seed_rows(path: Path) -> list[dict[str, str]]:
    """Load seed-repeat fixture rows from disk."""
    return _read_csv_rows(path)


def summarize_seed_stability(
    rows: Sequence[Mapping[str, str]],
    *,
    std_threshold: float = 0.35,
) -> list[StabilitySummary]:
    """Compute deterministic seed stability summary grouped by as-of/model/horizon."""
    grouped: dict[tuple[str, str, str], list[float]] = {}
    for row in rows:
        key = (row["asof"], row["model"], row["horizon"])
        grouped.setdefault(key, []).append(_as_float(row, "prediction", artifact="stability_seed_runs.csv"))

    summaries: list[StabilitySummary] = []
    for (asof, model, horizon), values in sorted(grouped.items()):
        if len(values) < 2:
            raise ModelDiagnosticsError(
                artifact="stability_seed_runs.csv",
                key=f"{asof}:{model}:{horizon}",
                reason_code="insufficient_seed_repeats",
            )

        std_value = float(pstdev(values))
        summaries.append(
            StabilitySummary(
                asof=asof,
                model=model,
                horizon=horizon,
                seed_count=len(values),
                mean_prediction=round(float(mean(values)), 6),
                std_prediction=round(std_value, 6),
                stable=std_value <= std_threshold,
            )
        )

    return summaries


def load_calibration_rows(path: Path) -> list[dict[str, str]]:
    """Load interval calibration fixture rows from disk."""
    return _read_csv_rows(path)


def summarize_interval_calibration(
    rows: Sequence[Mapping[str, str]],
    *,
    nominal_coverage: float = 0.8,
    tolerance: float = 0.2,
) -> list[CalibrationSummary]:
    """Compute empirical interval coverage and pass/fail by model+horizon."""
    grouped: dict[tuple[str, str], list[bool]] = {}
    for row in rows:
        lower = _as_float(row, "lower", artifact="calibration_windows.csv")
        upper = _as_float(row, "upper", artifact="calibration_windows.csv")
        actual = _as_float(row, "actual", artifact="calibration_windows.csv")
        covered = lower <= actual <= upper
        key = (row["model"], row["horizon"])
        grouped.setdefault(key, []).append(covered)

    summaries: list[CalibrationSummary] = []
    for (model, horizon), covered_rows in sorted(grouped.items()):
        obs = len(covered_rows)
        empirical = sum(1 for item in covered_rows if item) / obs
        passed = abs(empirical - nominal_coverage) <= tolerance
        summaries.append(
            CalibrationSummary(
                model=model,
                horizon=horizon,
                observations=obs,
                nominal_coverage=nominal_coverage,
                empirical_coverage=round(empirical, 6),
                passed=passed,
            )
        )

    return summaries


def load_divergence_rows(path: Path) -> list[dict[str, str]]:
    """Load champion/challenger divergence rows from disk."""
    return _read_csv_rows(path)


def detect_champion_challenger_divergence(
    rows: Sequence[Mapping[str, str]],
    *,
    spread_threshold: float = 0.8,
) -> list[DivergenceAlert]:
    """Flag rows where champion/challenger point spread breaches threshold."""
    alerts: list[DivergenceAlert] = []
    for row in sorted(rows, key=lambda item: (item["asof"], item["horizon"])):
        champion_point = _as_float(
            row,
            "champion_point",
            artifact="champion_challenger_divergence.csv",
        )
        challenger_point = _as_float(
            row,
            "challenger_point",
            artifact="champion_challenger_divergence.csv",
        )
        spread = abs(champion_point - challenger_point)
        alerts.append(
            DivergenceAlert(
                asof=row["asof"],
                horizon=row["horizon"],
                champion_point=champion_point,
                challenger_point=challenger_point,
                abs_spread=round(spread, 6),
                alert=spread > spread_threshold,
            )
        )

    return alerts
