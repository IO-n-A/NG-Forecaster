"""Parallel_Lead N4 diagnostics: seed stability, calibration regression, divergence checks."""

from __future__ import annotations

from pathlib import Path

from tests.helpers.model_diagnostics_harness import (
    detect_champion_challenger_divergence,
    load_calibration_rows,
    load_divergence_rows,
    load_seed_rows,
    summarize_interval_calibration,
    summarize_seed_stability,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "models"


def test_seed_stability_flags_unstable_challenger_group() -> None:
    rows = load_seed_rows(FIXTURE_ROOT / "stability_seed_runs.csv")
    first = summarize_seed_stability(rows, std_threshold=0.35)
    second = summarize_seed_stability(rows, std_threshold=0.35)

    assert first == second
    assert len(first) == 2

    champion = next(item for item in first if item.model == "champion")
    challenger = next(item for item in first if item.model == "challenger")

    assert champion.seed_count == 3
    assert champion.stable is True
    assert challenger.seed_count == 3
    assert challenger.stable is False


def test_calibration_regression_passes_champion_and_fails_challenger() -> None:
    rows = load_calibration_rows(FIXTURE_ROOT / "calibration_windows.csv")
    summary = summarize_interval_calibration(rows, nominal_coverage=0.8, tolerance=0.25)
    assert len(summary) == 2

    champion = next(item for item in summary if item.model == "champion")
    challenger = next(item for item in summary if item.model == "challenger")

    assert champion.empirical_coverage == 1.0
    assert champion.passed is True
    assert challenger.empirical_coverage == 0.25
    assert challenger.passed is False


def test_divergence_alert_detects_large_spread_row() -> None:
    rows = load_divergence_rows(FIXTURE_ROOT / "champion_challenger_divergence.csv")
    alerts = detect_champion_challenger_divergence(rows, spread_threshold=0.8)

    assert len(alerts) == 2
    assert alerts[0].alert is False
    assert alerts[1].alert is True
    assert alerts[1].abs_spread == 1.15
