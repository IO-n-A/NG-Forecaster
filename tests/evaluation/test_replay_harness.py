"""Tests for reusable replay harness fixture helpers."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.replay_harness import (
    REASON_FUTURE_FEATURE,
    REASON_TARGET_NOT_LAGGED,
    LeakageContractError,
    build_replay_schedule,
    load_and_validate_asof_fixture,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"


def test_valid_asof_fixture_passes_leakage_checks() -> None:
    rows = load_and_validate_asof_fixture(FIXTURE_ROOT / "leakage" / "asof_valid.csv")
    assert len(rows) == 4
    assert [row["row_key"] for row in rows] == ["prod_001", "prod_002", "prod_003", "prod_004"]


def test_invalid_future_feature_fixture_raises_reason_code() -> None:
    with pytest.raises(LeakageContractError) as exc_info:
        load_and_validate_asof_fixture(
            FIXTURE_ROOT / "leakage" / "asof_invalid_future_feature.csv"
        )

    err = exc_info.value
    assert err.reason_code == REASON_FUTURE_FEATURE
    assert err.asof == "2025-01-31"
    assert err.key == "prod_fut"


def test_invalid_target_lag_fixture_raises_reason_code() -> None:
    with pytest.raises(LeakageContractError) as exc_info:
        load_and_validate_asof_fixture(FIXTURE_ROOT / "leakage" / "asof_invalid_target_lag.csv")

    err = exc_info.value
    assert err.reason_code == REASON_TARGET_NOT_LAGGED
    assert err.asof == "2025-02-28"
    assert err.key == "prod_tlag"


def test_replay_schedule_has_complete_t_minus_1_and_t_grid() -> None:
    schedule = build_replay_schedule(FIXTURE_ROOT / "replay" / "checkpoints.csv")
    assert len(schedule) == 4

    expected = {
        "2025-01-31": {"T-1", "T"},
        "2025-02-28": {"T-1", "T"},
    }
    for checkpoint_date, expected_horizons in expected.items():
        horizons = {
            item.horizon
            for item in schedule
            if item.checkpoint_date == checkpoint_date
        }
        assert horizons == expected_horizons
