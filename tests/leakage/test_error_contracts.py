"""Stable leakage error message contract tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from tests.helpers.replay_harness import (
    REASON_FUTURE_FEATURE,
    REASON_TARGET_NOT_LAGGED,
    LeakageContractError,
    load_and_validate_asof_fixture,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "leakage"


def test_error_message_contract_uses_asof_key_reason_code() -> None:
    error = LeakageContractError(
        asof="2025-01-31",
        key="prod_001",
        reason_code=REASON_FUTURE_FEATURE,
    )
    assert str(error) == "asof=2025-01-31 | key=prod_001 | reason_code=future_feature_timestamp"


def test_future_feature_violation_emits_stable_contract_fields() -> None:
    with pytest.raises(LeakageContractError) as exc_info:
        load_and_validate_asof_fixture(FIXTURE_ROOT / "asof_invalid_future_feature.csv")

    message = str(exc_info.value)
    assert "asof=2025-01-31" in message
    assert "key=prod_fut" in message
    assert "reason_code=future_feature_timestamp" in message


def test_target_lag_violation_emits_stable_contract_fields() -> None:
    with pytest.raises(LeakageContractError) as exc_info:
        load_and_validate_asof_fixture(FIXTURE_ROOT / "asof_invalid_target_lag.csv")

    message = str(exc_info.value)
    assert "asof=2025-02-28" in message
    assert "key=prod_tlag" in message
    assert "reason_code=target_not_lagged" in message
    assert exc_info.value.reason_code == REASON_TARGET_NOT_LAGGED
