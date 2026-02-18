from __future__ import annotations

import pandas as pd
import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.lag_guard import (
    ensure_no_future_feature_rows,
    ensure_target_lag_eligibility,
)


def test_ensure_no_future_feature_rows_passes_for_eligible_rows() -> None:
    frame = pd.DataFrame(
        {
            "feature_name": ["hh_last", "stor_last"],
            "feature_timestamp": ["2024-01-08", "2024-01-10"],
            "value": [2.5, 110.0],
        }
    )

    ensure_no_future_feature_rows(frame, asof="2024-01-10")


def test_ensure_no_future_feature_rows_fails_with_reason_contract() -> None:
    frame = pd.DataFrame(
        {
            "feature_name": ["hh_last"],
            "feature_timestamp": ["2024-01-11"],
            "value": [2.8],
        }
    )

    with pytest.raises(ContractViolation, match="reason_code=feature_after_asof") as exc:
        ensure_no_future_feature_rows(frame, asof="2024-01-10")

    msg = str(exc.value)
    assert "asof=2024-01-10T00:00:00" in msg
    assert "key=feature_name=hh_last" in msg


def test_ensure_target_lag_eligibility_fails_when_target_is_too_fresh() -> None:
    frame = pd.DataFrame(
        {
            "target_name": ["ng_prod"],
            "target_timestamp": ["2024-01-10"],
            "target_value": [101.0],
        }
    )

    with pytest.raises(ContractViolation, match="reason_code=target_lag_violation"):
        ensure_target_lag_eligibility(frame, asof="2024-01-10", min_lag_days=1)


def test_ensure_target_lag_eligibility_passes_for_cutoff_records() -> None:
    frame = pd.DataFrame(
        {
            "target_name": ["ng_prod", "ng_prod"],
            "target_timestamp": ["2024-01-08", "2024-01-09"],
            "target_value": [100.0, 101.0],
        }
    )

    ensure_target_lag_eligibility(frame, asof="2024-01-10", min_lag_days=1)
