from __future__ import annotations

import pandas as pd
import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.vintage_builder import build_vintage_panel


@pytest.fixture()
def target() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_timestamp": ["2024-01-08", "2024-01-09", "2024-01-10"],
            "target_value": [100.0, 101.0, 102.0],
        }
    )


def test_builder_uses_latest_eligible_value_without_future_fill(target: pd.DataFrame) -> None:
    features = pd.DataFrame(
        {
            "feature_name": ["hh_last", "hh_last", "hh_last"],
            "feature_timestamp": ["2024-01-09", "2024-01-10", "2024-01-20"],
            "value": [3.0, 3.1, 999.0],
        }
    )
    policy = {
        "version": 1,
        "default": {"max_age_days": 14},
        "features": {
            "hh_last": {
                "source_frequency": "daily",
                "aggregation": "last",
                "max_age_days": 14,
            }
        },
    }

    result = build_vintage_panel(features, target, asof="2024-01-10", feature_policy=policy)
    t_now = result.slices["T"].iloc[0]
    assert t_now["hh_last"] == 3.1


def test_builder_rejects_stale_feature_rows(target: pd.DataFrame) -> None:
    features = pd.DataFrame(
        {
            "feature_name": ["hh_last"],
            "feature_timestamp": ["2024-01-01"],
            "value": [2.1],
        }
    )
    strict_policy = {
        "version": 1,
        "default": {"max_age_days": 3},
        "features": {
            "hh_last": {
                "source_frequency": "daily",
                "aggregation": "last",
                "max_age_days": 3,
            }
        },
    }

    with pytest.raises(ContractViolation, match="reason_code=feature_ineligible"):
        build_vintage_panel(features, target, asof="2024-01-10", feature_policy=strict_policy)
