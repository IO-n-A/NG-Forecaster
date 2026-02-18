from __future__ import annotations

import pandas as pd
import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.vintage_builder import build_vintage_panel


@pytest.fixture()
def policy() -> dict[str, object]:
    return {
        "version": 1,
        "default": {"max_age_days": 30},
        "features": {
            "hh_last": {
                "source_frequency": "daily",
                "aggregation": "last",
                "max_age_days": 7,
            },
            "stor_last": {
                "source_frequency": "weekly",
                "aggregation": "last",
                "max_age_days": 14,
            },
        },
    }


@pytest.fixture()
def features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_name": ["hh_last", "hh_last", "hh_last", "stor_last", "stor_last"],
            "feature_timestamp": [
                "2024-01-08",
                "2024-01-10",
                "2024-01-12",  # future for asof=2024-01-10, must be ignored
                "2024-01-05",
                "2024-01-09",
            ],
            "value": [2.0, 3.0, 999.0, 100.0, 105.0],
        }
    )


@pytest.fixture()
def target() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_timestamp": ["2024-01-07", "2024-01-08", "2024-01-09", "2024-01-10"],
            "target_value": [50.0, 51.0, 52.0, 53.0],
        }
    )


def test_build_vintage_panel_enforces_asof_filtering_before_selection(
    features: pd.DataFrame,
    target: pd.DataFrame,
    policy: dict[str, object],
) -> None:
    result = build_vintage_panel(features, target, asof="2024-01-10", feature_policy=policy)

    t_minus_1 = result.slices["T-1"].iloc[0]
    t_now = result.slices["T"].iloc[0]

    assert t_minus_1["hh_last"] == 2.0
    assert t_now["hh_last"] == 3.0
    assert t_now["stor_last"] == 105.0
    assert t_now["target_value"] == 52.0
    assert t_minus_1["target_value"] == 51.0
    assert result.lineage["T"] == t_now["lineage_id"]


def test_build_vintage_panel_blocks_when_preprocess_gate_fails(
    features: pd.DataFrame,
    target: pd.DataFrame,
    policy: dict[str, object],
) -> None:
    with pytest.raises(ContractViolation, match="reason_code=preprocess_gate_failed"):
        build_vintage_panel(
            features,
            target,
            asof="2024-01-10",
            feature_policy=policy,
            preprocessing_status="failed",
        )


def test_build_vintage_panel_fails_for_stale_feature(
    target: pd.DataFrame,
) -> None:
    stale_features = pd.DataFrame(
        {
            "feature_name": ["hh_last", "stor_last"],
            "feature_timestamp": ["2024-01-01", "2024-01-01"],
            "value": [2.0, 100.0],
        }
    )
    strict_policy = {
        "version": 1,
        "default": {"max_age_days": 1},
        "features": {
            "hh_last": {
                "source_frequency": "daily",
                "aggregation": "last",
                "max_age_days": 1,
            },
            "stor_last": {
                "source_frequency": "weekly",
                "aggregation": "last",
                "max_age_days": 1,
            },
        },
    }

    with pytest.raises(ContractViolation, match="reason_code=feature_ineligible"):
        build_vintage_panel(
            stale_features,
            target,
            asof="2024-01-10",
            feature_policy=strict_policy,
        )
