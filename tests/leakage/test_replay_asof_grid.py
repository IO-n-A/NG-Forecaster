from __future__ import annotations

import pandas as pd
import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.evaluation.replay import run_replay


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
            "feature_name": ["hh_last", "hh_last", "stor_last", "stor_last"],
            "feature_timestamp": ["2024-01-08", "2024-01-10", "2024-01-05", "2024-01-09"],
            "value": [2.0, 3.0, 100.0, 105.0],
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


def test_run_replay_emits_horizon_specific_rows_with_traceability(
    features: pd.DataFrame,
    target: pd.DataFrame,
    policy: dict[str, object],
) -> None:
    result = run_replay(
        features,
        target,
        checkpoints=["2024-01-09", "2024-01-10"],
        feature_policy=policy,
    )

    frame = result.frame
    assert len(frame) == 4
    assert set(frame["horizon"]) == {"T-1", "T"}
    assert frame["replay_checkpoint"].tolist() == sorted(frame["replay_checkpoint"].tolist())
    assert frame["trace_id"].str.contains("::").all()


def test_run_replay_blocks_when_preprocess_gate_fails(
    features: pd.DataFrame,
    target: pd.DataFrame,
    policy: dict[str, object],
) -> None:
    with pytest.raises(ContractViolation, match="reason_code=preprocess_gate_failed"):
        run_replay(
            features,
            target,
            checkpoints=["2024-01-10"],
            feature_policy=policy,
            preprocessing_status="failed",
        )
