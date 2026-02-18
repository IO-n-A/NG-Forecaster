from __future__ import annotations

import pandas as pd

from ng_forecaster.evaluation.replay import run_replay


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_name": ["hh_last", "hh_last", "stor_last", "stor_last"],
            "feature_timestamp": [
                "2024-01-08",
                "2024-01-10",
                "2024-01-05",
                "2024-01-09",
            ],
            "value": [2.0, 3.0, 100.0, 105.0],
        }
    )


def _target() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_timestamp": [
                "2023-10-31",
                "2023-11-30",
                "2023-12-31",
                "2024-01-31",
            ],
            "target_value": [50.0, 51.0, 52.0, 53.0],
        }
    )


def _policy() -> dict[str, object]:
    return {
        "version": 1,
        "default": {"max_age_days": 90},
        "features": {
            "hh_last": {
                "source_frequency": "daily",
                "aggregation": "last",
                "max_age_days": 90,
            },
            "stor_last": {
                "source_frequency": "weekly",
                "aggregation": "last",
                "max_age_days": 90,
            },
        },
    }


def test_run_replay_emits_explicit_target_month_and_deterministic_snapshot() -> None:
    first = run_replay(
        _features(),
        _target(),
        checkpoints=["2024-01-10", "2024-02-10"],
        feature_policy=_policy(),
        min_target_lag_months=2,
        target_month_offset_months=2,
    ).frame
    second = run_replay(
        _features(),
        _target(),
        checkpoints=["2024-01-10", "2024-02-10"],
        feature_policy=_policy(),
        min_target_lag_months=2,
        target_month_offset_months=2,
    ).frame

    assert first.equals(second)
    assert set(first["horizon"]) == {"T-1", "T"}
    assert "target_month" in first.columns
    expected_months = {
        ("2024-01-10", "T-1"): "2023-11-30",
        ("2024-01-10", "T"): "2023-12-31",
        ("2024-02-10", "T-1"): "2023-12-31",
        ("2024-02-10", "T"): "2024-01-31",
    }
    for _, row in first.iterrows():
        key = (
            pd.Timestamp(row["replay_checkpoint"]).date().isoformat(),
            str(row["horizon"]),
        )
        assert str(row["target_month"]) == expected_months[key]
