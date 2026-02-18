from __future__ import annotations

import pandas as pd
import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.evaluation.replay import run_ablation_matrix


def _ablation_frame() -> pd.DataFrame:
    asof = pd.date_range("2025-01-31", periods=6, freq="ME")
    actual = [100.0, 101.2, 100.4, 102.1, 101.0, 100.8]

    experiment_offsets = {
        "B0_baseline": 1.8,
        "B1_plus_preprocessing": 1.3,
        "B2_plus_feature_expansion": 1.0,
        "B3_plus_challenger": 0.8,
        "B4_full_method": 0.5,
    }

    rows: list[dict[str, object]] = []
    for experiment_id, offset in experiment_offsets.items():
        for idx, (timestamp, actual_value) in enumerate(zip(asof, actual)):
            forecast = actual_value + (offset if idx % 2 == 0 else -offset)
            rows.append(
                {
                    "experiment_id": experiment_id,
                    "candidate_model": f"{experiment_id}_main",
                    "target": "ng_prod",
                    "asof": timestamp,
                    "horizon": 1,
                    "actual": actual_value,
                    "forecast": forecast,
                    "runtime_seconds": 30
                    + (5 * (list(experiment_offsets).index(experiment_id) + 1)),
                    "lineage_id": f"{experiment_id}_lineage",
                }
            )

    for idx, (timestamp, actual_value) in enumerate(zip(asof, actual)):
        forecast_alt = actual_value + (1.9 if idx % 2 == 0 else -1.9)
        rows.append(
            {
                "experiment_id": "B2_plus_feature_expansion",
                "candidate_model": "B2_plus_feature_expansion_alt",
                "target": "ng_prod",
                "asof": timestamp,
                "horizon": 1,
                "actual": actual_value,
                "forecast": forecast_alt,
                "runtime_seconds": 45,
                "lineage_id": "B2_plus_feature_expansion_lineage_alt",
            }
        )

    return pd.DataFrame(rows)


def _dm_policy_without_mapping() -> dict[str, object]:
    return {
        "version": 1,
        "loss": "mse",
        "sidedness": "two_sided",
        "alpha_levels": [0.05, 0.01],
        "small_sample_adjustment": "harvey",
        "multiple_comparison": "holm",
    }


def test_run_ablation_matrix_builds_scorecard_and_selection_wiring() -> None:
    frame = _ablation_frame()

    result = run_ablation_matrix(
        frame,
        config=None,
        dm_policy=_dm_policy_without_mapping(),
    )

    expected_experiments = {
        "B0_baseline",
        "B1_plus_preprocessing",
        "B2_plus_feature_expansion",
        "B3_plus_challenger",
        "B4_full_method",
    }
    assert set(result.scorecard["experiment_id"]) == expected_experiments
    assert {
        "experiment_id",
        "mae",
        "rmse",
        "mape",
        "dm_vs_baseline_p_value",
        "runtime_seconds",
        "lineage_id",
    }.issubset(result.scorecard.columns)

    baseline_row = result.scorecard[
        result.scorecard["experiment_id"] == "B0_baseline"
    ].iloc[0]
    assert float(baseline_row["dm_vs_baseline_p_value"]) == 1.0

    b2_selection = result.selection_summary[
        result.selection_summary["experiment_id"] == "B2_plus_feature_expansion"
    ].iloc[0]
    assert str(b2_selection["selected_model"]) == "B2_plus_feature_expansion_main"


def test_run_ablation_matrix_fails_when_configured_experiment_missing() -> None:
    frame = _ablation_frame()
    frame = frame[frame["experiment_id"] != "B3_plus_challenger"].reset_index(drop=True)

    with pytest.raises(
        ContractViolation, match="reason_code=missing_ablation_experiment"
    ):
        run_ablation_matrix(
            frame,
            config=None,
            dm_policy=_dm_policy_without_mapping(),
        )
