from __future__ import annotations

import pandas as pd
import pandas.testing as pdt
import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.evaluation.dm_test import (
    DM_REQUIRED_OUTPUT_COLUMNS,
    run_dm_tests,
    run_dm_tests_by_regime,
)


def _dm_policy() -> dict[str, object]:
    return {
        "version": 1,
        "loss": "mse",
        "sidedness": "two_sided",
        "alpha_levels": [0.05, 0.01],
        "small_sample_adjustment": "harvey",
        "benchmark_by_target": {"ng_prod": "baseline"},
        "multiple_comparison": "holm",
    }


def _forecast_frame() -> pd.DataFrame:
    asof = pd.date_range("2025-01-31", periods=8, freq="ME")
    actual = [100.0, 101.5, 99.3, 102.1, 101.0, 100.6, 102.8, 101.9]

    baseline_forecast = [
        101.6,
        103.0,
        100.9,
        103.8,
        102.7,
        102.2,
        104.5,
        103.4,
    ]
    candidate_a_forecast = [
        100.7,
        101.9,
        99.8,
        102.5,
        101.4,
        100.9,
        103.1,
        102.2,
    ]
    candidate_b_forecast = [
        101.9,
        103.4,
        101.2,
        104.1,
        103.0,
        102.4,
        104.9,
        103.8,
    ]

    rows: list[dict[str, object]] = []
    for model_name, forecast_values in (
        ("baseline", baseline_forecast),
        ("candidate_a", candidate_a_forecast),
        ("candidate_b", candidate_b_forecast),
    ):
        for timestamp, actual_value, forecast_value in zip(
            asof,
            actual,
            forecast_values,
        ):
            rows.append(
                {
                    "target": "ng_prod",
                    "model": model_name,
                    "asof": timestamp,
                    "horizon": 1,
                    "actual": actual_value,
                    "forecast": forecast_value,
                }
            )

    return pd.DataFrame(rows)


def test_run_dm_tests_emits_policy_schema_and_is_reproducible() -> None:
    frame = _forecast_frame()
    policy = _dm_policy()

    run_one = run_dm_tests(frame, policy)
    run_two = run_dm_tests(frame, policy)

    assert set(DM_REQUIRED_OUTPUT_COLUMNS).issubset(run_one.results.columns)
    assert set(run_one.results["candidate_model"]) == {"candidate_a", "candidate_b"}
    assert (run_one.results["adjusted_p_value"] >= run_one.results["p_value"]).all()

    pdt.assert_frame_equal(run_one.results, run_two.results)


def test_run_dm_tests_fails_when_benchmark_forecast_missing() -> None:
    frame = _forecast_frame()
    frame = frame[frame["model"] != "baseline"].reset_index(drop=True)

    with pytest.raises(
        ContractViolation, match="reason_code=missing_benchmark_forecast"
    ):
        run_dm_tests(frame, _dm_policy())


def test_run_dm_tests_supports_one_layer_vs_two_layer_model_labels() -> None:
    frame = _forecast_frame().replace(
        {
            "model": {
                "baseline": "wpd_vmd_lstm1",
                "candidate_a": "wpd_lstm_one_layer",
                "candidate_b": "wpd_vmd_lstm2",
            }
        }
    )
    policy = _dm_policy()
    policy["benchmark_by_target"] = {"ng_prod": "wpd_vmd_lstm1"}

    result = run_dm_tests(frame, policy)
    assert set(result.results["candidate_model"]) == {
        "wpd_lstm_one_layer",
        "wpd_vmd_lstm2",
    }


def test_run_dm_tests_respects_explicit_comparison_pairs_by_target() -> None:
    frame = _forecast_frame()
    policy = _dm_policy()
    policy["comparison_pairs_by_target"] = {
        "ng_prod": [["candidate_a", "candidate_b"], ["candidate_b", "baseline"]]
    }

    result = run_dm_tests(frame, policy)
    assert len(result.results) == 2
    assert set(result.results["candidate_model"]) == {"candidate_a", "candidate_b"}
    assert set(result.results["benchmark_model"]) == {"candidate_b", "baseline"}


def test_run_dm_tests_fails_when_configured_candidate_is_missing() -> None:
    frame = _forecast_frame()
    policy = _dm_policy()
    policy["comparison_pairs_by_target"] = {
        "ng_prod": [["candidate_missing", "baseline"]]
    }

    with pytest.raises(
        ContractViolation, match="reason_code=missing_candidate_forecast"
    ):
        run_dm_tests(frame, policy)


def test_run_dm_tests_by_regime_emits_regime_labeled_dm_outputs() -> None:
    frame = _forecast_frame().rename(
        columns={
            "model": "model_variant",
            "actual": "actual_released",
            "forecast": "fused_point",
        }
    )
    frame["target_month"] = frame["asof"].dt.to_period("M").dt.to_timestamp("M")
    frame["regime_label"] = [
        "normal" if idx % 2 == 0 else "freeze_off" for idx in range(len(frame))
    ]

    result = run_dm_tests_by_regime(frame, _dm_policy())
    ok_rows = result[result["status"] == "ok"]
    assert not ok_rows.empty
    assert set(ok_rows["regime_label"]) == {"freeze_off", "normal"}
