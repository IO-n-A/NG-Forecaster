from __future__ import annotations

from pathlib import Path

import pandas.testing as pdt
import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.evaluation.dm_test import DM_REQUIRED_OUTPUT_COLUMNS, run_dm_tests
from tests.helpers.dm_policy_matrix import (
    DMPolicyCase,
    frame_for_data_id,
    load_dm_forecast_matrix,
    load_dm_policy_cases,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "evaluation"
CASE_MATRIX_PATH = FIXTURE_ROOT / "dm_policy_case_matrix.csv"
FORECAST_MATRIX_PATH = FIXTURE_ROOT / "dm_policy_forecast_matrix.csv"

_CASES = load_dm_policy_cases(CASE_MATRIX_PATH)
_FORECAST_MATRIX = load_dm_forecast_matrix(FORECAST_MATRIX_PATH)


@pytest.mark.parametrize("case", _CASES, ids=lambda item: item.case_id)
def test_dm_policy_fixture_matrix_contracts(case: DMPolicyCase) -> None:
    frame = frame_for_data_id(_FORECAST_MATRIX, case.data_id)
    policy = case.to_policy()

    if case.expected_status == "fail":
        with pytest.raises(
            ContractViolation,
            match=f"reason_code={case.expected_reason_code}",
        ):
            run_dm_tests(frame, policy)
        return

    run_one = run_dm_tests(frame, policy)
    run_two = run_dm_tests(frame, policy)

    pdt.assert_frame_equal(run_one.results, run_two.results)
    assert set(DM_REQUIRED_OUTPUT_COLUMNS).issubset(run_one.results.columns)
    assert (run_one.results["adjusted_p_value"] >= run_one.results["p_value"]).all()

    if case.sidedness == "less":
        assert (run_one.results["p_value"] < 0.5).all()
    if case.sidedness == "greater":
        assert (run_one.results["p_value"] > 0.5).all()

    if case.multiple_comparison == "none":
        assert (run_one.results["adjusted_p_value"] == run_one.results["p_value"]).all()


def test_multiple_comparison_corrections_are_monotonic_vs_none() -> None:
    policy_by_method = {
        "none": {
            "version": 1,
            "loss": "mse",
            "sidedness": "two_sided",
            "alpha_levels": [0.05, 0.01],
            "small_sample_adjustment": "harvey",
            "benchmark_by_target": {"ng_prod": "baseline"},
            "multiple_comparison": "none",
        },
        "holm": {
            "version": 1,
            "loss": "mse",
            "sidedness": "two_sided",
            "alpha_levels": [0.05, 0.01],
            "small_sample_adjustment": "harvey",
            "benchmark_by_target": {"ng_prod": "baseline"},
            "multiple_comparison": "holm",
        },
        "bonferroni": {
            "version": 1,
            "loss": "mse",
            "sidedness": "two_sided",
            "alpha_levels": [0.05, 0.01],
            "small_sample_adjustment": "harvey",
            "benchmark_by_target": {"ng_prod": "baseline"},
            "multiple_comparison": "bonferroni",
        },
    }
    frame = frame_for_data_id(_FORECAST_MATRIX, "core")

    result_none = run_dm_tests(frame, policy_by_method["none"]).results.set_index(
        "candidate_model"
    )
    result_holm = run_dm_tests(frame, policy_by_method["holm"]).results.set_index(
        "candidate_model"
    )
    result_bonf = run_dm_tests(frame, policy_by_method["bonferroni"]).results.set_index(
        "candidate_model"
    )

    assert (result_none["adjusted_p_value"] == result_none["p_value"]).all()
    assert (
        result_holm["adjusted_p_value"] >= result_none["adjusted_p_value"]
    ).all()
    assert (
        result_bonf["adjusted_p_value"] >= result_holm["adjusted_p_value"]
    ).all()
