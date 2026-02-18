from __future__ import annotations

from pathlib import Path

import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.qa.n5_policy_audit import audit_n5_policy

_VALID_POLICY = """version: 1
loss: mse
sidedness: two_sided
alpha_levels:
  - 0.05
  - 0.01
small_sample_adjustment: harvey
benchmark_by_target:
  ng_prod: champion
multiple_comparison: holm
"""

_PAIR_POLICY = """version: 1
loss: mse
sidedness: two_sided
alpha_levels:
  - 0.05
  - 0.01
small_sample_adjustment: harvey
benchmark_by_target:
  ng_prod: wpd_vmd_lstm1
multiple_comparison: holm
comparison_pairs_by_target:
  ng_prod:
    - [wpd_lstm_one_layer, wpd_vmd_lstm1]
    - [wpd_vmd_lstm2, wpd_vmd_lstm1]
"""


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


def _write_valid_dm_results(path: Path) -> None:
    _write(
        path,
        "\n".join(
            [
                "target,candidate_model,benchmark_model,d_bar,dm_stat,p_value,significant_0_05,significant_0_01,adjusted_p_value",
                "ng_prod,challenger,champion,-0.12,-2.5,0.012,true,false,0.020",
            ]
        ),
    )


def test_n5_policy_audit_passes_with_compliant_inputs(tmp_path: Path) -> None:
    policy = tmp_path / "evaluation.yaml"
    dm_results = tmp_path / "dm_results.csv"

    _write(policy, _VALID_POLICY)
    _write_valid_dm_results(dm_results)

    result = audit_n5_policy(policy, dm_results)
    assert result.passed
    assert result.target_count == 1


def test_n5_policy_audit_fails_on_significance_mismatch(tmp_path: Path) -> None:
    policy = tmp_path / "evaluation.yaml"
    dm_results = tmp_path / "dm_results.csv"

    _write(policy, _VALID_POLICY)
    _write(
        dm_results,
        "\n".join(
            [
                "target,candidate_model,benchmark_model,d_bar,dm_stat,p_value,significant_0_05,significant_0_01,adjusted_p_value",
                "ng_prod,challenger,champion,-0.12,-2.5,0.012,false,false,0.020",
            ]
        ),
    )

    with pytest.raises(ContractViolation, match="reason_code=significance_mismatch"):
        audit_n5_policy(policy, dm_results)


def test_n5_policy_audit_accepts_configured_comparison_pairs(tmp_path: Path) -> None:
    policy = tmp_path / "evaluation.yaml"
    dm_results = tmp_path / "dm_results.csv"

    _write(policy, _PAIR_POLICY)
    _write(
        dm_results,
        "\n".join(
            [
                "target,candidate_model,benchmark_model,d_bar,dm_stat,p_value,significant_0_05,significant_0_01,adjusted_p_value",
                "ng_prod,wpd_lstm_one_layer,wpd_vmd_lstm1,-0.03,-1.3,0.19,false,false,0.19",
                "ng_prod,wpd_vmd_lstm2,wpd_vmd_lstm1,0.01,0.2,0.84,false,false,0.84",
            ]
        ),
    )

    result = audit_n5_policy(policy, dm_results)
    assert result.passed
