from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.reporting.exporters import (
    export_ablation_scorecard,
    export_dm_results,
)


def test_export_dm_results_writes_sorted_schema_compliant_csv(tmp_path: Path) -> None:
    dm_results = pd.DataFrame(
        {
            "target": ["ng_prod", "ng_prod"],
            "candidate_model": ["candidate_b", "candidate_a"],
            "benchmark_model": ["baseline", "baseline"],
            "d_bar": [0.12, -0.20],
            "dm_stat": [1.1, -2.4],
            "p_value": [0.24, 0.02],
            "significant_0_05": [False, True],
            "significant_0_01": [False, False],
            "adjusted_p_value": [0.24, 0.04],
            "n_obs": [12, 12],
        }
    )

    output = export_dm_results(dm_results, tmp_path)
    assert output.exists()

    loaded = pd.read_csv(output)
    assert loaded.iloc[0]["candidate_model"] == "candidate_a"


def test_export_ablation_scorecard_rejects_non_positive_runtime(tmp_path: Path) -> None:
    scorecard = pd.DataFrame(
        {
            "experiment_id": ["B0_baseline"],
            "target": ["ng_prod"],
            "mae": [1.2],
            "rmse": [1.5],
            "mape": [0.02],
            "dm_vs_baseline_p_value": [1.0],
            "runtime_seconds": [0],
            "lineage_id": ["lineage_b0"],
        }
    )

    with pytest.raises(
        ContractViolation, match="reason_code=invalid_ablation_scorecard"
    ):
        export_ablation_scorecard(scorecard, tmp_path)
