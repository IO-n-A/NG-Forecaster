from __future__ import annotations

from pathlib import Path

import pandas as pd
import pandas.testing as pdt
import pytest

from ng_forecaster.data.validators import load_yaml
from ng_forecaster.errors import ContractViolation
from ng_forecaster.evaluation.replay import run_ablation_matrix
from ng_forecaster.qa.n6_adoption_gate import check_n6_adoption_readiness
from ng_forecaster.reporting.exporters import export_ablation_scorecard
from tests.helpers.ablation_replay_fixture import (
    canonicalize_ablation_scorecard,
    canonicalize_dm_results,
    canonicalize_selection_summary,
    load_ablation_replay_frame,
    load_expected_frame,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "evaluation"


def test_ablation_replay_pack_matches_expected_regression_snapshot() -> None:
    frame = load_ablation_replay_frame(FIXTURE_ROOT / "ablation_replay_matrix.csv")
    config = load_yaml("configs/experiments/nowcast_ablation.yaml")
    dm_policy = load_yaml("configs/evaluation.yaml")

    run_one = run_ablation_matrix(frame, config=config, dm_policy=dm_policy)
    run_two = run_ablation_matrix(frame, config=config, dm_policy=dm_policy)

    score_one = canonicalize_ablation_scorecard(run_one.scorecard)
    score_two = canonicalize_ablation_scorecard(run_two.scorecard)
    dm_one = canonicalize_dm_results(run_one.dm_results)
    dm_two = canonicalize_dm_results(run_two.dm_results)
    sel_one = canonicalize_selection_summary(run_one.selection_summary)
    sel_two = canonicalize_selection_summary(run_two.selection_summary)

    pdt.assert_frame_equal(score_one, score_two)
    pdt.assert_frame_equal(dm_one, dm_two)
    pdt.assert_frame_equal(sel_one, sel_two)

    expected_score = canonicalize_ablation_scorecard(
        load_expected_frame(FIXTURE_ROOT / "expected_ablation_scorecard.csv")
    )
    expected_dm = canonicalize_dm_results(
        load_expected_frame(FIXTURE_ROOT / "expected_ablation_dm_results.csv")
    )
    expected_sel = canonicalize_selection_summary(
        load_expected_frame(FIXTURE_ROOT / "expected_ablation_selection_summary.csv")
    )

    pdt.assert_frame_equal(score_one, expected_score)
    pdt.assert_frame_equal(dm_one, expected_dm)
    pdt.assert_frame_equal(sel_one, expected_sel)

    b2_row = sel_one[sel_one["experiment_id"] == "B2_plus_feature_expansion"].iloc[0]
    assert str(b2_row["selected_model"]) == "B2_plus_feature_expansion_main"


def test_ablation_replay_pack_fails_when_required_experiment_missing() -> None:
    frame = load_ablation_replay_frame(FIXTURE_ROOT / "ablation_replay_matrix.csv")
    frame = frame[frame["experiment_id"] != "B3_plus_challenger"].reset_index(drop=True)

    config = load_yaml("configs/experiments/nowcast_ablation.yaml")
    dm_policy = load_yaml("configs/evaluation.yaml")

    with pytest.raises(ContractViolation, match="reason_code=missing_ablation_experiment"):
        run_ablation_matrix(frame, config=config, dm_policy=dm_policy)


def test_ablation_replay_scorecard_exercises_n6_gate_contracts(tmp_path: Path) -> None:
    frame = load_ablation_replay_frame(FIXTURE_ROOT / "ablation_replay_matrix.csv")
    config = load_yaml("configs/experiments/nowcast_ablation.yaml")
    dm_policy = load_yaml("configs/evaluation.yaml")

    result = run_ablation_matrix(frame, config=config, dm_policy=dm_policy)
    scorecard_path = export_ablation_scorecard(result.scorecard, tmp_path)

    with pytest.raises(
        ContractViolation,
        match="reason_code=insufficient_ablation_significance",
    ):
        check_n6_adoption_readiness(scorecard_path)

    gate = check_n6_adoption_readiness(scorecard_path, max_dm_p_value=1.0)
    assert gate.passed

    stored = pd.read_csv(scorecard_path)
    assert set(
        [
            "experiment_id",
            "mae",
            "rmse",
            "mape",
            "dm_vs_baseline_p_value",
            "runtime_seconds",
            "lineage_id",
        ]
    ).issubset(stored.columns)
