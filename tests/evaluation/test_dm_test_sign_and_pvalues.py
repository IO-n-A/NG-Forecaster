from __future__ import annotations

import pandas as pd

from ng_forecaster.evaluation.dm_test import run_dm_tests


def _dm_frame() -> pd.DataFrame:
    asof = pd.date_range("2025-01-31", periods=12, freq="ME")
    actual = [100.0 + idx * 0.5 for idx in range(12)]
    rows: list[dict[str, object]] = []
    for idx, (ts, y) in enumerate(zip(asof, actual)):
        baseline_forecast = y + (1.5 if idx % 2 == 0 else 2.5)
        better_forecast = y + (0.2 if idx % 3 == 0 else 0.8)
        worse_forecast = y + (3.0 if idx % 2 == 0 else 4.0)
        rows.extend(
            [
                {
                    "target": "ng_prod",
                    "model": "baseline",
                    "asof": ts,
                    "horizon": 1,
                    "actual": y,
                    "forecast": baseline_forecast,
                },
                {
                    "target": "ng_prod",
                    "model": "candidate_better",
                    "asof": ts,
                    "horizon": 1,
                    "actual": y,
                    "forecast": better_forecast,
                },
                {
                    "target": "ng_prod",
                    "model": "candidate_worse",
                    "asof": ts,
                    "horizon": 1,
                    "actual": y,
                    "forecast": worse_forecast,
                },
            ]
        )
    return pd.DataFrame(rows)


def test_dm_sign_and_one_sided_improve_pvalues_match_direction() -> None:
    result = run_dm_tests(
        _dm_frame(),
        {
            "benchmark_by_target": {"ng_prod": "baseline"},
            "sidedness": "two_sided",
        },
    ).results
    better = result[result["candidate_model"] == "candidate_better"].iloc[0]
    worse = result[result["candidate_model"] == "candidate_worse"].iloc[0]
    assert float(better["loss_diff_mean"]) < 0
    assert float(worse["loss_diff_mean"]) > 0
    assert float(better["dm_p_value_one_sided_improve"]) < 0.10
    assert float(worse["dm_p_value_one_sided_improve"]) > 0.90
    assert 0.0 <= float(better["dm_p_value_two_sided"]) <= 1.0
    assert 0.0 <= float(worse["dm_p_value_two_sided"]) <= 1.0


def test_dm_configured_less_sidedness_matches_one_sided_improve_column() -> None:
    result = run_dm_tests(
        _dm_frame(),
        {
            "benchmark_by_target": {"ng_prod": "baseline"},
            "sidedness": "less",
            "multiple_comparison": "none",
        },
    ).results
    assert (
        result["p_value"].round(12) == result["dm_p_value_one_sided_improve"].round(12)
    ).all()
