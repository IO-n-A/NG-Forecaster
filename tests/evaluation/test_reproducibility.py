from __future__ import annotations

import pandas as pd

from ng_forecaster.evaluation.reproducibility import (
    resolve_seed_schedule,
    summarize_trial_scorecard,
)


def test_resolve_seed_schedule_extends_when_schedule_is_short() -> None:
    seeds = resolve_seed_schedule(n_trials=5, seed_schedule="7,9")
    assert seeds == [7, 9, 10, 11, 12]


def test_summarize_trial_scorecard_emits_mean_std_columns() -> None:
    frame = pd.DataFrame(
        [
            {
                "model_variant": "wpd_vmd_lstm1",
                "trial_id": 1,
                "mae": 10.0,
                "rmse": 11.0,
                "mape_pct": 1.2,
                "mean_signed_error": 0.4,
                "mean_abs_error": 10.0,
                "interval_hit_rate_95": 1.0,
            },
            {
                "model_variant": "wpd_vmd_lstm1",
                "trial_id": 2,
                "mae": 10.5,
                "rmse": 11.2,
                "mape_pct": 1.1,
                "mean_signed_error": 0.2,
                "mean_abs_error": 10.5,
                "interval_hit_rate_95": 1.0,
            },
        ]
    )

    summary = summarize_trial_scorecard(frame, std_tolerance=2.0)
    assert len(summary) == 1
    assert summary.iloc[0]["n_trials"] == 2
    assert float(summary.iloc[0]["mae_mean"]) == 10.25
    assert float(summary.iloc[0]["mae_std"]) > 0.0
    assert bool(summary.iloc[0]["stability_flag"])
