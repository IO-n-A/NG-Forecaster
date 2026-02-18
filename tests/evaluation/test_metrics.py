from __future__ import annotations

import pandas as pd

from ng_forecaster.evaluation.metrics import score_point_forecasts


def test_score_point_forecasts_grouped_scorecard() -> None:
    frame = pd.DataFrame(
        {
            "horizon": [1, 1, 2, 2],
            "actual": [10.0, 12.0, 9.0, 11.0],
            "forecast": [10.5, 11.5, 8.5, 10.5],
        }
    )

    score = score_point_forecasts(frame, group_cols=["horizon"])
    assert set(score.columns) == {"horizon", "n_obs", "mae", "rmse", "mape"}
    assert set(score["horizon"]) == {1, 2}
