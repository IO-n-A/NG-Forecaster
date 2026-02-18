from __future__ import annotations

import pandas as pd

from ng_forecaster.evaluation.block_importance import score_block_ablations


def test_score_block_ablations_marks_inert_blocks() -> None:
    asof_grid = pd.date_range("2025-01-31", periods=6, freq="ME")
    rows: list[dict[str, object]] = []
    for idx, asof in enumerate(asof_grid):
        actual = 100.0 + idx
        baseline = 101.0 + idx
        rows.append(
            {
                "experiment_id": "baseline_full",
                "target": "ng_prod",
                "asof": asof.date().isoformat(),
                "horizon": 1,
                "actual": actual,
                "forecast": baseline,
                "runtime_seconds": 10.0,
                "lineage_id": f"baseline-{idx}",
                "block_id": "baseline_full",
                "ablation_mode": "none",
            }
        )
        rows.append(
            {
                "experiment_id": "block_drop::inert",
                "target": "ng_prod",
                "asof": asof.date().isoformat(),
                "horizon": 1,
                "actual": actual,
                "forecast": baseline,
                "runtime_seconds": 11.0,
                "lineage_id": f"inert-{idx}",
                "block_id": "inert",
                "ablation_mode": "block_drop",
            }
        )
        rows.append(
            {
                "experiment_id": "block_drop::active",
                "target": "ng_prod",
                "asof": asof.date().isoformat(),
                "horizon": 1,
                "actual": actual,
                "forecast": baseline + 2.0,
                "runtime_seconds": 12.0,
                "lineage_id": f"active-{idx}",
                "block_id": "active",
                "ablation_mode": "block_drop",
            }
        )

    scorecard, _ = score_block_ablations(pd.DataFrame(rows))
    inert = scorecard[scorecard["experiment_id"] == "block_drop::inert"].iloc[0]
    active = scorecard[scorecard["experiment_id"] == "block_drop::active"].iloc[0]
    baseline = scorecard[scorecard["experiment_id"] == "baseline_full"].iloc[0]

    assert float(inert["max_abs_delta_point"]) == 0.0
    assert int(inert["block_inert"]) == 1

    assert float(active["max_abs_delta_point"]) == 2.0
    assert int(active["block_inert"]) == 0

    assert float(baseline["max_abs_delta_point"]) == 0.0
    assert int(baseline["block_inert"]) == 0
