from __future__ import annotations

from pathlib import Path

import pandas as pd

from ng_forecaster.ingest.adapters.bakerhughes_rigs import (
    build_monthly_rig_features,
    parse_bakerhughes_rig_history,
)


def test_parse_bakerhughes_rig_history_from_csv(tmp_path: Path) -> None:
    source = tmp_path / "rigs.csv"
    pd.DataFrame(
        {
            "Report Date": ["2026-01-03", "2026-01-10", "2026-01-17"],
            "Oil Rigs": [480, 482, 485],
            "Gas Rigs": [102, 103, 104],
        }
    ).to_csv(source, index=False)

    parsed = parse_bakerhughes_rig_history(source)
    assert len(parsed) == 3
    assert set(parsed.columns) >= {
        "timestamp",
        "oil_rig_count",
        "gas_rig_count",
        "available_timestamp",
    }


def test_build_monthly_rig_features_emits_oil_side_feature_rows() -> None:
    history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-12-01", periods=6, freq="W-SAT"),
            "oil_rig_count": [470, 472, 474, 476, 478, 480],
            "gas_rig_count": [98, 99, 100, 101, 102, 103],
        }
    )
    features = build_monthly_rig_features(history, asof="2026-01-20")
    assert set(features["feature_name"]) == {
        "oil_rigs_last",
        "oil_rigs_mean_4w",
        "oil_rigs_slope_4w",
        "gas_rigs_last",
        "gas_rigs_mean_4w",
        "gas_rigs_slope_4w",
    }
    assert (features["block_id"] == "oil_side").all()

