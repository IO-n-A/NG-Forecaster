from __future__ import annotations

import pandas as pd

from ng_forecaster.features.transfer_priors import (
    build_transfer_prior_feature_rows,
    upsert_transfer_priors_panel,
)


def test_transfer_priors_upsert_and_feature_rows(tmp_path) -> None:
    gold_root = tmp_path / "gold"
    panel = pd.DataFrame(
        [
            {
                "asof": "2026-02-28",
                "target_month": "2025-12-31",
                "horizon": 1,
                "transfer_prior_us_bcfd": 19.5,
                "transfer_prior_dispersion": 2.1,
                "transfer_prior_basin_count": 6,
                "available_timestamp": "2026-02-14",
                "lineage_id": "lineage-1",
                "source_model": "tl_basin_dnn",
            },
            {
                "asof": "2026-02-28",
                "target_month": "2026-01-31",
                "horizon": 2,
                "transfer_prior_us_bcfd": 20.2,
                "transfer_prior_dispersion": 2.3,
                "transfer_prior_basin_count": 6,
                "available_timestamp": "2026-02-14",
                "lineage_id": "lineage-1",
                "source_model": "tl_basin_dnn",
            },
        ]
    )
    upsert_transfer_priors_panel(panel, gold_root=gold_root)

    features, meta = build_transfer_prior_feature_rows(
        asof="2026-02-14",
        target_month="2025-12-31",
        gold_root=gold_root,
    )

    assert meta["status"] == "transfer_prior_features_loaded"
    assert len(features) == 6
    assert set(features["feature_name"]) == {
        "transfer_prior_us_bcfd_t",
        "transfer_prior_us_bcfd_t_plus_1",
        "transfer_prior_dispersion_t",
        "transfer_prior_dispersion_t_plus_1",
        "transfer_prior_basin_count_t",
        "transfer_prior_basin_count_t_plus_1",
    }


def test_transfer_priors_feature_rows_not_ready_when_panel_missing(tmp_path) -> None:
    features, meta = build_transfer_prior_feature_rows(
        asof="2026-02-14",
        target_month="2025-12-31",
        gold_root=tmp_path / "gold",
    )

    assert features.empty
    assert meta["status"] == "transfer_priors_not_ready"
