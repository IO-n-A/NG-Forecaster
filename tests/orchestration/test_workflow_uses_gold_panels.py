from __future__ import annotations

import pandas as pd
import pytest

import ng_forecaster.orchestration.airflow.workflow_support as workflow_support
from ng_forecaster.errors import ContractViolation


def test_load_steo_gold_feature_rows_reads_latest_vintage(tmp_path) -> None:
    gold_root = tmp_path / "gold"
    gold_root.mkdir(parents=True, exist_ok=True)

    observation = pd.DataFrame(
        {
            "vintage_month": ["2026-01-31", "2026-01-31", "2026-02-28", "2026-02-28"],
            "timestamp": ["2025-12-31", "2026-01-31", "2025-12-31", "2026-01-31"],
            "is_forecast": [True, True, True, True],
            "dry_prod_bcfd": [100.0, 101.0, 102.0, 103.0],
            "marketed_prod_bcfd": [110.0, 111.0, 112.0, 113.0],
            "dry_to_marketed_ratio": [0.91, 0.91, 0.91, 0.91],
            "extraction_loss_ratio": [0.09, 0.09, 0.09, 0.09],
        }
    )
    driver = pd.DataFrame(
        {
            "vintage_month": ["2026-02-28"],
            "timestamp": ["2025-12-31"],
            "active_rigs": [500.0],
            "new_wells_drilled": [900.0],
            "new_wells_completed": [880.0],
            "duc_inventory": [1200.0],
            "new_well_gas_production": [65.0],
            "existing_gas_production_change": [-1.5],
        }
    )
    shale = pd.DataFrame(
        {
            "vintage_month": ["2026-02-28", "2026-02-28"],
            "timestamp": ["2025-12-31", "2026-01-31"],
            "is_forecast": [True, True],
            "shale_dry_prod_bcfd": [80.0, 81.0],
            "shale_component_sum_bcfd": [79.5, 80.5],
            "dry_prod_bcfd": [102.0, 103.0],
            "non_shale_dry_prod_bcfd": [22.0, 22.0],
            "shale_share_of_dry": [0.7843, 0.7864],
        }
    )

    observation.to_parquet(gold_root / "steo_observation_panel.parquet", index=False)
    driver.to_parquet(gold_root / "steo_driver_panel.parquet", index=False)
    shale.to_parquet(gold_root / "steo_shale_split_panel.parquet", index=False)

    features, meta = workflow_support.load_steo_gold_feature_rows(
        asof=pd.Timestamp("2026-02-14"),
        target_month=pd.Timestamp("2025-12-31"),
        gold_root=gold_root,
    )

    assert not features.empty
    assert meta["status"] == "gold_features_loaded"
    assert meta["latest_vintage_month"] == "2026-02-28"
    assert features["feature_name"].str.startswith("steo_").all()


def test_load_market_inputs_wires_gold_features(monkeypatch) -> None:
    def _fake_gold_features(*, asof, target_month, gold_root=None):
        _ = (asof, target_month, gold_root)
        frame = pd.DataFrame(
            {
                "feature_name": ["steo_driver_active_rigs"],
                "feature_timestamp": [pd.Timestamp("2025-12-31")],
                "available_timestamp": [pd.Timestamp("2026-02-05")],
                "block_id": ["steo_driver"],
                "value": [500.0],
            }
        )
        return frame, {
            "status": "gold_features_loaded",
            "latest_vintage_month": "2026-02-28",
            "feature_row_count": 1,
            "gold_root": "data/gold",
        }

    monkeypatch.setattr(
        workflow_support,
        "load_steo_gold_feature_rows",
        _fake_gold_features,
    )

    payload = workflow_support.load_market_inputs(pd.Timestamp("2026-02-14"))

    assert payload["steo_gold_meta"]["status"] == "gold_features_loaded"
    assert "+gold_steo" in payload["feature_source"]
    assert "steo_driver_active_rigs" in payload["feature_policy"]["features"]


def test_load_market_inputs_wires_transfer_prior_features(monkeypatch) -> None:
    def _fake_transfer_features(*, asof, target_month, gold_root="data/gold"):
        _ = (asof, target_month, gold_root)
        frame = pd.DataFrame(
            {
                "feature_name": [
                    "transfer_prior_us_bcfd_t",
                    "transfer_prior_dispersion_t",
                    "transfer_prior_basin_count_t",
                    "transfer_prior_us_bcfd_t_plus_1",
                    "transfer_prior_dispersion_t_plus_1",
                    "transfer_prior_basin_count_t_plus_1",
                ],
                "feature_timestamp": [pd.Timestamp("2025-12-31")] * 3
                + [pd.Timestamp("2026-01-31")] * 3,
                "available_timestamp": [pd.Timestamp("2026-02-14")] * 6,
                "block_id": ["transfer_priors"] * 6,
                "value": [20.0, 2.4, 6.0, 20.4, 2.6, 6.0],
            }
        )
        return frame, {
            "status": "transfer_prior_features_loaded",
            "feature_row_count": 6,
            "selected_asof": "2026-02-28",
        }

    monkeypatch.setattr(
        workflow_support,
        "build_transfer_prior_feature_rows",
        _fake_transfer_features,
    )

    payload = workflow_support.load_market_inputs(pd.Timestamp("2026-02-14"))

    assert payload["transfer_priors_meta"]["status"] == "transfer_prior_features_loaded"
    assert "transfer_prior_us_bcfd_t" in payload["feature_policy"]["features"]
    assert "+transfer_priors" in payload["feature_source"]


def test_load_market_inputs_wires_bakerhughes_oil_side_features(monkeypatch) -> None:
    def _fake_bakerhughes_features(*, asof, gold_root=None):
        _ = (asof, gold_root)
        frame = pd.DataFrame(
            {
                "feature_name": [
                    "oil_rigs_last",
                    "oil_rigs_mean_4w",
                    "oil_rigs_slope_4w",
                    "gas_rigs_last",
                    "gas_rigs_mean_4w",
                    "gas_rigs_slope_4w",
                ],
                "feature_timestamp": [pd.Timestamp("2025-12-31")] * 6,
                "available_timestamp": [pd.Timestamp("2026-02-14")] * 6,
                "block_id": ["oil_side"] * 6,
                "value": [500, 495, 2.2, 100, 98, 1.1],
            }
        )
        return frame, {
            "status": "bakerhughes_features_loaded",
            "feature_row_count": 6,
        }

    monkeypatch.setattr(
        workflow_support,
        "load_bakerhughes_gold_feature_rows",
        _fake_bakerhughes_features,
    )

    payload = workflow_support.load_market_inputs(pd.Timestamp("2026-02-14"))

    assert payload["bakerhughes_meta"]["status"] == "bakerhughes_features_loaded"
    assert "+oil_side" in payload["feature_source"]
    assert "oil_rigs_last" in payload["feature_policy"]["features"]


def test_load_steo_gold_feature_rows_allows_pre_10ab_observation_only(
    tmp_path,
) -> None:
    gold_root = tmp_path / "gold"
    gold_root.mkdir(parents=True, exist_ok=True)

    observation = pd.DataFrame(
        {
            "vintage_month": ["2024-02-29", "2024-02-29"],
            "timestamp": ["2023-12-31", "2024-01-31"],
            "is_forecast": [True, True],
            "dry_prod_bcfd": [100.0, 101.0],
            "marketed_prod_bcfd": [110.0, 111.0],
            "dry_to_marketed_ratio": [0.9091, 0.9099],
            "extraction_loss_ratio": [0.0909, 0.0901],
        }
    )
    observation.to_parquet(gold_root / "steo_observation_panel.parquet", index=False)
    pd.DataFrame(
        columns=[
            "vintage_month",
            "timestamp",
            "active_rigs",
            "new_wells_drilled",
        ]
    ).to_parquet(gold_root / "steo_driver_panel.parquet", index=False)
    pd.DataFrame(
        columns=[
            "vintage_month",
            "timestamp",
            "shale_dry_prod_bcfd",
            "shale_share_of_dry",
            "non_shale_dry_prod_bcfd",
        ]
    ).to_parquet(gold_root / "steo_shale_split_panel.parquet", index=False)

    features, meta = workflow_support.load_steo_gold_feature_rows(
        asof=pd.Timestamp("2024-02-14"),
        target_month=pd.Timestamp("2023-12-31"),
        gold_root=gold_root,
    )

    assert not features.empty
    assert meta["status"] == "gold_features_observation_only_pre_10ab"
    assert meta["driver_panel_available"] is False
    assert meta["shale_panel_available"] is False
    assert not features["feature_name"].str.contains("driver_").any()
    assert not features["feature_name"].str.contains("shale_").any()


def test_load_steo_gold_feature_rows_fails_post_10ab_when_panels_missing(
    tmp_path,
) -> None:
    gold_root = tmp_path / "gold"
    gold_root.mkdir(parents=True, exist_ok=True)

    observation = pd.DataFrame(
        {
            "vintage_month": ["2024-07-31", "2024-07-31"],
            "timestamp": ["2024-05-31", "2024-06-30"],
            "is_forecast": [True, True],
            "dry_prod_bcfd": [100.0, 101.0],
            "marketed_prod_bcfd": [110.0, 111.0],
            "dry_to_marketed_ratio": [0.9091, 0.9099],
            "extraction_loss_ratio": [0.0909, 0.0901],
        }
    )
    observation.to_parquet(gold_root / "steo_observation_panel.parquet", index=False)
    pd.DataFrame(
        columns=[
            "vintage_month",
            "timestamp",
            "active_rigs",
            "new_wells_drilled",
        ]
    ).to_parquet(gold_root / "steo_driver_panel.parquet", index=False)
    pd.DataFrame(
        columns=[
            "vintage_month",
            "timestamp",
            "shale_dry_prod_bcfd",
            "shale_share_of_dry",
            "non_shale_dry_prod_bcfd",
        ]
    ).to_parquet(gold_root / "steo_shale_split_panel.parquet", index=False)

    with pytest.raises(ContractViolation, match="source_schema_drift"):
        workflow_support.load_steo_gold_feature_rows(
            asof=pd.Timestamp("2024-07-14"),
            target_month=pd.Timestamp("2024-05-31"),
            gold_root=gold_root,
        )


def test_load_weather_gold_feature_rows_builds_weighted_features(tmp_path) -> None:
    gold_root = tmp_path / "gold"
    gold_root.mkdir(parents=True, exist_ok=True)
    panel = pd.DataFrame(
        {
            "basin_id": ["appalachia", "permian"],
            "basin_name": ["Appalachia", "Permian"],
            "timestamp": [pd.Timestamp("2026-02-28"), pd.Timestamp("2026-02-28")],
            "available_timestamp": [
                pd.Timestamp("2026-02-14"),
                pd.Timestamp("2026-02-14"),
            ],
            "freeze_days": [10, 2],
            "freeze_event_share": [0.5, 0.1],
            "freeze_intensity_c": [3.0, 1.0],
            "coverage_fraction": [1.0, 0.5],
            "source_id": ["nasa_power_t2m_min", "nasa_power_t2m_min"],
            "lineage_id": ["a", "b"],
        }
    )
    panel.to_parquet(gold_root / "weather_freezeoff_panel.parquet", index=False)

    features, meta = workflow_support.load_weather_gold_feature_rows(
        asof=pd.Timestamp("2026-02-14"),
        gold_root=gold_root,
    )

    assert meta["status"] == "weather_features_loaded"
    assert set(features["feature_name"]) == {
        "coverage_fraction_mtd",
        "extreme_min_mtd",
        "freeze_days_mtd",
        "freeze_days_mtd_weighted",
        "freeze_event_flag",
        "freeze_event_intensity",
        "freeze_event_share_mtd_weighted",
        "freeze_intensity_mtd_weighted",
    }
