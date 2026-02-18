from __future__ import annotations

import pandas as pd

from ng_forecaster.features.regime_flags import (
    append_regime_flags,
    compute_regime_flags,
    load_regime_thresholds,
)


def test_compute_regime_flags_activates_expected_components() -> None:
    flags = compute_regime_flags(
        {
            "freeze_days_mtd_weighted": 9.0,
            "freeze_intensity_mtd_weighted": 2.0,
            "freeze_event_share_mtd_weighted": 0.35,
            "steo_regional_residential_spread_usd_mcf_t": 1.4,
            "transfer_prior_dispersion_t": 3.6,
        }
    )

    assert flags["regime_freeze_flag"] == 1.0
    assert flags["regime_basis_flag"] == 1.0
    assert flags["regime_transfer_dispersion_flag"] == 1.0
    assert flags["regime_any_flag"] == 1.0
    assert flags["regime_score"] > 1.0


def test_append_regime_flags_to_panel_rows() -> None:
    panel = pd.DataFrame(
        [
            {
                "freeze_days_mtd_weighted": 0.0,
                "freeze_intensity_mtd_weighted": 0.0,
                "freeze_event_share_mtd_weighted": 0.0,
                "steo_regional_residential_spread_usd_mcf_t": 0.2,
                "transfer_prior_dispersion_t": 0.5,
            },
            {
                "freeze_days_mtd_weighted": 10.0,
                "freeze_intensity_mtd_weighted": 2.4,
                "freeze_event_share_mtd_weighted": 0.4,
                "steo_regional_residential_spread_usd_mcf_t": 1.2,
                "transfer_prior_dispersion_t": 3.1,
            },
        ]
    )

    enriched = append_regime_flags(panel)

    assert "regime_any_flag" in enriched.columns
    assert enriched.loc[0, "regime_any_flag"] == 0.0
    assert enriched.loc[1, "regime_any_flag"] == 1.0


def test_load_regime_thresholds_from_repo_config() -> None:
    thresholds = load_regime_thresholds(config_path="configs/features.yaml")
    assert thresholds["freeze_days_high"] > 0
    assert thresholds["transfer_dispersion_high"] > 0
