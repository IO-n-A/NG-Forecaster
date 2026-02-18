from __future__ import annotations

from ng_forecaster.models.regime import classify_regime


def test_classify_regime_returns_normal_when_no_flags_active() -> None:
    regime = classify_regime(
        {
            "regime_freeze_flag": 0.0,
            "regime_basis_flag": 0.0,
            "regime_transfer_dispersion_flag": 0.0,
        }
    )
    assert regime == "normal"


def test_classify_regime_returns_multi_shock_when_multiple_flags_active() -> None:
    regime = classify_regime(
        {
            "regime_freeze_flag": 1.0,
            "regime_basis_flag": 1.0,
            "regime_transfer_dispersion_flag": 0.0,
        }
    )
    assert regime == "multi_shock"
