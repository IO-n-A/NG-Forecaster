from __future__ import annotations

import numpy as np
import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.models.decomposition.wpd import decompose_wpd_signal


def test_wpd_decomposition_produces_pf_components_and_reconstructs() -> None:
    x = np.linspace(0.0, 4.0 * np.pi, 96)
    signal = np.sin(x) + 0.3 * np.cos(3 * x)

    result = decompose_wpd_signal(
        signal,
        wavelet_family="db",
        wavelet_order=5,
        levels=3,
    )

    assert list(result.components.columns) == [f"PF{idx}" for idx in range(1, 9)]
    assert len(result.node_paths) == 8
    assert result.reconstruction_relative_l2 <= 1e-4
    assert set(result.energy.columns) == {"component_name", "energy"}


def test_wpd_rejects_unsupported_wavelet_family() -> None:
    with pytest.raises(ContractViolation, match="reason_code=invalid_model_policy"):
        decompose_wpd_signal([1.0, 2.0, 3.0, 4.0], wavelet_family="sym")
