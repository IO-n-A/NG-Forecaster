from __future__ import annotations

import numpy as np
import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.models.decomposition.vmd import (
    decompose_vmd_signal,
    select_vmd_k_by_energy_change,
)


def test_vmd_decomposition_reconstructs_signal() -> None:
    x = np.linspace(0.0, 8.0 * np.pi, 96)
    signal = np.sin(x) + 0.25 * np.sin(3.0 * x)

    result = decompose_vmd_signal(signal, k=4)
    reconstructed = result.modes.sum(axis=1).to_numpy(dtype=float)

    assert result.modes.shape[1] == 4
    assert np.allclose(reconstructed, signal, atol=1e-6)
    assert set(result.mode_energy.columns) == {"mode_name", "energy"}


def test_vmd_k_selection_returns_first_stable_theta_candidate() -> None:
    x = np.linspace(0.0, 6.0 * np.pi, 84)
    signal = np.sin(x) + 0.1 * np.cos(5.0 * x)

    selection = select_vmd_k_by_energy_change(
        signal,
        k_min=2,
        k_max=6,
        theta_threshold=0.25,
    )
    assert 2 <= selection.selected_k <= 6
    assert len(selection.search_table) == 5
    assert set(selection.search_table.columns) == {
        "k",
        "energy",
        "theta_k_k_minus_1",
    }


def test_vmd_k_selection_rejects_invalid_bounds() -> None:
    with pytest.raises(ContractViolation, match="reason_code=invalid_model_policy"):
        select_vmd_k_by_energy_change([1.0, 2.0, 3.0], k_min=3, k_max=2)
