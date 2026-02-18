from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.reporting.exporters import export_model_diagnostics


def test_export_model_diagnostics_writes_required_files(tmp_path: Path) -> None:
    diagnostics = {"model_family": "wpd_vmd_lstm", "seed": 42, "lookback": 36}
    stability = pd.DataFrame(
        {
            "horizon": [1, 2],
            "mean": [100.0, 101.0],
            "std": [0.1, 0.2],
            "min": [99.9, 100.8],
            "max": [100.2, 101.3],
            "spread": [0.3, 0.5],
        }
    )

    paths = export_model_diagnostics(diagnostics, stability, tmp_path)
    assert paths["model_diagnostics"].exists()
    assert paths["seed_stability_summary"].exists()


def test_export_model_diagnostics_validates_payload() -> None:
    with pytest.raises(
        ContractViolation, match="reason_code=invalid_diagnostics_payload"
    ):
        export_model_diagnostics({}, pd.DataFrame(), "/tmp")
