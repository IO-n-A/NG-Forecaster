from __future__ import annotations

from pathlib import Path

import importlib.util
import numpy as np
import pandas as pd
import pytest

from ng_forecaster.models.neural.train_lstm import forecast_component_with_pytorch_lstm


@pytest.mark.skipif(
    importlib.util.find_spec("torch") is None,
    reason="torch not installed",
)
def test_forecast_component_with_pytorch_lstm_emits_artifacts(tmp_path: Path) -> None:
    values = pd.Series(100.0 + np.linspace(0.0, 8.0, 64) + np.sin(np.linspace(0, 6, 64)))
    result = forecast_component_with_pytorch_lstm(
        values,
        horizons=[1, 2],
        lstm_config={
            "lookback": 24,
            "hidden_units": 8,
            "batch_size": 8,
            "learning_rate": 0.01,
            "repeat_runs": 2,
            "dropout": 0.1,
            "max_epochs": 20,
            "early_stopping_patience": 5,
            "val_fraction": 0.2,
            "min_delta": 1e-4,
        },
        component_name="mode_test",
        base_seed=17,
        artifact_dir=tmp_path,
    )

    assert len(result.point_forecast) == 2
    assert set(result.point_forecast["horizon"]) == {1, 2}
    assert len(result.run_forecasts) == 4
    assert Path(result.diagnostics["model_path"]).exists()
    assert Path(result.diagnostics["manifest_path"]).exists()

