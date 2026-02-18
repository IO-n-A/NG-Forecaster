from __future__ import annotations

import pandas as pd
import pytest

from ng_forecaster.data.validators import (
    validate_weather_coverage,
    validate_weather_lineage,
)
from ng_forecaster.errors import ContractViolation


def _weather_panel() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "basin_id": ["appalachia"],
            "timestamp": [pd.Timestamp("2025-01-31")],
            "coverage_fraction": [0.9],
            "source_id": ["nasa_power_t2m_min"],
            "lineage_id": ["lineage-1"],
            "available_timestamp": [pd.Timestamp("2025-01-31")],
        }
    )


def test_validate_weather_coverage_and_lineage_passes() -> None:
    panel = _weather_panel()
    validate_weather_coverage(panel, min_coverage=0.8)
    validate_weather_lineage(panel)


def test_validate_weather_coverage_fails_when_insufficient() -> None:
    panel = _weather_panel()
    panel.loc[:, "coverage_fraction"] = 0.4
    with pytest.raises(
        ContractViolation, match="reason_code=insufficient_source_coverage"
    ):
        validate_weather_coverage(panel, min_coverage=0.8)


def test_validate_weather_lineage_fails_when_missing_metadata() -> None:
    panel = _weather_panel()
    panel.loc[:, "source_id"] = ""
    with pytest.raises(ContractViolation, match="reason_code=missing_lineage_metadata"):
        validate_weather_lineage(panel)
