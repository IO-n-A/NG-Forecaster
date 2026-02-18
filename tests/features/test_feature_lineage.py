from __future__ import annotations

from pathlib import Path

import pandas as pd

from ng_forecaster.features.vintage_builder import build_vintage_panel
from ng_forecaster.reporting.exporters import export_feature_lineage


def _feature_policy() -> dict[str, object]:
    return {
        "version": 1,
        "default": {"max_age_days": 30},
        "features": {
            "hh_last": {
                "source_frequency": "daily",
                "aggregation": "last",
                "max_age_days": 7,
            },
            "stor_last": {
                "source_frequency": "weekly",
                "aggregation": "last",
                "max_age_days": 14,
            },
        },
    }


def _features() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "feature_name": ["hh_last", "hh_last", "stor_last", "stor_last"],
            "feature_timestamp": ["2024-01-09", "2024-01-10", "2024-01-08", "2024-01-09"],
            "value": [2.9, 3.0, 104.0, 105.0],
        }
    )


def _target() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "target_timestamp": ["2024-01-08", "2024-01-09"],
            "target_value": [51.0, 52.0],
        }
    )


def test_lineage_is_reproducible_for_identical_inputs(tmp_path: Path) -> None:
    a = build_vintage_panel(_features(), _target(), asof="2024-01-10", feature_policy=_feature_policy())
    b = build_vintage_panel(_features(), _target(), asof="2024-01-10", feature_policy=_feature_policy())

    assert a.lineage == b.lineage

    output = export_feature_lineage(a.lineage, tmp_path)
    assert output.exists()
    exported = pd.read_csv(output)
    assert "lineage_id" in exported.columns


def test_lineage_changes_when_feature_payload_changes() -> None:
    base = _features()
    changed = base.copy()
    changed.loc[changed["feature_name"] == "hh_last", "value"] = 3.2

    first = build_vintage_panel(base, _target(), asof="2024-01-10", feature_policy=_feature_policy())
    second = build_vintage_panel(changed, _target(), asof="2024-01-10", feature_policy=_feature_policy())

    assert first.lineage["T"] != second.lineage["T"]
