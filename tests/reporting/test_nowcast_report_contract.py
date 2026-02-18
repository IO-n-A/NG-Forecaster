from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import ng_forecaster.orchestration.airflow.runtime as runtime
from ng_forecaster.orchestration.airflow.dags.nowcast_pipeline_weekly import (
    run_nowcast_pipeline_weekly,
)


def _patch_idempotency_root(monkeypatch, tmp_path: Path) -> None:
    def _marker(*, dag_id: str, task_id: str, key: str, root: str | Path = "") -> Path:
        _ = root
        return tmp_path / "idempotency" / dag_id / task_id / f"{key}.json"

    monkeypatch.setattr(runtime, "idempotency_marker_path", _marker)


def _prepare_weather_gold(monkeypatch, tmp_path: Path) -> None:
    gold_root = tmp_path / "gold"
    gold_root.mkdir(parents=True, exist_ok=True)
    weather = pd.DataFrame(
        {
            "basin_id": ["appalachia"],
            "basin_name": ["Appalachia"],
            "timestamp": [pd.Timestamp("2026-02-28")],
            "available_timestamp": [pd.Timestamp("2026-02-14")],
            "freeze_days": [6],
            "freeze_event_share": [0.3],
            "freeze_intensity_c": [2.2],
            "coverage_fraction": [1.0],
            "source_id": ["nasa_power_t2m_min"],
            "lineage_id": ["fixture-weather"],
        }
    )
    weather.to_parquet(gold_root / "weather_freezeoff_panel.parquet", index=False)
    monkeypatch.setenv("NGF_GOLD_ROOT", str(gold_root))


def test_nowcast_json_preserves_interval_and_target_month_label_contract(
    monkeypatch,
    tmp_path: Path,
) -> None:
    _patch_idempotency_root(monkeypatch, tmp_path)
    _prepare_weather_gold(monkeypatch, tmp_path)
    run_nowcast_pipeline_weekly(asof="2026-02-14")

    payload = json.loads(
        Path("data/artifacts/nowcast/2026-02-14/nowcast.json").read_text(
            encoding="utf-8"
        )
    )
    rows = payload["nowcasts"]
    assert rows
    assert [str(row["horizon_label"]) for row in rows] == ["T-1", "T"]

    root_target_month = pd.Timestamp(payload["target_month"])
    expected_target_month_by_horizon = {
        "T-1": root_target_month.to_period("M").to_timestamp("M"),
        "T": (root_target_month.to_period("M") + 1).to_timestamp("M"),
    }
    expected_target_label_by_horizon = {
        "T-1": "target_month",
        "T": "target_month_plus_1",
    }

    for row in rows:
        horizon_label = str(row["horizon_label"])
        ts = pd.Timestamp(row["target_month"])
        assert ts == ts.to_period("M").to_timestamp("M")
        assert ts == expected_target_month_by_horizon[horizon_label]
        assert (
            str(row["target_month_label"])
            == expected_target_label_by_horizon[horizon_label]
        )

        low = float(row["fused_lower_95"])
        point = float(row["fused_point"])
        high = float(row["fused_upper_95"])
        assert low <= point <= high
