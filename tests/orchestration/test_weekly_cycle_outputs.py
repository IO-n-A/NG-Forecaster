from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import ng_forecaster.orchestration.airflow.runtime as runtime
from ng_forecaster.orchestration.airflow.dags.eia_api_ingest import run_eia_api_ingest
from ng_forecaster.orchestration.airflow.dags.eia_bulk_ingest import run_eia_bulk_ingest
from ng_forecaster.orchestration.airflow.dags.eia_metadata_refresh import (
    run_eia_metadata_refresh,
)
from ng_forecaster.orchestration.airflow.dags.nowcast_ablation_runner import (
    run_nowcast_ablation_runner,
)
from ng_forecaster.orchestration.airflow.dags.nowcast_pipeline_weekly import (
    run_nowcast_pipeline_weekly,
)
from ng_forecaster.orchestration.airflow.dags.realtime_backtest_runner import (
    run_realtime_backtest_runner,
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


def test_weekly_cycle_emits_acceptance_artifacts(monkeypatch, tmp_path: Path) -> None:
    _patch_idempotency_root(monkeypatch, tmp_path)
    _prepare_weather_gold(monkeypatch, tmp_path)
    asof = "2026-02-14"

    run_eia_metadata_refresh(asof=asof)
    run_eia_api_ingest(asof=asof)
    run_eia_bulk_ingest(asof=asof)
    run_nowcast_pipeline_weekly(asof=asof)
    run_realtime_backtest_runner(asof=asof)
    run_nowcast_ablation_runner(asof=asof)

    nowcast_dir = Path("data/artifacts/nowcast/2026-02-14")
    assert (nowcast_dir / "nowcast.json").exists()
    assert (nowcast_dir / "diagnostics.json").exists()
    assert (nowcast_dir / "lineage.json").exists()
    assert (nowcast_dir / "preprocess_summary.json").exists()
    assert (nowcast_dir / "release_history_36m.csv").exists()
    release_history = pd.read_csv(nowcast_dir / "release_history_36m.csv")
    assert len(release_history) >= 36

    nowcast_payload = json.loads(
        (nowcast_dir / "nowcast.json").read_text(encoding="utf-8")
    )
    assert nowcast_payload["target_month"] == "2025-12-31"
    assert nowcast_payload["release_policy"]["target_month"] == "2025-12-31"

    backtest_report = json.loads(
        Path("data/reports/backtest_report.json").read_text(encoding="utf-8")
    )
    assert backtest_report["target_month"] == "2025-12-31"

    assert Path("data/reports/dm_results.csv").exists()
    assert Path("data/reports/ablation_scorecard.csv").exists()
