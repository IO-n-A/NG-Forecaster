from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import ng_forecaster.orchestration.airflow.runtime as runtime
from ng_forecaster.orchestration.airflow.dags.realtime_backtest_runner import (
    run_realtime_backtest_runner,
)


def _patch_idempotency_root(monkeypatch, tmp_path: Path) -> None:
    def _marker(*, dag_id: str, task_id: str, key: str, root: str | Path = "") -> Path:
        _ = root
        return tmp_path / "idempotency" / dag_id / task_id / f"{key}.json"

    monkeypatch.setattr(runtime, "idempotency_marker_path", _marker)


def test_realtime_backtest_emits_penultimate_month_context(
    monkeypatch, tmp_path: Path
) -> None:
    _patch_idempotency_root(monkeypatch, tmp_path)

    result = run_realtime_backtest_runner(asof="2026-02-14")
    assert result["target_month"] == "2025-12-31"
    assert result["release_policy"]["target_month"] == "2025-12-31"

    report = json.loads(
        Path("data/reports/backtest_report.json").read_text(encoding="utf-8")
    )
    assert report["target_month"] == "2025-12-31"
    assert report["release_policy"]["target_month"] == "2025-12-31"

    scorecard = pd.read_csv(Path("data/reports/backtest_scorecard.csv"))
    assert "target_month" in scorecard.columns

    t_minus_1_month = str(
        scorecard.loc[scorecard["horizon"] == "T-1", "target_month"].iloc[0]
    )
    t_month = str(scorecard.loc[scorecard["horizon"] == "T", "target_month"].iloc[0])
    assert t_minus_1_month == "2025-12-31"
    assert t_month == "2026-01-31"
