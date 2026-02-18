from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml

import ng_forecaster.orchestration.airflow.runtime as runtime
from ng_forecaster.errors import ContractViolation
from ng_forecaster.data.validators import load_yaml
from ng_forecaster.models.champion_wpd_vmd_lstm import run_champion_pipeline
from ng_forecaster.orchestration.airflow.dags.nowcast_pipeline_weekly import (
    run_nowcast_pipeline_weekly,
)
from ng_forecaster.orchestration.airflow.workflow_support import enforce_release_policy
from ng_forecaster.orchestration.airflow.workflow_support import load_market_inputs


def _patch_idempotency_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _marker(*, dag_id: str, task_id: str, key: str, root: str | Path = "") -> Path:
        _ = root
        return tmp_path / "idempotency" / dag_id / task_id / f"{key}.json"

    monkeypatch.setattr(runtime, "idempotency_marker_path", _marker)


def test_release_policy_blocks_when_target_month_is_already_released() -> None:
    with pytest.raises(ContractViolation, match="reason_code=lag_policy_violated"):
        enforce_release_policy("2026-03-31")


def test_release_policy_blocks_runs_outside_admissible_day_window(
    tmp_path: Path,
) -> None:
    payload = yaml.safe_load(Path("configs/sources.yaml").read_text(encoding="utf-8"))
    payload["release_calendar"]["admissible_day_window"] = {
        "start_day": 15,
        "end_day": 20,
    }
    custom = tmp_path / "sources_custom.yaml"
    custom.write_text(yaml.safe_dump(payload, sort_keys=True), encoding="utf-8")

    with pytest.raises(
        ContractViolation, match="reason_code=run_outside_release_window"
    ):
        enforce_release_policy("2026-02-14", catalog_path=custom)


def test_nowcast_pipeline_is_blocked_when_lag_policy_is_violated(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_idempotency_root(monkeypatch, tmp_path)

    with pytest.raises(ContractViolation, match="reason_code=lag_policy_violated"):
        run_nowcast_pipeline_weekly(asof="2026-03-31")


def test_released_only_training_cutoff_differs_by_asof() -> None:
    jan_inputs = load_market_inputs(pd.Timestamp("2026-01-14"))
    feb_inputs = load_market_inputs(pd.Timestamp("2026-02-14"))

    jan_max = pd.Timestamp(jan_inputs["target_history"]["timestamp"].max())
    feb_max = pd.Timestamp(feb_inputs["target_history"]["timestamp"].max())
    jan_full_max = pd.Timestamp(jan_inputs["target_history_full"]["timestamp"].max())
    jan_latest_released = pd.Timestamp(jan_inputs["latest_released_month"])

    assert jan_max == pd.Timestamp("2025-10-31")
    assert feb_max == pd.Timestamp("2025-11-30")
    assert jan_max < feb_max
    assert jan_full_max == pd.Timestamp("2025-11-30")
    assert jan_max <= jan_latest_released
    assert len(jan_inputs["target_history"]) < len(jan_inputs["target_history_full"])


def test_released_only_contract_changes_champion_forecast_by_asof() -> None:
    config = load_yaml("configs/model_champion.yaml")
    jan_inputs = load_market_inputs(pd.Timestamp("2026-01-14"))
    feb_inputs = load_market_inputs(pd.Timestamp("2026-02-14"))

    jan_run = run_champion_pipeline(
        jan_inputs["target_history"],
        config,
        timestamp_col="timestamp",
        target_col="target_value",
    )
    feb_run = run_champion_pipeline(
        feb_inputs["target_history"],
        config,
        timestamp_col="timestamp",
        target_col="target_value",
    )

    assert not jan_run.point_forecast["point_forecast"].equals(
        feb_run.point_forecast["point_forecast"]
    )
