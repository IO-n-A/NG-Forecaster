from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import ng_forecaster.orchestration.airflow.runtime as runtime
import ng_forecaster.orchestration.airflow.dags.nowcast_pipeline_weekly as pipeline_weekly
from ng_forecaster.orchestration.airflow.dags.nowcast_pipeline_weekly import (
    run_nowcast_pipeline_weekly,
)
from ng_forecaster.data.validators import load_yaml
from ng_forecaster.orchestration.airflow.workflow_support import resolve_target_month


def _patch_idempotency_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _marker(*, dag_id: str, task_id: str, key: str, root: str | Path = "") -> Path:
        _ = root
        return tmp_path / "idempotency" / dag_id / task_id / f"{key}.json"

    monkeypatch.setattr(runtime, "idempotency_marker_path", _marker)


def _prepare_weather_gold(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
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


def _prepare_transfer_priors_gold(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    gold_root = tmp_path / "gold"
    gold_root.mkdir(parents=True, exist_ok=True)
    panel = pd.DataFrame(
        [
            {
                "asof": "2026-02-28",
                "target_month": "2025-12-31",
                "horizon": 1,
                "transfer_prior_us_bcfd": 3200.0,
                "transfer_prior_dispersion": 240.0,
                "transfer_prior_basin_count": 6,
                "available_timestamp": "2026-02-14",
                "lineage_id": "fixture-transfer",
                "source_model": "tl_basin_dnn",
            },
            {
                "asof": "2026-02-28",
                "target_month": "2026-01-31",
                "horizon": 2,
                "transfer_prior_us_bcfd": 3300.0,
                "transfer_prior_dispersion": 250.0,
                "transfer_prior_basin_count": 6,
                "available_timestamp": "2026-02-14",
                "lineage_id": "fixture-transfer",
                "source_model": "tl_basin_dnn",
            },
        ]
    )
    panel.to_parquet(gold_root / "transfer_priors_panel.parquet", index=False)
    monkeypatch.setenv("NGF_GOLD_ROOT", str(gold_root))


@pytest.mark.parametrize(
    ("asof", "expected"),
    [
        ("2026-02-14", "2025-12-31"),
        ("2026-01-01", "2025-11-30"),
        ("2024-03-15", "2024-01-31"),
    ],
)
def test_resolve_target_month_penultimate_contract(asof: str, expected: str) -> None:
    resolved = resolve_target_month(asof=asof, lag_months=2)
    assert resolved.date().isoformat() == expected


def test_nowcast_pipeline_emits_target_month_contract(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    _patch_idempotency_root(monkeypatch, tmp_path)
    _prepare_weather_gold(monkeypatch, tmp_path)
    asof = "2026-02-14"

    result = run_nowcast_pipeline_weekly(asof=asof)
    assert result["target_month"] == "2025-12-31"

    nowcast_path = Path("data/artifacts/nowcast/2026-02-14/nowcast.json")
    payload = json.loads(nowcast_path.read_text(encoding="utf-8"))
    assert payload["target_month"] == "2025-12-31"
    assert payload["nowcasts"][0]["target_month"] == "2025-12-31"
    assert payload["nowcasts"][1]["target_month"] == "2026-01-31"

    report_path = Path("data/reports/nowcast_pipeline_report.json")
    report_payload = json.loads(report_path.read_text(encoding="utf-8"))
    assert report_payload["target_month"] == "2025-12-31"


def test_nowcast_pipeline_supports_champion_variant_override(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_idempotency_root(monkeypatch, tmp_path)
    _prepare_weather_gold(monkeypatch, tmp_path)
    run_nowcast_pipeline_weekly(
        asof="2026-02-14",
        champion_config_override={
            "model": {"variant": "wpd_lstm_one_layer"},
            "strategy": "wpd_lstm_one_layer",
        },
    )
    diagnostics = json.loads(
        Path("data/artifacts/nowcast/2026-02-14/model_diagnostics.json").read_text(
            encoding="utf-8"
        )
    )
    assert diagnostics["champion"]["model_variant"] == "wpd_lstm_one_layer"


def test_nowcast_pipeline_wires_transfer_priors_into_champion_and_challenger(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    _patch_idempotency_root(monkeypatch, tmp_path)
    _prepare_weather_gold(monkeypatch, tmp_path)
    _prepare_transfer_priors_gold(monkeypatch, tmp_path)

    def _patched_load_yaml(path: str):
        payload = load_yaml(path)
        if str(path).endswith("model_challenger.yaml"):
            payload["exogenous"] = {
                "transfer_priors": {
                    "enabled": True,
                    "prior_weight": 5000.0,
                    "dispersion_weight": 1000.0,
                    "prior_scale": 1000.0,
                    "dispersion_scale": 100.0,
                }
            }
        return payload

    monkeypatch.setattr(pipeline_weekly, "load_yaml", _patched_load_yaml)

    run_nowcast_pipeline_weekly(
        asof="2026-02-14",
        champion_config_override={
            "exogenous": {
                "transfer_priors": {
                    "enabled": True,
                    "prior_weight": 5000.0,
                    "dispersion_weight": 1000.0,
                    "prior_scale": 1000.0,
                    "dispersion_scale": 100.0,
                }
            }
        },
    )
    diagnostics = json.loads(
        Path("data/artifacts/nowcast/2026-02-14/model_diagnostics.json").read_text(
            encoding="utf-8"
        )
    )

    assert diagnostics["champion"]["exogenous_transfer_priors"]["enabled"] is True
    assert diagnostics["challenger"]["exogenous_transfer_priors"]["enabled"] is True
