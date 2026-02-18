from __future__ import annotations

from pathlib import Path

import ng_forecaster.orchestration.airflow.dags.eia_api_ingest as api_ingest
import ng_forecaster.orchestration.airflow.dags.eia_bulk_ingest as bulk_ingest


def _empty_bootstrap_payload(tmp_path: Path) -> dict[str, object]:
    raw_dir = tmp_path / "raw"
    report_dir = tmp_path / "reports"
    raw_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return {
        "bootstrap_root": str(tmp_path),
        "raw_dir": str(raw_dir),
        "report_dir": str(report_dir),
        "raw_file_count": 0,
        "report_file_count": 0,
        "raw_files": [],
        "report_files": [],
        "available": False,
    }


def test_api_ingest_supports_missing_bootstrap_on_first_run(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        api_ingest,
        "bootstrap_status",
        lambda: _empty_bootstrap_payload(tmp_path),
    )
    monkeypatch.setenv("NGF_API_SOURCE_ROOT", str(tmp_path / "live_api"))
    monkeypatch.setenv("NGF_BULK_SOURCE_ROOT", str(tmp_path / "live_bulk"))
    monkeypatch.setenv("NGF_FIXTURE_SOURCE_ROOT", str(tmp_path / "fixture"))

    result = api_ingest.run_eia_api_ingest(asof="2030-01-01")
    payload = result["tasks"]["ingest_bootstrap_payloads"]

    assert payload["bootstrap_available"] is False
    assert payload["copied_file_count"] == 0


def test_bulk_ingest_supports_missing_bootstrap_on_first_run(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        bulk_ingest,
        "bootstrap_status",
        lambda: _empty_bootstrap_payload(tmp_path),
    )
    monkeypatch.setenv("NGF_API_SOURCE_ROOT", str(tmp_path / "live_api"))
    monkeypatch.setenv("NGF_BULK_SOURCE_ROOT", str(tmp_path / "live_bulk"))
    monkeypatch.setenv("NGF_FIXTURE_SOURCE_ROOT", str(tmp_path / "fixture"))

    result = bulk_ingest.run_eia_bulk_ingest(asof="2030-01-02")
    payload = result["tasks"]["ingest_bulk_payloads"]

    assert payload["copied_file_count"] == 0
    assert payload["dead_letter_count"] == 0
