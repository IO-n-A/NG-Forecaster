from __future__ import annotations

import json
from pathlib import Path

import pytest

import ng_forecaster.orchestration.airflow.dags.eia_api_ingest as api_ingest
import ng_forecaster.orchestration.airflow.dags.eia_bulk_ingest as bulk_ingest
import ng_forecaster.orchestration.airflow.runtime as runtime
from ng_forecaster.orchestration.airflow.runtime import (
    RetryPolicy,
    TransientHTTPError,
    run_with_retries,
)
from ng_forecaster.orchestration.airflow.workflow_support import sha256_file
from tests.helpers.orchestration_fixture_harness import (
    build_bootstrap_status_payload,
    load_retry_backoff_cases,
    stage_bootstrap_raw_fixture,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "orchestration"


def _patch_idempotency_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    def _marker(*, dag_id: str, task_id: str, key: str, root: str | Path = "") -> Path:
        _ = root
        return tmp_path / "idempotency" / dag_id / task_id / f"{key}.json"

    monkeypatch.setattr(runtime, "idempotency_marker_path", _marker)


@pytest.mark.parametrize(
    "case",
    load_retry_backoff_cases(FIXTURE_ROOT / "retry_backoff_matrix.csv"),
    ids=lambda item: item.case_id,
)
def test_retry_backoff_matrix_contract(case) -> None:
    statuses = list(case.status_sequence)
    attempts = 0
    idx = 0
    sleep_calls: list[float] = []

    def _sleep(delay: float) -> None:
        sleep_calls.append(round(float(delay), 6))

    def _runner() -> dict[str, int]:
        nonlocal attempts, idx
        attempts += 1
        status = statuses[min(idx, len(statuses) - 1)]
        idx += 1
        if status >= 400:
            raise TransientHTTPError(
                status_code=status,
                detail=f"fixture_case={case.case_id}",
            )
        return {"status_code": status}

    policy = RetryPolicy(
        max_attempts=case.max_attempts,
        initial_delay_seconds=case.initial_delay_seconds,
        backoff_multiplier=case.backoff_multiplier,
    )

    if case.expected_status == "pass":
        result = run_with_retries(_runner, retry_policy=policy, sleep=_sleep)
        assert int(result["status_code"]) == 200
    else:
        with pytest.raises(TransientHTTPError):
            run_with_retries(_runner, retry_policy=policy, sleep=_sleep)

    assert attempts == case.expected_attempts
    assert sleep_calls == [round(value, 6) for value in case.expected_sleep_sequence]


def test_api_ingest_manifest_integrity_with_fixture_payloads(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "bootstrap" / "raw"
    report_dir = tmp_path / "bootstrap" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    stage_bootstrap_raw_fixture(FIXTURE_ROOT / "bootstrap_raw", raw_dir)
    status_payload = build_bootstrap_status_payload(raw_dir, report_dir)

    monkeypatch.setattr(api_ingest, "bootstrap_status", lambda: status_payload)
    monkeypatch.setenv("EIA_API_KEY", "EIA_TEST_KEY_123456")
    _patch_idempotency_root(monkeypatch, tmp_path)

    result = api_ingest.run_eia_api_ingest(asof="2032-01-05")
    manifest_path = Path(result["tasks"]["publish_ingest_manifest"]["manifest_path"])

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert payload["bootstrap_available"] is True
    assert payload["api_secret_redacted"] != "EIA_TEST_KEY_123456"

    target_root = Path(payload["target_root"])
    listed_entries = list(payload["files"])
    source_ids = [str(entry["source_id"]) for entry in listed_entries]
    assert source_ids == sorted(source_ids)
    assert set(source_ids) == {"weekly_meta", "weekly_prices"}
    assert {str(entry["filename"]) for entry in listed_entries} == {
        "weekly_meta.json",
        "weekly_prices.csv",
    }

    for entry in listed_entries:
        copied_file = target_root / str(entry["filename"])
        assert copied_file.exists()
        assert entry["sha256"] == sha256_file(copied_file)
        assert int(entry["size_bytes"]) == copied_file.stat().st_size


def test_bulk_ingest_manifest_integrity_and_dead_letter_routing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    raw_dir = tmp_path / "bootstrap" / "raw"
    report_dir = tmp_path / "bootstrap" / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)

    stage_bootstrap_raw_fixture(FIXTURE_ROOT / "bootstrap_raw", raw_dir)
    status_payload = build_bootstrap_status_payload(raw_dir, report_dir)

    monkeypatch.setattr(bulk_ingest, "bootstrap_status", lambda: status_payload)
    _patch_idempotency_root(monkeypatch, tmp_path)

    result = bulk_ingest.run_eia_bulk_ingest(asof="2032-01-06")
    manifest_path = Path(result["tasks"]["publish_bulk_manifest"]["manifest_path"])

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    copied_entries = list(payload["files"])
    dead_entries = list(payload["dead_lettered"])

    copied_source_ids = [str(entry["source_id"]) for entry in copied_entries]
    assert copied_source_ids == sorted(copied_source_ids)
    assert {"ng_prod_target", "dpr_data", "duc_data"}.issubset(set(copied_source_ids))
    assert any(str(entry["source_id"]) == "bootstrap_readme" for entry in dead_entries)
    assert all(
        Path(str(entry["filename"])).suffix.lower()
        not in {".xls", ".xlsx", ".csv", ".json"}
        for entry in dead_entries
    )

    target_root = Path(payload["target_root"])
    for entry in copied_entries:
        copied_file = target_root / str(entry["filename"])
        assert copied_file.exists()
        assert entry["sha256"] == sha256_file(copied_file)

    dead_root = Path(payload["dead_letter_root"])
    for entry in dead_entries:
        assert (dead_root / str(entry["filename"])).exists()
