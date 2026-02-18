from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import ng_forecaster.ingest.api_client as api_client
from ng_forecaster.errors import ContractViolation
from ng_forecaster.ingest.api_client import stage_sources_for_stream
from ng_forecaster.ingest.catalog import build_ingestion_plan, validate_source_catalog


def _base_catalog_payload(bootstrap_root: Path) -> dict[str, object]:
    return {
        "version": 1,
        "defaults": {
            "freshness_max_age_days": 90,
            "bootstrap_root": str(bootstrap_root),
        },
        "release_calendar": {
            "lag_months": 2,
            "release_day_of_month": 30,
            "admissible_day_window": {"start_day": 1, "end_day": 31},
        },
        "sources": [
            {
                "id": "target",
                "role": "target_history",
                "ingest_stream": "bulk",
                "retrieval_mode": "bulk_snapshot",
                "filename": "target.xls",
                "required": True,
            },
            {
                "id": "api_a",
                "role": "driver",
                "ingest_stream": "api",
                "retrieval_mode": "api_snapshot",
                "filename": "a.csv",
                "required": True,
            },
            {
                "id": "api_b",
                "role": "driver",
                "ingest_stream": "api",
                "retrieval_mode": "bootstrap_file",
                "filename": "b.csv",
                "required": True,
            },
        ],
    }


def test_catalog_rejects_invalid_retrieval_mode(tmp_path: Path) -> None:
    payload = _base_catalog_payload(tmp_path / "bootstrap")
    payload["sources"][1]["retrieval_mode"] = "invalid_mode"  # type: ignore[index]

    with pytest.raises(ContractViolation, match="reason_code=invalid_source_catalog"):
        validate_source_catalog(payload)


def test_build_ingestion_plan_prefers_live_snapshot_then_bootstrap(
    tmp_path: Path,
) -> None:
    bootstrap_root = tmp_path / "bootstrap" / "raw"
    bootstrap_root.mkdir(parents=True, exist_ok=True)
    (bootstrap_root / "b.csv").write_text("from_bootstrap\n", encoding="utf-8")

    live_api_root = tmp_path / "bronze" / "api"
    (live_api_root / "asof=2025-01-31").mkdir(parents=True, exist_ok=True)
    (live_api_root / "asof=2025-01-31" / "a.csv").write_text(
        "from_live\n", encoding="utf-8"
    )

    catalog = validate_source_catalog(_base_catalog_payload(bootstrap_root))
    plan = build_ingestion_plan(
        catalog=catalog,
        ingest_stream="api",
        asof=pd.Timestamp("2025-02-14"),
        bootstrap_root=bootstrap_root,
        stream_roots={"api": live_api_root, "bulk": tmp_path / "bronze" / "bulk"},
        fixture_root=tmp_path / "fixture",
    )

    assert [item.source_id for item in plan] == ["api_a", "api_b"]
    assert plan[0].origin == "api_snapshot"
    assert plan[0].path.read_text(encoding="utf-8").strip() == "from_live"
    assert plan[1].origin == "bootstrap_raw"
    assert plan[1].path.read_text(encoding="utf-8").strip() == "from_bootstrap"


def test_stage_sources_for_stream_is_deterministic(tmp_path: Path) -> None:
    bootstrap_root = tmp_path / "bootstrap" / "raw"
    bootstrap_root.mkdir(parents=True, exist_ok=True)
    (bootstrap_root / "target.xls").write_text("placeholder\n", encoding="utf-8")
    (bootstrap_root / "a.csv").write_text("a\n", encoding="utf-8")
    (bootstrap_root / "b.csv").write_text("b\n", encoding="utf-8")

    payload = _base_catalog_payload(bootstrap_root)
    payload["sources"] = [  # type: ignore[assignment]
        payload["sources"][0],  # target
        {
            "id": "z_source",
            "role": "driver",
            "ingest_stream": "api",
            "retrieval_mode": "bootstrap_file",
            "filename": "b.csv",
            "required": True,
        },
        {
            "id": "a_source",
            "role": "driver",
            "ingest_stream": "api",
            "retrieval_mode": "bootstrap_file",
            "filename": "a.csv",
            "required": True,
        },
    ]
    catalog = validate_source_catalog(payload)

    staged = stage_sources_for_stream(
        asof=pd.Timestamp("2025-02-14"),
        ingest_stream="api",
        target_root=tmp_path / "target",
        catalog=catalog,
        bootstrap_root=bootstrap_root,
        stream_roots={"api": tmp_path / "bronze" / "api"},
        fixture_root=tmp_path / "fixture",
    )

    assert [item.source_id for item in staged] == ["a_source", "z_source"]
    assert [item.filename for item in staged] == ["a.csv", "b.csv"]


def test_parse_target_history_negative_contracts(monkeypatch, tmp_path: Path) -> None:
    source = tmp_path / "target.xls"
    source.write_text("placeholder\n", encoding="utf-8")

    monkeypatch.setattr(
        api_client.pd, "read_excel", lambda *args, **kwargs: pd.DataFrame()
    )
    with pytest.raises(
        ContractViolation, match="reason_code=missing_target_history_rows"
    ):
        api_client.parse_eia_target_history(source)

    monkeypatch.setattr(
        api_client.pd,
        "read_excel",
        lambda *args, **kwargs: pd.DataFrame({"foo": ["x", "y"], "bar": ["a", "b"]}),
    )
    with pytest.raises(ContractViolation, match="reason_code=source_schema_drift"):
        api_client.parse_eia_target_history(source)


def test_build_ingestion_plan_supports_archive_manifest_mode(tmp_path: Path) -> None:
    payload = _base_catalog_payload(tmp_path / "bootstrap")
    payload["sources"].append(  # type: ignore[attr-defined]
        {
            "id": "steo_manifest",
            "role": "driver",
            "ingest_stream": "bulk",
            "retrieval_mode": "archive_manifest",
            "filename": "manifest.csv",
            "required": True,
            "parse": {"archive_manifest": "steo_vintages/manifest.csv"},
        }
    )

    archive_manifest = tmp_path / "bronze" / "bulk" / "steo_vintages" / "manifest.csv"
    archive_manifest.parent.mkdir(parents=True, exist_ok=True)
    archive_manifest.write_text("relative_path,sha256,bytes\n", encoding="utf-8")

    bootstrap_root = tmp_path / "bootstrap" / "raw"
    bootstrap_root.mkdir(parents=True, exist_ok=True)
    (bootstrap_root / "target.xls").write_text("placeholder\n", encoding="utf-8")

    catalog = validate_source_catalog(payload)
    plan = build_ingestion_plan(
        catalog=catalog,
        ingest_stream="bulk",
        asof=pd.Timestamp("2026-02-14"),
        bootstrap_root=bootstrap_root,
        stream_roots={"bulk": tmp_path / "bronze" / "bulk"},
        fixture_root=tmp_path / "fixture",
    )

    manifest_row = [row for row in plan if row.source_id == "steo_manifest"][0]
    assert manifest_row.path == archive_manifest
    assert manifest_row.origin == "bulk_archive_manifest"


def test_build_ingestion_plan_supports_archive_glob_mode(tmp_path: Path) -> None:
    payload = _base_catalog_payload(tmp_path / "bootstrap")
    payload["sources"].append(  # type: ignore[attr-defined]
        {
            "id": "steo_latest_workbook",
            "role": "driver",
            "ingest_stream": "bulk",
            "retrieval_mode": "archive_glob",
            "filename": "steo_m.xlsx",
            "required": True,
            "parse": {"glob": "steo_vintages/vintage_month=*/steo_m.xlsx"},
        }
    )

    workbook = (
        tmp_path
        / "bronze"
        / "bulk"
        / "steo_vintages"
        / "vintage_month=2026-02"
        / "steo_m.xlsx"
    )
    workbook.parent.mkdir(parents=True, exist_ok=True)
    workbook.write_bytes(b"xlsx")

    bootstrap_root = tmp_path / "bootstrap" / "raw"
    bootstrap_root.mkdir(parents=True, exist_ok=True)
    (bootstrap_root / "target.xls").write_text("placeholder\n", encoding="utf-8")

    catalog = validate_source_catalog(payload)
    plan = build_ingestion_plan(
        catalog=catalog,
        ingest_stream="bulk",
        asof=pd.Timestamp("2026-02-14"),
        bootstrap_root=bootstrap_root,
        stream_roots={"bulk": tmp_path / "bronze" / "bulk"},
        fixture_root=tmp_path / "fixture",
    )

    workbook_row = [row for row in plan if row.source_id == "steo_latest_workbook"][0]
    assert workbook_row.path == workbook
    assert workbook_row.origin == "bulk_archive_glob"
