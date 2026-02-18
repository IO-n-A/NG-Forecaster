"""Single integration test covering N1 leakage, N3 gate, and N2 lineage determinism."""

from __future__ import annotations

import csv
import json
from pathlib import Path

from tests.helpers.artifact_schema import validate_preprocess_artifact_bundle
from tests.helpers.feature_reference import (
    build_monthly_features,
    load_daily_price_rows,
    load_expected_feature_row,
    load_weekly_storage_rows,
)
from tests.helpers.reference_preprocess import (
    load_series_fixture,
    run_reference_preprocess,
)
from tests.helpers.replay_harness import (
    build_replay_schedule,
    load_and_validate_asof_fixture,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures"


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        raise ValueError("rows must be non-empty")

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_n1_n3_n2_pipeline_contract(tmp_path: Path) -> None:
    leakage_rows = load_and_validate_asof_fixture(
        FIXTURE_ROOT / "leakage" / "asof_valid.csv"
    )
    assert len(leakage_rows) == 4

    replay_schedule = build_replay_schedule(FIXTURE_ROOT / "replay" / "checkpoints.csv")
    assert any(
        item.asof == "2025-02-28" and item.horizon == "T" for item in replay_schedule
    )

    preprocess_rows = load_series_fixture(
        FIXTURE_ROOT / "preprocess" / "mixed_case.csv"
    )
    preprocess_output = run_reference_preprocess(preprocess_rows, asof="2025-02-28")
    assert preprocess_output.summary["status"] == "passed"

    summary_path = tmp_path / "preprocess_summary.json"
    missing_path = tmp_path / "missing_value_flags.csv"
    outlier_path = tmp_path / "outlier_flags.csv"

    summary_path.write_text(json.dumps(preprocess_output.summary), encoding="utf-8")
    _write_csv(missing_path, preprocess_output.missing_flags)
    _write_csv(outlier_path, preprocess_output.outlier_flags)
    validate_preprocess_artifact_bundle(summary_path, missing_path, outlier_path)

    daily_rows = load_daily_price_rows(
        FIXTURE_ROOT / "features" / "input_daily_price.csv"
    )
    storage_rows = load_weekly_storage_rows(
        FIXTURE_ROOT / "features" / "input_weekly_storage.csv"
    )
    features = build_monthly_features(daily_rows, storage_rows, asof="2025-02-28")

    expected = load_expected_feature_row(
        FIXTURE_ROOT / "features" / "expected_monthly_features_asof_2025-02-28.csv"
    )
    assert features == expected
