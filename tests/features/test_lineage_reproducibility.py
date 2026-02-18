"""Lineage reproducibility and sensitivity checks."""

from __future__ import annotations

from pathlib import Path

from tests.helpers.feature_reference import (
    build_monthly_features,
    load_daily_price_rows,
    load_expected_feature_row,
    load_weekly_storage_rows,
)

FIXTURE_ROOT = Path(__file__).resolve().parents[1] / "fixtures" / "features"


def test_monthly_features_match_gold_fixtures_for_two_asofs() -> None:
    daily_rows = load_daily_price_rows(FIXTURE_ROOT / "input_daily_price.csv")
    storage_rows = load_weekly_storage_rows(FIXTURE_ROOT / "input_weekly_storage.csv")

    expected_jan = load_expected_feature_row(
        FIXTURE_ROOT / "expected_monthly_features_asof_2025-01-31.csv"
    )
    expected_feb = load_expected_feature_row(
        FIXTURE_ROOT / "expected_monthly_features_asof_2025-02-28.csv"
    )

    assert build_monthly_features(daily_rows, storage_rows, asof="2025-01-31") == expected_jan
    assert build_monthly_features(daily_rows, storage_rows, asof="2025-02-28") == expected_feb


def test_lineage_is_invariant_to_input_order() -> None:
    daily_rows = load_daily_price_rows(FIXTURE_ROOT / "input_daily_price.csv")
    storage_rows = load_weekly_storage_rows(FIXTURE_ROOT / "input_weekly_storage.csv")

    baseline = build_monthly_features(daily_rows, storage_rows, asof="2025-02-28")
    reordered = build_monthly_features(
        list(reversed(daily_rows)),
        list(reversed(storage_rows)),
        asof="2025-02-28",
    )

    assert reordered == baseline


def test_lineage_changes_when_eligible_input_changes() -> None:
    daily_rows = load_daily_price_rows(FIXTURE_ROOT / "input_daily_price.csv")
    storage_rows = load_weekly_storage_rows(FIXTURE_ROOT / "input_weekly_storage.csv")

    baseline = build_monthly_features(daily_rows, storage_rows, asof="2025-02-28")

    mutated_rows = [dict(row) for row in daily_rows]
    for row in mutated_rows:
        if row["timestamp"] == "2025-02-14":
            row["hh_price"] = 3.95

    mutated = build_monthly_features(mutated_rows, storage_rows, asof="2025-02-28")

    assert mutated["lineage_id"] != baseline["lineage_id"]
    assert mutated["hh_mtd_mean"] != baseline["hh_mtd_mean"]
