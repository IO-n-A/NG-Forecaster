"""Deterministic feature and lineage reference helpers."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import date, datetime
from pathlib import Path
from typing import Any, Mapping


def _parse_date(value: str) -> date:
    return datetime.strptime(value, "%Y-%m-%d").date()


def _mean(values: list[float]) -> float:
    if not values:
        raise ValueError("values must be non-empty")
    return sum(values) / len(values)


def load_daily_price_rows(path: Path) -> list[dict[str, object]]:
    """Load daily Henry Hub price rows from a fixture."""
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "hh_price": float(row["hh_price"]),
                }
            )

    return sorted(rows, key=lambda row: str(row["timestamp"]))


def load_weekly_storage_rows(path: Path) -> list[dict[str, object]]:
    """Load weekly storage rows from a fixture."""
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            rows.append(
                {
                    "timestamp": row["timestamp"],
                    "storage_bcf": float(row["storage_bcf"]),
                }
            )

    return sorted(rows, key=lambda row: str(row["timestamp"]))


def _canonical_payload(payload: Mapping[str, Any]) -> dict[str, Any]:
    canonical: dict[str, Any] = {}
    for key, value in payload.items():
        if key == "lineage_id":
            continue
        if isinstance(value, float):
            canonical[key] = f"{value:.6f}"
        else:
            canonical[key] = value
    return canonical


def compute_lineage_id(payload: Mapping[str, Any]) -> str:
    """Compute a deterministic lineage hash over canonicalized feature payload."""
    serialized = json.dumps(
        _canonical_payload(payload),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:16]


def build_monthly_features(
    daily_price_rows: list[dict[str, object]],
    weekly_storage_rows: list[dict[str, object]],
    *,
    asof: str,
) -> dict[str, object]:
    """Build deterministic monthly feature values for a given as-of date."""
    asof_date = _parse_date(asof)
    month_start = asof_date.replace(day=1)

    price_values: list[float] = []
    ordered_price_rows = sorted(daily_price_rows, key=lambda row: str(row["timestamp"]))
    for row in ordered_price_rows:
        row_date = _parse_date(str(row["timestamp"]))
        if month_start <= row_date <= asof_date:
            price_values.append(float(row["hh_price"]))

    storage_values: list[float] = []
    ordered_storage_rows = sorted(weekly_storage_rows, key=lambda row: str(row["timestamp"]))
    for row in ordered_storage_rows:
        row_date = _parse_date(str(row["timestamp"]))
        if month_start <= row_date <= asof_date:
            storage_values.append(float(row["storage_bcf"]))

    if not price_values:
        raise ValueError(f"no daily price rows for asof={asof}")
    if not storage_values:
        raise ValueError(f"no storage rows for asof={asof}")

    features: dict[str, object] = {
        "asof": asof,
        "month": asof_date.strftime("%Y-%m"),
        "hh_mtd_mean": round(_mean(price_values), 6),
        "hh_last": round(price_values[-1], 6),
        "stor_mean": round(_mean(storage_values), 6),
        "stor_last": round(storage_values[-1], 6),
        "stor_slope": round(storage_values[-1] - storage_values[0], 6),
    }
    features["lineage_id"] = compute_lineage_id(features)
    return features


def load_expected_feature_row(path: Path) -> dict[str, object]:
    """Load a single-row expected feature fixture."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))

    if len(rows) != 1:
        raise ValueError(f"expected one row in fixture: {path}")

    raw = rows[0]
    return {
        "asof": raw["asof"],
        "month": raw["month"],
        "hh_mtd_mean": float(raw["hh_mtd_mean"]),
        "hh_last": float(raw["hh_last"]),
        "stor_mean": float(raw["stor_mean"]),
        "stor_last": float(raw["stor_last"]),
        "stor_slope": float(raw["stor_slope"]),
        "lineage_id": raw["lineage_id"],
    }
