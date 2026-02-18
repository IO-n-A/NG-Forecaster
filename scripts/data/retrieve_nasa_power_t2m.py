#!/usr/bin/env python3
"""Retrieve NASA POWER daily T2M_MIN weather rows for CP3 freeze-off features."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ng_forecaster.errors import ContractViolation
from ng_forecaster.ingest.adapters.nasa_power import retrieve_nasa_power_t2m_min


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asof", required=True, help="Run as-of date (YYYY-MM-DD).")
    parser.add_argument(
        "--lookback-months",
        type=int,
        default=36,
        help="Daily retrieval lookback in months.",
    )
    parser.add_argument(
        "--basin-geojson",
        default="data/reference/dpr_basin_polygons.geojson",
        help="GeoJSON path with basin polygons.",
    )
    parser.add_argument(
        "--output-root",
        default="data/bronze/weather/nasa_power_t2m_min",
        help="Bronze output root.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="HTTP timeout in seconds.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    asof_ts = pd.Timestamp(args.asof)
    if pd.isna(asof_ts):
        print("FAIL: invalid --asof timestamp")
        return 1
    if args.lookback_months < 1:
        print("FAIL: --lookback-months must be >= 1")
        return 1

    start_month = (asof_ts.to_period("M") - args.lookback_months + 1).to_timestamp()
    token = os.getenv("NASA_EARTHDATA_TOKEN", "")
    try:
        frame = retrieve_nasa_power_t2m_min(
            asof=asof_ts,
            start_date=start_month,
            basin_geojson_path=args.basin_geojson,
            token=token,
            timeout_seconds=args.timeout_seconds,
        )
    except ContractViolation as exc:
        print(f"FAIL: {exc}")
        return 1

    asof_label = asof_ts.date().isoformat()
    output_dir = Path(args.output_root) / f"asof={asof_label}"
    output_dir.mkdir(parents=True, exist_ok=True)

    parquet_path = output_dir / "nasa_power_t2m_min_daily.parquet"
    frame.to_parquet(parquet_path, index=False)

    report_payload = {
        "asof": asof_label,
        "start_date": start_month.date().isoformat(),
        "output_path": str(parquet_path),
        "row_count": int(len(frame)),
        "basin_count": int(frame["basin_id"].nunique()),
        "min_date": pd.Timestamp(frame["date"].min()).date().isoformat(),
        "max_date": pd.Timestamp(frame["date"].max()).date().isoformat(),
    }
    report_path = output_dir / "manifest.json"
    report_path.write_text(
        json.dumps(report_payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    report_payload["manifest_path"] = str(report_path)
    print(json.dumps(report_payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
