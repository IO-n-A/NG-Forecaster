#!/usr/bin/env python3
"""Retrieve monthly STEO archive PDF/XLSX vintages with deterministic hash manifests."""

from __future__ import annotations

import argparse
from datetime import date
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ng_forecaster.data.bronze_writer import sha256_file  # noqa: E402
from ng_forecaster.errors import ContractViolation  # noqa: E402

try:  # pragma: no cover - network dependency
    import requests  # type: ignore[import-untyped]
except Exception:  # pragma: no cover
    requests = None


_ARCHIVE_BASE_URL = "https://www.eia.gov/outlooks/steo/archives"


def _month_range(start: pd.Timestamp, end: pd.Timestamp) -> list[pd.Timestamp]:
    months: list[pd.Timestamp] = []
    current = start.to_period("M").to_timestamp("M")
    final = end.to_period("M").to_timestamp("M")
    while current <= final:
        months.append(current)
        current = (current.to_period("M") + 1).to_timestamp("M")
    return months


def _token_for_month(month_end: pd.Timestamp) -> str:
    return month_end.strftime("%b%y").lower()


def _download(url: str, destination: Path, *, timeout_seconds: int) -> None:
    if requests is None:
        raise ContractViolation(
            "missing_dependency",
            key="requests",
            detail="requests package is required for STEO retrieval",
        )
    response = requests.get(url, timeout=float(timeout_seconds))
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)


def retrieve_steo_archives(
    *,
    start_month: object,
    end_month: object,
    output_root: str | Path = "data/bronze/eia_bulk/steo_vintages",
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Download STEO archive vintages and write retrieval manifest."""

    start = pd.Timestamp(start_month)
    end = pd.Timestamp(end_month)
    if pd.isna(start) or pd.isna(end):
        raise ContractViolation(
            "invalid_timestamp",
            key="start_month/end_month",
            detail="unable to parse start_month or end_month",
        )
    if start > end:
        raise ContractViolation(
            "invalid_timestamp",
            key="start_month/end_month",
            detail="start_month must be <= end_month",
        )

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)
    months = _month_range(start, end)

    rows: list[dict[str, Any]] = []
    for month_end in months:
        token = _token_for_month(month_end)
        vintage = month_end.strftime("%Y-%m")
        vintage_dir = root / f"vintage_month={vintage}"
        pdf_path = vintage_dir / "steo_full.pdf"
        xlsx_path = vintage_dir / "steo_m.xlsx"

        pdf_url = f"{_ARCHIVE_BASE_URL}/{token}.pdf"
        xlsx_url = f"{_ARCHIVE_BASE_URL}/{token}_base.xlsx"

        _download(pdf_url, pdf_path, timeout_seconds=timeout_seconds)
        _download(xlsx_url, xlsx_path, timeout_seconds=timeout_seconds)

        rows.append(
            {
                "vintage_month": vintage,
                "relative_path": f"vintage_month={vintage}/steo_full.pdf",
                "url": pdf_url,
                "bytes": int(pdf_path.stat().st_size),
                "sha256": sha256_file(pdf_path),
            }
        )
        rows.append(
            {
                "vintage_month": vintage,
                "relative_path": f"vintage_month={vintage}/steo_m.xlsx",
                "url": xlsx_url,
                "bytes": int(xlsx_path.stat().st_size),
                "sha256": sha256_file(xlsx_path),
            }
        )

    manifest = pd.DataFrame(rows).sort_values(["vintage_month", "relative_path"])
    manifest_path = root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    payload = {
        "output_root": str(root),
        "start_month": pd.Timestamp(start)
        .to_period("M")
        .to_timestamp("M")
        .date()
        .isoformat(),
        "end_month": pd.Timestamp(end)
        .to_period("M")
        .to_timestamp("M")
        .date()
        .isoformat(),
        "vintage_count": int(len(months)),
        "file_count": int(len(manifest)),
        "manifest_path": str(manifest_path),
    }
    report_path = Path("data/reports/steo_archive_retrieval_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8"
    )
    payload["report_path"] = str(report_path)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--start-month",
        default="2020-01-31",
        help="Start month (inclusive) for vintage retrieval.",
    )
    parser.add_argument(
        "--end-month",
        default=date.today().isoformat(),
        help="End month (inclusive) for vintage retrieval.",
    )
    parser.add_argument(
        "--output-root",
        default="data/bronze/eia_bulk/steo_vintages",
        help="Destination root for retrieved archives.",
    )
    parser.add_argument(
        "--timeout-seconds",
        type=int,
        default=30,
        help="HTTP timeout in seconds for each download.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        result = retrieve_steo_archives(
            start_month=args.start_month,
            end_month=args.end_month,
            output_root=args.output_root,
            timeout_seconds=int(args.timeout_seconds),
        )
    except Exception as exc:
        print(f"FAIL: {exc}")
        return 1

    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
