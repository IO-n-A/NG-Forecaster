#!/usr/bin/env python3
"""Retrieve context-prior reports (IEA/AEO/peer) with deterministic URL manifest."""

from __future__ import annotations

import argparse
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


_DEFAULT_REPORTS: tuple[dict[str, str], ...] = (
    {
        "report_id": "iea_gas_market_q1_2026",
        "family": "iea",
        "filename": "GasMarketReport_Q1_2026.pdf",
        "url": "https://iea.blob.core.windows.net/assets/98d3c7fc-d2ee-479a-aa8f-c02da1c8a4b8/GasMarketReport%2CQ1-2026.pdf",
    },
    {
        "report_id": "iea_gas_2025",
        "family": "iea",
        "filename": "Gas2025.pdf",
        "url": "https://iea.blob.core.windows.net/assets/a281016b-edb9-4ba5-8114-57af264db8be/Gas2025.pdf",
    },
    {
        "report_id": "eia_aeo_ngmm_assumptions",
        "family": "aeo",
        "filename": "NGMM_Assumptions.pdf",
        "url": "https://www.eia.gov/outlooks/aeo/assumptions/pdf/NGMM_Assumptions.pdf",
    },
    {
        "report_id": "datapages_marcellus_gev",
        "family": "peer_reviewed",
        "filename": "BLTN21078.pdf",
        "url": "https://archives.datapages.com/data/bulletns/2024/01jan/BLTN21078/images/bltn21078.pdf",
    },
)


def _download(url: str, destination: Path, *, timeout_seconds: int) -> None:
    if requests is None:
        raise ContractViolation(
            "missing_dependency",
            key="requests",
            detail="requests package is required for context retrieval",
        )
    response = requests.get(url, timeout=float(timeout_seconds))
    response.raise_for_status()
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(response.content)


def retrieve_context_reports(
    *,
    output_root: str | Path = "data/bronze/context/external",
    timeout_seconds: int = 30,
) -> dict[str, Any]:
    """Retrieve context prior reports and emit manifest metadata."""

    root = Path(output_root)
    root.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for report in _DEFAULT_REPORTS:
        family = report["family"]
        filename = report["filename"]
        destination = root / family / filename
        _download(report["url"], destination, timeout_seconds=timeout_seconds)

        rows.append(
            {
                "report_id": report["report_id"],
                "family": family,
                "filename": filename,
                "relative_path": destination.relative_to(root).as_posix(),
                "url": report["url"],
                "bytes": int(destination.stat().st_size),
                "sha256": sha256_file(destination),
            }
        )

    manifest = (
        pd.DataFrame(rows).sort_values(["family", "filename"]).reset_index(drop=True)
    )
    manifest_path = root / "manifest.csv"
    manifest.to_csv(manifest_path, index=False)

    payload = {
        "output_root": str(root),
        "file_count": int(len(manifest)),
        "manifest_path": str(manifest_path),
    }
    report_path = Path("data/reports/context_report_retrieval_report.json")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8"
    )
    payload["report_path"] = str(report_path)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default="data/bronze/context/external",
        help="Destination root for context report downloads.",
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
    try:
        result = retrieve_context_reports(
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
