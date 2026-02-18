#!/usr/bin/env python3
"""Execute Sprint 4B data refresh cycle (retrieval + migration + silver + gold)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ng_forecaster.data.refresh import run_data_refresh_cycle  # noqa: E402
from ng_forecaster.errors import ContractViolation  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asof", required=True, help="As-of date for refresh cycle.")
    parser.add_argument(
        "--inventory",
        default="data/reports/sprint4b_data_new_inventory.csv",
        help="Inventory CSV for migration.",
    )
    parser.add_argument(
        "--source-root",
        default="data/new",
        help="Source directory for migration.",
    )
    parser.add_argument(
        "--bronze-root",
        default="data/bronze",
        help="Bronze root destination.",
    )
    parser.add_argument(
        "--run-steo-retrieval",
        action="store_true",
        help="Download STEO archives from EIA prior to normalization.",
    )
    parser.add_argument(
        "--run-context-retrieval",
        action="store_true",
        help="Download context prior reports prior to gold publish.",
    )
    parser.add_argument(
        "--retire-data-new",
        action="store_true",
        help="Delete data/new after successful migration+publish cycle.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        result = run_data_refresh_cycle(
            asof=args.asof,
            inventory_path=args.inventory,
            source_root=args.source_root,
            bronze_root=args.bronze_root,
            run_steo_retrieval=bool(args.run_steo_retrieval),
            run_context_retrieval=bool(args.run_context_retrieval),
            retire_source_dir=bool(args.retire_data_new),
        )
    except ContractViolation as exc:
        print(f"FAIL: {exc}")
        return 1

    print(json.dumps(result.as_dict(), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
