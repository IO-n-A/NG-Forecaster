#!/usr/bin/env python3
"""Migrate evaluated `data/new` inventory payloads into bronze/context storage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ng_forecaster.data.migration import migrate_data_new_inventory  # noqa: E402
from ng_forecaster.errors import ContractViolation  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--inventory",
        default="data/reports/sprint4b_data_new_inventory.csv",
        help="Path to evaluated data/new inventory CSV.",
    )
    parser.add_argument(
        "--source-root",
        default="data/new",
        help="Root folder containing staged Sprint4B source files.",
    )
    parser.add_argument(
        "--bronze-root",
        default="data/bronze",
        help="Bronze root destination.",
    )
    parser.add_argument(
        "--priorities",
        default="P0,P1",
        help="Comma-separated priorities to migrate.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    priorities = tuple(
        sorted(
            {
                token.strip().upper()
                for token in str(args.priorities).split(",")
                if token.strip()
            }
        )
    )
    try:
        result = migrate_data_new_inventory(
            inventory_path=args.inventory,
            source_root=args.source_root,
            bronze_root=args.bronze_root,
            include_priorities=priorities,
        )
    except ContractViolation as exc:
        print(f"FAIL: {exc}")
        return 1

    print(
        json.dumps(
            {
                **result.as_dict(),
                "priorities": priorities,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
