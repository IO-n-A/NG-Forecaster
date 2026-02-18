#!/usr/bin/env python3
"""Run target-month and interval-order QA gate against nowcast artifacts."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ng_forecaster.errors import ContractViolation
from ng_forecaster.qa.target_month_gate import check_target_month_gate


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--nowcast-json",
        default=None,
        help="Path to nowcast.json. Defaults to latest artifact directory.",
    )
    parser.add_argument(
        "--artifact-root",
        default="data/artifacts/nowcast",
        help="Root directory used when --nowcast-json is omitted.",
    )
    parser.add_argument(
        "--lag-months",
        type=int,
        default=2,
        help="Expected target-month lag from asof month.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        result = check_target_month_gate(
            args.nowcast_json,
            artifact_root=args.artifact_root,
            lag_months=args.lag_months,
        )
    except ContractViolation as exc:
        print(f"FAIL: {exc}")
        return 1

    print("PASS: target-month gate passed")
    print(json.dumps(result.as_dict(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
