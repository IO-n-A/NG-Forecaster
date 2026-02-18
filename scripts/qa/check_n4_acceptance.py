#!/usr/bin/env python3
"""Run N4 acceptance gate for stability/calibration/reporting evidence."""

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
from ng_forecaster.qa.n4_calibration_gate import check_n4_acceptance


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--seed-stability",
        default="data/reports/n4_seed_stability.csv",
    )
    parser.add_argument(
        "--calibration",
        default="data/reports/n4_calibration_summary.csv",
    )
    parser.add_argument(
        "--outputs",
        default="data/reports/n4_nowcast_outputs.csv",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        result = check_n4_acceptance(
            args.seed_stability,
            args.calibration,
            args.outputs,
        )
    except ContractViolation as exc:
        print(f"FAIL: {exc}")
        return 1

    print("PASS: N4 acceptance gate passed")
    print(json.dumps(result.as_dict(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
