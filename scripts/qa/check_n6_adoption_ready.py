#!/usr/bin/env python3
"""Run N6 ablation adoption-readiness gate."""

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
from ng_forecaster.qa.n6_adoption_gate import check_n6_adoption_readiness


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--scorecard",
        default="data/reports/ablation_scorecard.csv",
    )
    parser.add_argument("--baseline-id", default="B0_baseline")
    parser.add_argument("--full-method-id", default="B4_full_method")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        result = check_n6_adoption_readiness(
            args.scorecard,
            baseline_id=args.baseline_id,
            full_method_id=args.full_method_id,
        )
    except ContractViolation as exc:
        print(f"FAIL: {exc}")
        return 1

    print("PASS: N6 adoption-readiness gate passed")
    print(json.dumps(result.as_dict(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
