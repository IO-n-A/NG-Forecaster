#!/usr/bin/env python3
"""Run N5 DM policy compliance audit."""

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
from ng_forecaster.qa.n5_policy_audit import audit_n5_policy


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy", default="configs/evaluation.yaml")
    parser.add_argument("--dm-results", default="data/reports/dm_results.csv")
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        result = audit_n5_policy(args.policy, args.dm_results)
    except ContractViolation as exc:
        print(f"FAIL: {exc}")
        return 1

    print("PASS: N5 policy audit passed")
    print(json.dumps(result.as_dict(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
