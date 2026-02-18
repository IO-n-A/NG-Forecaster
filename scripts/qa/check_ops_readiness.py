#!/usr/bin/env python3
"""Run operations readiness gate for weekly governance controls."""

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
from ng_forecaster.qa.ops_readiness import check_ops_readiness


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dag-dir",
        default="src/ng_forecaster/orchestration/airflow/dags",
    )
    parser.add_argument(
        "--runbook",
        default="docs/operations/runbook_weekly.md",
    )
    parser.add_argument(
        "--incident-log",
        default="data/reports/incidents_log.jsonl",
    )
    parser.add_argument(
        "--release-approval",
        default="data/reports/release_approval.json",
    )
    parser.add_argument(
        "--feature-blocks",
        default="configs/feature_blocks.yaml",
    )
    parser.add_argument(
        "--source-catalog",
        default="configs/sources.yaml",
    )
    parser.add_argument(
        "--gold-root",
        default="data/gold",
    )
    parser.add_argument(
        "--forbid-path",
        action="append",
        default=["data/new"],
        help="Path that must not exist for release readiness (repeatable).",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    try:
        result = check_ops_readiness(
            dag_dir=args.dag_dir,
            runbook_path=args.runbook,
            incident_log_path=args.incident_log,
            release_approval_path=args.release_approval,
            feature_blocks_path=args.feature_blocks,
            source_catalog_path=args.source_catalog,
            gold_root=args.gold_root,
            forbidden_paths=tuple(str(item) for item in args.forbid_path),
        )
    except ContractViolation as exc:
        print(f"FAIL: {exc}")
        return 1

    print("PASS: operations readiness gate passed")
    print(json.dumps(result.as_dict(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
