#!/usr/bin/env python3
"""Lint backlog cadence and blocker completeness contracts."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ng_forecaster.qa.backlog_linter import lint_backlog  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backlog", default="docs/coding/backlog.md")
    parser.add_argument("--max-gap-minutes", type=int, default=90)
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    report = lint_backlog(args.backlog, max_gap_minutes=args.max_gap_minutes)

    print(report.to_text())
    print(report.to_json())
    return 0 if report.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
