#!/usr/bin/env python3
"""Validate handoff evidence triad completeness for N1."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ng_forecaster.qa.handoff_validator import validate_handoff_evidence  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backlog",
        default="docs/coding/backlog.md",
        help="Path to backlog markdown",
    )
    parser.add_argument(
        "--handoff-id",
        default="handoff-n1",
        help="Handoff identifier (e.g., handoff-n1)",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = validate_handoff_evidence(args.backlog, handoff_id=args.handoff_id)

    print(result.to_text())
    print(json.dumps(result.as_dict(), sort_keys=True, indent=2))
    return 0 if result.complete else 1


if __name__ == "__main__":
    raise SystemExit(main())
