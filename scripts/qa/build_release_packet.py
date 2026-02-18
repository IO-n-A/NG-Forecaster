#!/usr/bin/env python3
"""Build Sprint 1 release packet and enforce release gate checks."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ng_forecaster.qa.release_gate import build_release_packet  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--backlog", default="docs/coding/backlog.md")
    parser.add_argument(
        "--output",
        default="data/reports/sprint1_release_packet.json",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = build_release_packet(backlog_path=args.backlog, output_path=args.output)
    print(result.to_text())
    print(f"release_packet={Path(args.output)}")
    return 0 if result.passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
