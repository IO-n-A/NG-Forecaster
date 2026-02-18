#!/usr/bin/env python3
"""Compare two rerun artifact snapshots for deterministic equality."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ng_forecaster.qa.determinism import compare_rerun_snapshots  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "left_snapshot",
        nargs="?",
        default="data/artifacts/nowcast/2024-01-10",
        help="First snapshot directory",
    )
    parser.add_argument(
        "right_snapshot",
        nargs="?",
        default="data/artifacts/nowcast/2024-01-12",
        help="Second snapshot directory",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = compare_rerun_snapshots(args.left_snapshot, args.right_snapshot)
    print(result.to_text())
    print(json.dumps(result.as_dict(), sort_keys=True, indent=2))
    return 0 if result.matches else 1


if __name__ == "__main__":
    raise SystemExit(main())
