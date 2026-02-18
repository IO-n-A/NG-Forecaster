#!/usr/bin/env python3
"""Run preprocess artifact gate checks for latest or explicit run directory."""

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

from ng_forecaster.errors import ContractViolation  # noqa: E402
from ng_forecaster.qa.preprocess_gate import (
    check_preprocess_gate,
    resolve_latest_nowcast_artifact_dir,
)  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact-dir",
        default=None,
        help="Optional artifact directory. Defaults to latest under data/artifacts/nowcast.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()

    artifact_dir = (
        Path(args.artifact_dir)
        if args.artifact_dir
        else resolve_latest_nowcast_artifact_dir()
    )
    try:
        result = check_preprocess_gate(artifact_dir)
    except ContractViolation as exc:
        print(f"FAIL: {exc}")
        return 1

    print(f"PASS: preprocess gate passed for {artifact_dir}")
    print(json.dumps(result.as_dict(), sort_keys=True, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
