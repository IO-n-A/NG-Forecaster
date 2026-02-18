#!/usr/bin/env python3
"""Build transfer-priors gold panel using the basin DNN training routine."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ng_forecaster.errors import ContractViolation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asof",
        default=None,
        help="As-of date (YYYY-MM-DD). Defaults to current UTC day.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1, 2],
    )
    parser.add_argument("--silver-root", default="data/silver/steo_vintages")
    parser.add_argument("--gold-root", default="data/gold")
    parser.add_argument("--artifact-root", default="data/artifacts/models")
    return parser


def _resolve_asof(raw: str | None) -> str:
    if raw is None:
        return str(pd.Timestamp.utcnow().normalize().date().isoformat())
    parsed = pd.Timestamp(raw)
    if pd.isna(parsed):
        raise ContractViolation(
            "invalid_timestamp",
            key="asof",
            detail="provided --asof could not be parsed",
        )
    return str(parsed.normalize().date().isoformat())


def main() -> int:
    args = _build_parser().parse_args()
    asof = _resolve_asof(args.asof)
    train_script = ROOT / "scripts/models/train_transfer_basin_dnn.py"
    command = [
        sys.executable,
        str(train_script),
        "--asof",
        asof,
        "--silver-root",
        str(args.silver_root),
        "--gold-root",
        str(args.gold_root),
        "--artifact-root",
        str(args.artifact_root),
        "--horizons",
        *[str(int(value)) for value in args.horizons],
    ]
    completed = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise ContractViolation(
            "source_schema_drift",
            key="train_transfer_basin_dnn",
            detail="expected JSON output payload from transfer training script",
        )
    payload = json.loads(lines[-1])
    panel_path = Path(str(payload.get("transfer_priors_panel_path", "")))
    if not panel_path.exists():
        raise ContractViolation(
            "missing_source_file",
            key=str(panel_path),
            detail="transfer priors panel was not written",
        )
    output = {
        "asof": asof,
        "horizons": [int(value) for value in args.horizons],
        "transfer_priors_panel_path": str(panel_path),
        "manifest_path": str(payload.get("manifest_path", "")),
        "lineage_id": str(payload.get("lineage_id", "")),
        "status": "transfer_priors_panel_built",
    }
    print(json.dumps(output, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
