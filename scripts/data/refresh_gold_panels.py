#!/usr/bin/env python3
"""Refresh CP2/CP3 gold panels from silver/bronze inputs."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ng_forecaster.data.gold_publish import (
    publish_steo_gold_marts,
    publish_weather_freezeoff_panel,
)
from ng_forecaster.errors import ContractViolation


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--only",
        choices=("steo", "weather_freezeoff", "all"),
        default="steo",
        help="Select which gold panel families to publish.",
    )
    parser.add_argument(
        "--asof",
        default=None,
        help="As-of date for weather-freeze panel publication.",
    )
    parser.add_argument(
        "--silver-root",
        default="data/silver/steo_vintages",
    )
    parser.add_argument(
        "--bronze-weather-root",
        default="data/bronze/weather/nasa_power_t2m_min",
    )
    parser.add_argument(
        "--gold-root",
        default="data/gold",
    )
    parser.add_argument(
        "--report-root",
        default="data/reports",
    )
    parser.add_argument(
        "--train-transfer-priors",
        type=int,
        default=1,
        help="Set to 1 to build transfer priors panel during steo/all refresh.",
    )
    parser.add_argument(
        "--transfer-horizons",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Horizon list for transfer priors training wrapper.",
    )
    return parser


def _resolve_weather_asof(raw_asof: str | None, weather_root: Path) -> pd.Timestamp:
    if raw_asof is not None:
        parsed = pd.Timestamp(raw_asof)
        if pd.isna(parsed):
            raise ContractViolation(
                "invalid_timestamp",
                key="asof",
                detail="provided --asof could not be parsed",
            )
        return parsed

    candidates: list[pd.Timestamp] = []
    for path in sorted(weather_root.glob("asof=*"), key=lambda item: item.name):
        token = path.name.split("=", 1)[-1]
        parsed = pd.to_datetime(token, errors="coerce")
        if pd.isna(parsed):
            continue
        candidates.append(pd.Timestamp(parsed))
    if candidates:
        return sorted(candidates)[-1]
    return pd.Timestamp.utcnow().normalize()


def _resolve_asof(raw_asof: str | None) -> pd.Timestamp:
    if raw_asof is None:
        return pd.Timestamp.utcnow().normalize()
    parsed = pd.Timestamp(raw_asof)
    if pd.isna(parsed):
        raise ContractViolation(
            "invalid_timestamp",
            key="asof",
            detail="provided --asof could not be parsed",
        )
    return parsed.normalize()


def _run_transfer_priors_panel(
    *,
    asof: pd.Timestamp,
    horizons: list[int],
    silver_root: str,
    gold_root: str,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(ROOT / "scripts/models/build_transfer_priors_panel.py"),
        "--asof",
        asof.date().isoformat(),
        "--silver-root",
        silver_root,
        "--gold-root",
        gold_root,
        "--horizons",
        *[str(int(value)) for value in horizons],
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
            key="build_transfer_priors_panel",
            detail="expected JSON output payload from transfer-priors wrapper",
        )
    payload = json.loads(lines[-1])
    panel_path = Path(str(payload.get("transfer_priors_panel_path", "")))
    if not panel_path.exists():
        raise ContractViolation(
            "missing_source_file",
            key=str(panel_path),
            detail="transfer priors panel path was not produced by refresh workflow",
        )
    return dict(payload)


def main() -> int:
    args = _build_parser().parse_args()
    results: dict[str, Any] = {"only": args.only}
    try:
        if args.only in {"steo", "all"}:
            results["steo"] = publish_steo_gold_marts(
                silver_root=args.silver_root,
                gold_root=args.gold_root,
                report_root=args.report_root,
            )
            if bool(int(args.train_transfer_priors)):
                transfer_asof = _resolve_asof(args.asof)
                results["transfer_priors"] = _run_transfer_priors_panel(
                    asof=transfer_asof,
                    horizons=[int(value) for value in args.transfer_horizons],
                    silver_root=str(args.silver_root),
                    gold_root=str(args.gold_root),
                )

        if args.only in {"weather_freezeoff", "all"}:
            asof_ts = _resolve_weather_asof(
                args.asof,
                weather_root=Path(args.bronze_weather_root),
            )
            results["weather_freezeoff"] = publish_weather_freezeoff_panel(
                asof=asof_ts,
                bronze_root=args.bronze_weather_root,
                gold_root=args.gold_root,
                report_root=args.report_root,
            )
    except ContractViolation as exc:
        print(f"FAIL: {exc}")
        return 1

    print(json.dumps(results, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
