#!/usr/bin/env python3
"""Run lookback sweep with DM scoring and winner promotion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ng_forecaster.evaluation.lookback_sweep import run_lookback_sweep  # noqa: E402


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--end-target-month",
        default="2025-11-30",
        help="Final replay target month (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--variant",
        default="wpd_vmd_lstm1",
        help="Champion variant to evaluate.",
    )
    parser.add_argument(
        "--lookbacks",
        nargs="+",
        type=int,
        default=[24, 30, 36, 40, 48],
        help="Lookback grid values.",
    )
    parser.add_argument(
        "--replay_months",
        type=int,
        default=24,
        help="Number of replay months.",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Forecast horizons to score.",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=20,
        help="Deterministic trial count per lookback.",
    )
    parser.add_argument(
        "--asof_day",
        type=int,
        default=14,
        help="Preferred as-of day when deriving replay dates.",
    )
    parser.add_argument(
        "--dm_baseline_lookback",
        type=int,
        default=36,
        help="Baseline lookback used in DM comparisons.",
    )
    parser.add_argument(
        "--report-root",
        default="data/reports",
        help="Output report directory.",
    )
    parser.add_argument(
        "--promote-winner",
        type=int,
        default=1,
        help="Set to 1 to write winner lookback into model champion config.",
    )
    parser.add_argument(
        "--champion-config",
        default="configs/model_champion.yaml",
        help="Champion config path for winner promotion.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = run_lookback_sweep(
        end_target_month=args.end_target_month,
        lookbacks=args.lookbacks,
        replay_months=int(args.replay_months),
        horizons=args.horizons,
        n_trials=int(args.n_trials),
        variant=str(args.variant),
        asof_day=int(args.asof_day),
        dm_baseline_lookback=int(args.dm_baseline_lookback),
        report_root=args.report_root,
        promote_winner=bool(int(args.promote_winner)),
        champion_config_path=args.champion_config,
    )
    print(
        json.dumps(
            {
                "scorecard_rows": int(len(result.scorecard)),
                "error_rows": int(len(result.errors)),
                "dm_rows": int(len(result.dm_results)),
                "selected_lookback": int(result.winner["selected_lookback"]),
                "winner_path": str(result.winner["winner_path"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

