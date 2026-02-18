#!/usr/bin/env python3
"""Execute and score 24 rolling nowcast runs against released monthly values."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ng_forecaster.evaluation.validation_24m import (  # noqa: E402
    DEFAULT_VARIANTS,
    run_24_month_validation,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--end-target-month",
        default="2025-11-30",
        help="Month-end target month for the final run (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--runs",
        type=int,
        default=24,
        help="Number of rolling monthly runs.",
    )
    parser.add_argument(
        "--asof-day",
        type=int,
        default=14,
        help="Preferred day-of-month for derived asof dates.",
    )
    parser.add_argument(
        "--variants",
        default=",".join(DEFAULT_VARIANTS),
        help="Comma-separated champion variants to evaluate.",
    )
    parser.add_argument(
        "--report-root",
        default="data/reports",
        help="Output directory for validation score files.",
    )
    parser.add_argument(
        "--dump_feature_row",
        type=int,
        default=0,
        help="Set to 1 to export the first feature-matrix row from each run.",
    )
    parser.add_argument(
        "--n_trials",
        type=int,
        default=1,
        help="Number of deterministic trial repeats per model variant.",
    )
    parser.add_argument(
        "--seed_schedule",
        default="",
        help="Comma-separated seed list; extends deterministically if shorter than n_trials.",
    )
    parser.add_argument(
        "--weight_search",
        type=int,
        default=0,
        help="Set to 1 to evaluate fusion weight grid search and export selection artifacts.",
    )
    parser.add_argument(
        "--regime_split",
        type=int,
        default=0,
        help="Set to 1 to export scorecards/DM results split by regime buckets.",
    )
    parser.add_argument(
        "--block_importance",
        type=int,
        default=0,
        help="Set to 1 to compute block-level ablation and influence artifacts.",
    )
    parser.add_argument(
        "--fusion_config",
        default="configs/fusion.yaml",
        help="Fusion policy YAML used for regime-aware weights and weight-search grids.",
    )
    parser.add_argument(
        "--ablation_config",
        default="configs/experiments/nowcast_ablation.yaml",
        help="Block-ablation configuration YAML for CP5 scorecards.",
    )
    parser.add_argument(
        "--extra_block",
        default="",
        help=(
            "Optional block id override for ablation/block-importance runs "
            "(e.g. steo_drilling_metrics, weather_freezeoff_enriched, oil_side)."
        ),
    )
    parser.add_argument(
        "--fusion_constraints",
        default="",
        help=(
            "Optional YAML constraints for Sprint ablations "
            "(e.g. force_bsts_off or regime gating toggles)."
        ),
    )
    parser.add_argument(
        "--interval_metrics",
        type=int,
        default=0,
        help="Set to 1 to export interval-quality scorecards/calibration tables.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    variants = [item.strip() for item in str(args.variants).split(",") if item.strip()]
    result = run_24_month_validation(
        end_target_month=args.end_target_month,
        runs=int(args.runs),
        variants=variants,
        asof_day=int(args.asof_day),
        report_root=args.report_root,
        dump_feature_row=bool(int(args.dump_feature_row)),
        n_trials=int(args.n_trials),
        seed_schedule=(args.seed_schedule or None),
        weight_search=bool(int(args.weight_search)),
        regime_split=bool(int(args.regime_split)),
        block_importance=bool(int(args.block_importance)),
        fusion_config_path=args.fusion_config,
        ablation_config_path=args.ablation_config,
        fusion_constraints_path=(args.fusion_constraints or None),
        extra_block=(args.extra_block or None),
        interval_metrics=bool(int(args.interval_metrics)),
    )
    print(
        json.dumps(
            {
                "point_rows": int(len(result.point_estimates)),
                "scorecard_rows": int(len(result.scorecard)),
                "summary_path": result.summary["summary_path"],
                "feature_rows_path": result.summary.get("feature_rows_path", ""),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
