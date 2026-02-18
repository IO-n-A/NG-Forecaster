#!/usr/bin/env python3
"""Execute the full weekly orchestration cycle without requiring a running Airflow instance."""

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

from ng_forecaster.orchestration.airflow.dags.eia_metadata_refresh import (
    run_eia_metadata_refresh,
)
from ng_forecaster.orchestration.airflow.dags.eia_api_ingest import run_eia_api_ingest
from ng_forecaster.orchestration.airflow.dags.eia_bulk_ingest import run_eia_bulk_ingest
from ng_forecaster.orchestration.airflow.dags.nowcast_pipeline_weekly import (
    run_nowcast_pipeline_weekly,
)
from ng_forecaster.orchestration.airflow.dags.realtime_backtest_runner import (
    run_realtime_backtest_runner,
)
from ng_forecaster.orchestration.airflow.dags.nowcast_ablation_runner import (
    run_nowcast_ablation_runner,
)
from ng_forecaster.data.refresh import run_data_refresh_cycle
from ng_forecaster.orchestration.airflow.workflow_support import (
    bootstrap_status,
    resolve_weekly_asof,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asof",
        default=None,
        help="As-of timestamp for the weekly cycle (default resolves from fixtures).",
    )
    parser.add_argument(
        "--output",
        default="data/reports/weekly_ops_cycle_report.json",
        help="Output path for cycle report JSON.",
    )
    parser.add_argument(
        "--skip-data-refresh",
        action="store_true",
        help="Skip Sprint 4B data refresh cycle before DAG execution.",
    )
    parser.add_argument(
        "--refresh-retrieve-steo",
        action="store_true",
        help="Run network STEO retrieval during data refresh.",
    )
    parser.add_argument(
        "--refresh-retrieve-context",
        action="store_true",
        help="Run context-report retrieval during data refresh.",
    )
    parser.add_argument(
        "--retire-data-new",
        action="store_true",
        help="Delete data/new after successful data refresh.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    asof_ts = resolve_weekly_asof(args.asof)

    bootstrap = bootstrap_status()
    data_refresh_summary = None
    if not args.skip_data_refresh:
        data_refresh_summary = run_data_refresh_cycle(
            asof=asof_ts.isoformat(),
            run_steo_retrieval=bool(args.refresh_retrieve_steo),
            run_context_retrieval=bool(args.refresh_retrieve_context),
            retire_source_dir=bool(args.retire_data_new),
        ).as_dict()

    runs = [
        run_eia_metadata_refresh(asof=asof_ts.isoformat()),
        run_eia_api_ingest(asof=asof_ts.isoformat()),
        run_eia_bulk_ingest(asof=asof_ts.isoformat()),
        run_nowcast_pipeline_weekly(asof=asof_ts.isoformat()),
        run_realtime_backtest_runner(asof=asof_ts.isoformat()),
        run_nowcast_ablation_runner(asof=asof_ts.isoformat()),
    ]

    target_month: str | None = None
    release_policy: dict[str, object] | None = None
    for run in runs:
        if str(run.get("dag_id", "")) == "nowcast_pipeline_weekly":
            value = run.get("target_month")
            if value is not None:
                target_month = str(value)
            policy_payload = run.get("release_policy")
            if isinstance(policy_payload, dict):
                release_policy = dict(policy_payload)
            break

    payload = {
        "asof": asof_ts.date().isoformat(),
        "target_month": target_month,
        "release_policy": release_policy,
        "bootstrap_available": bool(bootstrap["available"]),
        "bootstrap_raw_file_count": int(bootstrap["raw_file_count"]),
        "data_refresh": {
            "enabled": not args.skip_data_refresh,
            "summary": data_refresh_summary,
        },
        "dag_runs": runs,
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2), encoding="utf-8"
    )

    print(f"PASS: weekly ops cycle completed for asof={asof_ts.date().isoformat()}")
    print(
        json.dumps({"output": str(output_path), "dag_count": len(runs)}, sort_keys=True)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
