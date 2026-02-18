#!/usr/bin/env python3
"""Run the full NG-Forecaster pipeline without requiring weekly Airflow DAG scheduling."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

ROOT = Path(__file__).resolve().parent
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from ng_forecaster.errors import ContractViolation
from ng_forecaster.data.refresh import run_data_refresh_cycle
from ng_forecaster.orchestration.airflow.dags.eia_api_ingest import run_eia_api_ingest
from ng_forecaster.orchestration.airflow.dags.eia_bulk_ingest import run_eia_bulk_ingest
from ng_forecaster.orchestration.airflow.dags.eia_metadata_refresh import (
    run_eia_metadata_refresh,
)
from ng_forecaster.orchestration.airflow.dags.nowcast_ablation_runner import (
    run_nowcast_ablation_runner,
)
from ng_forecaster.orchestration.airflow.dags.nowcast_pipeline_weekly import (
    run_nowcast_pipeline_weekly,
)
from ng_forecaster.orchestration.airflow.dags.realtime_backtest_runner import (
    run_realtime_backtest_runner,
)
from ng_forecaster.orchestration.airflow.workflow_support import (
    bootstrap_status,
    resolve_weekly_asof,
)
from ng_forecaster.qa.n4_calibration_gate import check_n4_acceptance
from ng_forecaster.qa.n5_policy_audit import audit_n5_policy
from ng_forecaster.qa.n6_adoption_gate import check_n6_adoption_readiness
from ng_forecaster.qa.ops_readiness import check_ops_readiness
from ng_forecaster.qa.preprocess_gate import (
    check_preprocess_gate,
    resolve_latest_nowcast_artifact_dir,
)
from ng_forecaster.qa.target_month_gate import check_target_month_gate
from ops.viz.generate_pipeline_visuals import generate_all_visuals


def _run_workflow(asof_iso: str) -> list[dict[str, Any]]:
    return [
        run_eia_metadata_refresh(asof=asof_iso),
        run_eia_api_ingest(asof=asof_iso),
        run_eia_bulk_ingest(asof=asof_iso),
        run_nowcast_pipeline_weekly(asof=asof_iso),
        run_realtime_backtest_runner(asof=asof_iso),
        run_nowcast_ablation_runner(asof=asof_iso),
    ]


def _gate_result_payload(result: Any) -> dict[str, Any]:
    if hasattr(result, "as_dict"):
        return dict(result.as_dict())
    if isinstance(result, dict):
        return dict(result)
    return {"result": str(result)}


def _run_qa_gates() -> tuple[dict[str, Any], bool]:
    gates: dict[str, Any] = {}
    all_passed = True

    def _record(name: str, fn) -> None:
        nonlocal all_passed
        try:
            result = fn()
            gates[name] = {"passed": True, **_gate_result_payload(result)}
        except ContractViolation as exc:
            all_passed = False
            gates[name] = {
                "passed": False,
                "error": str(exc),
            }

    _record(
        "target_month_gate",
        lambda: check_target_month_gate(),
    )
    _record(
        "preprocess_gate",
        lambda: check_preprocess_gate(resolve_latest_nowcast_artifact_dir()),
    )
    _record(
        "n4_acceptance",
        lambda: check_n4_acceptance(
            "data/reports/n4_seed_stability.csv",
            "data/reports/n4_calibration_summary.csv",
            "data/reports/n4_nowcast_outputs.csv",
        ),
    )
    _record(
        "n5_policy_audit",
        lambda: audit_n5_policy(
            "configs/evaluation.yaml", "data/reports/dm_results.csv"
        ),
    )
    _record(
        "n6_adoption",
        lambda: check_n6_adoption_readiness("data/reports/ablation_scorecard.csv"),
    )
    _record("ops_readiness", lambda: check_ops_readiness())

    return gates, all_passed


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asof",
        default=None,
        help="As-of date/time for pipeline execution.",
    )
    parser.add_argument(
        "--output",
        default="data/reports/pipeline_run_summary.json",
        help="Summary JSON output path.",
    )
    parser.add_argument(
        "--skip-qa",
        action="store_true",
        help="Skip post-run QA gate execution.",
    )
    parser.add_argument(
        "--skip-viz",
        action="store_true",
        help="Skip visualization generation under ops/viz.",
    )
    parser.add_argument(
        "--skip-data-refresh",
        action="store_true",
        help="Skip Sprint 4B data refresh cycle (migration/silver/gold).",
    )
    parser.add_argument(
        "--refresh-retrieve-steo",
        action="store_true",
        help="Run network retrieval for STEO archives before refresh publish.",
    )
    parser.add_argument(
        "--refresh-retrieve-context",
        action="store_true",
        help="Run network retrieval for context reports before refresh publish.",
    )
    parser.add_argument(
        "--retire-data-new",
        action="store_true",
        help="Delete data/new after a successful refresh cycle.",
    )
    parser.add_argument(
        "--viz-output-dir",
        default="ops/viz/output",
        help="Visualization output directory.",
    )
    parser.add_argument(
        "--strict-gates",
        action="store_true",
        default=True,
        help="Exit non-zero if any QA gate fails (default: enabled).",
    )
    parser.add_argument(
        "--no-strict-gates",
        dest="strict_gates",
        action="store_false",
        help="Do not fail process if QA gates fail.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    asof_ts = resolve_weekly_asof(args.asof)
    asof_iso = asof_ts.isoformat()

    bootstrap = bootstrap_status()
    data_refresh_summary: dict[str, Any] | None = None
    if not args.skip_data_refresh:
        data_refresh_summary = run_data_refresh_cycle(
            asof=asof_iso,
            run_steo_retrieval=bool(args.refresh_retrieve_steo),
            run_context_retrieval=bool(args.refresh_retrieve_context),
            retire_source_dir=bool(args.retire_data_new),
        ).as_dict()
    workflow_runs = _run_workflow(asof_iso)

    target_month: str | None = None
    release_policy: dict[str, Any] | None = None
    for run in workflow_runs:
        if str(run.get("dag_id", "")) == "nowcast_pipeline_weekly":
            value = run.get("target_month")
            if value is not None:
                target_month = str(value)
            policy_payload = run.get("release_policy")
            if isinstance(policy_payload, dict):
                release_policy = dict(policy_payload)
            break

    qa_results: dict[str, Any] = {}
    qa_passed = True
    if not args.skip_qa:
        qa_results, qa_passed = _run_qa_gates()

    viz_summary: dict[str, Any] | None = None
    if not args.skip_viz:
        viz_summary = generate_all_visuals(output_dir=args.viz_output_dir)

    summary = {
        "asof": asof_ts.date().isoformat(),
        "target_month": target_month,
        "release_policy": release_policy,
        "bootstrap_available": bool(bootstrap["available"]),
        "bootstrap_raw_file_count": int(bootstrap["raw_file_count"]),
        "data_refresh": {
            "enabled": not args.skip_data_refresh,
            "summary": data_refresh_summary,
        },
        "dag_runs": workflow_runs,
        "qa": {
            "enabled": not args.skip_qa,
            "passed": qa_passed,
            "gates": qa_results,
        },
        "visualization": {
            "enabled": not args.skip_viz,
            "summary": viz_summary,
        },
    }

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    print(f"Pipeline run complete for asof={asof_ts.date().isoformat()}")
    print(
        json.dumps(
            {
                "output": str(output_path),
                "dag_count": len(workflow_runs),
                "qa_passed": qa_passed,
                "viz_enabled": not args.skip_viz,
            },
            sort_keys=True,
        )
    )

    if args.strict_gates and not args.skip_qa and not qa_passed:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
