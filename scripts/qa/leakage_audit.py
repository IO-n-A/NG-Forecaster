#!/usr/bin/env python3
"""Leakage audit over rolling validation months using feature availability timestamps."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from ng_forecaster.data.validators import load_yaml
from ng_forecaster.evaluation.validation_24m import (
    build_target_month_grid,
    derive_policy_admissible_asof,
)
from ng_forecaster.orchestration.airflow.workflow_support import load_market_inputs


def run_leakage_audit(
    *,
    end_target_month: str,
    runs: int,
    asof_day: int,
    report_root: Path,
    fail_on_violations: bool,
) -> dict[str, object]:
    source_cfg = load_yaml("configs/sources.yaml")
    release_cfg = source_cfg["release_calendar"]
    lag_months = int(release_cfg["lag_months"])
    release_day = int(release_cfg["release_day_of_month"])

    months = build_target_month_grid(end_target_month=end_target_month, runs=int(runs))
    violations: list[dict[str, object]] = []
    month_rows: list[dict[str, object]] = []

    for target_month in months:
        asof = derive_policy_admissible_asof(
            target_month=target_month,
            lag_months=lag_months,
            release_day_of_month=release_day,
            preferred_day=int(asof_day),
        )
        inputs = load_market_inputs(asof)
        features = inputs.get("features")
        target_history = inputs.get("target_history")
        latest_released = pd.Timestamp(inputs.get("latest_released_month"))

        month_violations = 0
        if isinstance(features, pd.DataFrame) and not features.empty:
            feature_payload = features.copy()
            feature_payload["feature_timestamp"] = pd.to_datetime(
                feature_payload.get("feature_timestamp"),
                errors="coerce",
            )
            feature_payload["available_timestamp"] = pd.to_datetime(
                feature_payload.get("available_timestamp"),
                errors="coerce",
            )
            for _, row in feature_payload.iterrows():
                feature_name = str(row.get("feature_name", ""))
                block_id = str(row.get("block_id", ""))
                if (
                    pd.notna(row.get("feature_timestamp"))
                    and pd.Timestamp(row["feature_timestamp"]) > asof
                ):
                    month_violations += 1
                    violations.append(
                        {
                            "asof": asof.date().isoformat(),
                            "target_month": target_month.date().isoformat(),
                            "violation_type": "feature_timestamp_after_asof",
                            "feature_name": feature_name,
                            "block_id": block_id,
                            "feature_timestamp": pd.Timestamp(
                                row["feature_timestamp"]
                            ).isoformat(),
                            "available_timestamp": (
                                pd.Timestamp(row["available_timestamp"]).isoformat()
                                if pd.notna(row.get("available_timestamp"))
                                else ""
                            ),
                        }
                    )
                if (
                    pd.notna(row.get("available_timestamp"))
                    and pd.Timestamp(row["available_timestamp"]) > asof
                ):
                    month_violations += 1
                    violations.append(
                        {
                            "asof": asof.date().isoformat(),
                            "target_month": target_month.date().isoformat(),
                            "violation_type": "available_timestamp_after_asof",
                            "feature_name": feature_name,
                            "block_id": block_id,
                            "feature_timestamp": (
                                pd.Timestamp(row["feature_timestamp"]).isoformat()
                                if pd.notna(row.get("feature_timestamp"))
                                else ""
                            ),
                            "available_timestamp": pd.Timestamp(
                                row["available_timestamp"]
                            ).isoformat(),
                        }
                    )

        if isinstance(target_history, pd.DataFrame) and not target_history.empty:
            target_payload = target_history.copy()
            target_payload["timestamp"] = pd.to_datetime(
                target_payload.get("timestamp"),
                errors="coerce",
            )
            max_ts = pd.Timestamp(target_payload["timestamp"].max())
            if pd.notna(max_ts) and max_ts > latest_released:
                month_violations += 1
                violations.append(
                    {
                        "asof": asof.date().isoformat(),
                        "target_month": target_month.date().isoformat(),
                        "violation_type": "target_history_after_latest_released_month",
                        "feature_name": "",
                        "block_id": "target_history",
                        "feature_timestamp": max_ts.isoformat(),
                        "available_timestamp": latest_released.isoformat(),
                    }
                )

        month_rows.append(
            {
                "asof": asof.date().isoformat(),
                "target_month": target_month.date().isoformat(),
                "violations": int(month_violations),
                "has_violation": bool(month_violations > 0),
            }
        )

    report_root.mkdir(parents=True, exist_ok=True)
    violations_df = pd.DataFrame(
        violations,
        columns=[
            "asof",
            "target_month",
            "violation_type",
            "feature_name",
            "block_id",
            "feature_timestamp",
            "available_timestamp",
        ],
    )
    summary_month = pd.DataFrame(month_rows)
    summary_type = (
        violations_df.groupby("violation_type", sort=True)
        .size()
        .rename("count")
        .reset_index()
        if not violations_df.empty
        else pd.DataFrame(columns=["violation_type", "count"])
    )
    summary_totals = pd.DataFrame(
        [
            {
                "runs": int(len(months)),
                "months_with_violations": int(summary_month["has_violation"].sum()),
                "total_violations": int(len(violations_df)),
            }
        ]
    )

    violations_path = report_root / "leakage_audit_violations.csv"
    summary_month_path = report_root / "leakage_audit_summary_by_month.csv"
    summary_type_path = report_root / "leakage_audit_summary_by_type.csv"
    summary_totals_path = report_root / "leakage_audit_summary.csv"
    violations_df.to_csv(violations_path, index=False)
    summary_month.to_csv(summary_month_path, index=False)
    summary_type.to_csv(summary_type_path, index=False)
    summary_totals.to_csv(summary_totals_path, index=False)

    payload: dict[str, object] = {
        "violations_path": str(violations_path),
        "summary_path": str(summary_totals_path),
        "summary_by_month_path": str(summary_month_path),
        "summary_by_type_path": str(summary_type_path),
        "runs": int(len(months)),
        "total_violations": int(len(violations_df)),
    }
    (report_root / "leakage_audit_summary.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    if fail_on_violations and len(violations_df) > 0:
        raise SystemExit(2)
    return payload


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--end-target-month", default="2025-11-30")
    parser.add_argument("--runs", type=int, default=24)
    parser.add_argument("--asof-day", type=int, default=14)
    parser.add_argument(
        "--report-root",
        default="data/reports/sprint9/wpdlstm2_sanity",
        help="Output folder for leakage audit exports.",
    )
    parser.add_argument(
        "--fail-on-violations",
        type=int,
        default=1,
        help="Set to 1 to exit non-zero when violations are found.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    payload = run_leakage_audit(
        end_target_month=str(args.end_target_month),
        runs=int(args.runs),
        asof_day=int(args.asof_day),
        report_root=Path(args.report_root),
        fail_on_violations=bool(int(args.fail_on_violations)),
    )
    print(json.dumps(payload, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
