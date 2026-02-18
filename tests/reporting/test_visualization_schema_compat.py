from __future__ import annotations

import json
from pathlib import Path

from ops.viz.generate_pipeline_visuals import generate_all_visuals


def _write_csv(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _seed_report_files(report_root: Path) -> None:
    _write_csv(
        report_root / "ablation_scorecard.csv",
        "\n".join(
            [
                "experiment_id,mae,rmse,mape,dm_vs_baseline_p_value,runtime_seconds,lineage_id",
                "B0_baseline,1.2,1.5,0.01,1.0,40.0,lineage_0",
                "B4_full_method,1.0,1.2,0.01,0.08,72.0,lineage_4",
            ]
        ),
    )
    _write_csv(
        report_root / "backtest_scorecard.csv",
        "\n".join(
            [
                "horizon,target_month,mae,rmse,mape",
                "T-1,2025-12-31,0.5,0.6,0.01",
                "T,2026-01-31,0.6,0.7,0.01",
            ]
        ),
    )
    _write_csv(
        report_root / "dm_results.csv",
        "\n".join(
            [
                "target,candidate_model,benchmark_model,d_bar,dm_stat,p_value,significant_0_05,significant_0_01,adjusted_p_value",
                "ng_prod,challenger,champion,-0.2,-1.6,0.09,false,false,0.09",
            ]
        ),
    )
    _write_json(
        report_root / "weekly_ops_cycle_report.json",
        {
            "asof": "2026-02-14",
            "dag_runs": [
                {"dag_id": "nowcast_pipeline_weekly", "tasks": {"a": {}, "b": {}}}
            ],
        },
    )


def _seed_bronze_files(bronze_root: Path) -> None:
    _write_csv(
        bronze_root / "eia_api" / "asof=2026-02-14" / "weekly_prices.csv",
        "\n".join(["timestamp,value", "2026-01-31,101.2", "2026-02-28,102.1"]),
    )
    _write_csv(
        bronze_root / "eia_bulk" / "asof=2026-02-14" / "weekly_prices.csv",
        "\n".join(["timestamp,value", "2026-01-31,101.1", "2026-02-28,102.0"]),
    )
    _write_csv(
        bronze_root / "dead_letter" / "asof=2026-02-14" / "readme.txt",
        "dead-letter-placeholder",
    )


def test_visualization_generator_supports_target_month_schema(tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    report_root = tmp_path / "reports"
    output_root = tmp_path / "viz"
    _seed_report_files(report_root)

    _write_json(
        artifact_root / "nowcast" / "2026-02-14" / "nowcast.json",
        {
            "asof": "2026-02-14",
            "target_month": "2025-12-31",
            "nowcasts": [
                {
                    "horizon_label": "T-1",
                    "target_month_label": "target_month",
                    "target_month": "2025-12-31",
                    "fused_point": 100.0,
                    "fused_lower_95": 99.0,
                    "fused_upper_95": 101.0,
                },
                {
                    "horizon_label": "T",
                    "target_month_label": "target_month_plus_1",
                    "target_month": "2026-01-31",
                    "fused_point": 101.0,
                    "fused_lower_95": 100.0,
                    "fused_upper_95": 102.0,
                },
            ],
        },
    )

    summary = generate_all_visuals(
        report_root=report_root,
        artifact_root=artifact_root,
        output_dir=output_root,
    )

    assert "nowcast_intervals" in summary["generated"]
    assert Path(summary["generated"]["nowcast_intervals"]).exists()


def test_visualization_generator_remains_backward_compatible_with_legacy_nowcast_schema(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    report_root = tmp_path / "reports"
    output_root = tmp_path / "viz"
    _seed_report_files(report_root)

    _write_json(
        artifact_root / "nowcast" / "2025-02-28" / "nowcast.json",
        {
            "asof": "2025-02-28",
            "nowcasts": [
                {
                    "horizon_label": "T-1",
                    "fused_point": 99.5,
                    "fused_lower_95": 98.2,
                    "fused_upper_95": 100.8,
                },
                {
                    "horizon_label": "T",
                    "fused_point": 100.3,
                    "fused_lower_95": 99.1,
                    "fused_upper_95": 101.7,
                },
            ],
        },
    )

    summary = generate_all_visuals(
        report_root=report_root,
        artifact_root=artifact_root,
        output_dir=output_root,
    )

    assert "nowcast_intervals" in summary["generated"]
    assert Path(summary["generated"]["nowcast_intervals"]).exists()


def test_visualization_generator_builds_ingestion_time_series_from_bronze_data(
    tmp_path: Path,
) -> None:
    artifact_root = tmp_path / "artifacts"
    report_root = tmp_path / "reports"
    bronze_root = tmp_path / "bronze"
    output_root = tmp_path / "viz"
    _seed_report_files(report_root)
    _seed_bronze_files(bronze_root)

    _write_json(
        artifact_root / "nowcast" / "2026-02-14" / "nowcast.json",
        {
            "asof": "2026-02-14",
            "nowcasts": [
                {
                    "horizon_label": "T-1",
                    "fused_point": 100.0,
                    "fused_lower_95": 99.0,
                    "fused_upper_95": 101.0,
                }
            ],
        },
    )

    summary = generate_all_visuals(
        report_root=report_root,
        artifact_root=artifact_root,
        output_dir=output_root,
    )

    assert "ingest_file_counts" in summary["generated"]
    assert "ingested_weekly_prices" in summary["generated"]
    assert Path(summary["generated"]["ingest_file_counts"]).exists()
    assert Path(summary["generated"]["ingested_weekly_prices"]).exists()
    assert summary["dataset_rows"]["ingest_inventory"] > 0
    assert summary["dataset_rows"]["ingested_prices"] > 0
