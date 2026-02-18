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


def test_visualization_summary_contains_metadata_and_freshness(
    tmp_path: Path,
) -> None:
    report_root = tmp_path / "reports"
    artifact_root = tmp_path / "artifacts"
    output_root = tmp_path / "viz"

    _write_csv(
        report_root / "validation_24m_point_estimates.csv",
        "\n".join(
            [
                "model_variant,asof,target_month,fused_point,actual_released",
                "wpd_lstm_one_layer,2026-01-14,2025-11-30,3304.0,3303.0",
            ]
        ),
    )
    _write_csv(
        artifact_root / "nowcast" / "2026-01-14" / "release_history_36m.csv",
        "\n".join(
            [
                "timestamp,target_value,asof",
                "2025-10-31,3303.0,2026-01-14",
            ]
        ),
    )
    _write_json(
        artifact_root / "nowcast" / "2026-01-14" / "nowcast.json",
        {
            "asof": "2026-01-14",
            "nowcasts": [
                {
                    "horizon_label": "T-1",
                    "target_month": "2025-11-30",
                    "fused_point": 3304.0,
                    "fused_lower_95": 3299.0,
                    "fused_upper_95": 3309.0,
                }
            ],
        },
    )

    summary = generate_all_visuals(
        report_root=report_root,
        artifact_root=artifact_root,
        output_dir=output_root,
    )

    assert "generated_at_utc" in summary
    assert "source_roots" in summary
    assert "freshness" in summary
    assert summary["freshness"]["validation_max_target_month"] == "2025-11-30"
    assert summary["missing_source_count"] == len(summary["missing_sources"])

    summary_path = Path(summary["generated"]["summary"])
    persisted = json.loads(summary_path.read_text(encoding="utf-8"))
    assert persisted["missing_source_count"] == len(persisted["missing_sources"])
    assert "generated_at_utc" in persisted


def test_visualization_cleanup_removes_stale_generated_files(tmp_path: Path) -> None:
    report_root = tmp_path / "reports"
    artifact_root = tmp_path / "artifacts"
    output_root = tmp_path / "viz"
    output_root.mkdir(parents=True, exist_ok=True)

    stale = output_root / "validation_24m_stale_plot.html"
    stale.write_text("stale", encoding="utf-8")

    generate_all_visuals(
        report_root=report_root,
        artifact_root=artifact_root,
        output_dir=output_root,
        clean=True,
    )
    assert not stale.exists()

    stale.write_text("stale", encoding="utf-8")
    generate_all_visuals(
        report_root=report_root,
        artifact_root=artifact_root,
        output_dir=output_root,
        clean=False,
    )
    assert stale.exists()


def test_latest_vs_previous_nowcast_delta_visual_is_generated(tmp_path: Path) -> None:
    report_root = tmp_path / "reports"
    artifact_root = tmp_path / "artifacts"
    output_root = tmp_path / "viz"

    _write_json(
        artifact_root / "nowcast" / "2026-01-14" / "nowcast.json",
        {
            "asof": "2026-01-14",
            "nowcasts": [
                {
                    "horizon_label": "T-1",
                    "target_month": "2025-11-30",
                    "fused_point": 3300.0,
                    "fused_lower_95": 3290.0,
                    "fused_upper_95": 3310.0,
                },
                {
                    "horizon_label": "T",
                    "target_month": "2025-12-31",
                    "fused_point": 3310.0,
                    "fused_lower_95": 3300.0,
                    "fused_upper_95": 3320.0,
                },
            ],
        },
    )
    _write_json(
        artifact_root / "nowcast" / "2026-02-14" / "nowcast.json",
        {
            "asof": "2026-02-14",
            "nowcasts": [
                {
                    "horizon_label": "T-1",
                    "target_month": "2025-12-31",
                    "fused_point": 3330.0,
                    "fused_lower_95": 3320.0,
                    "fused_upper_95": 3340.0,
                },
                {
                    "horizon_label": "T",
                    "target_month": "2026-01-31",
                    "fused_point": 3345.0,
                    "fused_lower_95": 3330.0,
                    "fused_upper_95": 3360.0,
                },
            ],
        },
    )

    summary = generate_all_visuals(
        report_root=report_root,
        artifact_root=artifact_root,
        output_dir=output_root,
    )

    key = "nowcast_latest_vs_previous_delta"
    assert key in summary["generated"]
    assert Path(summary["generated"][key]).exists()
    assert summary["dataset_rows"]["latest_vs_previous_delta"] == 2
