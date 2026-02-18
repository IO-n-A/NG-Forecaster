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


def test_validation_24m_visual_is_generated_when_sources_exist(tmp_path: Path) -> None:
    report_root = tmp_path / "reports"
    artifact_root = tmp_path / "artifacts"
    output_root = tmp_path / "viz"

    _write_csv(
        report_root / "validation_24m_point_estimates.csv",
        "\n".join(
            [
                "model_variant,asof,target_month,fused_point,fused_lower_95,fused_upper_95,actual_released,error,abs_error,ape_pct,interval_hit_95,release_history_rows,release_history_last",
                "wpd_lstm_one_layer,2026-01-14,2025-11-30,3304.0,3298.0,3310.0,3303.0,1.0,1.0,0.03,true,36,3303.0",
                "wpd_vmd_lstm1,2026-01-14,2025-11-30,3303.2,3299.0,3308.0,3303.0,0.2,0.2,0.01,true,36,3303.0",
            ]
        ),
    )
    _write_csv(
        artifact_root / "nowcast" / "2026-01-14" / "release_history_36m.csv",
        "\n".join(
            [
                "timestamp,target_value,asof,source",
                "2025-09-30,3290.0,2026-01-14,test",
                "2025-10-31,3303.0,2026-01-14,test",
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
                    "fused_point": 3303.0,
                    "fused_lower_95": 3295.0,
                    "fused_upper_95": 3311.0,
                }
            ],
        },
    )

    summary = generate_all_visuals(
        report_root=report_root,
        artifact_root=artifact_root,
        output_dir=output_root,
    )

    key = "validation_24m_point_vs_release_36m"
    assert key in summary["generated"]
    assert Path(summary["generated"][key]).exists()
    official_key = "validation_24m_point_vs_official_releases"
    assert official_key in summary["generated"]
    assert Path(summary["generated"][official_key]).exists()
    assert summary["dataset_rows"]["validation_24m_points"] == 2
    assert summary["dataset_rows"]["validation_release_36m"] == 2


def test_validation_24m_visual_handles_missing_sources(tmp_path: Path) -> None:
    report_root = tmp_path / "reports"
    artifact_root = tmp_path / "artifacts"
    output_root = tmp_path / "viz"
    report_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    summary = generate_all_visuals(
        report_root=report_root,
        artifact_root=artifact_root,
        output_dir=output_root,
    )

    key = "validation_24m_point_vs_release_36m"
    assert key in summary["generated"]
    assert Path(summary["generated"][key]).exists()
    official_key = "validation_24m_point_vs_official_releases"
    assert official_key in summary["generated"]
    assert Path(summary["generated"][official_key]).exists()
    assert (
        str(report_root / "validation_24m_point_estimates.csv")
        in summary["missing_sources"]
    )
