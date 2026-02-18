from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import ng_forecaster.evaluation.validation_24m as validation_24m
from ng_forecaster.evaluation.month_length_effect import (
    build_calendar_calibration_report,
    build_month_length_by_regime_report,
    build_month_length_effect_report,
)


def test_month_length_effect_builders_emit_expected_columns() -> None:
    frame = pd.DataFrame(
        {
            "model_variant": ["v1"] * 8,
            "target_month_days": [30, 31, 30, 31, 30, 31, 30, 31],
            "target_month": pd.date_range("2025-01-31", periods=8, freq="ME"),
            "regime_label": ["normal", "normal", "freeze_off", "freeze_off"] * 2,
            "ape_pct": [2.0, 2.2, 3.0, 3.4, 1.8, 2.1, 2.9, 3.1],
            "ape_pct_raw": [2.1, 2.4, 3.1, 3.5, 1.9, 2.2, 3.0, 3.2],
            "abs_error": [10.0, 11.0, 14.0, 16.0, 9.0, 10.0, 13.0, 15.0],
            "abs_error_raw": [10.5, 12.0, 14.5, 16.5, 9.3, 10.5, 13.5, 15.5],
            "calendar_calibration_applied": [1, 1, 1, 1, 0, 0, 1, 1],
            "day_count_class": ["30d", "31d", "30d", "31d", "30d", "31d", "30d", "31d"],
        }
    )
    effect = build_month_length_effect_report(frame)
    by_regime = build_month_length_by_regime_report(frame)
    calibration = build_calendar_calibration_report(frame)
    assert {
        "raw_gap_31d_minus_30d_ape_pct",
        "controlled_day31_coef_ape_pct",
    }.issubset(effect.columns)
    assert {"regime_label", "gap_31d_minus_30d_ape_pct"}.issubset(by_regime.columns)
    assert {
        "mean_ape_pct_raw",
        "mean_ape_pct_calibrated",
        "delta_ape_pct_calibrated_minus_raw",
    }.issubset(calibration.columns)


def test_run_24m_validation_exports_month_length_effect_and_calibration(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def _fake_enforce_release_policy(asof: object) -> dict[str, object]:
        asof_ts = pd.Timestamp(asof)
        target_month = (asof_ts.to_period("M") - 2).to_timestamp("M")
        return {
            "asof": asof_ts.date().isoformat(),
            "target_month": target_month.date().isoformat(),
            "policy_passed": True,
        }

    def _fake_load_market_inputs(asof: object) -> dict[str, pd.DataFrame]:
        asof_ts = pd.Timestamp(asof)
        full_history = pd.DataFrame(
            {
                "timestamp": pd.date_range("2023-01-31", periods=40, freq="ME"),
                "target_value": [3000.0 + idx for idx in range(40)],
            }
        )
        features = pd.DataFrame(
            [
                {
                    "feature_name": "freeze_days_mtd_weighted",
                    "feature_timestamp": asof_ts,
                    "available_timestamp": asof_ts,
                    "value": 1.0,
                }
            ]
        )
        monthly_release = full_history.tail(36).reset_index(drop=True)
        return {
            "target_history_full": full_history,
            "target_history": full_history.copy(),
            "features": features,
            "monthly_release_history": monthly_release,
        }

    def _fake_run_nowcast_pipeline_weekly(
        *,
        asof: str | None = None,
        champion_config_override: dict[str, object] | None = None,
        fusion_config_override: dict[str, object] | None = None,
        idempotency_token: str | None = None,
    ) -> dict[str, object]:
        _ = champion_config_override
        _ = fusion_config_override
        _ = idempotency_token
        asof_value = str(asof or "2026-01-14")
        target_month = (pd.Timestamp(asof_value).to_period("M") - 2).to_timestamp("M")
        artifact_dir = Path("data/artifacts/nowcast") / asof_value
        artifact_dir.mkdir(parents=True, exist_ok=True)
        (artifact_dir / "nowcast.json").write_text(
            json.dumps(
                {
                    "asof": asof_value,
                    "target_month": target_month.date().isoformat(),
                    "nowcasts": [
                        {
                            "target_month": target_month.date().isoformat(),
                            "fused_point": 3110.0,
                            "fused_point_pre_calendar_calibration": 3098.0,
                            "calendar_calibration_delta": 12.0,
                            "calendar_calibration_applied": True,
                            "fused_lower_95": 3000.0,
                            "fused_upper_95": 3200.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        pd.DataFrame(
            {
                "timestamp": pd.date_range("2022-01-31", periods=36, freq="ME"),
                "target_value": [2900.0 + idx for idx in range(36)],
            }
        ).to_csv(artifact_dir / "release_history_36m.csv", index=False)
        pd.DataFrame(
            [
                {
                    "horizon": 1,
                    "point_forecast": 3110.4,
                    "mean_forecast": 3109.8,
                    "challenger_lower_95": 3000.0,
                    "challenger_upper_95": 3200.0,
                    "applied_champion_weight": 0.7,
                }
            ]
        ).to_csv(artifact_dir / "fusion_inputs.csv", index=False)
        return {"target_month": target_month.date().isoformat()}

    monkeypatch.setattr(
        validation_24m, "enforce_release_policy", _fake_enforce_release_policy
    )
    monkeypatch.setattr(validation_24m, "load_market_inputs", _fake_load_market_inputs)
    monkeypatch.setattr(
        validation_24m,
        "run_nowcast_pipeline_weekly",
        _fake_run_nowcast_pipeline_weekly,
    )

    result = validation_24m.run_24_month_validation(
        end_target_month="2025-11-30",
        runs=1,
        variants=("wpd_lstm_one_layer",),
        report_root=tmp_path / "reports",
    )
    month_effect_path = Path(result.summary["month_length_effect_path"])
    month_by_regime_path = Path(result.summary["month_length_by_regime_path"])
    calibration_path = Path(result.summary["calendar_calibration_path"])
    assert month_effect_path.exists()
    assert month_by_regime_path.exists()
    assert calibration_path.exists()
    assert len(pd.read_csv(month_effect_path)) >= 1
    assert len(pd.read_csv(month_by_regime_path)) >= 1
    assert len(pd.read_csv(calibration_path)) >= 1
