from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import ng_forecaster.evaluation.validation_24m as validation_24m
from ng_forecaster.errors import ContractViolation


def test_build_target_month_grid_is_deterministic() -> None:
    grid = validation_24m.build_target_month_grid(
        end_target_month="2025-11-30",
        runs=3,
    )
    assert [item.date().isoformat() for item in grid] == [
        "2025-09-30",
        "2025-10-31",
        "2025-11-30",
    ]


def test_resolve_validation_variants_supports_full_plus_prototype_alias() -> None:
    resolved = validation_24m.resolve_validation_variants(("full_plus_prototype",))
    assert resolved == list(validation_24m.DEFAULT_VARIANTS)


def test_resolve_validation_variants_supports_challenger_bsts_aliases() -> None:
    resolved_single = validation_24m.resolve_validation_variants(("challenger_bsts",))
    assert resolved_single == ["challenger_bsts"]

    resolved_bundle = validation_24m.resolve_validation_variants(("full_plus_regime",))
    assert "challenger_bsts" in resolved_bundle


def test_derive_policy_admissible_asof_matches_release_target_month() -> None:
    asof = validation_24m.derive_policy_admissible_asof(
        target_month=pd.Timestamp("2025-11-30"),
        lag_months=2,
        release_day_of_month=30,
        preferred_day=14,
    )
    assert asof.date().isoformat() == "2026-01-14"


def test_run_24_month_validation_exports_sorted_outputs(
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
                    "value": 0.0,
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
        _ = fusion_config_override
        _ = idempotency_token
        asof_value = str(asof or "2026-01-14")
        override = champion_config_override or {}
        model_cfg = override.get("model", {})
        variant = (
            str(model_cfg.get("variant", "wpd_vmd_lstm1"))
            if isinstance(model_cfg, dict)
            else "wpd_vmd_lstm1"
        )
        offset = {
            "wpd_lstm_one_layer": 12.0,
            "wpd_vmd_lstm1": 8.0,
            "wpd_vmd_lstm2": 10.0,
        }[variant]
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
                            "fused_point": 3100.0 + offset,
                            "fused_lower_95": 3000.0,
                            "fused_upper_95": 3200.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        history = pd.DataFrame(
            {
                "timestamp": pd.date_range("2022-01-31", periods=36, freq="ME"),
                "target_value": [2900.0 + idx for idx in range(36)],
                "asof": asof_value,
                "source": "test",
            }
        )
        history.to_csv(artifact_dir / "release_history_36m.csv", index=False)
        pd.DataFrame(
            [
                {
                    "horizon": 1,
                    "point_forecast": 3100.0 + offset + 0.4,
                    "mean_forecast": 3100.0 + offset - 0.2,
                    "challenger_lower_95": 3000.0,
                    "challenger_upper_95": 3200.0,
                    "applied_champion_weight": 0.7,
                },
                {
                    "horizon": 2,
                    "point_forecast": 3110.0 + offset + 0.4,
                    "mean_forecast": 3110.0 + offset - 0.2,
                    "challenger_lower_95": 3010.0,
                    "challenger_upper_95": 3210.0,
                    "applied_champion_weight": 0.68,
                },
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

    report_root = tmp_path / "reports"
    result = validation_24m.run_24_month_validation(
        end_target_month="2025-11-30",
        runs=2,
        variants=("wpd_lstm_one_layer", "wpd_vmd_lstm1"),
        report_root=report_root,
    )

    assert len(result.point_estimates) == 4
    assert len(result.scorecard) == 2
    assert result.point_estimates.equals(
        result.point_estimates.sort_values(
            ["model_variant", "target_month", "asof"]
        ).reset_index(drop=True)
    )
    assert {"target_month_days", "day_count_class", "is_february"}.issubset(
        set(result.point_estimates.columns)
    )
    for _, row in result.point_estimates.iterrows():
        error = float(row["error"])
        fused_point = float(row["fused_point"])
        actual = float(row["actual_released"])
        assert error == fused_point - actual
        assert float(row["abs_error"]) == abs(error)
        expected_ape = abs(error) / abs(actual) * 100.0
        assert float(row["ape_pct"]) == expected_ape

    assert set(result.point_estimates["day_count_class"]) == {"30d", "31d"}
    assert not result.point_estimates["is_february"].astype(bool).any()
    assert {"mean_signed_error", "mean_abs_error", "february_run_count"}.issubset(
        set(result.scorecard.columns)
    )
    assert (result.scorecard["mean_signed_error"] > 0).all()
    assert (result.scorecard["mean_abs_error"] > 0).all()
    assert (result.scorecard["february_run_count"] == 0).all()

    assert (report_root / "validation_24m_point_estimates.csv").exists()
    assert (report_root / "validation_24m_scorecard.csv").exists()
    assert (report_root / "validation_24m_month_length_diagnostics.csv").exists()
    assert (report_root / "validation_24m_summary.json").exists()


def test_run_24_month_validation_supports_alias_and_feature_dump(
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
                    "value": 0.0,
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
        _ = fusion_config_override
        _ = idempotency_token
        asof_value = str(asof or "2026-01-14")
        override = champion_config_override or {}
        model_cfg = override.get("model", {})
        variant = (
            str(model_cfg.get("variant", "wpd_vmd_lstm1"))
            if isinstance(model_cfg, dict)
            else "wpd_vmd_lstm1"
        )
        offset = {
            "wpd_lstm_one_layer": 12.0,
            "wpd_vmd_lstm1": 8.0,
            "wpd_vmd_lstm2": 10.0,
        }[variant]
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
                            "fused_point": 3100.0 + offset,
                            "fused_lower_95": 3000.0,
                            "fused_upper_95": 3200.0,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        history = pd.DataFrame(
            {
                "timestamp": pd.date_range("2022-01-31", periods=36, freq="ME"),
                "target_value": [2900.0 + idx for idx in range(36)],
                "asof": asof_value,
                "source": "test",
            }
        )
        history.to_csv(artifact_dir / "release_history_36m.csv", index=False)
        pd.DataFrame(
            [
                {
                    "horizon": 1,
                    "point_forecast": 3100.0 + offset + 0.4,
                    "mean_forecast": 3100.0 + offset - 0.2,
                    "challenger_lower_95": 3000.0,
                    "challenger_upper_95": 3200.0,
                    "applied_champion_weight": 0.7,
                },
                {
                    "horizon": 2,
                    "point_forecast": 3110.0 + offset + 0.4,
                    "mean_forecast": 3110.0 + offset - 0.2,
                    "challenger_lower_95": 3010.0,
                    "challenger_upper_95": 3210.0,
                    "applied_champion_weight": 0.68,
                },
            ]
        ).to_csv(artifact_dir / "fusion_inputs.csv", index=False)
        pd.DataFrame(
            [
                {
                    "horizon": "T-1",
                    "target_month": target_month.date().isoformat(),
                    "lineage_id": "lineage-1",
                    "transfer_prior_us_bcfd_t": 20.1,
                    "regime_any_flag": 1.0,
                }
            ]
        ).to_csv(artifact_dir / "feature_matrix.csv", index=False)
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

    report_root = tmp_path / "reports"
    result = validation_24m.run_24_month_validation(
        end_target_month="2025-11-30",
        runs=1,
        variants=("full",),
        dump_feature_row=True,
        report_root=report_root,
    )

    assert len(result.point_estimates) == len(validation_24m.DEFAULT_VARIANTS)
    feature_rows_path = Path(result.summary["feature_rows_path"])
    assert feature_rows_path.exists()
    exported = pd.read_csv(feature_rows_path)
    assert "transfer_prior_us_bcfd_t" in exported.columns
    assert "regime_any_flag" in exported.columns


def test_run_24_month_validation_supports_challenger_bsts_cli_token(
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
                    "value": 0.0,
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
        _ = fusion_config_override
        _ = idempotency_token
        asof_value = str(asof or "2026-01-14")
        override = champion_config_override or {}
        model_cfg = override.get("model", {})
        variant = (
            str(model_cfg.get("variant", "wpd_lstm_one_layer"))
            if isinstance(model_cfg, dict)
            else "wpd_lstm_one_layer"
        )
        offset = {"wpd_lstm_one_layer": 12.0, "wpd_vmd_lstm1": 8.0}.get(variant, 12.0)
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
                            "fused_point": 3100.0 + offset,
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
                "asof": asof_value,
                "source": "test",
            }
        ).to_csv(artifact_dir / "release_history_36m.csv", index=False)
        pd.DataFrame(
            [
                {
                    "horizon": 1,
                    "point_forecast": 3100.0 + offset + 0.4,
                    "mean_forecast": 3100.0 + offset - 0.2,
                    "challenger_lower_95": 3000.0,
                    "challenger_upper_95": 3200.0,
                    "applied_champion_weight": 0.7,
                },
                {
                    "horizon": 2,
                    "point_forecast": 3110.0 + offset + 0.4,
                    "mean_forecast": 3110.0 + offset - 0.2,
                    "challenger_lower_95": 3010.0,
                    "challenger_upper_95": 3210.0,
                    "applied_champion_weight": 0.68,
                },
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

    report_root = tmp_path / "reports"
    result = validation_24m.run_24_month_validation(
        end_target_month="2025-11-30",
        runs=1,
        variants=("challenger_bsts",),
        report_root=report_root,
    )

    assert len(result.point_estimates) == 1
    row = result.point_estimates.iloc[0]
    assert str(row["model_variant"]) == "challenger_bsts"
    assert str(row["forecast_source"]) == "challenger"

    summary_path = Path(result.summary["summary_path"])
    summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
    assert summary_payload["target_month_end"] == "2025-11-30"
    assert len(summary_payload["asof_schedule"]) == 1


def test_run_24_month_validation_fails_loud_on_unit_scale_mismatch(
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
                    "value": 0.0,
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
                            "fused_point": 25_000_000.0,
                            "fused_lower_95": 20_000_000.0,
                            "fused_upper_95": 30_000_000.0,
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
                    "point_forecast": 25_000_000.0,
                    "mean_forecast": 25_000_000.0,
                    "challenger_lower_95": 20_000_000.0,
                    "challenger_upper_95": 30_000_000.0,
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

    with pytest.raises(ContractViolation, match="reason_code=unit_scale_mismatch"):
        validation_24m.run_24_month_validation(
            end_target_month="2025-11-30",
            runs=1,
            variants=("wpd_lstm_one_layer",),
            report_root=tmp_path / "reports",
        )


def test_run_24_month_validation_exports_regime_conditioned_weight_search_v7(
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
                    "value": 0.0,
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
        _ = fusion_config_override
        _ = idempotency_token
        asof_value = str(asof or "2026-01-14")
        override = champion_config_override or {}
        model_cfg = override.get("model", {})
        variant = (
            str(model_cfg.get("variant", "wpd_lstm_one_layer"))
            if isinstance(model_cfg, dict)
            else "wpd_lstm_one_layer"
        )
        offset = {"wpd_lstm_one_layer": 12.0}.get(variant, 12.0)
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
                            "fused_point": 3100.0 + offset,
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
                "asof": asof_value,
                "source": "test",
            }
        ).to_csv(artifact_dir / "release_history_36m.csv", index=False)
        pd.DataFrame(
            [
                {
                    "horizon": 1,
                    "point_forecast": 3100.0 + offset + 0.4,
                    "mean_forecast": 3100.0 + offset - 0.2,
                    "challenger_lower_95": 3000.0,
                    "challenger_upper_95": 3200.0,
                    "applied_champion_weight": 0.7,
                },
                {
                    "horizon": 2,
                    "point_forecast": 3110.0 + offset + 0.4,
                    "mean_forecast": 3110.0 + offset - 0.2,
                    "challenger_lower_95": 3010.0,
                    "challenger_upper_95": 3210.0,
                    "applied_champion_weight": 0.68,
                },
            ]
        ).to_csv(artifact_dir / "fusion_inputs.csv", index=False)
        return {"target_month": target_month.date().isoformat()}

    def _fake_resolve_regime_snapshot(
        feature_rows: pd.DataFrame, *, asof: pd.Timestamp, thresholds: dict[str, object]
    ) -> dict[str, float | str]:
        _ = feature_rows
        _ = thresholds
        label = "shock" if int(pd.Timestamp(asof).month) % 2 == 0 else "normal"
        return {
            "regime_freeze_flag": 1.0 if label == "shock" else 0.0,
            "regime_basis_flag": 0.0,
            "regime_transfer_dispersion_flag": 0.0,
            "regime_any_flag": 1.0 if label == "shock" else 0.0,
            "regime_score": 1.0 if label == "shock" else 0.0,
            "regime_label": label,
        }

    monkeypatch.setattr(
        validation_24m, "enforce_release_policy", _fake_enforce_release_policy
    )
    monkeypatch.setattr(validation_24m, "load_market_inputs", _fake_load_market_inputs)
    monkeypatch.setattr(
        validation_24m,
        "run_nowcast_pipeline_weekly",
        _fake_run_nowcast_pipeline_weekly,
    )
    monkeypatch.setattr(
        validation_24m,
        "_resolve_regime_snapshot",
        _fake_resolve_regime_snapshot,
    )

    result = validation_24m.run_24_month_validation(
        end_target_month="2025-11-30",
        runs=2,
        variants=("wpd_lstm_one_layer",),
        weight_search=True,
        report_root=tmp_path / "reports",
    )

    search_v7_path = Path(result.summary["fusion_weight_search_v7_path"])
    selected_v7_path = Path(result.summary["fusion_weight_selected_v7_path"])
    assert search_v7_path.exists()
    assert selected_v7_path.exists()

    search_v7 = pd.read_csv(search_v7_path)
    assert {
        "model_variant",
        "regime_label",
        "champion_weight",
        "release_anchor_weight",
    }.issubset(set(search_v7.columns))
    assert set(search_v7["regime_label"]) == {"normal", "shock"}

    selected_v7 = json.loads(selected_v7_path.read_text(encoding="utf-8"))
    selected_rows = selected_v7["selected_by_variant_regime"]["wpd_lstm_one_layer"]
    assert set(selected_rows.keys()) == {"normal", "shock"}


def test_run_24_month_validation_exports_interval_metrics(
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
                    "value": 0.0,
                }
            ]
        )
        return {
            "target_history_full": full_history,
            "target_history": full_history.copy(),
            "features": features,
            "monthly_release_history": full_history.tail(36).reset_index(drop=True),
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
                            "fused_point": 3100.0,
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
                "asof": asof_value,
                "source": "test",
            }
        ).to_csv(artifact_dir / "release_history_36m.csv", index=False)
        pd.DataFrame(
            [
                {
                    "horizon": 1,
                    "point_forecast": 3100.0,
                    "mean_forecast": 3099.5,
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
        interval_metrics=True,
        report_root=tmp_path / "reports",
    )
    assert Path(result.summary["interval_scorecard_path"]).exists()
    assert Path(result.summary["interval_scorecard_by_regime_path"]).exists()
    assert Path(result.summary["interval_calibration_table_path"]).exists()
    assert Path(result.summary["interval_calibration_by_regime_path"]).exists()


def test_run_24_month_validation_applies_force_bsts_off_constraints(
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
                    "value": 0.0,
                }
            ]
        )
        return {
            "target_history_full": full_history,
            "target_history": full_history.copy(),
            "features": features,
            "monthly_release_history": full_history.tail(36).reset_index(drop=True),
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
                            "fused_point": 3100.0,
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
                "asof": asof_value,
                "source": "test",
            }
        ).to_csv(artifact_dir / "release_history_36m.csv", index=False)
        pd.DataFrame(
            [
                {
                    "horizon": 1,
                    "point_forecast": 3100.0,
                    "mean_forecast": 3098.0,
                    "challenger_lower_95": 3000.0,
                    "challenger_upper_95": 3200.0,
                    "steo_point_forecast": 3099.0,
                    "prototype_point_forecast": 3097.0,
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

    constraints = tmp_path / "constraints.yaml"
    constraints.write_text("force_bsts_off: true\n", encoding="utf-8")
    result = validation_24m.run_24_month_validation(
        end_target_month="2025-11-30",
        runs=1,
        variants=("wpd_lstm_one_layer",),
        weight_search=True,
        report_root=tmp_path / "reports",
        fusion_constraints_path=constraints,
    )
    search = pd.read_csv(Path(result.summary["fusion_weight_search_path"]))
    assert not search.empty
    assert (search["bsts_weight"].abs() <= 1e-9).all()
    assert bool(result.summary["force_bsts_off"])
