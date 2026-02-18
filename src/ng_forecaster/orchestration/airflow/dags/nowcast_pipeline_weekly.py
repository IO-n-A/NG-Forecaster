"""Airflow DAG: weekly nowcast pipeline with policy gates and artifact publishing."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd

from ng_forecaster.data.preprocess import run_preprocessing
from ng_forecaster.data.validators import (
    load_and_validate_preprocessing_policy,
    load_yaml,
)
from ng_forecaster.errors import ContractViolation
from ng_forecaster.evaluation.dm_test import load_and_validate_dm_policy, run_dm_tests
from ng_forecaster.features.vintage_builder import build_vintage_panel
from ng_forecaster.models.challenger_bsts import run_challenger_model
from ng_forecaster.models.champion_wpd_vmd_lstm import (
    run_champion_pipeline,
    run_champion_seed_repeats,
)
from ng_forecaster.models.fusion import (
    build_fusion_result,
    load_fusion_policy,
    resolve_calendar_calibration_for_regime,
    resolve_fusion_weights_for_regime_full,
)
from ng_forecaster.models.prototypes import (
    build_cohort_kernel_forecast,
    fit_kernel_parameters,
    load_drilling_metrics_history,
)
from ng_forecaster.models.regime import classify_regime
from ng_forecaster.models.steo_forecaster import build_steo_forecast
from ng_forecaster.orchestration.airflow.runtime import (
    DAGSpec,
    TaskSpec,
    execute_task_graph,
)
from ng_forecaster.orchestration.airflow.workflow_support import (
    build_calibration_realized,
    build_dm_runtime_frame,
    enforce_release_policy,
    load_market_inputs,
    load_weather_gold_feature_rows,
    resolve_weekly_asof,
    write_json,
)
from ng_forecaster.reporting.exporters import (
    export_dm_results,
    export_feature_lineage,
    export_model_diagnostics,
    export_preprocess_artifacts,
)

DAG_ID = "nowcast_pipeline_weekly"
WEEKLY_SCHEDULE = "@weekly"

DAG_SPEC = DAGSpec(
    dag_id=DAG_ID,
    schedule=WEEKLY_SCHEDULE,
    task_specs=(
        TaskSpec(task_id="resolve_asof"),
        TaskSpec(
            task_id="resolve_release_calendar_window",
            upstream_task_ids=("resolve_asof",),
        ),
        TaskSpec(
            task_id="ensure_weather_gold_ready",
            upstream_task_ids=("resolve_release_calendar_window",),
        ),
        TaskSpec(
            task_id="build_vintage_panel",
            upstream_task_ids=("ensure_weather_gold_ready",),
        ),
        TaskSpec(
            task_id="run_preprocessing_gate",
            upstream_task_ids=("build_vintage_panel",),
        ),
        TaskSpec(
            task_id="build_feature_matrix",
            upstream_task_ids=("run_preprocessing_gate",),
        ),
        TaskSpec(
            task_id="train_or_load_champion",
            upstream_task_ids=("build_feature_matrix",),
        ),
        TaskSpec(
            task_id="run_champion_seed_repeats",
            upstream_task_ids=("train_or_load_champion",),
        ),
        TaskSpec(
            task_id="train_or_load_challenger",
            upstream_task_ids=("run_champion_seed_repeats",),
        ),
        TaskSpec(
            task_id="run_challenger_intervals",
            upstream_task_ids=("train_or_load_challenger",),
        ),
        TaskSpec(
            task_id="run_calibration_checks",
            upstream_task_ids=("run_challenger_intervals",),
        ),
        TaskSpec(
            task_id="apply_dm_policy",
            upstream_task_ids=("run_calibration_checks",),
        ),
        TaskSpec(
            task_id="publish_nowcast_artifacts",
            upstream_task_ids=("apply_dm_policy",),
        ),
    ),
)


def _merge_nested_dict(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merged[key] = _merge_nested_dict(base[key], value)
        else:
            merged[key] = value
    return merged


def run_nowcast_pipeline_weekly(
    *,
    asof: str | None = None,
    champion_config_override: dict[str, Any] | None = None,
    fusion_config_override: dict[str, Any] | None = None,
    idempotency_token: str | None = None,
) -> dict[str, Any]:
    """Execute weekly nowcast pipeline with N1/N3/N5 hard-policy gates."""

    asof_ts = resolve_weekly_asof(asof)
    run_state: dict[str, Any] = {
        "idempotency_token": str(idempotency_token or ""),
    }

    def _model_exogenous_features() -> dict[str, float]:
        frame = run_state.get("feature_matrix")
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return {}
        candidate = frame[frame["horizon"] == "T"]
        row = candidate.iloc[0] if not candidate.empty else frame.iloc[0]
        allow_keys = {
            "freeze_days_mtd_weighted",
            "freeze_intensity_mtd_weighted",
            "freeze_event_share_mtd_weighted",
            "extreme_min_mtd",
            "coverage_fraction_mtd",
        }
        exogenous: dict[str, float] = {}
        for key, value in row.to_dict().items():
            key_str = str(key)
            if not (key_str.startswith("transfer_prior_") or key_str in allow_keys):
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if pd.isna(numeric):
                continue
            exogenous[key_str] = numeric
        return exogenous

    def _resolve_regime_label() -> str:
        frame = run_state.get("feature_matrix")
        if not isinstance(frame, pd.DataFrame) or frame.empty:
            return "normal"
        candidate = frame[frame["horizon"] == "T-1"]
        row = candidate.iloc[0] if not candidate.empty else frame.iloc[0]
        return classify_regime(row.to_dict())

    def _resolve_asof(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        run_state["asof"] = asof_ts
        return {"asof": asof_ts.date().isoformat()}

    def _resolve_release_calendar_window(
        _: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        asof_value = pd.Timestamp(run_state.get("asof", asof_ts))
        run_state["asof"] = asof_value
        policy_context = enforce_release_policy(asof_value)
        target_month = pd.Timestamp(policy_context["target_month"])
        release_start = pd.Timestamp(policy_context["latest_released_month"])
        release_end = target_month

        run_state["release_policy"] = policy_context
        run_state["target_month"] = target_month
        run_state["release_window"] = {
            "start": release_start,
            "end": release_end,
        }
        return {
            "target_month": target_month.date().isoformat(),
            "policy_passed": bool(policy_context["policy_passed"]),
            "lag_months": int(policy_context["lag_months"]),
            "effective_lag_months": int(policy_context["effective_lag_months"]),
            "release_window_start": release_start.date().isoformat(),
            "release_window_end": release_end.date().isoformat(),
            "release_policy": policy_context,
        }

    def _build_vintage_panel(task_results: dict[str, dict[str, Any]]) -> dict[str, Any]:
        if "asof" not in run_state:
            resolve_payload = task_results.get("resolve_asof", {})
            run_state["asof"] = pd.Timestamp(resolve_payload.get("asof", asof_ts))
        if "release_policy" not in run_state:
            release_payload = task_results.get("resolve_release_calendar_window", {})
            policy = release_payload.get("release_policy")
            if isinstance(policy, dict):
                run_state["release_policy"] = dict(policy)
        if "target_month" not in run_state:
            release_payload = task_results.get("resolve_release_calendar_window", {})
            if release_payload.get("target_month") is not None:
                run_state["target_month"] = pd.Timestamp(
                    release_payload["target_month"]
                )
            elif "release_policy" in run_state:
                run_state["target_month"] = pd.Timestamp(
                    run_state["release_policy"]["target_month"]
                )
            else:
                run_state["target_month"] = (
                    pd.Timestamp(run_state["asof"]).to_period("M") - 2
                ).to_timestamp("M")

        inputs = load_market_inputs(run_state["asof"])
        run_state["inputs"] = inputs

        if pd.Timestamp(inputs["target_month"]) != pd.Timestamp(
            run_state["target_month"]
        ):
            raise ContractViolation(
                "target_month_mismatch",
                asof=run_state["asof"].to_pydatetime(),
                key="target_month",
                detail=(
                    "market input target_month does not match release policy: "
                    f"{pd.Timestamp(inputs['target_month']).date().isoformat()} vs "
                    f"{pd.Timestamp(run_state['target_month']).date().isoformat()}"
                ),
            )

        vintage = build_vintage_panel(
            inputs["features"],
            inputs["target_for_vintage"],
            asof=run_state["asof"],
            preprocessing_status="passed",
            min_target_lag_months=int(
                run_state["release_policy"]["effective_lag_months"]
            ),
            feature_policy=inputs["feature_policy"],
            target_month=run_state["target_month"],
        )
        run_state["vintage"] = vintage
        return {
            "lineage_t_minus_1": vintage.lineage["T-1"],
            "lineage_t": vintage.lineage["T"],
            "target_month_t_minus_1": vintage.target_months["T-1"].date().isoformat(),
            "target_month_t": vintage.target_months["T"].date().isoformat(),
        }

    def _ensure_weather_gold_ready(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        weather_features, weather_meta = load_weather_gold_feature_rows(
            asof=pd.Timestamp(run_state.get("asof", asof_ts))
        )
        status = str(weather_meta.get("status", "")).strip().lower()
        validation_mode = str(run_state.get("idempotency_token", "")).startswith(
            "validation24::"
        )
        if status != "weather_features_loaded" or weather_features.empty:
            if validation_mode:
                asof_value = pd.Timestamp(run_state.get("asof", asof_ts))
                fallback_ts = asof_value.to_period("M").to_timestamp("M")
                fallback = pd.DataFrame(
                    [
                        {
                            "feature_name": "freeze_days_mtd_weighted",
                            "feature_timestamp": fallback_ts,
                            "available_timestamp": asof_value,
                            "block_id": "weather_freezeoff",
                            "value": 0.0,
                        },
                        {
                            "feature_name": "freeze_event_share_mtd_weighted",
                            "feature_timestamp": fallback_ts,
                            "available_timestamp": asof_value,
                            "block_id": "weather_freezeoff",
                            "value": 0.0,
                        },
                        {
                            "feature_name": "freeze_intensity_mtd_weighted",
                            "feature_timestamp": fallback_ts,
                            "available_timestamp": asof_value,
                            "block_id": "weather_freezeoff",
                            "value": 0.0,
                        },
                    ]
                )
                run_state["weather_features"] = fallback
                run_state["weather_meta"] = {
                    **weather_meta,
                    "status": "weather_backfill_for_validation",
                }
                return {
                    "status": "weather_backfill_for_validation",
                    "feature_row_count": int(len(fallback)),
                    "available_timestamp_max": asof_value.date().isoformat(),
                }
            raise ContractViolation(
                "missing_source_file",
                asof=pd.Timestamp(run_state.get("asof", asof_ts)).to_pydatetime(),
                key="weather_freezeoff_panel",
                detail=(
                    "weather gold features are required before weekly nowcast run; "
                    f"status={status or '<none>'}"
                ),
            )
        max_available = pd.Timestamp(weather_features["available_timestamp"].max())
        stale_days = int(
            (
                pd.Timestamp(run_state.get("asof", asof_ts)).normalize()
                - max_available.normalize()
            ).days
        )
        if stale_days > 45:
            if validation_mode:
                run_state["weather_features"] = weather_features
                run_state["weather_meta"] = {
                    **weather_meta,
                    "status": "weather_stale_but_allowed_for_validation",
                }
                return {
                    "status": "weather_stale_but_allowed_for_validation",
                    "feature_row_count": int(len(weather_features)),
                    "available_timestamp_max": max_available.date().isoformat(),
                }
            raise ContractViolation(
                "stale_source",
                asof=pd.Timestamp(run_state.get("asof", asof_ts)).to_pydatetime(),
                key="weather_freezeoff_panel",
                detail=f"weather features are stale by {stale_days} days",
            )
        run_state["weather_features"] = weather_features
        run_state["weather_meta"] = weather_meta
        return {
            "status": "weather_ready",
            "feature_row_count": int(len(weather_features)),
            "available_timestamp_max": max_available.date().isoformat(),
        }

    def _run_preprocessing_gate(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        inputs = run_state["inputs"]
        preprocess_input = pd.concat(
            [
                inputs["daily"].assign(series_id="hh_last"),
                inputs["weekly"].assign(series_id="stor_last"),
            ],
            ignore_index=True,
        )
        policy = load_and_validate_preprocessing_policy("configs/preprocessing.yaml")
        result = run_preprocessing(
            preprocess_input,
            policy,
            value_col="value",
            series_col="series_id",
            timestamp_col="timestamp",
        )
        if result.status != "passed":
            raise ContractViolation(
                "preprocess_gate_failed",
                key="preprocess_status",
                detail=f"status={result.status}",
            )

        run_state["preprocess"] = result
        return {
            "status": result.status,
            "missing_flag_count": int(len(result.missing_flags)),
            "outlier_flag_count": int(len(result.outlier_flags)),
        }

    def _build_feature_matrix(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        vintage = run_state["vintage"]
        policy_features = sorted(
            run_state["inputs"]["feature_policy"]["features"].keys()
        )
        rows: list[dict[str, Any]] = []
        for horizon in ("T-1", "T"):
            panel = vintage.slices[horizon].iloc[0]
            row: dict[str, Any] = {
                "horizon": horizon,
                "target_month": str(panel.get("target_month", "")),
                "lineage_id": str(panel["lineage_id"]),
            }
            for feature_name in policy_features:
                if feature_name in panel.index:
                    row[feature_name] = float(panel.get(feature_name, 0.0))
            for column in panel.index:
                if str(column).startswith("regime_"):
                    row[str(column)] = float(panel.get(column, 0.0))
            rows.append(row)
        feature_matrix = (
            pd.DataFrame(rows).sort_values("horizon").reset_index(drop=True)
        )
        run_state["feature_matrix"] = feature_matrix
        return {
            "row_count": int(len(feature_matrix)),
            "horizons": feature_matrix["horizon"].tolist(),
            "feature_columns": sorted(
                column
                for column in feature_matrix.columns
                if column not in {"horizon", "target_month", "lineage_id"}
            ),
        }

    def _assert_released_only_target_history() -> None:
        history = run_state["inputs"]["target_history"]
        max_training_timestamp = pd.Timestamp(history["timestamp"].max())
        latest_released_month = pd.Timestamp(
            run_state["inputs"]["latest_released_month"]
        )
        if max_training_timestamp > latest_released_month:
            raise ContractViolation(
                "leakage_detected",
                asof=run_state["asof"].to_pydatetime(),
                key="target_history",
                detail=(
                    f"training max timestamp {max_training_timestamp.date().isoformat()} "
                    f"exceeds latest released month {latest_released_month.date().isoformat()}"
                ),
            )

    def _train_or_load_champion(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        _assert_released_only_target_history()
        config = load_yaml("configs/model_champion.yaml")
        if champion_config_override:
            config = _merge_nested_dict(config, dict(champion_config_override))
        run_state["champion_config"] = config
        run_state["model_exogenous"] = _model_exogenous_features()
        champion = run_champion_pipeline(
            run_state["inputs"]["target_history"],
            config,
            timestamp_col="timestamp",
            target_col="target_value",
            exogenous_features=run_state.get("model_exogenous"),
            artifact_root="data/artifacts/models/lstm",
            artifact_tag=run_state["asof"].date().isoformat(),
        )
        run_state["champion"] = champion
        return {
            "forecast_rows": int(len(champion.point_forecast)),
            "model_variant": str(champion.diagnostics["model_variant"]),
            "exogenous_feature_count": int(len(run_state.get("model_exogenous", {}))),
        }

    def _run_champion_seed_repeats(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        config = dict(run_state.get("champion_config", {})) or load_yaml(
            "configs/model_champion.yaml"
        )
        if champion_config_override:
            config = _merge_nested_dict(config, dict(champion_config_override))
        repeat_runs = int(config.get("lstm", {}).get("repeat_runs", 5))
        base_seed = int(config.get("training", {}).get("seed", 42))
        seeds = [base_seed + run_idx for run_idx in range(repeat_runs)]
        seed_runs = run_champion_seed_repeats(
            run_state["inputs"]["target_history"],
            config,
            seeds=seeds,
            timestamp_col="timestamp",
            target_col="target_value",
            exogenous_features=run_state.get("model_exogenous"),
        )
        run_state["champion_seed_runs"] = seed_runs
        return {
            "seed_count": int(seed_runs.diagnostics["seed_count"]),
            "forecast_rows": int(len(seed_runs.forecasts)),
        }

    def _train_or_load_challenger(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        _assert_released_only_target_history()
        config = load_yaml("configs/model_challenger.yaml")
        challenger = run_challenger_model(
            run_state["inputs"]["target_history"],
            config,
            timestamp_col="timestamp",
            target_col="target_value",
            exogenous_features=run_state.get("model_exogenous"),
            artifact_root="data/artifacts/posteriors/bsts",
            artifact_tag=run_state["asof"].date().isoformat(),
        )
        run_state["challenger"] = challenger
        return {
            "forecast_rows": int(len(challenger.forecast)),
            "model_variant": str(challenger.diagnostics["model_variant"]),
        }

    def _run_challenger_intervals(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        challenger = run_state["challenger"]
        interval_width = (
            challenger.forecast["upper_95"] - challenger.forecast["lower_95"]
        )
        return {
            "mean_interval_width": float(interval_width.mean()),
            "max_interval_width": float(interval_width.max()),
        }

    def _run_calibration_checks(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        realized = build_calibration_realized(run_state["inputs"]["target_history"])
        monthly_release_history = run_state["inputs"].get("monthly_release_history")
        if not isinstance(monthly_release_history, pd.DataFrame):
            monthly_release_history = run_state["inputs"].get("monthly_releases_24m")
        fusion_policy = load_fusion_policy("configs/fusion.yaml")
        if fusion_config_override:
            fusion_policy = _merge_nested_dict(
                fusion_policy,
                dict(fusion_config_override),
            )
        regime_label = _resolve_regime_label()
        (
            horizon_weights,
            anchor_weight,
            steo_weight,
            prototype_weight,
            month_length_bias_weight,
        ) = resolve_fusion_weights_for_regime_full(
            fusion_policy,
            regime_label=regime_label,
        )
        calendar_calibration_cfg = resolve_calendar_calibration_for_regime(
            fusion_policy,
            regime_label=regime_label,
        )
        steo_forecast = None
        try:
            steo_result = build_steo_forecast(
                run_state["inputs"]["features"],
                target_month=run_state["target_month"],
                horizons=[1, 2],
                release_history=(
                    monthly_release_history
                    if isinstance(monthly_release_history, pd.DataFrame)
                    else None
                ),
            )
            steo_forecast = steo_result.forecast
            run_state["steo_forecast_diag"] = {
                **steo_result.diagnostics,
                "status": "loaded",
            }
        except ContractViolation as exc:
            run_state["steo_forecast_diag"] = {
                "status": "not_available",
                "reason_code": str(exc.context.reason_code),
            }
        prototype_forecast = None
        try:
            drilling_history = load_drilling_metrics_history(
                asof=run_state["asof"],
                lookback_months=36,
            )
            kernel_params = fit_kernel_parameters(
                release_history=(
                    monthly_release_history
                    if isinstance(monthly_release_history, pd.DataFrame)
                    else run_state["inputs"]["target_history"]
                ),
                drilling_history=drilling_history,
            )
            prototype_result = build_cohort_kernel_forecast(
                drilling_history=drilling_history,
                target_month=run_state["target_month"],
                release_history=(
                    monthly_release_history
                    if isinstance(monthly_release_history, pd.DataFrame)
                    else run_state["inputs"]["target_history"]
                ),
                horizons=[1, 2],
                kernel_params=kernel_params,
            )
            prototype_forecast = prototype_result.forecast
            run_state["prototype_forecast_diag"] = {
                **prototype_result.diagnostics,
                "status": "loaded",
            }
        except ContractViolation as exc:
            run_state["prototype_forecast_diag"] = {
                "status": "not_available",
                "reason_code": str(exc.context.reason_code),
            }
        default_champion_weight = float(
            fusion_policy.get("base", {}).get("champion_weight", 0.70)
        )
        fusion = build_fusion_result(
            run_state["champion"].point_forecast,
            run_state["challenger"].forecast,
            run_state["champion_seed_runs"].forecasts,
            realized,
            champion_weight=default_champion_weight,
            release_history=(
                monthly_release_history
                if isinstance(monthly_release_history, pd.DataFrame)
                else None
            ),
            release_anchor_weight=anchor_weight,
            steo_forecast=steo_forecast,
            steo_weight=float(steo_weight),
            prototype_forecast=prototype_forecast,
            prototype_weight=float(prototype_weight),
            month_length_bias_weight=float(month_length_bias_weight),
            calendar_calibration=calendar_calibration_cfg,
            horizon_weights=horizon_weights,
            regime_label=regime_label,
        )
        run_state["fusion"] = fusion
        run_state["fusion_policy_applied"] = {
            "regime_label": regime_label,
            "champion_weight": default_champion_weight,
            "horizon_weights": horizon_weights,
            "release_anchor_weight": anchor_weight,
            "steo_weight": float(steo_weight),
            "prototype_weight": float(prototype_weight),
            "month_length_bias_weight": float(month_length_bias_weight),
            "calendar_calibration": dict(calendar_calibration_cfg),
        }
        return {
            "coverage_rate": float(fusion.calibration_summary["coverage_rate"]),
            "mean_abs_divergence": float(
                fusion.divergence_summary["mean_abs_divergence"]
            ),
            "regime_label": regime_label,
            "release_anchor_weight": float(anchor_weight),
            "steo_weight": float(steo_weight),
            "prototype_weight": float(prototype_weight),
            "month_length_bias_weight": float(month_length_bias_weight),
            "calendar_calibration_enabled": bool(
                calendar_calibration_cfg.get("enabled", False)
            ),
            "horizon_weights": {
                str(key): float(value) for key, value in sorted(horizon_weights.items())
            },
            "release_history_rows": int(
                len(monthly_release_history)
                if isinstance(monthly_release_history, pd.DataFrame)
                else 0
            ),
            "release_history_source": str(
                run_state["inputs"]
                .get(
                    "monthly_release_history_meta",
                    run_state["inputs"].get("monthly_releases_24m_meta", {}),
                )
                .get("source", "unknown")
            ),
            "steo_gold_status": str(
                run_state["inputs"].get("steo_gold_meta", {}).get("status", "missing")
            ),
            "steo_gold_vintage_month": str(
                run_state["inputs"]
                .get("steo_gold_meta", {})
                .get("latest_vintage_month", "")
            ),
            "steo_forecaster_status": str(
                run_state.get("steo_forecast_diag", {}).get("status", "not_available")
            ),
            "prototype_forecaster_status": str(
                run_state.get("prototype_forecast_diag", {}).get(
                    "status",
                    "not_available",
                )
            ),
        }

    def _apply_dm_policy(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        policy = load_and_validate_dm_policy("configs/evaluation.yaml")
        dm_input = build_dm_runtime_frame(
            run_state["inputs"]["target_history"],
            target_month=run_state["target_month"],
        )
        dm_run = run_dm_tests(dm_input, policy)
        run_state["dm_results"] = dm_run.results
        return {
            "row_count": int(len(dm_run.results)),
            "significant_0_05_count": int(dm_run.results["significant_0_05"].sum()),
        }

    def _publish_nowcast_artifacts(_: dict[str, dict[str, Any]]) -> dict[str, Any]:
        asof_label = run_state["asof"].date().isoformat()
        artifact_dir = Path("data/artifacts/nowcast") / asof_label
        report_dir = Path("data/reports")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        report_dir.mkdir(parents=True, exist_ok=True)

        preprocess_paths = export_preprocess_artifacts(
            run_state["preprocess"], artifact_dir
        )
        lineage_path = export_feature_lineage(
            run_state["vintage"].lineage,
            artifact_dir,
            filename="feature_lineage.csv",
        )

        diagnostics_payload = {
            "model_family": "wpd_vmd_lstm",
            "champion": run_state["champion"].diagnostics,
            "challenger": run_state["challenger"].diagnostics,
            "divergence_summary": run_state["fusion"].divergence_summary,
            "calibration_summary": run_state["fusion"].calibration_summary,
            "fusion_policy_applied": run_state.get("fusion_policy_applied", {}),
            "release_history_meta": run_state["inputs"].get(
                "monthly_release_history_meta",
                run_state["inputs"].get("monthly_releases_24m_meta", {}),
            ),
            "release_history_36m_meta": run_state["inputs"].get(
                "monthly_release_history_meta",
                run_state["inputs"].get("monthly_releases_24m_meta", {}),
            ),
            "release_anchor_applied_count": int(
                run_state["fusion"]
                .forecast["release_anchor_applied"]
                .astype(bool)
                .sum()
            ),
            "steo_gold_meta": run_state["inputs"].get("steo_gold_meta", {}),
            "steo_forecast": run_state.get("steo_forecast_diag", {}),
            "prototype_forecast": run_state.get("prototype_forecast_diag", {}),
        }
        model_diag_paths = export_model_diagnostics(
            diagnostics_payload,
            run_state["fusion"].stability_summary,
            artifact_dir,
        )

        dm_path = export_dm_results(run_state["dm_results"], artifact_dir)
        export_dm_results(run_state["dm_results"], report_dir)

        release_history = run_state["inputs"].get("monthly_release_history")
        if not isinstance(release_history, pd.DataFrame):
            release_history = run_state["inputs"].get("monthly_releases_24m")
        if not isinstance(release_history, pd.DataFrame):
            raise ContractViolation(
                "missing_source_file",
                key="monthly_release_history",
                detail="monthly release history is required for nowcast publication",
            )
        if not {"timestamp", "target_value"}.issubset(release_history.columns):
            raise ContractViolation(
                "source_schema_drift",
                key="monthly_release_history",
                detail="release history must include timestamp and target_value columns",
            )

        release_history_export = release_history[["timestamp", "target_value"]].copy()
        release_history_export["timestamp"] = pd.to_datetime(
            release_history_export["timestamp"], errors="coerce"
        )
        release_history_export["target_value"] = pd.to_numeric(
            release_history_export["target_value"], errors="coerce"
        )
        release_history_export = release_history_export[
            release_history_export["timestamp"].notna()
            & release_history_export["target_value"].notna()
        ].copy()
        release_history_export = release_history_export.sort_values("timestamp")
        release_history_export = release_history_export.drop_duplicates(
            "timestamp", keep="last"
        )
        release_history_export = release_history_export.tail(36).reset_index(drop=True)
        if len(release_history_export) < 36:
            raise ContractViolation(
                "insufficient_release_history",
                key="release_history_36m",
                detail=(
                    "release history export requires at least 36 rows; "
                    f"received={len(release_history_export)}"
                ),
            )
        release_history_export["asof"] = asof_label
        release_history_export["source"] = str(
            run_state["inputs"]
            .get(
                "monthly_release_history_meta",
                run_state["inputs"].get("monthly_releases_24m_meta", {}),
            )
            .get("source", "unknown")
        )
        release_history_path = artifact_dir / "release_history_36m.csv"
        release_history_export.to_csv(release_history_path, index=False)

        nowcast_table = run_state["fusion"].forecast.copy()
        nowcast_table["horizon_label"] = nowcast_table["horizon"].map(
            {1: "T-1", 2: "T"}
        )
        nowcast_table["target_month"] = nowcast_table["horizon"].map(
            lambda horizon: (
                pd.Timestamp(run_state["target_month"])
                + pd.DateOffset(months=int(horizon) - 1)
            )
            .to_period("M")
            .to_timestamp("M")
            .date()
            .isoformat()
        )
        nowcast_table["target_month_label"] = nowcast_table["horizon"].map(
            {1: "target_month", 2: "target_month_plus_1"}
        )

        invalid_interval = nowcast_table[
            (nowcast_table["fused_lower_95"] > nowcast_table["fused_point"])
            | (nowcast_table["fused_point"] > nowcast_table["fused_upper_95"])
        ]
        if not invalid_interval.empty:
            first = invalid_interval.iloc[0]
            raise ContractViolation(
                "interval_order_violation",
                asof=run_state["asof"].to_pydatetime(),
                key=f"horizon={int(first['horizon'])}",
                detail=(
                    f"lower={float(first['fused_lower_95'])}, "
                    f"point={float(first['fused_point'])}, "
                    f"upper={float(first['fused_upper_95'])}"
                ),
            )

        challenger_inputs = (
            run_state["challenger"]
            .forecast[["horizon", "lower_95", "upper_95"]]
            .rename(
                columns={
                    "lower_95": "challenger_lower_95",
                    "upper_95": "challenger_upper_95",
                }
            )
        )
        fusion_inputs = (
            run_state["fusion"]
            .forecast[
                [
                    "horizon",
                    "point_forecast",
                    "mean_forecast",
                    "fused_point",
                    "fused_lower_95",
                    "fused_upper_95",
                    "applied_champion_weight",
                    "applied_release_anchor_weight",
                    "applied_steo_weight",
                    "applied_prototype_weight",
                    "applied_month_length_bias_weight",
                    "fused_point_pre_calendar_calibration",
                    "calendar_calibration_delta",
                    "calendar_calibration_applied",
                    "regime_label",
                    "target_month_days",
                    "is_leap_february",
                    "release_anchor_point",
                    "anchor_month_end",
                    "release_anchor_applied",
                    "month_length_bias_applied",
                    "steo_point_forecast",
                    "steo_lower_95",
                    "steo_upper_95",
                    "steo_applied",
                    "prototype_point_forecast",
                    "prototype_lower_95",
                    "prototype_upper_95",
                    "prototype_applied",
                ]
            ]
            .merge(challenger_inputs, on="horizon", how="left")
        )
        fusion_inputs.insert(0, "asof", asof_label)
        fusion_inputs.insert(
            1,
            "target_month",
            pd.Timestamp(run_state["target_month"]).date().isoformat(),
        )
        fusion_inputs.to_csv(artifact_dir / "fusion_inputs.csv", index=False)

        write_json(
            artifact_dir / "nowcast.json",
            {
                "asof": asof_label,
                "schedule": WEEKLY_SCHEDULE,
                "target_month": pd.Timestamp(run_state["target_month"])
                .date()
                .isoformat(),
                "release_policy": run_state["release_policy"],
                "nowcasts": nowcast_table[
                    [
                        "horizon_label",
                        "target_month_label",
                        "target_month",
                        "fused_point",
                        "fused_lower_95",
                        "fused_upper_95",
                        "fused_point_pre_calendar_calibration",
                        "calendar_calibration_delta",
                        "calendar_calibration_applied",
                        "target_month_days",
                        "is_leap_february",
                        "regime_label",
                    ]
                ].to_dict(orient="records"),
            },
        )
        write_json(
            artifact_dir / "lineage.json",
            {
                "asof": asof_label,
                "target_month": pd.Timestamp(run_state["target_month"])
                .date()
                .isoformat(),
                "lineage": run_state["vintage"].lineage,
                "target_months": {
                    key: value.date().isoformat()
                    for key, value in run_state["vintage"].target_months.items()
                },
                "feature_lineage_csv": str(lineage_path),
            },
        )
        write_json(
            artifact_dir / "diagnostics.json",
            {
                "asof": asof_label,
                "target_month": pd.Timestamp(run_state["target_month"])
                .date()
                .isoformat(),
                "diagnostics": diagnostics_payload,
            },
        )
        feature_matrix_path = artifact_dir / "feature_matrix.csv"
        run_state["feature_matrix"].to_csv(feature_matrix_path, index=False)

        champion_rows = run_state["champion"].point_forecast.copy()
        champion_rows = champion_rows.rename(
            columns={
                "point_forecast": "point_forecast",
            }
        )
        champion_rows["model"] = "champion"
        champion_rows["interval_low"] = champion_rows["point_forecast"] - 0.5
        champion_rows["interval_high"] = champion_rows["point_forecast"] + 0.5

        challenger_rows = run_state["challenger"].forecast.copy()
        challenger_rows["model"] = "challenger"
        challenger_rows = challenger_rows.rename(
            columns={
                "mean_forecast": "point_forecast",
                "lower_95": "interval_low",
                "upper_95": "interval_high",
            }
        )

        n4_outputs = pd.concat(
            [
                champion_rows[
                    [
                        "horizon",
                        "model",
                        "point_forecast",
                        "interval_low",
                        "interval_high",
                    ]
                ],
                challenger_rows[
                    [
                        "horizon",
                        "model",
                        "point_forecast",
                        "interval_low",
                        "interval_high",
                    ]
                ],
            ],
            ignore_index=True,
        )
        n4_outputs.insert(0, "asof", asof_label)
        n4_outputs.insert(
            1,
            "target_month",
            pd.Timestamp(run_state["target_month"]).date().isoformat(),
        )
        n4_outputs.to_csv(report_dir / "n4_nowcast_outputs.csv", index=False)

        stability = run_state["fusion"].stability_summary
        seed_count = int(run_state["champion_seed_runs"].diagnostics["seed_count"])
        champion_seed_summary = {
            "model": "champion",
            "seed_count": seed_count,
            "point_mean": float(stability["mean"].mean()),
            "point_std": float(stability["std"].mean()),
            "interval_width_mean": float(
                (
                    run_state["challenger"].forecast["upper_95"]
                    - run_state["challenger"].forecast["lower_95"]
                ).mean()
            ),
        }
        challenger_seed_summary = {
            "model": "challenger",
            "seed_count": seed_count,
            "point_mean": float(
                run_state["challenger"].forecast["mean_forecast"].mean()
            ),
            "point_std": 0.0,
            "interval_width_mean": float(
                (
                    run_state["challenger"].forecast["upper_95"]
                    - run_state["challenger"].forecast["lower_95"]
                ).mean()
            ),
        }
        pd.DataFrame([champion_seed_summary, challenger_seed_summary]).to_csv(
            report_dir / "n4_seed_stability.csv",
            index=False,
        )

        coverage = float(run_state["fusion"].calibration_summary["coverage_rate"])
        n4_calibration = pd.DataFrame(
            [
                {
                    "model": "champion",
                    "nominal_coverage": 0.95,
                    "empirical_coverage": coverage,
                    "calibration_error": coverage - 0.95,
                },
                {
                    "model": "challenger",
                    "nominal_coverage": 0.95,
                    "empirical_coverage": coverage,
                    "calibration_error": coverage - 0.95,
                },
            ]
        )
        n4_calibration.to_csv(report_dir / "n4_calibration_summary.csv", index=False)

        return {
            "artifact_dir": str(artifact_dir),
            "target_month": pd.Timestamp(run_state["target_month"]).date().isoformat(),
            "preprocess_summary": str(preprocess_paths["preprocess_summary"]),
            "model_diagnostics": str(model_diag_paths["model_diagnostics"]),
            "dm_results": str(dm_path),
            "lineage_path": str(lineage_path),
            "release_history_36m": str(release_history_path),
            "feature_matrix_path": str(feature_matrix_path),
        }

    result = execute_task_graph(
        dag_spec=DAG_SPEC,
        context={
            "asof": asof_ts.date().isoformat(),
            "champion_override": (
                json.dumps(champion_config_override, sort_keys=True)
                if champion_config_override
                else ""
            ),
            "idempotency_token": str(idempotency_token or ""),
        },
        task_functions={
            "resolve_asof": _resolve_asof,
            "resolve_release_calendar_window": _resolve_release_calendar_window,
            "ensure_weather_gold_ready": _ensure_weather_gold_ready,
            "build_vintage_panel": _build_vintage_panel,
            "run_preprocessing_gate": _run_preprocessing_gate,
            "build_feature_matrix": _build_feature_matrix,
            "train_or_load_champion": _train_or_load_champion,
            "run_champion_seed_repeats": _run_champion_seed_repeats,
            "train_or_load_challenger": _train_or_load_challenger,
            "run_challenger_intervals": _run_challenger_intervals,
            "run_calibration_checks": _run_calibration_checks,
            "apply_dm_policy": _apply_dm_policy,
            "publish_nowcast_artifacts": _publish_nowcast_artifacts,
        },
    )

    release_task = result["tasks"].get("resolve_release_calendar_window", {})
    target_month_value = run_state.get("target_month")
    if target_month_value is None:
        release_target = release_task.get("target_month")
        if release_target is None:
            release_target = (
                result["tasks"].get("publish_nowcast_artifacts", {}).get("target_month")
            )
        if release_target is not None:
            target_month_value = pd.Timestamp(release_target)
        else:
            target_month_value = (asof_ts.to_period("M") - 2).to_timestamp("M")
    target_month_value = (
        pd.Timestamp(target_month_value).to_period("M").to_timestamp("M")
    )
    run_state["target_month"] = target_month_value

    if "release_policy" not in run_state:
        release_policy_payload = release_task.get("release_policy")
        if isinstance(release_policy_payload, dict):
            run_state["release_policy"] = dict(release_policy_payload)
        else:
            run_state["release_policy"] = {
                "asof": asof_ts.date().isoformat(),
                "target_month": target_month_value.date().isoformat(),
            }

    report_packet = {
        "asof": asof_ts.date().isoformat(),
        "target_month": target_month_value.date().isoformat(),
        "release_policy": run_state["release_policy"],
        "dag_id": DAG_ID,
        "schedule": WEEKLY_SCHEDULE,
        "task_status": {
            name: payload.get("status", "done")
            for name, payload in result["tasks"].items()
        },
    }
    Path("data/reports").mkdir(parents=True, exist_ok=True)
    Path("data/reports/nowcast_pipeline_report.json").write_text(
        json.dumps(report_packet, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    result["target_month"] = target_month_value.date().isoformat()
    result["release_policy"] = run_state["release_policy"]

    return result


try:  # pragma: no cover - optional Airflow import for runtime integration
    from airflow import DAG
    from airflow.operators.empty import EmptyOperator

    with DAG(
        dag_id=DAG_ID,
        schedule=WEEKLY_SCHEDULE,
        catchup=False,
        tags=["weekly", "nowcast"],
    ):
        resolve_asof = EmptyOperator(task_id="resolve_asof")
        resolve_release_calendar_window = EmptyOperator(
            task_id="resolve_release_calendar_window"
        )
        ensure_weather_gold_ready = EmptyOperator(task_id="ensure_weather_gold_ready")
        build_vintage_panel = EmptyOperator(task_id="build_vintage_panel")
        run_preprocessing_gate = EmptyOperator(task_id="run_preprocessing_gate")
        build_feature_matrix = EmptyOperator(task_id="build_feature_matrix")
        train_or_load_champion = EmptyOperator(task_id="train_or_load_champion")
        run_champion_seed_repeats = EmptyOperator(task_id="run_champion_seed_repeats")
        train_or_load_challenger = EmptyOperator(task_id="train_or_load_challenger")
        run_challenger_intervals = EmptyOperator(task_id="run_challenger_intervals")
        run_calibration_checks = EmptyOperator(task_id="run_calibration_checks")
        apply_dm_policy = EmptyOperator(task_id="apply_dm_policy")
        publish_nowcast_artifacts = EmptyOperator(task_id="publish_nowcast_artifacts")

        (
            resolve_asof
            >> resolve_release_calendar_window
            >> ensure_weather_gold_ready
            >> build_vintage_panel
            >> run_preprocessing_gate
            >> build_feature_matrix
            >> train_or_load_champion
            >> run_champion_seed_repeats
            >> train_or_load_challenger
            >> run_challenger_intervals
            >> run_calibration_checks
            >> apply_dm_policy
            >> publish_nowcast_artifacts
        )
except Exception:  # pragma: no cover
    pass
