#!/usr/bin/env python3
"""Generate Plotly visualizations (HTML + PNG) for ingestion and nowcast outputs."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import re
from typing import Any, cast

import pandas as pd
import pypalettes  # type: ignore[import-untyped]
import plotly.express as px  # type: ignore[import-untyped]
import plotly.graph_objects as go  # type: ignore[import-untyped]

_TIME_TEMPLATE = "%Y-%m-%d"
_LEGACY_SVG_FILES = (
    "nowcast_intervals.svg",
    "ablation_mae.svg",
    "ablation_dm_pvalue.svg",
    "backtest_rmse.svg",
    "dm_policy_adjusted_pvalue.svg",
    "weekly_dag_task_counts.svg",
)
_CLEANABLE_PREFIXES = (
    "nowcast_",
    "ingest_",
    "ingested_",
    "preprocess_",
    "backtest_",
    "dm_",
    "ablation_",
    "weekly_",
    "validation_24m_",
    "steo_",
    "dashboard",
    "visualization_summary",
)
_CLEANABLE_EXTS = {".html", ".json", ".md", ".svg", ".png"}


def _load_coconut_palette(*, min_len: int = 12) -> list[str]:
    raw = pypalettes.load_palette("Coconut", keep_first_n=6)
    base = [str(color)[:7] for color in raw]
    if len(base) >= min_len:
        return base[:min_len]
    repeats = (min_len + len(base) - 1) // len(base)
    expanded = (base * repeats)[:min_len]
    return expanded


_COCONUT_COLORS = _load_coconut_palette()


def _ensure_dir(path: str | Path) -> Path:
    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def _cleanup_legacy_outputs(output_dir: Path) -> None:
    for filename in _LEGACY_SVG_FILES:
        candidate = output_dir / filename
        if candidate.exists():
            candidate.unlink()


def _cleanup_stale_generated_outputs(output_dir: Path, keep_files: set[str]) -> None:
    """Delete stale generated visualization artifacts from earlier schema versions."""

    for candidate in output_dir.iterdir():
        if not candidate.is_file():
            continue
        if candidate.name in keep_files:
            continue
        if candidate.suffix not in _CLEANABLE_EXTS:
            continue
        if not any(candidate.name.startswith(prefix) for prefix in _CLEANABLE_PREFIXES):
            continue
        candidate.unlink()


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"expected JSON object at {path}")
    return cast(dict[str, Any], payload)


def _load_csv_frame(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def _safe_float(value: Any, *, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _extract_asof(name: str) -> str:
    if name.startswith("asof="):
        return name.split("=", 1)[1]
    return name


def _iter_dirs(path: Path) -> list[Path]:
    if not path.exists():
        return []
    return sorted(
        [item for item in path.iterdir() if item.is_dir()], key=lambda item: item.name
    )


def _format_timestamp(value: Any) -> str:
    if value is None:
        return ""
    ts = pd.to_datetime(value, errors="coerce")
    if pd.isna(ts):
        return ""
    return str(cast(pd.Timestamp, ts).strftime(_TIME_TEMPLATE))


def _collect_nowcast_series(artifact_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for asof_dir in _iter_dirs(artifact_root / "nowcast"):
        nowcast_path = asof_dir / "nowcast.json"
        if not nowcast_path.exists():
            continue

        payload = _load_json(nowcast_path)
        asof_str = str(payload.get("asof", _extract_asof(asof_dir.name)))
        asof = pd.to_datetime(asof_str, errors="coerce")
        if pd.isna(asof):
            continue

        nowcasts = payload.get("nowcasts", [])
        if not isinstance(nowcasts, list):
            continue

        for rec in nowcasts:
            if not isinstance(rec, dict):
                continue

            lower = _safe_float(rec.get("fused_lower_95"))
            upper = _safe_float(rec.get("fused_upper_95"))
            interval_low = min(lower, upper)
            interval_high = max(lower, upper)

            rows.append(
                {
                    "asof": asof,
                    "horizon": str(rec.get("horizon_label", "unknown")),
                    "target_month": str(
                        rec.get("target_month", payload.get("target_month", ""))
                    ),
                    "target_label": str(
                        rec.get("target_month_label", rec.get("horizon_label", ""))
                    ),
                    "fused_point": _safe_float(rec.get("fused_point")),
                    "interval_low": interval_low,
                    "interval_high": interval_high,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=[
                "asof",
                "horizon",
                "target_month",
                "target_label",
                "fused_point",
                "interval_low",
                "interval_high",
            ]
        )

    frame = pd.DataFrame(rows)
    frame["target_month"] = frame["target_month"].replace("", pd.NA)
    frame = frame.sort_values(["asof", "horizon"]).reset_index(drop=True)
    return frame


def _collect_preprocess_summary_series(artifact_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for asof_dir in _iter_dirs(artifact_root / "nowcast"):
        summary_path = asof_dir / "preprocess_summary.json"
        if not summary_path.exists():
            continue

        payload = _load_json(summary_path)
        asof = pd.to_datetime(_extract_asof(asof_dir.name), errors="coerce")
        if pd.isna(asof):
            continue

        rows.append(
            {
                "asof": asof,
                "status": str(payload.get("status", "unknown")),
                "missing_flag_count": int(
                    _safe_float(payload.get("missing_flag_count"))
                ),
                "outlier_flag_count": int(
                    _safe_float(payload.get("outlier_flag_count"))
                ),
                "unresolved_missing_count": int(
                    _safe_float(payload.get("unresolved_missing_count"))
                ),
                "low_coverage_series_count": int(
                    _safe_float(payload.get("low_coverage_series_count"))
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "asof",
                "status",
                "missing_flag_count",
                "outlier_flag_count",
                "unresolved_missing_count",
                "low_coverage_series_count",
            ]
        )

    frame = pd.DataFrame(rows).sort_values("asof").reset_index(drop=True)
    return frame


def _collect_nowcast_latest_vs_previous_delta(artifact_root: Path) -> pd.DataFrame:
    """Build horizon-level delta between latest and previous nowcast runs."""

    run_snapshots: list[dict[str, Any]] = []
    for asof_dir in _iter_dirs(artifact_root / "nowcast"):
        nowcast_path = asof_dir / "nowcast.json"
        if not nowcast_path.exists():
            continue

        payload = _load_json(nowcast_path)
        asof_str = str(payload.get("asof", _extract_asof(asof_dir.name)))
        asof = pd.to_datetime(asof_str, errors="coerce")
        if pd.isna(asof):
            continue

        nowcasts = payload.get("nowcasts", [])
        if not isinstance(nowcasts, list):
            continue

        by_horizon: dict[str, dict[str, Any]] = {}
        for rec in nowcasts:
            if not isinstance(rec, dict):
                continue

            horizon = str(rec.get("horizon_label", "unknown"))
            lower = _safe_float(rec.get("fused_lower_95"), default=float("nan"))
            upper = _safe_float(rec.get("fused_upper_95"), default=float("nan"))
            interval_width = (
                abs(upper - lower)
                if math.isfinite(lower) and math.isfinite(upper)
                else float("nan")
            )
            by_horizon[horizon] = {
                "target_month": str(rec.get("target_month", "")),
                "fused_point": _safe_float(
                    rec.get("fused_point"), default=float("nan")
                ),
                "interval_width_95": interval_width,
            }

        if by_horizon:
            run_snapshots.append(
                {"asof": cast(pd.Timestamp, asof), "by_horizon": by_horizon}
            )

    columns = [
        "horizon",
        "asof_previous",
        "asof_latest",
        "previous_target_month",
        "latest_target_month",
        "previous_fused_point",
        "latest_fused_point",
        "delta_fused_point",
        "delta_fused_point_pct",
        "previous_interval_width_95",
        "latest_interval_width_95",
        "delta_interval_width_95",
    ]
    if len(run_snapshots) < 2:
        return pd.DataFrame(columns=columns)

    ordered = sorted(run_snapshots, key=lambda item: cast(pd.Timestamp, item["asof"]))
    previous = ordered[-2]
    latest = ordered[-1]
    previous_map = cast(dict[str, dict[str, Any]], previous["by_horizon"])
    latest_map = cast(dict[str, dict[str, Any]], latest["by_horizon"])
    all_horizons = sorted(set(previous_map) | set(latest_map))

    rows: list[dict[str, Any]] = []
    for horizon in all_horizons:
        prev_row = previous_map.get(horizon, {})
        latest_row = latest_map.get(horizon, {})

        prev_point = _safe_float(prev_row.get("fused_point"), default=float("nan"))
        latest_point = _safe_float(latest_row.get("fused_point"), default=float("nan"))
        delta_point = (
            latest_point - prev_point
            if math.isfinite(prev_point) and math.isfinite(latest_point)
            else float("nan")
        )
        delta_point_pct = (
            (delta_point / prev_point) * 100.0
            if math.isfinite(delta_point)
            and math.isfinite(prev_point)
            and prev_point != 0.0
            else float("nan")
        )

        prev_width = _safe_float(
            prev_row.get("interval_width_95"), default=float("nan")
        )
        latest_width = _safe_float(
            latest_row.get("interval_width_95"), default=float("nan")
        )
        delta_width = (
            latest_width - prev_width
            if math.isfinite(prev_width) and math.isfinite(latest_width)
            else float("nan")
        )

        rows.append(
            {
                "horizon": horizon,
                "asof_previous": cast(pd.Timestamp, previous["asof"]),
                "asof_latest": cast(pd.Timestamp, latest["asof"]),
                "previous_target_month": str(prev_row.get("target_month", "")),
                "latest_target_month": str(latest_row.get("target_month", "")),
                "previous_fused_point": prev_point,
                "latest_fused_point": latest_point,
                "delta_fused_point": delta_point,
                "delta_fused_point_pct": delta_point_pct,
                "previous_interval_width_95": prev_width,
                "latest_interval_width_95": latest_width,
                "delta_interval_width_95": delta_width,
            }
        )

    return (
        pd.DataFrame(rows, columns=columns)
        .sort_values("horizon")
        .reset_index(drop=True)
    )


def _collect_nowcast_diagnostics_series(artifact_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for asof_dir in _iter_dirs(artifact_root / "nowcast"):
        diagnostics_path = asof_dir / "diagnostics.json"
        if not diagnostics_path.exists():
            continue

        payload = _load_json(diagnostics_path)
        diagnostics = payload.get("diagnostics")
        if isinstance(diagnostics, dict):
            payload = diagnostics

        calibration = payload.get("calibration_summary", {})
        divergence = payload.get("divergence_summary", {})
        asof = pd.to_datetime(_extract_asof(asof_dir.name), errors="coerce")
        if pd.isna(asof):
            continue

        rows.append(
            {
                "asof": asof,
                "coverage_rate": _safe_float(calibration.get("coverage_rate")),
                "mean_interval_width": _safe_float(
                    calibration.get("mean_interval_width")
                ),
                "mean_abs_error": _safe_float(calibration.get("mean_abs_error")),
                "mean_abs_divergence": _safe_float(
                    divergence.get("mean_abs_divergence")
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "asof",
                "coverage_rate",
                "mean_interval_width",
                "mean_abs_error",
                "mean_abs_divergence",
            ]
        )

    frame = pd.DataFrame(rows).sort_values("asof").reset_index(drop=True)
    return frame


def _collect_dm_history(report_root: Path, artifact_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []

    for asof_dir in _iter_dirs(artifact_root / "nowcast"):
        dm_path = asof_dir / "dm_results.csv"
        if not dm_path.exists():
            continue

        frame = _load_csv_frame(dm_path)
        asof = pd.to_datetime(_extract_asof(asof_dir.name), errors="coerce")
        if pd.isna(asof):
            continue

        for _, rec in frame.iterrows():
            rows.append(
                {
                    "asof": asof,
                    "target": str(rec.get("target", "unknown")),
                    "candidate_model": str(rec.get("candidate_model", "unknown")),
                    "benchmark_model": str(rec.get("benchmark_model", "unknown")),
                    "adjusted_p_value": _safe_float(rec.get("adjusted_p_value")),
                    "p_value": _safe_float(rec.get("p_value")),
                }
            )

    if not rows:
        dm_report_path = report_root / "dm_results.csv"
        if dm_report_path.exists():
            frame = _load_csv_frame(dm_report_path)
            nowcast_dirs = _iter_dirs(artifact_root / "nowcast")
            fallback_asof = (
                pd.to_datetime(_extract_asof(nowcast_dirs[-1].name), errors="coerce")
                if nowcast_dirs
                else pd.NaT
            )
            for _, rec in frame.iterrows():
                rows.append(
                    {
                        "asof": fallback_asof,
                        "target": str(rec.get("target", "unknown")),
                        "candidate_model": str(rec.get("candidate_model", "unknown")),
                        "benchmark_model": str(rec.get("benchmark_model", "unknown")),
                        "adjusted_p_value": _safe_float(rec.get("adjusted_p_value")),
                        "p_value": _safe_float(rec.get("p_value")),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "asof",
                "target",
                "candidate_model",
                "benchmark_model",
                "adjusted_p_value",
                "p_value",
            ]
        )

    dm = (
        pd.DataFrame(rows)
        .sort_values(["asof", "candidate_model"])
        .reset_index(drop=True)
    )
    dm["asof"] = pd.to_datetime(dm["asof"], errors="coerce")
    return dm


def _collect_ingest_file_inventory(bronze_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stream in ("eia_api", "eia_bulk", "dead_letter"):
        stream_root = bronze_root / stream
        for asof_dir in _iter_dirs(stream_root):
            asof = pd.to_datetime(_extract_asof(asof_dir.name), errors="coerce")
            if pd.isna(asof):
                continue

            files = [path for path in asof_dir.rglob("*") if path.is_file()]
            total_bytes = int(sum(path.stat().st_size for path in files))
            rows.append(
                {
                    "asof": asof,
                    "stream": stream,
                    "file_count": len(files),
                    "total_bytes": total_bytes,
                }
            )

    if not rows:
        return pd.DataFrame(columns=["asof", "stream", "file_count", "total_bytes"])

    frame = pd.DataFrame(rows).sort_values(["asof", "stream"]).reset_index(drop=True)
    return frame


def _collect_ingested_weekly_prices(bronze_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for stream in ("eia_api", "eia_bulk"):
        stream_root = bronze_root / stream
        for asof_dir in _iter_dirs(stream_root):
            asof = pd.to_datetime(_extract_asof(asof_dir.name), errors="coerce")
            if pd.isna(asof):
                continue

            weekly_prices_path = asof_dir / "weekly_prices.csv"
            if not weekly_prices_path.exists():
                continue

            frame = _load_csv_frame(weekly_prices_path)
            if "timestamp" not in frame.columns or "value" not in frame.columns:
                continue

            frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
            frame["value"] = pd.to_numeric(frame["value"], errors="coerce")
            frame = frame.dropna(subset=["timestamp", "value"])

            for _, rec in frame.iterrows():
                rows.append(
                    {
                        "asof": asof,
                        "stream": stream,
                        "timestamp": rec["timestamp"],
                        "value": float(rec["value"]),
                    }
                )

    if not rows:
        return pd.DataFrame(columns=["asof", "stream", "timestamp", "value"])

    frame = (
        pd.DataFrame(rows)
        .sort_values(["stream", "timestamp", "asof"])
        .reset_index(drop=True)
    )
    latest = frame.groupby(["stream", "timestamp"], as_index=False).tail(1)
    return latest.sort_values(["timestamp", "stream"]).reset_index(drop=True)


def _collect_backtest_series(report_root: Path) -> pd.DataFrame:
    backtest_path = report_root / "backtest_scorecard.csv"
    if not backtest_path.exists():
        return pd.DataFrame(columns=["horizon", "target_month", "mae", "rmse", "mape"])

    frame = _load_csv_frame(backtest_path)
    if "target_month" in frame.columns:
        frame["target_month"] = pd.to_datetime(frame["target_month"], errors="coerce")
    for metric in ("mae", "rmse", "mape"):
        if metric in frame.columns:
            frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
    return (
        frame.dropna(subset=["target_month"])
        .sort_values("target_month")
        .reset_index(drop=True)
    )


def _parse_ablation_run_end(label: str) -> pd.Timestamp | None:
    match = re.match(r"^\d{4}-\d{2}-\d{2}_(\d{4}-\d{2}-\d{2})$", label)
    if match is None:
        return None
    ts = pd.to_datetime(match.group(1), errors="coerce")
    if pd.isna(ts):
        return None
    return cast(pd.Timestamp, ts)


def _collect_ablation_history(artifact_root: Path, report_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for run_dir in _iter_dirs(artifact_root / "ablation"):
        run_end = _parse_ablation_run_end(run_dir.name)
        if run_end is None:
            continue

        scorecard_path = run_dir / "ablation_scorecard.csv"
        if not scorecard_path.exists():
            continue

        frame = _load_csv_frame(scorecard_path)
        for _, rec in frame.iterrows():
            rows.append(
                {
                    "run_end": run_end,
                    "experiment_id": str(rec.get("experiment_id", "unknown")),
                    "mae": _safe_float(rec.get("mae")),
                    "dm_vs_baseline_p_value": _safe_float(
                        rec.get("dm_vs_baseline_p_value")
                    ),
                    "runtime_seconds": _safe_float(rec.get("runtime_seconds")),
                }
            )

    if not rows:
        report_scorecard = report_root / "ablation_scorecard.csv"
        if report_scorecard.exists():
            frame = _load_csv_frame(report_scorecard)
            fallback_end = pd.to_datetime("today").normalize()
            for _, rec in frame.iterrows():
                rows.append(
                    {
                        "run_end": fallback_end,
                        "experiment_id": str(rec.get("experiment_id", "unknown")),
                        "mae": _safe_float(rec.get("mae")),
                        "dm_vs_baseline_p_value": _safe_float(
                            rec.get("dm_vs_baseline_p_value")
                        ),
                        "runtime_seconds": _safe_float(rec.get("runtime_seconds")),
                    }
                )

    if not rows:
        return pd.DataFrame(
            columns=[
                "run_end",
                "experiment_id",
                "mae",
                "dm_vs_baseline_p_value",
                "runtime_seconds",
            ]
        )

    frame = pd.DataFrame(rows)
    return frame.sort_values(["run_end", "experiment_id"]).reset_index(drop=True)


def _collect_weekly_dag_tasks(report_root: Path) -> pd.DataFrame:
    report_path = report_root / "weekly_ops_cycle_report.json"
    if not report_path.exists():
        return pd.DataFrame(columns=["dag_id", "task_count", "schedule", "asof"])

    payload = _load_json(report_path)
    rows: list[dict[str, Any]] = []
    for item in payload.get("dag_runs", []):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "dag_id": str(item.get("dag_id", "unknown")),
                "task_count": float(len(item.get("tasks", {}))),
                "schedule": str(item.get("schedule", "")),
                "asof": str(payload.get("asof", "")),
            }
        )

    if not rows:
        return pd.DataFrame(columns=["dag_id", "task_count", "schedule", "asof"])

    return pd.DataFrame(rows).sort_values("dag_id").reset_index(drop=True)


def _collect_validation_point_estimates(report_root: Path) -> pd.DataFrame:
    path = report_root / "validation_24m_point_estimates.csv"
    if not path.exists():
        return pd.DataFrame(
            columns=[
                "model_variant",
                "asof",
                "target_month",
                "fused_point",
                "actual_released",
            ]
        )

    frame = _load_csv_frame(path)
    for column in ("asof", "target_month"):
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
    for column in (
        "fused_point",
        "fused_lower_95",
        "fused_upper_95",
        "actual_released",
        "release_history_last",
    ):
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
    frame = frame.dropna(subset=["target_month", "fused_point"]).copy()
    return frame.sort_values(["model_variant", "target_month"]).reset_index(drop=True)


def _collect_validation_release_history_36m(
    artifact_root: Path,
    validation_points: pd.DataFrame,
) -> pd.DataFrame:
    nowcast_root = artifact_root / "nowcast"
    if validation_points.empty:
        return pd.DataFrame(columns=["timestamp", "target_value", "asof"])

    asof_series = pd.to_datetime(validation_points["asof"], errors="coerce")
    latest_asof = asof_series.dropna().max()
    candidate_paths: list[Path] = []
    if pd.notna(latest_asof):
        candidate_paths.append(
            nowcast_root
            / cast(pd.Timestamp, latest_asof).date().isoformat()
            / "release_history_36m.csv"
        )
    for nowcast_dir in reversed(_iter_dirs(nowcast_root)):
        candidate_paths.append(nowcast_dir / "release_history_36m.csv")

    for path in candidate_paths:
        if not path.exists():
            continue
        frame = _load_csv_frame(path)
        if not {"timestamp", "target_value"}.issubset(frame.columns):
            continue
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
        frame["target_value"] = pd.to_numeric(frame["target_value"], errors="coerce")
        frame = frame.dropna(subset=["timestamp", "target_value"]).copy()
        frame = frame.sort_values("timestamp").reset_index(drop=True)
        if "asof" not in frame.columns:
            frame["asof"] = path.parent.name
        return frame

    return pd.DataFrame(columns=["timestamp", "target_value", "asof"])


def _collect_steo_vintage_coverage(silver_root: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for vintage_dir in _iter_dirs(silver_root):
        if not vintage_dir.name.startswith("vintage_month="):
            continue
        vintage = vintage_dir.name.split("=", 1)[1]
        for table_id in ("table_5a", "table_10a", "table_10b"):
            table_path = vintage_dir / f"{table_id}.parquet"
            row_count = 0
            if table_path.exists():
                try:
                    row_count = int(len(pd.read_parquet(table_path)))
                except Exception:
                    row_count = 0
            rows.append(
                {
                    "vintage_month": pd.to_datetime(vintage, errors="coerce"),
                    "table_id": table_id,
                    "row_count": row_count,
                    "status": "parsed" if row_count > 0 else "missing_or_empty",
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["vintage_month", "table_id", "row_count", "status"]
        )
    frame = pd.DataFrame(rows)
    frame = frame.dropna(subset=["vintage_month"]).copy()
    frame = frame.sort_values(["vintage_month", "table_id"]).reset_index(drop=True)
    return frame


def _write_figure(
    fig: go.Figure,
    output_path: Path,
    *,
    title: str,
    x_title: str,
    y_title: str,
    legend_orientation: str = "h",
) -> None:
    vertical_legend = str(legend_orientation).strip().lower() == "v"
    legend_config: dict[str, Any]
    if vertical_legend:
        legend_config = {
            "orientation": "v",
            "yanchor": "top",
            "y": 1.0,
            "xanchor": "left",
            "x": 1.02,
            "traceorder": "normal",
        }
    else:
        legend_config = {
            "orientation": "h",
            "yanchor": "bottom",
            "y": 1.02,
            "x": 0.0,
            "traceorder": "normal",
        }

    fig.update_layout(
        colorway=_COCONUT_COLORS,
        template="plotly_white",
        title={"text": title, "x": 0.02},
        hovermode="x unified",
        xaxis_title=x_title,
        yaxis_title=y_title,
        legend=legend_config,
        margin={"l": 70, "r": 220 if vertical_legend else 30, "t": 70, "b": 60},
    )
    fig.update_xaxes(showgrid=True, gridcolor="#e5e7eb")
    fig.update_yaxes(showgrid=True, gridcolor="#e5e7eb")
    fig.write_html(
        output_path,
        include_plotlyjs="cdn",
        full_html=True,
        div_id=output_path.stem,
    )
    png_path = output_path.with_suffix(".png")
    fig.write_image(png_path, format="png", scale=2, width=1600, height=900)


def _plot_nowcast_intervals(frame: pd.DataFrame, output_path: Path) -> None:
    fig = go.Figure()
    if frame.empty:
        fig.add_annotation(
            text="No nowcast data available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
    else:
        palette = _COCONUT_COLORS
        horizons = sorted(frame["horizon"].dropna().unique().tolist())
        for idx, horizon in enumerate(horizons):
            subset = frame[frame["horizon"] == horizon].sort_values("asof")
            color = palette[idx % len(palette)]
            err_plus = (subset["interval_high"] - subset["fused_point"]).clip(lower=0)
            err_minus = (subset["fused_point"] - subset["interval_low"]).clip(lower=0)

            fig.add_trace(
                go.Scatter(
                    x=subset["asof"],
                    y=subset["fused_point"],
                    mode="lines+markers",
                    name=f"{horizon} fused point",
                    line={"width": 2.5, "color": color},
                    marker={"size": 7},
                    error_y={
                        "type": "data",
                        "array": err_plus,
                        "arrayminus": err_minus,
                        "thickness": 1,
                        "width": 3,
                        "color": color,
                    },
                    customdata=subset[["target_month", "target_label"]].fillna(""),
                    hovertemplate=(
                        "asof=%{x|%Y-%m-%d}<br>"
                        "horizon=" + horizon + "<br>"
                        "point=%{y:.4f}<br>"
                        "target_month=%{customdata[0]}<br>"
                        "target_label=%{customdata[1]}<extra></extra>"
                    ),
                )
            )

    _write_figure(
        fig,
        output_path,
        title="Nowcast Point and 95% Interval by As-of Date",
        x_title="As-of Date",
        y_title="Forecast Value",
    )


def _plot_nowcast_latest_vs_previous_delta(
    frame: pd.DataFrame,
    output_path: Path,
) -> None:
    fig = go.Figure()
    if frame.empty:
        fig.add_annotation(
            text="Need at least two nowcast runs to compute delta",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
        title = "Latest vs Previous Nowcast Delta"
    else:
        asof_previous = _format_timestamp(frame["asof_previous"].iloc[0])
        asof_latest = _format_timestamp(frame["asof_latest"].iloc[0])
        title = f"Latest vs Previous Nowcast Delta ({asof_previous} -> {asof_latest})"

        fig.add_trace(
            go.Bar(
                x=frame["horizon"],
                y=frame["previous_fused_point"],
                name=f"previous ({asof_previous})",
                marker={"color": "#94a3b8"},
                customdata=frame[
                    ["previous_target_month", "previous_interval_width_95"]
                ],
                hovertemplate=(
                    "horizon=%{x}<br>"
                    "run=previous<br>"
                    "point=%{y:.4f}<br>"
                    "target_month=%{customdata[0]}<br>"
                    "interval_width_95=%{customdata[1]:.4f}<extra></extra>"
                ),
            )
        )
        fig.add_trace(
            go.Bar(
                x=frame["horizon"],
                y=frame["latest_fused_point"],
                name=f"latest ({asof_latest})",
                marker={"color": "#2563eb"},
                customdata=frame[
                    [
                        "latest_target_month",
                        "latest_interval_width_95",
                        "delta_fused_point",
                        "delta_fused_point_pct",
                        "delta_interval_width_95",
                    ]
                ],
                hovertemplate=(
                    "horizon=%{x}<br>"
                    "run=latest<br>"
                    "point=%{y:.4f}<br>"
                    "target_month=%{customdata[0]}<br>"
                    "interval_width_95=%{customdata[1]:.4f}<br>"
                    "delta_point=%{customdata[2]:+.4f}<br>"
                    "delta_point_pct=%{customdata[3]:+.2f}%<br>"
                    "delta_interval_width=%{customdata[4]:+.4f}<extra></extra>"
                ),
            )
        )
        fig.update_layout(barmode="group")

        for _, rec in frame.iterrows():
            prev_point = _safe_float(
                rec.get("previous_fused_point"), default=float("nan")
            )
            latest_point = _safe_float(
                rec.get("latest_fused_point"), default=float("nan")
            )
            if not (math.isfinite(prev_point) and math.isfinite(latest_point)):
                continue

            delta_point = _safe_float(
                rec.get("delta_fused_point"), default=float("nan")
            )
            delta_pct = _safe_float(
                rec.get("delta_fused_point_pct"), default=float("nan")
            )
            delta_text = (
                f"Δ {delta_point:+,.0f} ({delta_pct:+.2f}%)"
                if math.isfinite(delta_pct)
                else f"Δ {delta_point:+,.0f}"
            )
            fig.add_annotation(
                x=rec["horizon"],
                y=max(prev_point, latest_point),
                text=delta_text,
                showarrow=False,
                yshift=12,
                font={"size": 11, "color": "#334155"},
            )

    _write_figure(
        fig,
        output_path,
        title=title,
        x_title="Horizon",
        y_title="Forecast Value",
    )


def _plot_ingest_file_counts(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No ingestion inventory found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
    else:
        fig = px.line(
            frame,
            x="asof",
            y="file_count",
            color="stream",
            markers=True,
            line_group="stream",
            color_discrete_sequence=_COCONUT_COLORS,
        )

    _write_figure(
        cast(go.Figure, fig),
        output_path,
        title="Ingestion File Count by Stream Over Time",
        x_title="As-of Date",
        y_title="File Count",
    )


def _plot_ingested_weekly_prices(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No ingested weekly_prices.csv found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
    else:
        fig = px.line(
            frame,
            x="timestamp",
            y="value",
            color="stream",
            markers=True,
            color_discrete_sequence=_COCONUT_COLORS,
        )

    _write_figure(
        cast(go.Figure, fig),
        output_path,
        title="Ingested Weekly Price Time Series (Latest Snapshot per Stream)",
        x_title="Market Timestamp",
        y_title="Price Value",
    )


def _plot_preprocess_quality(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No preprocess summary artifacts found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
    else:
        melted = frame.melt(
            id_vars=["asof"],
            value_vars=[
                "missing_flag_count",
                "outlier_flag_count",
                "unresolved_missing_count",
                "low_coverage_series_count",
            ],
            var_name="metric",
            value_name="value",
        )
        fig = px.line(
            melted,
            x="asof",
            y="value",
            color="metric",
            markers=True,
            color_discrete_sequence=_COCONUT_COLORS,
        )

    _write_figure(
        cast(go.Figure, fig),
        output_path,
        title="Preprocess Data-Quality Flags Over Time",
        x_title="As-of Date",
        y_title="Flag Count",
    )


def _plot_nowcast_diagnostics(frame: pd.DataFrame, output_path: Path) -> None:
    fig = go.Figure()
    if frame.empty:
        fig.add_annotation(
            text="No nowcast diagnostics found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=frame["asof"],
                y=frame["coverage_rate"],
                mode="lines+markers",
                name="coverage_rate",
                line={"color": "#0f766e", "width": 2.5},
                yaxis="y1",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=frame["asof"],
                y=frame["mean_interval_width"],
                mode="lines+markers",
                name="mean_interval_width",
                line={"color": "#1d4ed8", "width": 2.5},
                yaxis="y2",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=frame["asof"],
                y=frame["mean_abs_divergence"],
                mode="lines+markers",
                name="mean_abs_divergence",
                line={"color": "#be123c", "width": 2.5},
                yaxis="y2",
            )
        )
        fig.update_layout(
            yaxis2={
                "title": "Width / Divergence",
                "overlaying": "y",
                "side": "right",
                "showgrid": False,
            }
        )

    _write_figure(
        fig,
        output_path,
        title="Nowcast Calibration and Divergence Diagnostics Over Time",
        x_title="As-of Date",
        y_title="Coverage Rate",
    )


def _plot_backtest_rmse(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No backtest scorecard found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
    else:
        fig = px.line(
            frame,
            x="target_month",
            y="rmse",
            color="horizon",
            markers=True,
            color_discrete_sequence=_COCONUT_COLORS,
        )

    _write_figure(
        cast(go.Figure, fig),
        output_path,
        title="Backtest RMSE by Target Month and Horizon",
        x_title="Target Month",
        y_title="RMSE",
    )


def _plot_dm_adjusted_pvalue(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No DM result history found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
    else:
        fig = px.line(
            frame,
            x="asof",
            y="adjusted_p_value",
            color="candidate_model",
            markers=True,
            color_discrete_sequence=_COCONUT_COLORS,
        )
        fig.add_hline(y=0.05, line_dash="dash", line_color="#b91c1c")
        fig.add_hline(y=0.01, line_dash="dot", line_color="#7f1d1d")

    _write_figure(
        cast(go.Figure, fig),
        output_path,
        title="DM Adjusted p-value Over Time",
        x_title="As-of Date",
        y_title="Adjusted p-value",
    )


def _plot_ablation_mae(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No ablation history found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
    else:
        fig = px.line(
            frame,
            x="run_end",
            y="mae",
            color="experiment_id",
            markers=True,
            color_discrete_sequence=_COCONUT_COLORS,
        )

    _write_figure(
        cast(go.Figure, fig),
        output_path,
        title="Ablation MAE Trend by Experiment",
        x_title="Run End Date",
        y_title="MAE",
    )


def _plot_ablation_dm_pvalue(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No ablation history found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
    else:
        fig = px.line(
            frame,
            x="run_end",
            y="dm_vs_baseline_p_value",
            color="experiment_id",
            markers=True,
            color_discrete_sequence=_COCONUT_COLORS,
        )
        fig.add_hline(y=0.10, line_dash="dash", line_color="#92400e")

    _write_figure(
        cast(go.Figure, fig),
        output_path,
        title="Ablation DM vs Baseline p-value Trend",
        x_title="Run End Date",
        y_title="Adjusted p-value",
    )


def _plot_weekly_dag_task_counts(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No weekly ops report found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
    else:
        fig = px.bar(
            frame,
            x="dag_id",
            y="task_count",
            color="dag_id",
            color_discrete_sequence=_COCONUT_COLORS,
        )

    _write_figure(
        cast(go.Figure, fig),
        output_path,
        title="Weekly DAG Task Counts (Latest Cycle)",
        x_title="DAG",
        y_title="Task Count",
    )


def _plot_validation_24m_point_vs_release_36m(
    point_estimates: pd.DataFrame,
    release_history: pd.DataFrame,
    output_path: Path,
) -> None:
    fig = go.Figure()
    if point_estimates.empty:
        fig.add_annotation(
            text="No 24-month validation scorecard found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
    else:
        palette = _COCONUT_COLORS
        for idx, variant in enumerate(
            sorted(point_estimates["model_variant"].dropna().unique().tolist())
        ):
            subset = point_estimates[
                point_estimates["model_variant"] == variant
            ].sort_values("target_month")
            fig.add_trace(
                go.Scatter(
                    x=subset["target_month"],
                    y=subset["fused_point"],
                    mode="lines+markers",
                    name=f"{variant} fused_point",
                    line={
                        "width": 2.2,
                        "color": palette[idx % len(palette)],
                    },
                    marker={"size": 6},
                )
            )

        actual_series = (
            point_estimates[["target_month", "actual_released"]]
            .drop_duplicates(subset=["target_month"])
            .sort_values("target_month")
        )
        if not actual_series.empty:
            fig.add_trace(
                go.Scatter(
                    x=actual_series["target_month"],
                    y=actual_series["actual_released"],
                    mode="lines+markers",
                    name="released_actual",
                    line={"width": 2.8, "color": "#0f172a"},
                    marker={"size": 5},
                )
            )

    if not release_history.empty:
        source_label = str(release_history.iloc[-1].get("asof", "latest"))
        fig.add_trace(
            go.Scatter(
                x=release_history["timestamp"],
                y=release_history["target_value"],
                mode="lines",
                name=f"release_history_36m ({source_label})",
                line={"width": 2.0, "dash": "dash", "color": "#475569"},
            )
        )

    _write_figure(
        fig,
        output_path,
        title="24-Run Point Estimates vs 36-Month Release History",
        x_title="Target Month",
        y_title="U.S. Dry Natural Gas Production",
        legend_orientation="v",
    )


def _plot_validation_24m_point_vs_official_releases(
    point_estimates: pd.DataFrame,
    output_path: Path,
) -> None:
    fig = go.Figure()
    if point_estimates.empty:
        fig.add_annotation(
            text="No 24-month validation scorecard found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
    else:
        palette = _COCONUT_COLORS
        for idx, variant in enumerate(
            sorted(point_estimates["model_variant"].dropna().unique().tolist())
        ):
            subset = point_estimates[
                point_estimates["model_variant"] == variant
            ].sort_values("target_month")
            fig.add_trace(
                go.Scatter(
                    x=subset["target_month"],
                    y=subset["fused_point"],
                    mode="lines+markers",
                    name=f"{variant} fused_point",
                    line={
                        "width": 2.2,
                        "color": palette[idx % len(palette)],
                    },
                    marker={"size": 6},
                )
            )

        actual_series = (
            point_estimates[["target_month", "actual_released"]]
            .dropna(subset=["actual_released"])
            .drop_duplicates(subset=["target_month"])
            .sort_values("target_month")
        )
        if not actual_series.empty:
            fig.add_trace(
                go.Scatter(
                    x=actual_series["target_month"],
                    y=actual_series["actual_released"],
                    mode="lines+markers",
                    name="official_release",
                    line={"width": 2.8, "color": "#0f172a"},
                    marker={"size": 5},
                )
            )

    _write_figure(
        fig,
        output_path,
        title="24-Run Point Estimates vs Official Releases",
        x_title="Target Month",
        y_title="U.S. Dry Natural Gas Production",
        legend_orientation="v",
    )


def _plot_steo_vintage_coverage(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No silver STEO vintage tables found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
    else:
        fig = px.line(
            frame,
            x="vintage_month",
            y="row_count",
            color="table_id",
            markers=True,
            color_discrete_sequence=_COCONUT_COLORS,
        )

    _write_figure(
        cast(go.Figure, fig),
        output_path,
        title="STEO Silver Table Coverage by Vintage Month",
        x_title="Vintage Month",
        y_title="Parsed Row Count",
    )


def _plot_steo_parser_status(frame: pd.DataFrame, output_path: Path) -> None:
    if frame.empty:
        fig = go.Figure()
        fig.add_annotation(
            text="No STEO parser status rows found",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
    else:
        status = (
            frame.groupby("status", as_index=False)["table_id"]
            .count()
            .rename(columns={"table_id": "count"})
        )
        fig = px.bar(
            status,
            x="status",
            y="count",
            color="status",
            color_discrete_sequence=_COCONUT_COLORS,
        )

    _write_figure(
        cast(go.Figure, fig),
        output_path,
        title="STEO Parser Status Counts",
        x_title="Parser Status",
        y_title="Count",
    )


def _plot_validation_gold_provenance(
    point_estimates: pd.DataFrame,
    output_path: Path,
) -> None:
    fig = go.Figure()
    if point_estimates.empty or "steo_gold_status" not in point_estimates.columns:
        fig.add_annotation(
            text="No validation provenance columns available",
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.5,
            showarrow=False,
            font={"size": 16},
        )
    else:
        provenance = (
            point_estimates.groupby(
                ["model_variant", "steo_gold_status"], as_index=False
            )["target_month"]
            .count()
            .rename(columns={"target_month": "run_count"})
        )
        fig = px.bar(
            provenance,
            x="model_variant",
            y="run_count",
            color="steo_gold_status",
            barmode="group",
            color_discrete_sequence=_COCONUT_COLORS,
        )

    _write_figure(
        fig,
        output_path,
        title="24-Run Gold Source Provenance by Model Variant",
        x_title="Model Variant",
        y_title="Run Count",
        legend_orientation="v",
    )


def generate_all_visuals(
    *,
    report_root: str | Path = "data/reports",
    artifact_root: str | Path = "data/artifacts",
    output_dir: str | Path = "ops/viz/output",
    clean: bool = True,
) -> dict[str, Any]:
    """Generate Plotly visual outputs (HTML + PNG) from ingestion and nowcast artifacts."""

    report_path = Path(report_root)
    artifact_path = Path(artifact_root)
    bronze_path = artifact_path.parent / "bronze"
    silver_path = artifact_path.parent / "silver" / "steo_vintages"
    out = _ensure_dir(output_dir)
    _cleanup_legacy_outputs(out)

    generated: dict[str, str] = {}
    summary: dict[str, Any] = {
        "generated_at_utc": datetime.now(timezone.utc).strftime(
            "%Y-%m-%d %H:%M:%S UTC"
        ),
        "source_roots": {
            "report_root": str(report_path),
            "artifact_root": str(artifact_path),
            "bronze_root": str(bronze_path),
            "silver_root": str(silver_path),
        },
        "output_dir": str(out),
        "generated": generated,
        "missing_sources": [],
        "dataset_rows": {},
    }

    def _register_visual_output(name: str, html_path: Path) -> None:
        generated[name] = str(html_path)
        generated[f"{name}_png"] = str(html_path.with_suffix(".png"))

    nowcast = _collect_nowcast_series(artifact_path)
    preprocess = _collect_preprocess_summary_series(artifact_path)
    diagnostics = _collect_nowcast_diagnostics_series(artifact_path)
    latest_vs_previous_delta = _collect_nowcast_latest_vs_previous_delta(artifact_path)
    dm_history = _collect_dm_history(report_path, artifact_path)
    ingest_inventory = _collect_ingest_file_inventory(bronze_path)
    ingested_prices = _collect_ingested_weekly_prices(bronze_path)
    backtest = _collect_backtest_series(report_path)
    ablation = _collect_ablation_history(artifact_path, report_path)
    dag_counts = _collect_weekly_dag_tasks(report_path)
    validation_points = _collect_validation_point_estimates(report_path)
    validation_release_36m = _collect_validation_release_history_36m(
        artifact_path,
        validation_points,
    )
    steo_coverage = _collect_steo_vintage_coverage(silver_path)

    summary["dataset_rows"] = {
        "nowcast": int(len(nowcast)),
        "preprocess": int(len(preprocess)),
        "diagnostics": int(len(diagnostics)),
        "latest_vs_previous_delta": int(len(latest_vs_previous_delta)),
        "dm_history": int(len(dm_history)),
        "ingest_inventory": int(len(ingest_inventory)),
        "ingested_prices": int(len(ingested_prices)),
        "backtest": int(len(backtest)),
        "ablation": int(len(ablation)),
        "dag_counts": int(len(dag_counts)),
        "validation_24m_points": int(len(validation_points)),
        "validation_release_36m": int(len(validation_release_36m)),
        "steo_vintage_coverage": int(len(steo_coverage)),
    }
    summary["freshness"] = {
        "nowcast_max_asof": (
            _format_timestamp(nowcast["asof"].max()) if not nowcast.empty else ""
        ),
        "validation_max_target_month": (
            _format_timestamp(validation_points["target_month"].max())
            if not validation_points.empty
            else ""
        ),
        "validation_max_asof": (
            _format_timestamp(validation_points["asof"].max())
            if not validation_points.empty and "asof" in validation_points.columns
            else ""
        ),
        "release_history_36m_max_timestamp": (
            _format_timestamp(validation_release_36m["timestamp"].max())
            if not validation_release_36m.empty
            else ""
        ),
        "steo_vintage_max_month": (
            _format_timestamp(steo_coverage["vintage_month"].max())
            if not steo_coverage.empty
            else ""
        ),
    }

    nowcast_intervals_path = out / "nowcast_intervals.html"
    _plot_nowcast_intervals(nowcast, nowcast_intervals_path)
    _register_visual_output("nowcast_intervals", nowcast_intervals_path)
    if nowcast.empty:
        summary["missing_sources"].append("data/artifacts/nowcast/*/nowcast.json")

    ingest_counts_path = out / "ingest_file_counts.html"
    _plot_ingest_file_counts(ingest_inventory, ingest_counts_path)
    _register_visual_output("ingest_file_counts", ingest_counts_path)
    if ingest_inventory.empty:
        summary["missing_sources"].append("data/bronze/*/asof=*/")

    ingested_prices_path = out / "ingested_weekly_prices.html"
    _plot_ingested_weekly_prices(ingested_prices, ingested_prices_path)
    _register_visual_output("ingested_weekly_prices", ingested_prices_path)
    if ingested_prices.empty:
        summary["missing_sources"].append("data/bronze/*/asof=*/weekly_prices.csv")

    preprocess_quality_path = out / "preprocess_quality_timeseries.html"
    _plot_preprocess_quality(preprocess, preprocess_quality_path)
    _register_visual_output("preprocess_quality", preprocess_quality_path)
    if preprocess.empty:
        summary["missing_sources"].append(
            "data/artifacts/nowcast/*/preprocess_summary.json"
        )

    diagnostics_path = out / "nowcast_diagnostics_timeseries.html"
    _plot_nowcast_diagnostics(diagnostics, diagnostics_path)
    _register_visual_output("nowcast_diagnostics", diagnostics_path)
    if diagnostics.empty:
        summary["missing_sources"].append("data/artifacts/nowcast/*/diagnostics.json")

    latest_vs_previous_delta_path = out / "nowcast_latest_vs_previous_delta.html"
    _plot_nowcast_latest_vs_previous_delta(
        latest_vs_previous_delta,
        latest_vs_previous_delta_path,
    )
    _register_visual_output(
        "nowcast_latest_vs_previous_delta",
        latest_vs_previous_delta_path,
    )
    if latest_vs_previous_delta.empty:
        summary["missing_sources"].append(
            "data/artifacts/nowcast/*/nowcast.json (>=2 runs required for delta)"
        )

    backtest_rmse_path = out / "backtest_rmse.html"
    _plot_backtest_rmse(backtest, backtest_rmse_path)
    _register_visual_output("backtest_rmse", backtest_rmse_path)
    if backtest.empty:
        summary["missing_sources"].append(str(report_path / "backtest_scorecard.csv"))

    dm_path = out / "dm_policy_adjusted_pvalue.html"
    _plot_dm_adjusted_pvalue(dm_history, dm_path)
    _register_visual_output("dm_policy_adjusted_pvalue", dm_path)
    if dm_history.empty:
        summary["missing_sources"].append(str(report_path / "dm_results.csv"))

    ablation_mae_path = out / "ablation_mae.html"
    _plot_ablation_mae(ablation, ablation_mae_path)
    _register_visual_output("ablation_mae", ablation_mae_path)

    ablation_pvalue_path = out / "ablation_dm_pvalue.html"
    _plot_ablation_dm_pvalue(ablation, ablation_pvalue_path)
    _register_visual_output("ablation_dm_pvalue", ablation_pvalue_path)
    if ablation.empty:
        summary["missing_sources"].append(
            "data/artifacts/ablation/*/ablation_scorecard.csv"
        )

    dag_path = out / "weekly_dag_task_counts.html"
    _plot_weekly_dag_task_counts(dag_counts, dag_path)
    _register_visual_output("weekly_dag_task_counts", dag_path)
    if dag_counts.empty:
        summary["missing_sources"].append(
            str(report_path / "weekly_ops_cycle_report.json")
        )

    validation_path = out / "validation_24m_point_vs_release_36m.html"
    _plot_validation_24m_point_vs_release_36m(
        validation_points,
        validation_release_36m,
        validation_path,
    )
    _register_visual_output("validation_24m_point_vs_release_36m", validation_path)

    validation_official_path = out / "validation_24m_point_vs_official_releases.html"
    _plot_validation_24m_point_vs_official_releases(
        validation_points,
        validation_official_path,
    )
    _register_visual_output(
        "validation_24m_point_vs_official_releases",
        validation_official_path,
    )

    steo_coverage_path = out / "steo_vintage_coverage.html"
    _plot_steo_vintage_coverage(steo_coverage, steo_coverage_path)
    _register_visual_output("steo_vintage_coverage", steo_coverage_path)

    steo_parser_path = out / "steo_parser_status.html"
    _plot_steo_parser_status(steo_coverage, steo_parser_path)
    _register_visual_output("steo_parser_status", steo_parser_path)

    validation_prov_path = out / "validation_24m_gold_provenance.html"
    _plot_validation_gold_provenance(validation_points, validation_prov_path)
    _register_visual_output("validation_24m_gold_provenance", validation_prov_path)

    if validation_points.empty:
        summary["missing_sources"].append(
            str(report_path / "validation_24m_point_estimates.csv")
        )
    if validation_release_36m.empty:
        summary["missing_sources"].append(
            "data/artifacts/nowcast/*/release_history_36m.csv"
        )
    if steo_coverage.empty:
        summary["missing_sources"].append(
            "data/silver/steo_vintages/vintage_month=*/table_*.parquet"
        )

    dashboard_path = out / "dashboard.md"
    lines = [
        "# Pipeline Visual Dashboard",
        "",
        f"- Generated at (UTC): `{summary['generated_at_utc']}`",
        f"- Report root: `{report_path}`",
        f"- Artifact root: `{artifact_path}`",
        "",
        "## Time-Series Visuals",
    ]
    for name, path in sorted(generated.items()):
        rel = Path(path).as_posix()
        lines.append(f"- `{name}`: `{rel}`")

    if summary["missing_sources"]:
        lines.extend(["", "## Missing Sources"])
        for missing in summary["missing_sources"]:
            lines.append(f"- `{missing}`")

    lines.extend(["", "## Dataset Row Counts"])
    for key, value in sorted(summary["dataset_rows"].items()):
        lines.append(f"- `{key}`: `{value}`")
    lines.extend(["", "## Freshness"])
    for key, value in sorted(summary["freshness"].items()):
        lines.append(f"- `{key}`: `{value}`")

    dashboard_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    generated["dashboard"] = str(dashboard_path)

    summary["missing_sources"] = sorted(set(summary["missing_sources"]))
    summary["missing_source_count"] = int(len(summary["missing_sources"]))

    summary_path = out / "visualization_summary.json"
    summary_path.write_text(
        json.dumps(summary, sort_keys=True, indent=2), encoding="utf-8"
    )
    generated["summary"] = str(summary_path)

    if clean:
        keep_files = {Path(path).name for path in generated.values()}
        _cleanup_stale_generated_outputs(out, keep_files)

    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-root", default="data/reports")
    parser.add_argument("--artifact-root", default="data/artifacts")
    parser.add_argument("--output-dir", default="ops/viz/output")
    parser.add_argument(
        "--clean",
        type=int,
        default=1,
        choices=(0, 1),
        help="Set to 1 to remove stale generated visualization files in output-dir.",
    )
    parser.add_argument(
        "--strict-missing",
        action="store_true",
        help="Fail with non-zero exit code when missing_sources are detected.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    summary = generate_all_visuals(
        report_root=args.report_root,
        artifact_root=args.artifact_root,
        output_dir=args.output_dir,
        clean=bool(int(args.clean)),
    )

    payload = {
        "generated_count": len(summary["generated"]),
        "missing_source_count": summary["missing_source_count"],
        "missing_sources": summary["missing_sources"],
        "output_dir": summary["output_dir"],
    }
    print(json.dumps(payload, sort_keys=True))
    if args.strict_missing and summary["missing_source_count"] > 0:
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
