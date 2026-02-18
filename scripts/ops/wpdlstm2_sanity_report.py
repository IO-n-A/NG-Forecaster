#!/usr/bin/env python3
"""Sanity report for wpd_vmd_lstm2 before/after performance and replay invariance."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, cast

import pandas as pd


def _load_scorecard(root: Path) -> pd.DataFrame:
    path = root / "validation_24m_scorecard.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _load_summary(root: Path) -> dict[str, Any]:
    path = root / "validation_24m_summary.json"
    if not path.exists():
        raise FileNotFoundError(path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return cast(dict[str, Any], payload)


def _load_points(root: Path) -> pd.DataFrame:
    path = root / "validation_24m_point_estimates.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _window_fingerprint(summary: dict[str, Any]) -> str:
    payload = {
        "target_month_start": summary.get("target_month_start", ""),
        "target_month_end": summary.get("target_month_end", ""),
        "asof_schedule": summary.get("asof_schedule", []),
    }
    raw = json.dumps(payload, sort_keys=True).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def _resolve_window_payload(
    summary: dict[str, Any],
    points: pd.DataFrame,
) -> dict[str, Any]:
    target_start = str(summary.get("target_month_start", "")).strip()
    target_end = str(summary.get("target_month_end", "")).strip()
    asof_schedule = summary.get("asof_schedule", [])
    if not target_start and "target_month" in points.columns and not points.empty:
        target_start = str(points["target_month"].astype(str).min())
    if not target_end and "target_month" in points.columns and not points.empty:
        target_end = str(points["target_month"].astype(str).max())
    if (not isinstance(asof_schedule, list) or len(asof_schedule) == 0) and (
        "asof" in points.columns and not points.empty
    ):
        asof_schedule = sorted(points["asof"].astype(str).unique().tolist())
    return {
        "target_month_start": target_start,
        "target_month_end": target_end,
        "asof_schedule": asof_schedule if isinstance(asof_schedule, list) else [],
    }


def _pick_handcheck_months(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return frame
    frame = frame.copy()
    frame["ape_pct"] = pd.to_numeric(frame["ape_pct"], errors="coerce")
    frame["abs_error"] = pd.to_numeric(frame["abs_error"], errors="coerce")
    frame = frame.sort_values(["ape_pct", "target_month"], ascending=[False, True])

    picked: list[pd.DataFrame] = []
    for regime_label in ("multi_shock", "freeze_off", "normal"):
        subset = frame[frame["regime_label"].astype(str) == regime_label].head(2)
        if not subset.empty:
            picked.append(subset)

    selected = pd.concat(picked, ignore_index=True) if picked else pd.DataFrame()
    selected_keys = set(
        (
            str(row["asof"]),
            str(row["target_month"]),
            str(row["regime_label"]),
        )
        for _, row in selected.iterrows()
    )
    if len(selected) < 8:
        extras: list[pd.Series] = []
        for _, row in frame.iterrows():
            key = (str(row["asof"]), str(row["target_month"]), str(row["regime_label"]))
            if key in selected_keys:
                continue
            extras.append(row)
            if len(selected) + len(extras) >= 8:
                break
        if extras:
            selected = pd.concat([selected, pd.DataFrame(extras)], ignore_index=True)

    selected = (
        selected.drop_duplicates(subset=["asof", "target_month"])
        .sort_values("target_month")
        .reset_index(drop=True)
    )
    return selected[
        [
            "asof",
            "target_month",
            "regime_label",
            "actual_released",
            "fused_point",
            "error",
            "ape_pct",
            "target_month_days",
            "day_count_class",
            "regime_freeze_flag",
            "regime_basis_flag",
            "regime_transfer_dispersion_flag",
            "calendar_calibration_applied",
            "forecast_source",
        ]
    ]


def build_report(
    *,
    before_root: Path,
    after_root: Path,
    handcheck_root: Path,
    out_root: Path,
) -> dict[str, str]:
    before_score = _load_scorecard(before_root)
    after_score = _load_scorecard(after_root)
    before_summary = _load_summary(before_root)
    after_summary = _load_summary(after_root)
    before_points = _load_points(before_root)
    after_points = _load_points(after_root)
    before_window = _resolve_window_payload(before_summary, before_points)
    after_window = _resolve_window_payload(after_summary, after_points)

    model_id = "wpd_vmd_lstm2"
    before_row = before_score[before_score["model_variant"] == model_id]
    after_row = after_score[after_score["model_variant"] == model_id]
    if before_row.empty or after_row.empty:
        raise ValueError(f"missing {model_id} in scorecard roots")
    before_row = before_row.iloc[0]
    after_row = after_row.iloc[0]

    before_mape = float(before_row["mape_pct"])
    after_mape = float(after_row["mape_pct"])
    before_mae = float(before_row["mae"])
    after_mae = float(after_row["mae"])
    before_rmse = float(before_row["rmse"])
    after_rmse = float(after_row["rmse"])

    before_fp = _window_fingerprint(before_window)
    after_fp = _window_fingerprint(after_window)
    same_window = before_fp == after_fp

    before_after = pd.DataFrame(
        [
            {
                "model_variant": model_id,
                "before_root": str(before_root),
                "after_root": str(after_root),
                "before_mape_pct": before_mape,
                "after_mape_pct": after_mape,
                "delta_mape_pct_after_minus_before": after_mape - before_mape,
                "before_mae": before_mae,
                "after_mae": after_mae,
                "delta_mae_after_minus_before": after_mae - before_mae,
                "before_rmse": before_rmse,
                "after_rmse": after_rmse,
                "delta_rmse_after_minus_before": after_rmse - before_rmse,
                "before_target_month_start": str(before_window["target_month_start"]),
                "before_target_month_end": str(before_window["target_month_end"]),
                "after_target_month_start": str(after_window["target_month_start"]),
                "after_target_month_end": str(after_window["target_month_end"]),
                "before_asof_count": int(len(before_window["asof_schedule"])),
                "after_asof_count": int(len(after_window["asof_schedule"])),
                "same_window_fingerprint": bool(same_window),
            }
        ]
    )

    out_root.mkdir(parents=True, exist_ok=True)
    before_after_path = out_root / "wpdlstm2_before_after.csv"
    before_after.to_csv(before_after_path, index=False)

    fingerprint = {
        "model_variant": model_id,
        "before_root": str(before_root),
        "after_root": str(after_root),
        "before": {
            "target_month_start": str(before_window["target_month_start"]),
            "target_month_end": str(before_window["target_month_end"]),
            "asof_schedule": list(before_window["asof_schedule"]),
            "fingerprint": before_fp,
        },
        "after": {
            "target_month_start": str(after_window["target_month_start"]),
            "target_month_end": str(after_window["target_month_end"]),
            "asof_schedule": list(after_window["asof_schedule"]),
            "fingerprint": after_fp,
        },
        "same_window_fingerprint": bool(same_window),
    }
    fingerprint_path = out_root / "wpdlstm2_window_fingerprint.json"
    fingerprint_path.write_text(
        json.dumps(fingerprint, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    handcheck_points = _load_points(handcheck_root)
    handcheck_points = handcheck_points[
        handcheck_points["model_variant"].astype(str) == model_id
    ].copy()
    handcheck_table = _pick_handcheck_months(handcheck_points)
    handcheck_path = out_root / "wpdlstm2_handcheck_months.csv"
    handcheck_table.to_csv(handcheck_path, index=False)

    notes = out_root / "wpdlstm2_handcheck_notes.md"
    notes.write_text(
        "\n".join(
            [
                "# WPD-LSTM2 Handcheck Notes",
                "",
                f"- Source root: `{handcheck_root}`",
                f"- Selected rows: {int(len(handcheck_table))}",
                "- Selection policy: prefer 2 rows each from `multi_shock`, `freeze_off`, and `normal`; fill remaining slots by highest APE months.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "before_after_path": str(before_after_path),
        "window_fingerprint_path": str(fingerprint_path),
        "handcheck_path": str(handcheck_path),
        "handcheck_notes_path": str(notes),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--before_root",
        default="data/reports/sprint7/after/promoted_full_plus_regime",
        help="Baseline report root for before snapshot.",
    )
    parser.add_argument(
        "--after_root",
        default="data/reports/sprint8/after/promoted",
        help="After report root for comparison snapshot.",
    )
    parser.add_argument(
        "--handcheck_root",
        default="data/reports/sprint9/replay_24m_with_bsts",
        help="Report root used to sample month-level handcheck rows.",
    )
    parser.add_argument(
        "--out_root",
        default="data/reports/sprint9/wpdlstm2_sanity",
        help="Output root for sanity artifacts.",
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = build_report(
        before_root=Path(args.before_root),
        after_root=Path(args.after_root),
        handcheck_root=Path(args.handcheck_root),
        out_root=Path(args.out_root),
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
