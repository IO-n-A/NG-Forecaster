"""Compute BSTS marginal benefit from paired enabled vs forced-off replays."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ng_forecaster.evaluation.dm_test import run_dm_tests, run_dm_tests_by_regime
from ng_forecaster.errors import ContractViolation


def _load_point_estimates(root: Path) -> pd.DataFrame:
    path = root / "validation_24m_point_estimates.csv"
    if not path.exists():
        raise ContractViolation(
            "missing_source_file",
            key=str(path),
            detail="expected validation_24m_point_estimates.csv in report root",
        )
    frame = pd.read_csv(path)
    required = {
        "model_variant",
        "asof",
        "target_month",
        "fused_point",
        "actual_released",
        "regime_label",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ContractViolation(
            "missing_column",
            key=str(path),
            detail="point estimates missing columns: " + ", ".join(missing),
        )
    frame = frame[
        [
            "model_variant",
            "asof",
            "target_month",
            "fused_point",
            "actual_released",
            "regime_label",
        ]
    ].copy()
    frame["model_variant"] = frame["model_variant"].astype(str)
    frame["asof"] = frame["asof"].astype(str)
    frame["target_month"] = frame["target_month"].astype(str)
    frame["regime_label"] = frame["regime_label"].astype(str)
    frame["fused_point"] = pd.to_numeric(frame["fused_point"], errors="coerce")
    frame["actual_released"] = pd.to_numeric(frame["actual_released"], errors="coerce")
    frame = frame[
        frame["fused_point"].notna() & frame["actual_released"].notna()
    ].copy()
    return frame


def _build_dm_rows(merged: pd.DataFrame, *, variant: str) -> pd.DataFrame:
    if merged.empty:
        return pd.DataFrame(
            columns=[
                "regime_label",
                "model_variant",
                "target_month",
                "actual_released",
                "fused_point",
                "asof",
            ]
        )
    enabled_label = f"{variant}__enabled"
    forced_off_label = f"{variant}__forced_off"
    base_cols = ["asof", "target_month", "regime_label"]

    enabled = merged[
        base_cols + ["actual_released_enabled", "fused_point_enabled"]
    ].rename(
        columns={
            "actual_released_enabled": "actual_released",
            "fused_point_enabled": "fused_point",
        }
    )
    enabled["model_variant"] = enabled_label
    forced_off = merged[
        base_cols + ["actual_released_forced_off", "fused_point_forced_off"]
    ].rename(
        columns={
            "actual_released_forced_off": "actual_released",
            "fused_point_forced_off": "fused_point",
        }
    )
    forced_off["model_variant"] = forced_off_label
    return pd.concat([enabled, forced_off], ignore_index=True)


def _resolve_dm_overall(dm_rows: pd.DataFrame, *, variant: str) -> tuple[float, float]:
    if dm_rows.empty:
        return np.nan, np.nan
    policy = {
        "loss": "mse",
        "sidedness": "less",
        "multiple_comparison": "none",
        "benchmark_by_target": {"ng_prod": f"{variant}__forced_off"},
        "comparison_pairs_by_target": {
            "ng_prod": [(f"{variant}__enabled", f"{variant}__forced_off")]
        },
    }
    dm_input = dm_rows.rename(columns={"model_variant": "model"})
    dm_input["target"] = "ng_prod"
    dm_input["horizon"] = 1
    dm = run_dm_tests(
        dm_input[
            ["target", "model", "asof", "horizon", "actual_released", "fused_point"]
        ].rename(
            columns={
                "actual_released": "actual",
                "fused_point": "forecast",
            }
        ),
        policy,
        target_col="target",
        model_col="model",
        actual_col="actual",
        forecast_col="forecast",
        asof_col="asof",
        horizon_col="horizon",
    ).results
    if dm.empty:
        return np.nan, np.nan
    row = dm.iloc[0]
    return (
        float(row.get("dm_p_value_one_sided_improve", np.nan)),
        float(row.get("dm_p_value_two_sided", np.nan)),
    )


def _resolve_dm_by_regime(dm_rows: pd.DataFrame, *, variant: str) -> pd.DataFrame:
    if dm_rows.empty:
        return pd.DataFrame(
            columns=[
                "regime_label",
                "dm_p_value_one_sided_improve",
                "dm_p_value_two_sided",
                "dm_stat",
                "n_obs",
            ]
        )
    policy = {
        "loss": "mse",
        "sidedness": "less",
        "multiple_comparison": "none",
        "benchmark_by_target": {"ng_prod": f"{variant}__forced_off"},
        "comparison_pairs_by_target": {
            "ng_prod": [(f"{variant}__enabled", f"{variant}__forced_off")]
        },
    }
    dm = run_dm_tests_by_regime(
        dm_rows,
        policy,
        regime_col="regime_label",
        model_col="model_variant",
        actual_col="actual_released",
        forecast_col="fused_point",
        asof_col="asof",
        target_col="target_month",
        min_observations=3,
    )
    required_cols = {"status", "candidate_model", "benchmark_model"}
    if not required_cols.issubset(set(dm.columns)):
        return pd.DataFrame(
            columns=[
                "regime_label",
                "dm_p_value_one_sided_improve",
                "dm_p_value_two_sided",
                "dm_stat",
                "n_obs",
            ]
        )
    dm = dm[
        (dm.get("status", "").astype(str) == "ok")
        & (dm.get("candidate_model", "").astype(str) == f"{variant}__enabled")
        & (dm.get("benchmark_model", "").astype(str) == f"{variant}__forced_off")
    ].copy()
    if dm.empty:
        return pd.DataFrame(
            columns=[
                "regime_label",
                "dm_p_value_one_sided_improve",
                "dm_p_value_two_sided",
                "dm_stat",
                "n_obs",
            ]
        )
    return dm[
        [
            "regime_label",
            "dm_p_value_one_sided_improve",
            "dm_p_value_two_sided",
            "dm_stat",
            "n_obs",
        ]
    ].reset_index(drop=True)


def compute_bsts_marginal_benefit(
    *,
    enabled_root: str | Path,
    forced_off_root: str | Path,
    out_root: str | Path,
) -> dict[str, str]:
    enabled = _load_point_estimates(Path(enabled_root))
    forced_off = _load_point_estimates(Path(forced_off_root))

    common_variants = sorted(
        set(enabled["model_variant"].astype(str).unique().tolist()).intersection(
            set(forced_off["model_variant"].astype(str).unique().tolist())
        )
    )
    if not common_variants:
        raise ContractViolation(
            "missing_dm_candidates",
            key="model_variant",
            detail="no overlapping model variants between enabled and forced-off roots",
        )

    overall_rows: list[dict[str, Any]] = []
    regime_rows: list[dict[str, Any]] = []
    for variant in common_variants:
        enabled_variant = enabled[enabled["model_variant"] == variant].copy()
        forced_variant = forced_off[forced_off["model_variant"] == variant].copy()
        merged = enabled_variant.merge(
            forced_variant,
            on=["asof", "target_month"],
            suffixes=("_enabled", "_forced_off"),
            how="inner",
        )
        if merged.empty:
            continue
        merged["regime_label"] = merged["regime_label_enabled"].astype(str)
        merged["error_enabled"] = (
            merged["fused_point_enabled"] - merged["actual_released_enabled"]
        )
        merged["error_forced_off"] = (
            merged["fused_point_forced_off"] - merged["actual_released_forced_off"]
        )
        merged["abs_error_enabled"] = merged["error_enabled"].abs()
        merged["abs_error_forced_off"] = merged["error_forced_off"].abs()
        merged["ape_enabled"] = np.where(
            merged["actual_released_enabled"].abs() > 0,
            merged["abs_error_enabled"]
            / merged["actual_released_enabled"].abs()
            * 100.0,
            np.nan,
        )
        merged["ape_forced_off"] = np.where(
            merged["actual_released_forced_off"].abs() > 0,
            merged["abs_error_forced_off"]
            / merged["actual_released_forced_off"].abs()
            * 100.0,
            np.nan,
        )
        dm_rows = _build_dm_rows(merged, variant=variant)
        dm_p_one_sided, dm_p_two_sided = _resolve_dm_overall(dm_rows, variant=variant)
        overall_rows.append(
            {
                "model_variant": variant,
                "n_runs": int(len(merged)),
                "mape_pct_enabled": float(np.nanmean(merged["ape_enabled"])),
                "mape_pct_forced_off": float(np.nanmean(merged["ape_forced_off"])),
                "mae_enabled": float(np.nanmean(merged["abs_error_enabled"])),
                "mae_forced_off": float(np.nanmean(merged["abs_error_forced_off"])),
                "rmse_enabled": float(
                    np.sqrt(np.nanmean(np.square(merged["error_enabled"])))
                ),
                "rmse_forced_off": float(
                    np.sqrt(np.nanmean(np.square(merged["error_forced_off"])))
                ),
                "delta_mape_pct_enabled_minus_forced_off": float(
                    np.nanmean(merged["ape_enabled"])
                    - np.nanmean(merged["ape_forced_off"])
                ),
                "delta_mae_enabled_minus_forced_off": float(
                    np.nanmean(merged["abs_error_enabled"])
                    - np.nanmean(merged["abs_error_forced_off"])
                ),
                "delta_rmse_enabled_minus_forced_off": float(
                    np.sqrt(np.nanmean(np.square(merged["error_enabled"])))
                    - np.sqrt(np.nanmean(np.square(merged["error_forced_off"])))
                ),
                "dm_p_value_one_sided_improve": float(dm_p_one_sided),
                "dm_p_value_two_sided": float(dm_p_two_sided),
            }
        )

        dm_regime = _resolve_dm_by_regime(dm_rows, variant=variant)
        for regime_label, regime_group in merged.groupby("regime_label", sort=True):
            row = {
                "model_variant": variant,
                "regime_label": str(regime_label),
                "n_runs": int(len(regime_group)),
                "mape_pct_enabled": float(np.nanmean(regime_group["ape_enabled"])),
                "mape_pct_forced_off": float(
                    np.nanmean(regime_group["ape_forced_off"])
                ),
                "mae_enabled": float(np.nanmean(regime_group["abs_error_enabled"])),
                "mae_forced_off": float(
                    np.nanmean(regime_group["abs_error_forced_off"])
                ),
                "rmse_enabled": float(
                    np.sqrt(np.nanmean(np.square(regime_group["error_enabled"])))
                ),
                "rmse_forced_off": float(
                    np.sqrt(np.nanmean(np.square(regime_group["error_forced_off"])))
                ),
                "delta_mape_pct_enabled_minus_forced_off": float(
                    np.nanmean(regime_group["ape_enabled"])
                    - np.nanmean(regime_group["ape_forced_off"])
                ),
                "delta_mae_enabled_minus_forced_off": float(
                    np.nanmean(regime_group["abs_error_enabled"])
                    - np.nanmean(regime_group["abs_error_forced_off"])
                ),
                "delta_rmse_enabled_minus_forced_off": float(
                    np.sqrt(np.nanmean(np.square(regime_group["error_enabled"])))
                    - np.sqrt(np.nanmean(np.square(regime_group["error_forced_off"])))
                ),
                "dm_p_value_one_sided_improve": np.nan,
                "dm_p_value_two_sided": np.nan,
                "dm_stat": np.nan,
                "dm_n_obs": np.nan,
            }
            dm_match = dm_regime[dm_regime["regime_label"] == str(regime_label)]
            if not dm_match.empty:
                dm_row = dm_match.iloc[0]
                row["dm_p_value_one_sided_improve"] = float(
                    dm_row["dm_p_value_one_sided_improve"]
                )
                row["dm_p_value_two_sided"] = float(dm_row["dm_p_value_two_sided"])
                row["dm_stat"] = float(dm_row["dm_stat"])
                row["dm_n_obs"] = int(dm_row["n_obs"])
            regime_rows.append(row)

    overall = (
        pd.DataFrame(overall_rows).sort_values("model_variant").reset_index(drop=True)
    )
    by_regime = (
        pd.DataFrame(regime_rows)
        .sort_values(["model_variant", "regime_label"])
        .reset_index(drop=True)
    )

    out = Path(out_root)
    out.mkdir(parents=True, exist_ok=True)
    overall_path = out / "bsts_marginal_benefit_overall.csv"
    by_regime_path = out / "bsts_marginal_benefit_by_regime.csv"
    overall.to_csv(overall_path, index=False)
    by_regime.to_csv(by_regime_path, index=False)

    summary_path = out / "bsts_marginal_benefit_summary.json"
    summary = {
        "enabled_root": str(Path(enabled_root)),
        "forced_off_root": str(Path(forced_off_root)),
        "common_variants": common_variants,
        "overall_path": str(overall_path),
        "by_regime_path": str(by_regime_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "overall_path": str(overall_path),
        "by_regime_path": str(by_regime_path),
        "summary_path": str(summary_path),
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--enabled_root", required=True, help="Report root with BSTS enabled."
    )
    parser.add_argument(
        "--forced_off_root",
        required=True,
        help="Report root with forced-off BSTS constraints.",
    )
    parser.add_argument(
        "--out_root", required=True, help="Output root for marginal tables."
    )
    return parser


def main() -> int:
    args = _build_parser().parse_args()
    result = compute_bsts_marginal_benefit(
        enabled_root=args.enabled_root,
        forced_off_root=args.forced_off_root,
        out_root=args.out_root,
    )
    print(json.dumps(result, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
