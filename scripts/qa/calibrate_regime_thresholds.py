#!/usr/bin/env python3
"""Calibrate regime thresholds from historical feature-matrix artifacts."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.regime_flags import DEFAULT_REGIME_THRESHOLDS


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--feature-matrix-glob",
        default="data/artifacts/nowcast/*/feature_matrix.csv",
    )
    parser.add_argument(
        "--horizon",
        default="T",
        help="Feature-matrix horizon row used for calibration (default: T).",
    )
    parser.add_argument(
        "--quantile",
        type=float,
        default=0.85,
        help="Quantile used for threshold calibration (0,1).",
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=12,
        help="Minimum required sample rows across matched feature matrices.",
    )
    parser.add_argument(
        "--output-csv",
        default="data/reports/regime_threshold_calibration.csv",
    )
    parser.add_argument(
        "--output-json",
        default="data/reports/regime_thresholds_calibrated.json",
    )
    parser.add_argument(
        "--features-config",
        default="configs/features.yaml",
    )
    parser.add_argument(
        "--write-config",
        type=int,
        default=0,
        help="Set to 1 to update `regime_thresholds` in features config.",
    )
    return parser


def _pick_numeric(row: pd.Series, keys: tuple[str, ...]) -> float:
    for key in keys:
        if key not in row.index:
            continue
        value = pd.to_numeric(pd.Series([row[key]]), errors="coerce").iloc[0]
        if pd.isna(value):
            continue
        return float(value)
    return 0.0


def _collect_rows(path_glob: str, *, horizon: str) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for path in sorted(Path().glob(path_glob)):
        frame = pd.read_csv(path)
        if frame.empty:
            continue
        candidate = frame[frame["horizon"].astype(str) == str(horizon)]
        row = candidate.iloc[0] if not candidate.empty else frame.iloc[0]
        basis_proxy = _pick_numeric(
            row,
            (
                "regime_basis_spread_proxy",
                "steo_regional_residential_spread_usd_mcf_t",
                "steo_regional_commercial_spread_usd_mcf_t",
                "steo_regional_industrial_spread_usd_mcf_t",
            ),
        )
        transfer_proxy = _pick_numeric(
            row,
            (
                "regime_transfer_dispersion_proxy",
                "transfer_prior_dispersion_t",
                "transfer_prior_dispersion_t_plus_1",
            ),
        )
        rows.append(
            {
                "artifact_path": str(path),
                "asof": str(path.parent.name),
                "freeze_days_high": _pick_numeric(
                    row,
                    ("freeze_days_mtd_weighted", "freeze_days_mtd"),
                ),
                "freeze_intensity_high": _pick_numeric(
                    row,
                    ("freeze_intensity_mtd_weighted", "freeze_event_intensity"),
                ),
                "freeze_event_share_high": _pick_numeric(
                    row,
                    ("freeze_event_share_mtd_weighted", "freeze_event_flag"),
                ),
                "basis_spread_high": abs(float(basis_proxy)),
                "transfer_dispersion_high": abs(float(transfer_proxy)),
            }
        )
    return pd.DataFrame(rows)


def _validate_threshold_value(value: float, *, key: str) -> float:
    if not pd.notna(value) or float(value) <= 0:
        default = float(DEFAULT_REGIME_THRESHOLDS[key])
        return default
    return float(value)


def _build_thresholds(
    samples: pd.DataFrame,
    *,
    quantile: float,
) -> tuple[dict[str, float], pd.DataFrame]:
    thresholds: dict[str, float] = {}
    calibration_rows: list[dict[str, Any]] = []
    for key, default in DEFAULT_REGIME_THRESHOLDS.items():
        series = pd.to_numeric(samples[key], errors="coerce").dropna().astype(float)
        if series.empty:
            value = float(default)
            n_obs = 0
            non_zero = 0
            min_value = float("nan")
            median_value = float("nan")
            max_value = float("nan")
        else:
            raw = float(series.quantile(quantile))
            value = _validate_threshold_value(raw, key=key)
            n_obs = int(len(series))
            non_zero = int((series.abs() > 1e-12).sum())
            min_value = float(series.min())
            median_value = float(series.median())
            max_value = float(series.max())
        thresholds[key] = value
        calibration_rows.append(
            {
                "threshold_key": key,
                "quantile": float(quantile),
                "calibrated_value": float(value),
                "n_obs": int(n_obs),
                "non_zero_obs": int(non_zero),
                "min_value": min_value,
                "median_value": median_value,
                "max_value": max_value,
            }
        )
    return thresholds, pd.DataFrame(calibration_rows)


def _write_features_config(path: Path, *, thresholds: dict[str, float]) -> None:
    payload = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(payload, dict):
        raise ContractViolation(
            "invalid_feature_policy",
            key=str(path),
            detail="features config root must be a mapping",
        )
    payload["regime_thresholds"] = {
        key: float(value) for key, value in thresholds.items()
    }
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )


def main() -> int:
    args = _build_parser().parse_args()
    quantile = float(args.quantile)
    if quantile <= 0 or quantile >= 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="quantile",
            detail="quantile must be in the open interval (0, 1)",
        )

    samples = _collect_rows(args.feature_matrix_glob, horizon=str(args.horizon))
    if samples.empty:
        raise ContractViolation(
            "missing_source_file",
            key=str(args.feature_matrix_glob),
            detail="no feature_matrix.csv files matched calibration glob",
        )
    if len(samples) < int(args.min_samples):
        raise ContractViolation(
            "insufficient_release_history",
            key="regime_threshold_calibration",
            detail=(
                "insufficient calibration samples: "
                f"required={int(args.min_samples)} received={int(len(samples))}"
            ),
        )

    thresholds, calibration = _build_thresholds(samples, quantile=quantile)

    output_csv = Path(args.output_csv)
    output_json = Path(args.output_json)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    calibration.to_csv(output_csv, index=False)

    payload = {
        "quantile": quantile,
        "horizon": str(args.horizon),
        "sample_count": int(len(samples)),
        "thresholds": thresholds,
        "calibration_csv_path": str(output_csv),
        "source_glob": str(args.feature_matrix_glob),
    }
    output_json.write_text(
        json.dumps(payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )

    config_updated = False
    if bool(int(args.write_config)):
        config_path = Path(args.features_config)
        _write_features_config(config_path, thresholds=thresholds)
        config_updated = True

    print(
        json.dumps(
            {
                "status": "regime_thresholds_calibrated",
                "sample_count": int(len(samples)),
                "thresholds": thresholds,
                "calibration_csv_path": str(output_csv),
                "calibration_json_path": str(output_json),
                "features_config_updated": bool(config_updated),
                "features_config_path": str(args.features_config),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
