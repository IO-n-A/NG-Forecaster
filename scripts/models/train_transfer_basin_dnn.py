#!/usr/bin/env python3
"""Train basin transfer-learning priors and publish gold transfer panel."""

# ruff: noqa: E402

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ng_forecaster.features.transfer_priors import (  # noqa: E402
    build_transfer_lineage_id,
    upsert_transfer_priors_panel,
)
from ng_forecaster.models.transfer_learning import (  # noqa: E402
    TLTrainConfig,
    build_transfer_datasets,
    fit_source_encoder,
    fit_target_head,
    predict_with_transfer,
    save_encoder_state,
    save_head_state,
)
from ng_forecaster.orchestration.airflow.workflow_support import (  # noqa: E402
    resolve_release_policy_context,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--asof",
        required=True,
        help="As-of date for policy-safe transfer priors (YYYY-MM-DD).",
    )
    parser.add_argument(
        "--horizons",
        nargs="+",
        type=int,
        default=[1, 2],
        help="Forecast horizons in months ahead from target month.",
    )
    parser.add_argument(
        "--silver-root",
        default="data/silver/steo_vintages",
        help="Root directory with STEO silver vintages.",
    )
    parser.add_argument(
        "--gold-root",
        default="data/gold",
        help="Gold data root for transfer priors panel publishing.",
    )
    parser.add_argument(
        "--artifact-root",
        default="data/artifacts/models",
        help="Output directory for transfer model artifacts.",
    )
    parser.add_argument("--hidden-units", type=int, default=24)
    parser.add_argument("--learning-rate", type=float, default=0.01)
    parser.add_argument("--dropout-rate", type=float, default=0.10)
    parser.add_argument("--l2-penalty", type=float, default=1e-4)
    parser.add_argument("--max-epochs", type=int, default=600)
    parser.add_argument("--patience", type=int, default=40)
    parser.add_argument("--min-delta", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--min-history-months", type=int, default=36)
    parser.add_argument("--max-history-months", type=int, default=120)
    parser.add_argument("--eval-window-months", type=int, default=6)
    parser.add_argument("--min-source-rows", type=int, default=60)
    parser.add_argument("--min-target-train-rows", type=int, default=24)
    return parser


def _train_config_from_args(args: argparse.Namespace) -> TLTrainConfig:
    return TLTrainConfig(
        hidden_units=int(args.hidden_units),
        learning_rate=float(args.learning_rate),
        dropout_rate=float(args.dropout_rate),
        l2_penalty=float(args.l2_penalty),
        max_epochs=int(args.max_epochs),
        patience=int(args.patience),
        min_delta=float(args.min_delta),
        random_seed=int(args.seed),
    )


def _stable_seed(base_seed: int, *, basin_id: str, horizon: int) -> int:
    value = sum(ord(char) for char in basin_id) + int(horizon) * 997
    return int(base_seed + value)


def _build_transfer_panel_rows(
    *,
    asof: pd.Timestamp,
    release_target_month: pd.Timestamp,
    basin_predictions: pd.DataFrame,
    lineage_id: str,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    grouped = basin_predictions.groupby("horizon", sort=True)
    for horizon, group in grouped:
        horizon_int = int(horizon)
        target_month = (
            (release_target_month + pd.DateOffset(months=horizon_int - 1))
            .to_period("M")
            .to_timestamp("M")
        )
        rows.append(
            {
                "asof": asof.to_period("M").to_timestamp("M"),
                "target_month": target_month,
                "horizon": horizon_int,
                "transfer_prior_us_bcfd": float(group["prediction"].sum()),
                "transfer_prior_dispersion": float(group["prediction"].std(ddof=0)),
                "transfer_prior_basin_count": int(group["basin_id"].nunique()),
                "available_timestamp": asof.normalize(),
                "lineage_id": lineage_id,
                "source_model": "tl_basin_dnn",
            }
        )
    return pd.DataFrame(rows)


def main() -> int:
    args = _build_parser().parse_args()
    asof_ts = pd.Timestamp(args.asof)
    if pd.isna(asof_ts):
        raise ValueError("--asof must be parseable as a date")

    cfg = _train_config_from_args(args)
    bundle = build_transfer_datasets(
        asof=asof_ts,
        horizons=[int(h) for h in args.horizons],
        silver_root=args.silver_root,
        min_history_months=int(args.min_history_months),
        max_history_months=int(args.max_history_months),
        eval_window_months=int(args.eval_window_months),
        min_source_rows=int(args.min_source_rows),
        min_target_train_rows=int(args.min_target_train_rows),
    )
    policy_context = resolve_release_policy_context(asof_ts)
    release_target_month = (
        pd.Timestamp(policy_context["target_month"]).to_period("M").to_timestamp("M")
    )

    artifact_root = Path(args.artifact_root)
    encoder_root = artifact_root
    head_root = artifact_root / "tl_basin_heads"
    artifact_root.mkdir(parents=True, exist_ok=True)
    head_root.mkdir(parents=True, exist_ok=True)

    model_rows: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for dataset in bundle.datasets:
        seed = _stable_seed(
            cfg.random_seed, basin_id=dataset.basin_id, horizon=dataset.horizon
        )
        source_result = fit_source_encoder(
            dataset.source_x,
            dataset.source_y,
            config=cfg.__dict__,
            random_seed=seed,
        )
        target_result = fit_target_head(
            source_result.encoder,
            target_train_x=dataset.target_train_x,
            target_train_y=dataset.target_train_y,
            target_eval_x=dataset.target_eval_x,
            target_eval_y=dataset.target_eval_y,
            config=cfg.__dict__,
            random_seed=seed + 17,
        )
        prediction = float(
            predict_with_transfer(
                source_result.encoder,
                target_result.head,
                dataset.prediction_x,
            )[0]
        )
        prediction = max(prediction, 0.0)

        encoder_path = encoder_root / (
            f"tl_basin_encoder.{dataset.basin_id}.h{dataset.horizon}.npz"
        )
        head_path = head_root / f"{dataset.basin_id}.h{dataset.horizon}.npz"
        save_encoder_state(encoder_path, source_result.encoder)
        save_head_state(head_path, target_result.head)

        model_rows.append(
            {
                "basin_id": dataset.basin_id,
                "horizon": int(dataset.horizon),
                "encoder_path": str(encoder_path),
                "head_path": str(head_path),
                "source_rows": int(dataset.source_row_count),
                "target_train_rows": int(dataset.target_train_row_count),
                "target_eval_rows": int(dataset.target_eval_row_count),
                "source_train_rmse": float(source_result.train_rmse),
                "source_val_rmse": float(source_result.val_rmse),
                "target_train_rmse": float(target_result.train_rmse),
                "target_eval_rmse": float(target_result.eval_rmse),
                "prediction_timestamp": dataset.prediction_timestamp.date().isoformat(),
                "prediction_value": prediction,
            }
        )
        prediction_rows.append(
            {
                "basin_id": dataset.basin_id,
                "horizon": int(dataset.horizon),
                "prediction": prediction,
            }
        )

    predictions = pd.DataFrame(prediction_rows)
    lineage_id = build_transfer_lineage_id(
        {
            "asof": asof_ts.date().isoformat(),
            "horizons": sorted(
                {int(value) for value in predictions["horizon"].tolist()}
            ),
            "model_rows": model_rows,
            "vintage_month": bundle.vintage_month.date().isoformat(),
        }
    )
    panel_rows = _build_transfer_panel_rows(
        asof=asof_ts,
        release_target_month=release_target_month,
        basin_predictions=predictions,
        lineage_id=lineage_id,
    )
    panel_path = upsert_transfer_priors_panel(panel_rows, gold_root=args.gold_root)

    manifest_payload = {
        "asof": asof_ts.date().isoformat(),
        "release_target_month": release_target_month.date().isoformat(),
        "horizons": sorted({int(value) for value in predictions["horizon"].tolist()}),
        "config": cfg.__dict__,
        "source_panel_vintage_month": bundle.vintage_month.date().isoformat(),
        "source_panel_rows": int(len(bundle.panel)),
        "source_panel_basins": sorted(bundle.panel["basin_id"].unique().tolist()),
        "models": model_rows,
        "transfer_priors_panel_path": str(panel_path),
        "lineage_id": lineage_id,
    }
    manifest_path = artifact_root / "tl_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest_payload, indent=2, sort_keys=True),
        encoding="utf-8",
    )

    print(
        json.dumps(
            {
                "asof": asof_ts.date().isoformat(),
                "model_count": len(model_rows),
                "lineage_id": lineage_id,
                "manifest_path": str(manifest_path),
                "transfer_priors_panel_path": str(panel_path),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
