"""Transfer-prior panel publishing and runtime feature emission."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from ng_forecaster.errors import ContractViolation

_PANEL_FILE = "transfer_priors_panel.parquet"
_REQUIRED_PANEL_COLUMNS = {
    "asof",
    "target_month",
    "horizon",
    "transfer_prior_us_bcfd",
    "transfer_prior_dispersion",
    "transfer_prior_basin_count",
    "available_timestamp",
    "lineage_id",
    "source_model",
}


def _empty_feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        columns=[
            "feature_name",
            "feature_timestamp",
            "available_timestamp",
            "block_id",
            "value",
        ]
    )


def _ensure_columns(frame: pd.DataFrame, columns: Iterable[str], *, key: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ContractViolation(
            "missing_column",
            key=key,
            detail="missing columns: " + ", ".join(missing),
        )


def _normalize_panel(frame: pd.DataFrame) -> pd.DataFrame:
    _ensure_columns(frame, _REQUIRED_PANEL_COLUMNS, key="transfer_priors_panel")
    panel = frame.copy()
    panel["asof"] = pd.to_datetime(panel["asof"], errors="coerce")
    panel["target_month"] = (
        pd.to_datetime(panel["target_month"], errors="coerce")
        .dt.to_period("M")
        .dt.to_timestamp("M")
    )
    panel["available_timestamp"] = pd.to_datetime(
        panel["available_timestamp"], errors="coerce"
    )
    panel["horizon"] = pd.to_numeric(panel["horizon"], errors="coerce").astype("Int64")
    panel["transfer_prior_us_bcfd"] = pd.to_numeric(
        panel["transfer_prior_us_bcfd"], errors="coerce"
    )
    panel["transfer_prior_dispersion"] = pd.to_numeric(
        panel["transfer_prior_dispersion"], errors="coerce"
    )
    panel["transfer_prior_basin_count"] = pd.to_numeric(
        panel["transfer_prior_basin_count"], errors="coerce"
    ).astype("Int64")
    invalid = panel[
        panel["asof"].isna()
        | panel["target_month"].isna()
        | panel["available_timestamp"].isna()
        | panel["horizon"].isna()
        | panel["transfer_prior_us_bcfd"].isna()
        | panel["transfer_prior_dispersion"].isna()
        | panel["transfer_prior_basin_count"].isna()
        | (panel["lineage_id"].astype(str).str.strip() == "")
    ]
    if not invalid.empty:
        sample = invalid.iloc[0]
        raise ContractViolation(
            "source_schema_drift",
            key="transfer_priors_panel",
            detail=(
                "invalid transfer-prior row encountered: "
                f"asof={sample.get('asof')} target_month={sample.get('target_month')}"
            ),
        )
    return panel


def load_transfer_priors_panel(
    *,
    gold_root: str | Path = "data/gold",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load transfer priors panel from gold storage."""

    panel_path = Path(gold_root) / _PANEL_FILE
    if not panel_path.exists():
        return (
            pd.DataFrame(columns=sorted(_REQUIRED_PANEL_COLUMNS)),
            {
                "status": "missing_transfer_priors_panel",
                "panel_path": str(panel_path),
            },
        )
    panel = _normalize_panel(pd.read_parquet(panel_path))
    panel = panel.sort_values(["asof", "target_month", "horizon"]).reset_index(
        drop=True
    )
    return (
        panel,
        {
            "status": "transfer_priors_loaded",
            "panel_path": str(panel_path),
            "row_count": int(len(panel)),
            "latest_asof": (
                panel["asof"].max().date().isoformat() if not panel.empty else ""
            ),
        },
    )


def upsert_transfer_priors_panel(
    panel_rows: pd.DataFrame,
    *,
    gold_root: str | Path = "data/gold",
) -> Path:
    """Append/replace transfer prior rows by `(asof,target_month,horizon)` keys."""

    normalized = _normalize_panel(panel_rows)
    panel_path = Path(gold_root) / _PANEL_FILE
    panel_path.parent.mkdir(parents=True, exist_ok=True)

    existing = pd.DataFrame(columns=sorted(_REQUIRED_PANEL_COLUMNS))
    if panel_path.exists():
        existing = _normalize_panel(pd.read_parquet(panel_path))

    if existing.empty:
        merged = normalized.copy()
    else:
        merged = pd.concat([existing, normalized], ignore_index=True)
    merged = merged.sort_values(
        ["asof", "target_month", "horizon", "available_timestamp"]
    )
    merged = merged.drop_duplicates(["asof", "target_month", "horizon"], keep="last")
    merged = merged.reset_index(drop=True)
    merged.to_parquet(panel_path, index=False)
    return panel_path


def _suffix_for_horizon(horizon: int) -> str:
    if horizon == 1:
        return "t"
    if horizon == 2:
        return "t_plus_1"
    return f"t_plus_{horizon - 1}"


def build_transfer_prior_feature_rows(
    *,
    asof: object,
    target_month: object,
    gold_root: str | Path = "data/gold",
    required_horizons: tuple[int, ...] = (1, 2),
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build runtime feature rows from latest transfer-prior panel at or before asof."""

    asof_ts = pd.Timestamp(asof)
    target_month_ts = pd.Timestamp(target_month).to_period("M").to_timestamp("M")
    if pd.isna(asof_ts) or pd.isna(target_month_ts):
        raise ContractViolation(
            "invalid_timestamp",
            key="transfer_priors",
            detail="asof/target_month could not be parsed",
        )

    panel, meta = load_transfer_priors_panel(gold_root=gold_root)
    if panel.empty:
        return _empty_feature_frame(), {
            **meta,
            "status": "transfer_priors_not_ready",
            "asof": asof_ts.date().isoformat(),
        }

    eligible = panel[panel["available_timestamp"] <= asof_ts].copy()
    if eligible.empty:
        return _empty_feature_frame(), {
            **meta,
            "status": "transfer_priors_not_ready",
            "asof": asof_ts.date().isoformat(),
        }

    latest_asof = eligible["asof"].max()
    current = eligible[eligible["asof"] == latest_asof].copy()
    if current.empty:
        return _empty_feature_frame(), {
            **meta,
            "status": "transfer_priors_not_ready",
            "asof": asof_ts.date().isoformat(),
        }

    rows: list[dict[str, Any]] = []
    for horizon in required_horizons:
        horizon_int = int(horizon)
        expected_target = (
            (target_month_ts + pd.DateOffset(months=horizon_int - 1))
            .to_period("M")
            .to_timestamp("M")
        )
        match = current[
            (current["horizon"] == horizon_int)
            & (current["target_month"] == expected_target)
        ]
        if match.empty:
            raise ContractViolation(
                "missing_source_file",
                asof=asof_ts.to_pydatetime(),
                key="transfer_priors_panel",
                detail=(
                    "missing transfer-prior row for "
                    f"horizon={horizon_int} target_month={expected_target.date().isoformat()}"
                ),
            )
        row = match.sort_values("available_timestamp").iloc[-1]
        suffix = _suffix_for_horizon(horizon_int)
        feature_timestamp = pd.Timestamp(row["target_month"])
        available_timestamp = pd.Timestamp(row["available_timestamp"])

        rows.extend(
            [
                {
                    "feature_name": f"transfer_prior_us_bcfd_{suffix}",
                    "feature_timestamp": feature_timestamp,
                    "available_timestamp": available_timestamp,
                    "block_id": "transfer_priors",
                    "value": float(row["transfer_prior_us_bcfd"]),
                },
                {
                    "feature_name": f"transfer_prior_dispersion_{suffix}",
                    "feature_timestamp": feature_timestamp,
                    "available_timestamp": available_timestamp,
                    "block_id": "transfer_priors",
                    "value": float(row["transfer_prior_dispersion"]),
                },
                {
                    "feature_name": f"transfer_prior_basin_count_{suffix}",
                    "feature_timestamp": feature_timestamp,
                    "available_timestamp": available_timestamp,
                    "block_id": "transfer_priors",
                    "value": float(row["transfer_prior_basin_count"]),
                },
            ]
        )

    feature_rows = pd.DataFrame(rows).sort_values("feature_name").reset_index(drop=True)
    return feature_rows, {
        **meta,
        "status": "transfer_prior_features_loaded",
        "feature_row_count": int(len(feature_rows)),
        "selected_asof": pd.Timestamp(latest_asof).date().isoformat(),
    }


def build_transfer_lineage_id(payload: dict[str, Any]) -> str:
    """Build deterministic lineage hash for transfer-prior artifacts."""

    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return digest
