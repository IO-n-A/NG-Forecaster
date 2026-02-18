"""Baker Hughes rig-count adapter with monthly feature aggregation helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ng_forecaster.errors import ContractViolation


def _load_table(source_path: str | Path) -> pd.DataFrame:
    path = Path(source_path)
    if not path.exists():
        raise ContractViolation(
            "missing_source_file",
            key=str(path),
            detail="Baker Hughes source file does not exist",
        )
    if path.suffix.lower() in {".xls", ".xlsx"}:
        return pd.read_excel(path)
    return pd.read_csv(path)


def _match_column(columns: list[str], *, include: tuple[str, ...]) -> str | None:
    for column in columns:
        lowered = column.lower()
        if all(token in lowered for token in include):
            return column
    return None


def parse_bakerhughes_rig_history(source_path: str | Path) -> pd.DataFrame:
    """Parse Baker Hughes weekly rig history into canonical schema."""

    raw = _load_table(source_path)
    if raw.empty:
        raise ContractViolation(
            "missing_source_file",
            key=str(source_path),
            detail="Baker Hughes source is empty",
        )

    columns = [str(item) for item in raw.columns]
    date_col = _match_column(columns, include=("date",))
    oil_col = _match_column(columns, include=("oil", "rig"))
    gas_col = _match_column(columns, include=("gas", "rig"))
    if date_col is None or oil_col is None or gas_col is None:
        raise ContractViolation(
            "source_schema_drift",
            key=str(source_path),
            detail="unable to resolve date/oil rig/gas rig columns",
        )

    frame = pd.DataFrame(
        {
            "timestamp": pd.to_datetime(raw[date_col], errors="coerce"),
            "oil_rig_count": pd.to_numeric(raw[oil_col], errors="coerce"),
            "gas_rig_count": pd.to_numeric(raw[gas_col], errors="coerce"),
        }
    )
    frame = frame[
        frame["timestamp"].notna()
        & frame["oil_rig_count"].notna()
        & frame["gas_rig_count"].notna()
    ].copy()
    if frame.empty:
        raise ContractViolation(
            "source_schema_drift",
            key=str(source_path),
            detail="no valid Baker Hughes rows were parsed",
        )
    frame["timestamp"] = frame["timestamp"].dt.to_period("W").dt.to_timestamp("W")
    frame["available_timestamp"] = frame["timestamp"]
    frame["source_id"] = "baker_hughes_rig_count"
    frame = frame.sort_values("timestamp").drop_duplicates("timestamp", keep="last")
    return frame.reset_index(drop=True)


def _slope(values: pd.Series) -> float:
    if len(values) < 2:
        return 0.0
    x = np.arange(len(values), dtype=float)
    y = values.to_numpy(dtype=float)
    slope, _ = np.polyfit(x, y, 1)
    return float(slope)


def build_monthly_rig_features(
    history: pd.DataFrame,
    *,
    asof: object,
) -> pd.DataFrame:
    """Aggregate weekly rig history into monthly runtime features."""

    required = {"timestamp", "oil_rig_count", "gas_rig_count"}
    missing = sorted(required - set(history.columns))
    if missing:
        raise ContractViolation(
            "missing_column",
            key="bakerhughes_history",
            detail="missing columns: " + ", ".join(missing),
        )
    asof_ts = pd.Timestamp(asof)
    if pd.isna(asof_ts):
        raise ContractViolation(
            "invalid_timestamp",
            key="asof",
            detail="asof could not be parsed",
        )

    frame = history.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    frame["oil_rig_count"] = pd.to_numeric(frame["oil_rig_count"], errors="coerce")
    frame["gas_rig_count"] = pd.to_numeric(frame["gas_rig_count"], errors="coerce")
    frame = frame[
        frame["timestamp"].notna()
        & frame["oil_rig_count"].notna()
        & frame["gas_rig_count"].notna()
        & (frame["timestamp"] <= asof_ts)
    ].copy()
    if frame.empty:
        raise ContractViolation(
            "missing_feature_input",
            key="bakerhughes_history",
            detail="no Baker Hughes rows are available at or before asof",
        )

    tail = frame.tail(4).reset_index(drop=True)
    feature_ts = asof_ts.to_period("M").to_timestamp("M")
    available_ts = pd.Timestamp(tail["timestamp"].max())
    rows: list[dict[str, Any]] = [
        {
            "feature_name": "oil_rigs_last",
            "feature_timestamp": feature_ts,
            "available_timestamp": available_ts,
            "block_id": "oil_side",
            "value": float(tail.iloc[-1]["oil_rig_count"]),
            "source_frequency": "weekly",
        },
        {
            "feature_name": "oil_rigs_mean_4w",
            "feature_timestamp": feature_ts,
            "available_timestamp": available_ts,
            "block_id": "oil_side",
            "value": float(tail["oil_rig_count"].mean()),
            "source_frequency": "weekly",
        },
        {
            "feature_name": "oil_rigs_slope_4w",
            "feature_timestamp": feature_ts,
            "available_timestamp": available_ts,
            "block_id": "oil_side",
            "value": _slope(tail["oil_rig_count"]),
            "source_frequency": "weekly",
        },
        {
            "feature_name": "gas_rigs_last",
            "feature_timestamp": feature_ts,
            "available_timestamp": available_ts,
            "block_id": "oil_side",
            "value": float(tail.iloc[-1]["gas_rig_count"]),
            "source_frequency": "weekly",
        },
        {
            "feature_name": "gas_rigs_mean_4w",
            "feature_timestamp": feature_ts,
            "available_timestamp": available_ts,
            "block_id": "oil_side",
            "value": float(tail["gas_rig_count"].mean()),
            "source_frequency": "weekly",
        },
        {
            "feature_name": "gas_rigs_slope_4w",
            "feature_timestamp": feature_ts,
            "available_timestamp": available_ts,
            "block_id": "oil_side",
            "value": _slope(tail["gas_rig_count"]),
            "source_frequency": "weekly",
        },
    ]
    return pd.DataFrame(rows).sort_values("feature_name").reset_index(drop=True)

