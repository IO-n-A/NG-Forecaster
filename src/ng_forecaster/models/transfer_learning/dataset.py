"""Dataset builders for basin transfer-learning priors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ng_forecaster.errors import ContractViolation

TARGET_COLUMN = "new_well_gas_production"
DRIVER_FEATURE_COLUMNS = (
    "active_rigs",
    "duc_inventory",
    "new_wells_drilled",
    "new_wells_completed",
    "existing_gas_production_change",
    "new_wells_per_rig",
    "month_sin",
    "month_cos",
)

_METRIC_PREFIX_MAP = {
    "RIGS": "active_rigs",
    "DUCS": "duc_inventory",
    "NWD": "new_wells_drilled",
    "NWC": "new_wells_completed",
    "NGEOP": "existing_gas_production_change",
    "NWR": "new_wells_per_rig",
    "NGNW": TARGET_COLUMN,
}


@dataclass(frozen=True)
class BasinTransferDataset:
    """Transfer-learning dataset for one target basin/horizon pair."""

    basin_id: str
    horizon: int
    feature_columns: tuple[str, ...]
    target_column: str
    source_x: np.ndarray
    source_y: np.ndarray
    target_train_x: np.ndarray
    target_train_y: np.ndarray
    target_eval_x: np.ndarray
    target_eval_y: np.ndarray
    prediction_x: np.ndarray
    prediction_timestamp: pd.Timestamp
    source_row_count: int
    target_train_row_count: int
    target_eval_row_count: int
    vintage_month: pd.Timestamp


@dataclass(frozen=True)
class TransferDatasetBundle:
    """Full bundle for all basin/horizon transfer-learning datasets."""

    asof: pd.Timestamp
    vintage_month: pd.Timestamp
    panel: pd.DataFrame
    datasets: list[BasinTransferDataset]
    feature_columns: tuple[str, ...]
    target_column: str


def _parse_vintage_month(token: str) -> pd.Timestamp:
    parsed = pd.to_datetime(token, errors="coerce")
    if pd.isna(parsed):
        raise ContractViolation(
            "invalid_timestamp",
            key=token,
            detail="vintage month token could not be parsed",
        )
    return pd.Timestamp(parsed).to_period("M").to_timestamp("M")


def _resolve_latest_table_10a(
    *,
    asof: pd.Timestamp,
    silver_root: Path,
) -> tuple[pd.DataFrame, pd.Timestamp]:
    candidates: list[tuple[pd.Timestamp, Path]] = []
    for path in sorted(silver_root.glob("vintage_month=*/table_10a.parquet")):
        token = path.parent.name.split("=", 1)[-1]
        vintage = _parse_vintage_month(token)
        if vintage <= asof.to_period("M").to_timestamp("M"):
            candidates.append((vintage, path))

    if not candidates:
        raise ContractViolation(
            "missing_source_file",
            key=str(silver_root),
            detail="no policy-eligible table_10a parquet exists at or before asof",
        )

    vintage_month, table_path = candidates[-1]
    table = pd.read_parquet(table_path)
    if table.empty:
        raise ContractViolation(
            "missing_source_file",
            key=str(table_path),
            detail="latest policy-eligible table_10a parquet is empty",
        )
    return table, vintage_month


def _region_to_basin_id(description: object) -> str:
    value = str(description or "").strip().lower()
    if not value:
        return "unknown"
    cleaned = (
        value.replace("region", "")
        .replace("formation", "")
        .replace(",", " ")
        .replace("-", " ")
    )
    tokens = [token for token in cleaned.split() if token]
    if not tokens:
        return "unknown"
    return "_".join(tokens)


def _metric_for_series_id(series_id: object) -> str | None:
    token = str(series_id or "").strip().upper()
    for prefix, metric in _METRIC_PREFIX_MAP.items():
        if token.startswith(prefix):
            return metric
    return None


def _require_columns(frame: pd.DataFrame, columns: Iterable[str], *, key: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ContractViolation(
            "missing_column",
            key=key,
            detail="missing columns: " + ", ".join(missing),
        )


def build_basin_driver_panel(
    *,
    asof: object,
    silver_root: str | Path = "data/silver/steo_vintages",
    min_history_months: int = 36,
    max_history_months: int = 120,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build a basin/month driver panel from the latest eligible STEO table 10a."""

    asof_ts = pd.Timestamp(asof)
    if pd.isna(asof_ts):
        raise ContractViolation(
            "invalid_timestamp",
            key="asof",
            detail="asof could not be parsed",
        )
    if min_history_months < 12:
        raise ContractViolation(
            "invalid_model_policy",
            key="min_history_months",
            detail="min_history_months must be >= 12",
        )
    if max_history_months < min_history_months:
        raise ContractViolation(
            "invalid_model_policy",
            key="max_history_months",
            detail="max_history_months must be >= min_history_months",
        )

    root = Path(silver_root)
    table_10a, vintage_month = _resolve_latest_table_10a(asof=asof_ts, silver_root=root)
    _require_columns(
        table_10a,
        ("series_id", "description", "timestamp", "value", "is_forecast"),
        key="table_10a",
    )

    table = table_10a.copy()
    table["timestamp"] = pd.to_datetime(table["timestamp"], errors="coerce")
    table["value"] = pd.to_numeric(table["value"], errors="coerce")
    table["available_timestamp"] = pd.to_datetime(
        table.get("available_timestamp"), errors="coerce"
    )
    table["is_forecast"] = table["is_forecast"].astype(bool)
    table["metric"] = table["series_id"].map(_metric_for_series_id)
    table["basin_id"] = table["description"].map(_region_to_basin_id)

    table = table[
        table["metric"].notna()
        & table["timestamp"].notna()
        & table["value"].notna()
        & (~table["is_forecast"])
    ].copy()
    if table.empty:
        raise ContractViolation(
            "insufficient_training_data",
            key="table_10a",
            detail="no observed driver rows remained after filtering",
        )

    if table["available_timestamp"].isna().any():
        table["available_timestamp"] = pd.Timestamp(vintage_month)

    table = table[table["available_timestamp"] <= asof_ts].copy()
    if table.empty:
        raise ContractViolation(
            "lag_policy_violated",
            asof=asof_ts.to_pydatetime(),
            key="table_10a.available_timestamp",
            detail="no table_10a rows are policy-eligible at asof",
        )

    panel = (
        table.groupby(["basin_id", "timestamp", "metric"], as_index=False)["value"]
        .sum()
        .pivot_table(
            index=["basin_id", "timestamp"],
            columns="metric",
            values="value",
            aggfunc="last",
        )
        .reset_index()
    )
    availability = (
        table.groupby(["basin_id", "timestamp"], as_index=False)["available_timestamp"]
        .max()
        .rename(columns={"available_timestamp": "available_timestamp"})
    )
    panel = panel.merge(availability, on=["basin_id", "timestamp"], how="left")
    panel["month_sin"] = np.sin(2.0 * np.pi * panel["timestamp"].dt.month / 12.0)
    panel["month_cos"] = np.cos(2.0 * np.pi * panel["timestamp"].dt.month / 12.0)
    panel = panel.sort_values(["basin_id", "timestamp"]).reset_index(drop=True)

    required = list(DRIVER_FEATURE_COLUMNS) + [TARGET_COLUMN]
    panel = panel.dropna(subset=required).reset_index(drop=True)
    if panel.empty:
        raise ContractViolation(
            "insufficient_training_data",
            key="basin_driver_panel",
            detail="no basin rows contain complete driver+target columns",
        )

    trimmed_chunks: list[pd.DataFrame] = []
    for basin_id, group in panel.groupby("basin_id", sort=True):
        tail = group.sort_values("timestamp").tail(max_history_months).copy()
        if len(tail) < min_history_months:
            continue
        trimmed_chunks.append(tail)

    if not trimmed_chunks:
        raise ContractViolation(
            "insufficient_training_data",
            key="basin_driver_panel",
            detail=(
                "no basin has sufficient history after filtering; "
                f"required={min_history_months}"
            ),
        )
    panel = (
        pd.concat(trimmed_chunks, ignore_index=True)
        .sort_values(["basin_id", "timestamp"])
        .reset_index(drop=True)
    )
    basin_count = int(panel["basin_id"].nunique())
    if basin_count < 2:
        raise ContractViolation(
            "insufficient_training_data",
            key="basin_driver_panel",
            detail="transfer learning requires at least two basins",
        )

    metadata = {
        "asof": asof_ts.date().isoformat(),
        "vintage_month": vintage_month.date().isoformat(),
        "row_count": int(len(panel)),
        "basin_count": basin_count,
        "source_root": str(root),
    }
    return panel, metadata


def _build_supervised_rows(
    frame: pd.DataFrame,
    *,
    horizon: int,
    feature_columns: Sequence[str],
    target_column: str,
) -> pd.DataFrame:
    supervised = frame.sort_values("timestamp").copy()
    supervised["target_future"] = supervised[target_column].shift(-int(horizon))
    supervised = supervised.dropna(subset=[*feature_columns, "target_future"]).copy()
    supervised = supervised.reset_index(drop=True)
    return supervised


def _split_target_train_eval(
    supervised: pd.DataFrame,
    *,
    eval_window_months: int,
    min_target_train_rows: int,
    key: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if eval_window_months < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="eval_window_months",
            detail="eval_window_months must be >= 1",
        )
    if len(supervised) <= min_target_train_rows:
        raise ContractViolation(
            "insufficient_training_data",
            key=key,
            detail=(
                f"target supervised rows={len(supervised)} are below required "
                f"min_target_train_rows={min_target_train_rows + 1}"
            ),
        )

    eval_rows = min(max(1, eval_window_months), max(1, len(supervised) // 3))
    train_rows = len(supervised) - eval_rows
    if train_rows < min_target_train_rows:
        eval_rows = max(1, len(supervised) - min_target_train_rows)
        train_rows = len(supervised) - eval_rows
    if train_rows < min_target_train_rows:
        raise ContractViolation(
            "insufficient_training_data",
            key=key,
            detail=(
                f"target train rows={train_rows} are below min_target_train_rows="
                f"{min_target_train_rows}"
            ),
        )
    return supervised.iloc[:train_rows].copy(), supervised.iloc[train_rows:].copy()


def build_transfer_datasets(
    *,
    asof: object,
    horizons: Sequence[int],
    silver_root: str | Path = "data/silver/steo_vintages",
    min_history_months: int = 36,
    max_history_months: int = 120,
    eval_window_months: int = 6,
    min_source_rows: int = 60,
    min_target_train_rows: int = 24,
) -> TransferDatasetBundle:
    """Build source/target transfer datasets for all basins and horizons."""

    horizon_values = sorted({int(value) for value in horizons})
    if not horizon_values or horizon_values[0] < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="horizons",
            detail="horizons must include positive integers",
        )

    panel, metadata = build_basin_driver_panel(
        asof=asof,
        silver_root=silver_root,
        min_history_months=min_history_months,
        max_history_months=max_history_months,
    )
    asof_ts = pd.Timestamp(asof)
    basins = sorted(panel["basin_id"].unique().tolist())
    feature_columns = tuple(DRIVER_FEATURE_COLUMNS)

    datasets: list[BasinTransferDataset] = []
    for horizon in horizon_values:
        basin_supervised: dict[str, pd.DataFrame] = {}
        for basin_id in basins:
            basin_frame = panel[panel["basin_id"] == basin_id].copy()
            basin_supervised[basin_id] = _build_supervised_rows(
                basin_frame,
                horizon=horizon,
                feature_columns=feature_columns,
                target_column=TARGET_COLUMN,
            )

        for basin_id in basins:
            target_supervised = basin_supervised[basin_id]
            target_train, target_eval = _split_target_train_eval(
                target_supervised,
                eval_window_months=eval_window_months,
                min_target_train_rows=min_target_train_rows,
                key=f"{basin_id}:h{horizon}",
            )

            source_chunks = [
                basin_supervised[other_basin]
                for other_basin in basins
                if other_basin != basin_id
            ]
            source_frame = (
                pd.concat(source_chunks, ignore_index=True)
                .sort_values("timestamp")
                .reset_index(drop=True)
            )
            if len(source_frame) < min_source_rows:
                raise ContractViolation(
                    "insufficient_training_data",
                    key=f"{basin_id}:h{horizon}",
                    detail=(
                        f"source rows={len(source_frame)} are below "
                        f"min_source_rows={min_source_rows}"
                    ),
                )

            basin_history = panel[panel["basin_id"] == basin_id].sort_values(
                "timestamp"
            )
            latest_row = basin_history.iloc[-1]
            prediction_x = (
                latest_row.loc[list(feature_columns)]
                .to_numpy(dtype=float)
                .reshape(1, -1)
            )
            prediction_timestamp = (
                (pd.Timestamp(latest_row["timestamp"]) + pd.DateOffset(months=horizon))
                .to_period("M")
                .to_timestamp("M")
            )

            datasets.append(
                BasinTransferDataset(
                    basin_id=basin_id,
                    horizon=horizon,
                    feature_columns=feature_columns,
                    target_column=TARGET_COLUMN,
                    source_x=source_frame.loc[:, list(feature_columns)].to_numpy(
                        dtype=float
                    ),
                    source_y=source_frame["target_future"].to_numpy(dtype=float),
                    target_train_x=target_train.loc[:, list(feature_columns)].to_numpy(
                        dtype=float
                    ),
                    target_train_y=target_train["target_future"].to_numpy(dtype=float),
                    target_eval_x=target_eval.loc[:, list(feature_columns)].to_numpy(
                        dtype=float
                    ),
                    target_eval_y=target_eval["target_future"].to_numpy(dtype=float),
                    prediction_x=prediction_x,
                    prediction_timestamp=prediction_timestamp,
                    source_row_count=int(len(source_frame)),
                    target_train_row_count=int(len(target_train)),
                    target_eval_row_count=int(len(target_eval)),
                    vintage_month=pd.Timestamp(metadata["vintage_month"]),
                )
            )

    return TransferDatasetBundle(
        asof=asof_ts,
        vintage_month=pd.Timestamp(metadata["vintage_month"]),
        panel=panel,
        datasets=datasets,
        feature_columns=feature_columns,
        target_column=TARGET_COLUMN,
    )
