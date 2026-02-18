"""Gold-layer publishing for STEO marts, weather freeze-off panels, and priors."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ng_forecaster.data.validators import (
    validate_weather_coverage,
    validate_weather_lineage,
)
from ng_forecaster.errors import ContractViolation


def _load_silver_table(silver_root: Path, table_id: str) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for vintage_dir in sorted(
        silver_root.glob("vintage_month=*"), key=lambda item: item.name
    ):
        table_path = vintage_dir / f"{table_id}.parquet"
        if not table_path.exists():
            raise ContractViolation(
                "missing_source_file",
                key=str(table_path),
                detail="required silver table parquet is missing",
            )
        frame = pd.read_parquet(table_path)
        if frame.empty:
            continue
        frame = frame.copy()
        frame["vintage_month"] = str(vintage_dir.name.split("=", 1)[1])
        rows.append(frame)

    if not rows:
        raise ContractViolation(
            "missing_target_history_rows",
            key=table_id,
            detail="no rows available across silver vintages",
        )

    merged = pd.concat(rows, ignore_index=True)
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")
    if "available_timestamp" in merged.columns:
        merged["available_timestamp"] = pd.to_datetime(
            merged["available_timestamp"], errors="coerce"
        )
    else:
        merged["available_timestamp"] = pd.to_datetime(
            merged["vintage_month"], errors="coerce"
        )
    merged["available_timestamp"] = merged["available_timestamp"].fillna(
        pd.to_datetime(merged["vintage_month"], errors="coerce")
    )
    merged["value"] = pd.to_numeric(merged["value"], errors="coerce")
    merged = merged[
        merged["timestamp"].notna()
        & merged["value"].notna()
        & merged["available_timestamp"].notna()
    ].copy()
    merged = merged.sort_values(["vintage_month", "series_id", "timestamp"])
    return merged.reset_index(drop=True)


def _build_observation_panel(table_5a: pd.DataFrame) -> pd.DataFrame:
    required_series = {"NGPRPUS", "NGMPPUS"}
    present = set(table_5a["series_id"].dropna().astype(str).unique().tolist())
    missing = sorted(required_series - present)
    if missing:
        raise ContractViolation(
            "source_schema_drift",
            key="table_5a.series_id",
            detail=f"required STEO observation series are missing: {', '.join(missing)}",
        )

    subset = table_5a[table_5a["series_id"].isin(sorted(required_series))].copy()
    pivot = subset.pivot_table(
        index=["vintage_month", "timestamp", "is_forecast"],
        columns="series_id",
        values="value",
        aggfunc="last",
    ).reset_index()
    availability = (
        subset.groupby(["vintage_month", "timestamp"], as_index=False)[
            "available_timestamp"
        ]
        .max()
        .rename(columns={"available_timestamp": "available_timestamp"})
    )

    pivot = pivot.rename(
        columns={
            "NGPRPUS": "dry_prod_bcfd",
            "NGMPPUS": "marketed_prod_bcfd",
        }
    )
    pivot = pivot.merge(
        availability,
        on=["vintage_month", "timestamp"],
        how="left",
    )
    pivot["dry_to_marketed_ratio"] = (
        pivot["dry_prod_bcfd"] / pivot["marketed_prod_bcfd"]
    )
    pivot["extraction_loss_ratio"] = 1.0 - pivot["dry_to_marketed_ratio"]
    pivot["observation_source"] = "steo_table_5a"

    keep = [
        "vintage_month",
        "timestamp",
        "is_forecast",
        "available_timestamp",
        "dry_prod_bcfd",
        "marketed_prod_bcfd",
        "dry_to_marketed_ratio",
        "extraction_loss_ratio",
        "observation_source",
    ]
    panel = (
        pivot[keep].sort_values(["vintage_month", "timestamp"]).reset_index(drop=True)
    )
    return panel


def _region_from_description(description: object) -> str:
    text = str(description or "").strip()
    if not text:
        return "unknown"
    lowered = text.lower()
    if "region" in lowered:
        return text.replace("region", "").replace("Region", "").strip(" ,-()")
    if "formation" in lowered:
        return text.replace("formation", "").replace("formations", "").strip(" ,-()")
    return text.strip(" ,-()")


def _metric_group(series_id: str) -> str:
    token = str(series_id)
    prefixes = [
        ("CONWR", "existing_oil_well_change"),
        ("COEOP", "existing_oil_productivity"),
        ("NGEOP", "existing_gas_production_change"),
        ("NGNW", "new_well_gas_production"),
        ("DUCS", "duc_inventory"),
        ("RIGS", "active_rigs"),
        ("NWD", "new_wells_drilled"),
        ("NWC", "new_wells_completed"),
        ("NWR", "new_wells_per_rig"),
        ("CONW", "new_oil_well_production"),
    ]
    for prefix, label in prefixes:
        if token.startswith(prefix):
            return label
    return "other"


def _build_driver_panel(table_10a: pd.DataFrame) -> pd.DataFrame:
    table = table_10a.copy()
    table["metric_group"] = table["series_id"].astype(str).map(_metric_group)
    table["region"] = table["description"].map(_region_from_description)

    grouped = (
        table.groupby(["vintage_month", "timestamp", "metric_group"], as_index=False)[
            "value"
        ]
        .sum()
        .rename(columns={"value": "metric_value"})
    )
    pivot = grouped.pivot_table(
        index=["vintage_month", "timestamp"],
        columns="metric_group",
        values="metric_value",
        aggfunc="last",
    ).reset_index()

    observation = table_10a[
        ["vintage_month", "timestamp", "is_forecast", "available_timestamp"]
    ].drop_duplicates(["vintage_month", "timestamp"], keep="last")
    panel = pivot.merge(
        observation,
        on=["vintage_month", "timestamp"],
        how="left",
    )
    if {"new_wells_completed", "duc_inventory"}.issubset(panel.columns):
        denominator = pd.to_numeric(panel["duc_inventory"], errors="coerce").replace(
            0.0, np.nan
        )
        panel["completion_to_duc_ratio"] = (
            pd.to_numeric(panel["new_wells_completed"], errors="coerce") / denominator
        )
    else:
        panel["completion_to_duc_ratio"] = np.nan
    if {"new_well_gas_production", "existing_gas_production_change"}.issubset(
        panel.columns
    ):
        panel["new_well_minus_legacy"] = pd.to_numeric(
            panel["new_well_gas_production"], errors="coerce"
        ) - pd.to_numeric(panel["existing_gas_production_change"], errors="coerce")
    else:
        panel["new_well_minus_legacy"] = np.nan
    panel["driver_source"] = "steo_table_10a"
    panel = panel.sort_values(["vintage_month", "timestamp"]).reset_index(drop=True)
    return panel


def _build_shale_split_panel(
    table_10b: pd.DataFrame,
    observation_panel: pd.DataFrame,
) -> pd.DataFrame:
    shale = table_10b[table_10b["series_id"].astype(str).str.startswith("SNGPR")].copy()
    if shale.empty:
        raise ContractViolation(
            "source_schema_drift",
            key="table_10b.series_id",
            detail="no shale gas series found (expected SNGPR*)",
        )

    total = shale[shale["series_id"] == "SNGPRL48"].copy()
    if total.empty:
        raise ContractViolation(
            "source_schema_drift",
            key="table_10b.series_id",
            detail="missing total shale dry gas series SNGPRL48",
        )

    total = total.rename(columns={"value": "shale_dry_prod_bcfd"})
    others = shale[shale["series_id"] != "SNGPRL48"].copy()
    formation_sum = (
        others.groupby(["vintage_month", "timestamp"], as_index=False)["value"]
        .sum()
        .rename(columns={"value": "shale_component_sum_bcfd"})
    )

    panel = total[
        [
            "vintage_month",
            "timestamp",
            "is_forecast",
            "available_timestamp",
            "shale_dry_prod_bcfd",
        ]
    ].merge(
        formation_sum,
        on=["vintage_month", "timestamp"],
        how="left",
    )
    panel = panel.merge(
        observation_panel[["vintage_month", "timestamp", "dry_prod_bcfd"]],
        on=["vintage_month", "timestamp"],
        how="left",
    )
    panel["non_shale_dry_prod_bcfd"] = (
        panel["dry_prod_bcfd"] - panel["shale_dry_prod_bcfd"]
    )
    panel["shale_share_of_dry"] = panel["shale_dry_prod_bcfd"] / panel["dry_prod_bcfd"]
    panel["shale_source"] = "steo_table_10b"
    panel = panel.sort_values(["vintage_month", "timestamp"]).reset_index(drop=True)
    return panel


def _build_energy_prices_panel(table_2: pd.DataFrame) -> pd.DataFrame:
    series_map = {
        "WTIPUUS": "wti_spot_usd_bbl",
        "BREPUUS": "brent_spot_usd_bbl",
        "NGHHMCF": "hh_spot_usd_mcf",
        "NGHHUUS": "hh_spot_usd_mmbtu",
        "NGRCUUS": "residential_ng_usd_mcf",
        "NGCCUUS": "commercial_ng_usd_mcf",
        "NGICUUS": "industrial_ng_usd_mcf",
    }
    subset = table_2[table_2["series_id"].isin(sorted(series_map.keys()))].copy()
    if subset.empty:
        raise ContractViolation(
            "source_schema_drift",
            key="table_2.series_id",
            detail="table 2 does not contain required energy price series",
        )

    pivot = subset.pivot_table(
        index=["vintage_month", "timestamp", "is_forecast"],
        columns="series_id",
        values="value",
        aggfunc="last",
    ).reset_index()
    availability = (
        subset.groupby(["vintage_month", "timestamp"], as_index=False)[
            "available_timestamp"
        ]
        .max()
        .rename(columns={"available_timestamp": "available_timestamp"})
    )
    pivot = pivot.merge(availability, on=["vintage_month", "timestamp"], how="left")
    panel = pivot.rename(columns=series_map)
    panel["energy_prices_source"] = "steo_table_2"
    panel = panel.sort_values(["vintage_month", "timestamp"]).reset_index(drop=True)
    return panel


def _build_petroleum_supply_panel(table_4a: pd.DataFrame) -> pd.DataFrame:
    series_map = {
        "COPRPUS": "crude_prod_mmbd",
        "PASUPPLY": "total_supply_mmbd",
        "PATCPUSX": "total_consumption_mmbd",
        "PASXPUS": "commercial_inventory_mmb",
    }
    subset = table_4a[table_4a["series_id"].isin(sorted(series_map.keys()))].copy()
    if subset.empty:
        raise ContractViolation(
            "source_schema_drift",
            key="table_4a.series_id",
            detail="table 4a does not contain required petroleum supply series",
        )
    pivot = subset.pivot_table(
        index=["vintage_month", "timestamp", "is_forecast"],
        columns="series_id",
        values="value",
        aggfunc="last",
    ).reset_index()
    availability = (
        subset.groupby(["vintage_month", "timestamp"], as_index=False)[
            "available_timestamp"
        ]
        .max()
        .rename(columns={"available_timestamp": "available_timestamp"})
    )
    pivot = pivot.merge(availability, on=["vintage_month", "timestamp"], how="left")
    panel = pivot.rename(columns=series_map)
    panel["petroleum_supply_source"] = "steo_table_4a"
    panel = panel.sort_values(["vintage_month", "timestamp"]).reset_index(drop=True)
    return panel


def _build_regional_gas_prices_panel(table_5b: pd.DataFrame) -> pd.DataFrame:
    table = table_5b.copy()
    table["series_id"] = table["series_id"].astype(str)

    national_series = {
        "NGRCUUS": "residential_us_avg_usd_mcf",
        "NGCCUUS": "commercial_us_avg_usd_mcf",
        "NGICUUS": "industrial_us_avg_usd_mcf",
    }
    national = table[table["series_id"].isin(sorted(national_series.keys()))].copy()
    if national.empty:
        raise ContractViolation(
            "source_schema_drift",
            key="table_5b.series_id",
            detail="table 5b does not contain required national gas price series",
        )
    pivot = national.pivot_table(
        index=["vintage_month", "timestamp", "is_forecast"],
        columns="series_id",
        values="value",
        aggfunc="last",
    ).reset_index()
    pivot = pivot.rename(columns=national_series)

    spreads: list[pd.DataFrame] = []
    for prefix, out_col in (
        ("NGRCU_", "residential_spread_usd_mcf"),
        ("NGCCU_", "commercial_spread_usd_mcf"),
        ("NGICU_", "industrial_spread_usd_mcf"),
    ):
        regional = table[table["series_id"].str.startswith(prefix)].copy()
        regional = regional[~regional["series_id"].str.endswith("US")]
        if regional.empty:
            continue
        spread = (
            regional.groupby(["vintage_month", "timestamp"], as_index=False)["value"]
            .agg(lambda values: float(values.max() - values.min()))
            .rename(columns={"value": out_col})
        )
        spreads.append(spread)

    panel = pivot
    for spread in spreads:
        panel = panel.merge(
            spread,
            on=["vintage_month", "timestamp"],
            how="left",
        )

    availability = (
        table.groupby(["vintage_month", "timestamp"], as_index=False)[
            "available_timestamp"
        ]
        .max()
        .rename(columns={"available_timestamp": "available_timestamp"})
    )
    panel = panel.merge(availability, on=["vintage_month", "timestamp"], how="left")
    panel["regional_gas_prices_source"] = "steo_table_5b"
    panel = panel.sort_values(["vintage_month", "timestamp"]).reset_index(drop=True)
    return panel


def _build_context_priors(
    *,
    inventory_path: Path,
    bronze_context_root: Path,
) -> pd.DataFrame:
    inventory = pd.read_csv(inventory_path)
    inventory = inventory[
        inventory["integration_layer_target"].astype(str).str.lower() == "gold"
    ].copy()

    rows: list[dict[str, Any]] = []
    for _, rec in inventory.iterrows():
        relative = str(rec.get("relative_path", "")).strip()
        if not relative:
            continue

        base = bronze_context_root / relative
        if not base.exists():
            continue

        rows.append(
            {
                "relative_path": relative,
                "priority": str(rec.get("priority", "")),
                "report_family": str(rec.get("report_family", "")),
                "role": str(rec.get("role", "")),
                "integration_action": str(rec.get("integration_action", "")),
                "bytes": int(base.stat().st_size),
                "source_root": (
                    str(base.parents[2]) if len(base.parents) >= 3 else str(base.parent)
                ),
            }
        )

    if not rows:
        return pd.DataFrame(
            columns=[
                "relative_path",
                "priority",
                "report_family",
                "role",
                "integration_action",
                "bytes",
                "source_root",
            ]
        )

    return pd.DataFrame(rows).sort_values("relative_path").reset_index(drop=True)


def _resolve_weather_daily_snapshot(
    *,
    bronze_root: Path,
    asof: pd.Timestamp,
) -> Path:
    candidates: list[tuple[pd.Timestamp, Path]] = []
    for snapshot in sorted(bronze_root.glob("asof=*"), key=lambda path: path.name):
        token = snapshot.name.split("=", 1)[-1]
        parsed = pd.to_datetime(token, errors="coerce")
        if pd.isna(parsed):
            continue
        if pd.Timestamp(parsed) > asof:
            continue
        dataset_path = snapshot / "nasa_power_t2m_min_daily.parquet"
        if dataset_path.exists():
            candidates.append((pd.Timestamp(parsed), dataset_path))
    if not candidates:
        raise ContractViolation(
            "missing_source_file",
            key=str(bronze_root),
            detail="no weather daily snapshot is available at or before asof",
        )
    return sorted(candidates, key=lambda item: item[0])[-1][1]


def _build_weather_calibration(
    weather_daily: pd.DataFrame,
    *,
    report_root: Path,
    thresholds: tuple[float, ...] = (-12.0, -10.0, -8.0, -6.0, -5.0, -4.0, -2.0),
) -> tuple[float, Path]:
    rows: list[dict[str, Any]] = []
    frame = weather_daily.copy()
    frame["month"] = frame["date"].dt.to_period("M").dt.to_timestamp("M")
    for threshold in thresholds:
        marker = (frame["t2m_min_c"] <= threshold).astype(int)
        by_month = (
            frame.assign(freeze=marker)
            .groupby("month", as_index=False)["freeze"]
            .mean()
        )
        score = float(by_month["freeze"].std(ddof=0)) if not by_month.empty else 0.0
        rows.append(
            {
                "threshold_c": float(threshold),
                "freeze_share_std": score,
                "mean_freeze_share": (
                    float(by_month["freeze"].mean()) if not by_month.empty else 0.0
                ),
                "month_count": int(len(by_month)),
            }
        )
    calibration = pd.DataFrame(rows).sort_values(
        ["freeze_share_std", "threshold_c"], ascending=[False, False]
    )
    if calibration.empty:
        raise ContractViolation(
            "source_schema_drift",
            key="weather_threshold_calibration",
            detail="calibration search produced no thresholds",
        )
    best_threshold = float(calibration.iloc[0]["threshold_c"])
    calibration["selected"] = calibration["threshold_c"] == best_threshold
    report_root.mkdir(parents=True, exist_ok=True)
    output_path = report_root / "weather_threshold_calibration.csv"
    calibration.to_csv(output_path, index=False)
    return best_threshold, output_path


def _build_weather_freezeoff_panel(
    weather_daily: pd.DataFrame,
    *,
    threshold_c: float,
    asof: pd.Timestamp,
) -> pd.DataFrame:
    frame = weather_daily.copy()
    frame["timestamp"] = frame["date"].dt.to_period("M").dt.to_timestamp("M")
    frame["is_freeze_day"] = frame["t2m_min_c"] <= float(threshold_c)
    frame["freeze_intensity_c"] = (float(threshold_c) - frame["t2m_min_c"]).clip(
        lower=0.0
    )
    frame["is_mtd"] = frame["timestamp"] == asof.to_period("M").to_timestamp("M")
    frame.loc[~frame["is_mtd"], "is_mtd"] = False

    grouped = (
        frame.groupby(["basin_id", "basin_name", "timestamp"], as_index=False)
        .agg(
            observed_days=("date", "nunique"),
            freeze_days=("is_freeze_day", "sum"),
            freeze_intensity_c=("freeze_intensity_c", "mean"),
            extreme_min_c=("t2m_min_c", "min"),
            available_timestamp=("available_timestamp", "max"),
            source_id=("source_id", "last"),
            lineage_id=("lineage_id", "last"),
        )
        .sort_values(["basin_id", "timestamp"])
        .reset_index(drop=True)
    )
    grouped["calendar_days"] = grouped["timestamp"].dt.days_in_month.astype(int)
    current_month = asof.to_period("M").to_timestamp("M")
    grouped["expected_days"] = grouped["calendar_days"]
    grouped.loc[grouped["timestamp"] == current_month, "expected_days"] = int(asof.day)
    grouped["expected_days"] = grouped["expected_days"].clip(lower=1)
    grouped["coverage_fraction"] = grouped["observed_days"] / grouped["expected_days"]
    grouped["freeze_event_share"] = grouped["freeze_days"] / grouped["observed_days"]
    grouped["threshold_c"] = float(threshold_c)
    current_month = asof.to_period("M").to_timestamp("M")
    grouped["is_current_month"] = grouped["timestamp"] == current_month
    availability_anchor = grouped["timestamp"].dt.to_period("M").dt.to_timestamp(
        how="start"
    ) + pd.Timedelta(days=13)
    grouped["available_timestamp"] = availability_anchor
    current_anchor = current_month.to_period("M").to_timestamp(
        how="start"
    ) + pd.Timedelta(days=13)
    grouped.loc[grouped["timestamp"] == current_month, "available_timestamp"] = min(
        asof.normalize(), current_anchor
    )
    grouped["panel_source"] = "nasa_power_t2m_min"
    return grouped


def publish_weather_freezeoff_panel(
    *,
    asof: object,
    bronze_root: str | Path = "data/bronze/weather/nasa_power_t2m_min",
    gold_root: str | Path = "data/gold",
    report_root: str | Path = "data/reports",
) -> dict[str, Any]:
    """Build CP3 weather freeze-off gold panel from NASA POWER daily bronze rows."""

    asof_ts = pd.Timestamp(asof)
    if pd.isna(asof_ts):
        raise ContractViolation(
            "invalid_timestamp",
            key="asof",
            detail="asof could not be parsed",
        )

    bronze_path = Path(bronze_root)
    weather_snapshot = _resolve_weather_daily_snapshot(
        bronze_root=bronze_path, asof=asof_ts
    )
    weather_daily = pd.read_parquet(weather_snapshot)
    if weather_daily.empty:
        raise ContractViolation(
            "missing_source_file",
            key=str(weather_snapshot),
            detail="weather snapshot parquet is empty",
        )

    weather_daily = weather_daily.copy()
    weather_daily["date"] = pd.to_datetime(weather_daily["date"], errors="coerce")
    weather_daily["t2m_min_c"] = pd.to_numeric(
        weather_daily["t2m_min_c"], errors="coerce"
    )
    weather_daily["available_timestamp"] = pd.to_datetime(
        weather_daily["available_timestamp"], errors="coerce"
    )
    if weather_daily["date"].isna().any() or weather_daily["t2m_min_c"].isna().any():
        raise ContractViolation(
            "source_schema_drift",
            key=str(weather_snapshot),
            detail="weather daily dataset contains invalid date or temperature values",
        )

    report_path = Path(report_root)
    best_threshold, calibration_path = _build_weather_calibration(
        weather_daily,
        report_root=report_path,
    )
    panel = _build_weather_freezeoff_panel(
        weather_daily,
        threshold_c=best_threshold,
        asof=asof_ts,
    )
    validate_weather_coverage(panel, min_coverage=0.7)
    validate_weather_lineage(panel)

    target_root = Path(gold_root)
    target_root.mkdir(parents=True, exist_ok=True)
    panel_path = target_root / "weather_freezeoff_panel.parquet"
    panel.to_parquet(panel_path, index=False)

    payload = {
        "asof": asof_ts.date().isoformat(),
        "bronze_snapshot": str(weather_snapshot),
        "gold_panel_path": str(panel_path),
        "calibration_path": str(calibration_path),
        "row_count": int(len(panel)),
        "basin_count": int(panel["basin_id"].nunique()),
        "threshold_c": float(best_threshold),
    }
    publish_report = report_path / "weather_freezeoff_publish_report.json"
    publish_report.write_text(
        json.dumps(payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    payload["report_path"] = str(publish_report)
    return payload


def publish_steo_gold_marts(
    *,
    silver_root: str | Path = "data/silver/steo_vintages",
    gold_root: str | Path = "data/gold",
    inventory_path: str | Path = "data/reports/sprint4b_data_new_inventory.csv",
    bronze_context_root: str | Path = "data/bronze/context",
    report_root: str | Path = "data/reports",
) -> dict[str, Any]:
    """Publish Sprint 4B gold marts from silver STEO vintages and context priors."""

    silver_path = Path(silver_root)
    if not silver_path.exists():
        raise ContractViolation(
            "missing_source_file",
            key=str(silver_path),
            detail="silver STEO root does not exist",
        )

    table_2 = _load_silver_table(silver_path, "table_2")
    table_4a = _load_silver_table(silver_path, "table_4a")
    table_5a = _load_silver_table(silver_path, "table_5a")
    table_5b = _load_silver_table(silver_path, "table_5b")
    table_10a = _load_silver_table(silver_path, "table_10a")
    table_10b = _load_silver_table(silver_path, "table_10b")

    energy_prices_panel = _build_energy_prices_panel(table_2)
    petroleum_supply_panel = _build_petroleum_supply_panel(table_4a)
    observation_panel = _build_observation_panel(table_5a)
    regional_gas_prices_panel = _build_regional_gas_prices_panel(table_5b)
    driver_panel = _build_driver_panel(table_10a)
    shale_panel = _build_shale_split_panel(table_10b, observation_panel)

    context_panel = _build_context_priors(
        inventory_path=Path(inventory_path),
        bronze_context_root=Path(bronze_context_root),
    )

    target_root = Path(gold_root)
    target_root.mkdir(parents=True, exist_ok=True)

    output_paths = {
        "steo_energy_prices_panel": target_root / "steo_energy_prices_panel.parquet",
        "steo_petroleum_supply_panel": target_root
        / "steo_petroleum_supply_panel.parquet",
        "steo_observation_panel": target_root / "steo_observation_panel.parquet",
        "steo_regional_gas_prices_panel": target_root
        / "steo_regional_gas_prices_panel.parquet",
        "steo_driver_panel": target_root / "steo_driver_panel.parquet",
        "steo_drilling_metrics_panel": target_root
        / "steo_drilling_metrics_panel.parquet",
        "steo_shale_split_panel": target_root / "steo_shale_split_panel.parquet",
        "context_priors": target_root / "context_priors.parquet",
    }

    energy_prices_panel.to_parquet(
        output_paths["steo_energy_prices_panel"], index=False
    )
    petroleum_supply_panel.to_parquet(
        output_paths["steo_petroleum_supply_panel"], index=False
    )
    observation_panel.to_parquet(output_paths["steo_observation_panel"], index=False)
    regional_gas_prices_panel.to_parquet(
        output_paths["steo_regional_gas_prices_panel"], index=False
    )
    driver_panel.to_parquet(output_paths["steo_driver_panel"], index=False)
    driver_panel.to_parquet(output_paths["steo_drilling_metrics_panel"], index=False)
    shale_panel.to_parquet(output_paths["steo_shale_split_panel"], index=False)
    context_panel.to_parquet(output_paths["context_priors"], index=False)

    payload = {
        "silver_root": str(silver_path),
        "gold_root": str(target_root),
        "row_counts": {
            "steo_energy_prices_panel": int(len(energy_prices_panel)),
            "steo_petroleum_supply_panel": int(len(petroleum_supply_panel)),
            "steo_observation_panel": int(len(observation_panel)),
            "steo_regional_gas_prices_panel": int(len(regional_gas_prices_panel)),
            "steo_driver_panel": int(len(driver_panel)),
            "steo_drilling_metrics_panel": int(len(driver_panel)),
            "steo_shale_split_panel": int(len(shale_panel)),
            "context_priors": int(len(context_panel)),
        },
        "paths": {key: str(path) for key, path in output_paths.items()},
    }

    report_dir = Path(report_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    report_path = report_dir / "sprint4b_gold_publish_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(payload, sort_keys=True, indent=2),
        encoding="utf-8",
    )
    payload["report_path"] = str(report_path)
    return payload
