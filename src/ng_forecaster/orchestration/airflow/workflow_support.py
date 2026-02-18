"""Shared workflow helpers for weekly orchestration DAG implementations."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from ng_forecaster.errors import ContractViolation
from ng_forecaster.features.monthly_aggregations import (
    build_weighted_freezeoff_features,
)
from ng_forecaster.features.block_registry import load_feature_block_registry
from ng_forecaster.features.transfer_priors import build_transfer_prior_feature_rows
from ng_forecaster.ingest.adapters.bakerhughes_rigs import build_monthly_rig_features
from ng_forecaster.ingest.api_client import (
    fetch_eia_monthly_series_history,
    parse_eia_target_history,
    resolve_target_history_source,
)
from ng_forecaster.ingest.catalog import load_source_catalog

_FIXTURE_DAILY = Path("tests/fixtures/features/input_daily_price.csv")
_FIXTURE_WEEKLY = Path("tests/fixtures/features/input_weekly_storage.csv")
_MONTHLY_RELEASE_LOOKBACK_MONTHS = 36
_DEFAULT_GOLD_ROOT = "data/gold"


def resolve_weekly_asof(asof: str | None = None) -> pd.Timestamp:
    """Resolve a deterministic month-end as-of date for weekly orchestration runs."""

    if asof is not None:
        resolved = pd.Timestamp(asof)
        if pd.isna(resolved):
            raise ContractViolation(
                "invalid_timestamp",
                key="asof",
                detail="provided asof could not be parsed",
            )
        return resolved

    if not _FIXTURE_DAILY.exists() or not _FIXTURE_WEEKLY.exists():
        raise ContractViolation(
            "missing_bootstrap_inputs",
            key="fixtures",
            detail="weekly orchestration fallback fixtures are missing",
        )

    daily = pd.read_csv(_FIXTURE_DAILY)
    weekly = pd.read_csv(_FIXTURE_WEEKLY)
    max_daily = pd.to_datetime(daily["timestamp"], errors="coerce").max()
    max_weekly = pd.to_datetime(weekly["timestamp"], errors="coerce").max()
    resolved = min(max_daily, max_weekly)
    if pd.isna(resolved):
        raise ContractViolation(
            "invalid_timestamp",
            key="asof",
            detail="fixture timestamps are invalid",
        )

    return pd.Timestamp(resolved).to_period("M").to_timestamp("M")


def resolve_target_month(*, asof: object, lag_months: int = 2) -> pd.Timestamp:
    """Resolve the penultimate-month target for a given as-of timestamp."""

    if lag_months < 1:
        raise ContractViolation(
            "invalid_lag_policy",
            key="lag_months",
            detail="lag_months must be >= 1",
        )
    asof_ts = pd.Timestamp(asof)
    if pd.isna(asof_ts):
        raise ContractViolation(
            "invalid_timestamp",
            key="asof",
            detail="asof could not be parsed",
        )
    return (asof_ts.to_period("M") - lag_months).to_timestamp("M")


def _resolve_latest_released_month(
    *,
    asof: pd.Timestamp,
    lag_months: int,
    release_day_of_month: int,
) -> pd.Timestamp:
    """Resolve latest released target month from lag and release-day policy."""

    offset = lag_months if int(asof.day) >= release_day_of_month else lag_months + 1
    return (asof.to_period("M") - offset).to_timestamp("M")


def resolve_release_policy_context(
    asof: object,
    *,
    catalog_path: str | Path = "configs/sources.yaml",
) -> dict[str, Any]:
    """Resolve lag/release policy context and admissibility diagnostics."""

    asof_ts = pd.Timestamp(asof)
    if pd.isna(asof_ts):
        raise ContractViolation(
            "invalid_timestamp",
            key="asof",
            detail="asof could not be parsed",
        )

    catalog = load_source_catalog(catalog_path)
    release_cfg = catalog.release_calendar

    lag_months = int(release_cfg["lag_months"])
    release_day = int(release_cfg["release_day_of_month"])
    window = release_cfg["admissible_day_window"]
    start_day = int(window["start_day"])
    end_day = int(window["end_day"])

    target_month = resolve_target_month(asof=asof_ts, lag_months=lag_months)
    effective_lag_months = (
        lag_months if int(asof_ts.day) >= release_day else lag_months + 1
    )
    latest_released_month = _resolve_latest_released_month(
        asof=asof_ts,
        lag_months=lag_months,
        release_day_of_month=release_day,
    )

    day_admissible = start_day <= int(asof_ts.day) <= end_day
    lag_admissible = target_month > latest_released_month
    policy_passed = bool(day_admissible and lag_admissible)

    return {
        "asof": asof_ts.date().isoformat(),
        "target_month": target_month.date().isoformat(),
        "lag_months": lag_months,
        "effective_lag_months": int(effective_lag_months),
        "release_day_of_month": release_day,
        "admissible_window_start_day": start_day,
        "admissible_window_end_day": end_day,
        "day_of_month": int(asof_ts.day),
        "day_admissible": bool(day_admissible),
        "latest_released_month": latest_released_month.date().isoformat(),
        "lag_admissible": bool(lag_admissible),
        "policy_passed": bool(policy_passed),
    }


def enforce_release_policy(
    asof: object,
    *,
    catalog_path: str | Path = "configs/sources.yaml",
) -> dict[str, Any]:
    """Raise hard failure when release-calendar admissibility is violated."""

    context = resolve_release_policy_context(asof, catalog_path=catalog_path)
    if context["policy_passed"]:
        return context

    asof_value = pd.Timestamp(context["asof"]).to_pydatetime()
    if not bool(context["day_admissible"]):
        raise ContractViolation(
            "run_outside_release_window",
            asof=asof_value,
            key="admissible_day_window",
            detail=(
                f"day={context['day_of_month']} is outside "
                f"[{context['admissible_window_start_day']}, {context['admissible_window_end_day']}]"
            ),
        )

    raise ContractViolation(
        "lag_policy_violated",
        asof=asof_value,
        key="target_month",
        detail=(
            "target month must remain unreleased under lag policy; "
            f"target_month={context['target_month']} "
            f"latest_released_month={context['latest_released_month']}"
        ),
    )


def bootstrap_status(root: str | Path = "data/bootstrap") -> dict[str, Any]:
    """Return first-run bootstrap availability diagnostics."""

    base = Path(root)
    raw_dir = base / "raw"
    report_dir = base / "reports"
    raw_files = sorted(path.name for path in raw_dir.glob("*") if path.is_file())
    report_files = sorted(path.name for path in report_dir.glob("*") if path.is_file())

    return {
        "bootstrap_root": str(base),
        "raw_dir": str(raw_dir),
        "report_dir": str(report_dir),
        "raw_file_count": len(raw_files),
        "report_file_count": len(report_files),
        "raw_files": raw_files,
        "report_files": report_files,
        "available": bool(raw_files),
    }


def ensure_dir(path: str | Path) -> Path:
    """Create directory if needed and return the resolved path."""

    resolved = Path(path)
    resolved.mkdir(parents=True, exist_ok=True)
    return resolved


def write_json(path: str | Path, payload: Mapping[str, Any]) -> Path:
    """Write JSON payload with deterministic formatting."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(dict(payload), sort_keys=True, indent=2, default=str),
        encoding="utf-8",
    )
    return target


def sha256_file(path: str | Path) -> str:
    """Compute SHA-256 digest for a file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(65536)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _load_fixture_series() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not _FIXTURE_DAILY.exists() or not _FIXTURE_WEEKLY.exists():
        raise ContractViolation(
            "missing_bootstrap_inputs",
            key="fixtures",
            detail="required feature fixtures do not exist",
        )

    daily = pd.read_csv(_FIXTURE_DAILY).rename(columns={"hh_price": "value"})
    weekly = pd.read_csv(_FIXTURE_WEEKLY).rename(columns={"storage_bcf": "value"})

    daily["timestamp"] = pd.to_datetime(daily["timestamp"], errors="coerce")
    weekly["timestamp"] = pd.to_datetime(weekly["timestamp"], errors="coerce")
    if daily["timestamp"].isna().any() or weekly["timestamp"].isna().any():
        raise ContractViolation(
            "invalid_timestamp",
            key="market_inputs",
            detail="fixture timestamps contain invalid values",
        )

    daily = daily.sort_values("timestamp").reset_index(drop=True)
    weekly = weekly.sort_values("timestamp").reset_index(drop=True)
    return daily, weekly


def _build_feature_bundle_from_series(
    *,
    daily: pd.DataFrame,
    weekly: pd.DataFrame,
    feature_max_age_days: int,
    daily_frequency: str,
    weekly_frequency: str,
    feature_block_map: Mapping[str, str] | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    block_map = dict(feature_block_map or {})
    features = pd.concat(
        [
            daily.assign(feature_name="hh_last").rename(
                columns={"timestamp": "feature_timestamp"}
            ),
            weekly.assign(feature_name="stor_last").rename(
                columns={"timestamp": "feature_timestamp"}
            ),
        ],
        ignore_index=True,
    )
    features["available_timestamp"] = features["feature_timestamp"]
    features["block_id"] = features["feature_name"].map(block_map).fillna("market_core")
    features = features[
        [
            "feature_name",
            "feature_timestamp",
            "available_timestamp",
            "block_id",
            "value",
        ]
    ]
    features = features.sort_values(["feature_name", "feature_timestamp"]).reset_index(
        drop=True
    )

    feature_policy = {
        "version": 1,
        "default": {"max_age_days": int(feature_max_age_days)},
        "features": {
            "hh_last": {
                "source_frequency": daily_frequency,
                "aggregation": "last",
                "max_age_days": int(feature_max_age_days),
            },
            "stor_last": {
                "source_frequency": weekly_frequency,
                "aggregation": "last",
                "max_age_days": int(feature_max_age_days),
            },
        },
    }
    return features, feature_policy


def _default_gold_root() -> Path:
    return Path(os.getenv("NGF_GOLD_ROOT", _DEFAULT_GOLD_ROOT))


_STEO_DRIVER_SHALE_INTRO_MONTH = pd.Timestamp("2024-06-30")


def _load_gold_panel(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if frame.empty:
        return frame
    if "timestamp" in frame.columns:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], errors="coerce")
    if "available_timestamp" in frame.columns:
        frame["available_timestamp"] = pd.to_datetime(
            frame["available_timestamp"], errors="coerce"
        )
    else:
        if "vintage_month" in frame.columns:
            frame["available_timestamp"] = pd.to_datetime(
                frame["vintage_month"], errors="coerce"
            )
        else:
            frame["available_timestamp"] = pd.NaT
    if "vintage_month" in frame.columns:
        frame["vintage_month"] = (
            pd.to_datetime(frame["vintage_month"], errors="coerce")
            .dt.to_period("M")
            .dt.to_timestamp("M")
        )
    return frame


def load_steo_gold_feature_rows(
    *,
    asof: pd.Timestamp,
    target_month: pd.Timestamp,
    gold_root: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load STEO gold marts and convert them into runtime feature rows."""

    root = Path(gold_root) if gold_root is not None else _default_gold_root()
    energy_path = root / "steo_energy_prices_panel.parquet"
    petroleum_path = root / "steo_petroleum_supply_panel.parquet"
    observation_path = root / "steo_observation_panel.parquet"
    regional_path = root / "steo_regional_gas_prices_panel.parquet"
    driver_path = root / "steo_driver_panel.parquet"
    shale_path = root / "steo_shale_split_panel.parquet"

    energy = _load_gold_panel(energy_path)
    petroleum = _load_gold_panel(petroleum_path)
    observation = _load_gold_panel(observation_path)
    regional = _load_gold_panel(regional_path)
    driver = _load_gold_panel(driver_path)
    shale = _load_gold_panel(shale_path)

    if observation.empty:
        return (
            pd.DataFrame(
                columns=[
                    "feature_name",
                    "feature_timestamp",
                    "available_timestamp",
                    "block_id",
                    "value",
                ]
            ),
            {
                "status": "missing_gold_panels",
                "gold_root": str(root),
                "energy_exists": energy_path.exists(),
                "petroleum_exists": petroleum_path.exists(),
                "observation_exists": observation_path.exists(),
                "regional_exists": regional_path.exists(),
                "driver_exists": driver_path.exists(),
                "shale_exists": shale_path.exists(),
                "energy_row_count": int(len(energy)),
                "petroleum_row_count": int(len(petroleum)),
                "observation_row_count": int(len(observation)),
                "regional_row_count": int(len(regional)),
                "driver_row_count": int(len(driver)),
                "shale_row_count": int(len(shale)),
            },
        )

    asof_month = pd.Timestamp(asof).to_period("M").to_timestamp("M")
    available_vintages = observation["vintage_month"].dropna()
    available_vintages = available_vintages[available_vintages <= asof_month]
    if available_vintages.empty:
        raise ContractViolation(
            "missing_source_file",
            key="steo_observation_panel",
            detail="no gold vintage is available at or before run asof month",
        )
    latest_vintage = (
        pd.Timestamp(available_vintages.max()).to_period("M").to_timestamp("M")
    )

    def _slice_latest(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "vintage_month" not in frame.columns:
            return pd.DataFrame()
        return frame[frame["vintage_month"] == latest_vintage].copy()

    energy_slice = _slice_latest(energy)
    petroleum_slice = _slice_latest(petroleum)
    obs_slice = _slice_latest(observation)
    regional_slice = _slice_latest(regional)
    driver_slice = _slice_latest(driver)
    shale_slice = _slice_latest(shale)

    month_t = pd.Timestamp(target_month).to_period("M").to_timestamp("M")
    month_t1 = (month_t.to_period("M") + 1).to_timestamp("M")
    if obs_slice.empty:
        raise ContractViolation(
            "source_schema_drift",
            key=f"vintage_month={latest_vintage.date().isoformat()}",
            detail="gold observation panel is missing rows for the selected vintage",
        )

    has_driver_panel = not driver_slice.empty
    has_shale_panel = not shale_slice.empty
    has_10ab_coverage = bool(latest_vintage >= _STEO_DRIVER_SHALE_INTRO_MONTH)
    if has_10ab_coverage and (not has_driver_panel or not has_shale_panel):
        raise ContractViolation(
            "source_schema_drift",
            key=f"vintage_month={latest_vintage.date().isoformat()}",
            detail="gold marts are missing table_10a/table_10b rows for the selected vintage",
        )

    obs_target = obs_slice[obs_slice["timestamp"].isin([month_t, month_t1])].copy()
    shale_target = pd.DataFrame()
    if has_shale_panel:
        shale_target = shale_slice[
            shale_slice["timestamp"].isin([month_t, month_t1])
        ].copy()

    def _target_window_rows(frame: pd.DataFrame) -> pd.DataFrame:
        if frame.empty or "timestamp" not in frame.columns:
            return pd.DataFrame()
        return frame[frame["timestamp"].isin([month_t, month_t1])].copy()

    energy_target = _target_window_rows(energy_slice)
    petroleum_target = _target_window_rows(petroleum_slice)
    regional_target = _target_window_rows(regional_slice)

    if obs_target.empty:
        raise ContractViolation(
            "insufficient_release_history",
            key="steo_gold_target_rows",
            detail="gold observation panel is missing target-month STEO rows",
        )
    if has_shale_panel and shale_target.empty:
        raise ContractViolation(
            "insufficient_release_history",
            key="steo_shale_split_panel",
            detail="gold shale panel is missing target-month STEO rows",
        )
    if not energy_slice.empty and energy_target.empty:
        raise ContractViolation(
            "insufficient_release_history",
            key="steo_energy_prices_panel",
            detail="gold energy panel is missing target-month rows",
        )
    if not petroleum_slice.empty and petroleum_target.empty:
        raise ContractViolation(
            "insufficient_release_history",
            key="steo_petroleum_supply_panel",
            detail="gold petroleum panel is missing target-month rows",
        )
    if not regional_slice.empty and regional_target.empty:
        raise ContractViolation(
            "insufficient_release_history",
            key="steo_regional_gas_prices_panel",
            detail="gold regional gas panel is missing target-month rows",
        )

    driver_target = pd.DataFrame()
    if has_driver_panel:
        driver_target = driver_slice[driver_slice["timestamp"] <= month_t].copy()
    if has_driver_panel and driver_target.empty:
        raise ContractViolation(
            "insufficient_release_history",
            key="steo_driver_panel",
            detail="gold driver panel has no rows at or before target month",
        )
    latest_driver = (
        driver_target.sort_values("timestamp").iloc[-1] if has_driver_panel else None
    )

    def _available_ts(rec: pd.Series) -> pd.Timestamp:
        raw = rec.get("available_timestamp")
        parsed = pd.to_datetime(raw, errors="coerce")
        if pd.isna(parsed):
            parsed = pd.to_datetime(rec.get("vintage_month"), errors="coerce")
        if pd.isna(parsed):
            raise ContractViolation(
                "invalid_timestamp",
                key="available_timestamp",
                detail="gold panel row has no valid available timestamp",
            )
        return pd.Timestamp(parsed)

    rows: list[dict[str, Any]] = []
    for _, rec in obs_target.sort_values("timestamp").iterrows():
        ts = pd.Timestamp(rec["timestamp"]).to_period("M").to_timestamp("M")
        suffix = "t" if ts == month_t else "t_plus_1"
        available_ts = _available_ts(rec)
        for column in (
            "dry_prod_bcfd",
            "marketed_prod_bcfd",
            "dry_to_marketed_ratio",
            "extraction_loss_ratio",
        ):
            numeric = pd.to_numeric(pd.Series([rec.get(column)]), errors="coerce").iloc[
                0
            ]
            if pd.isna(numeric):
                continue
            rows.append(
                {
                    "feature_name": f"steo_{column}_{suffix}",
                    "feature_timestamp": ts,
                    "available_timestamp": available_ts,
                    "block_id": "steo_observation",
                    "value": float(numeric),
                }
            )

    if has_shale_panel:
        for _, rec in shale_target.sort_values("timestamp").iterrows():
            ts = pd.Timestamp(rec["timestamp"]).to_period("M").to_timestamp("M")
            suffix = "t" if ts == month_t else "t_plus_1"
            available_ts = _available_ts(rec)
            for column in (
                "shale_dry_prod_bcfd",
                "shale_share_of_dry",
                "non_shale_dry_prod_bcfd",
            ):
                numeric = pd.to_numeric(
                    pd.Series([rec.get(column)]), errors="coerce"
                ).iloc[0]
                if pd.isna(numeric):
                    continue
                rows.append(
                    {
                        "feature_name": f"steo_{column}_{suffix}",
                        "feature_timestamp": ts,
                        "available_timestamp": available_ts,
                        "block_id": "steo_shale",
                        "value": float(numeric),
                    }
                )

    if latest_driver is not None:
        available_ts = _available_ts(latest_driver)
        for column in (
            "active_rigs",
            "new_wells_drilled",
            "new_wells_completed",
            "duc_inventory",
            "new_well_gas_production",
            "existing_gas_production_change",
        ):
            numeric = pd.to_numeric(
                pd.Series([latest_driver.get(column)]), errors="coerce"
            ).iloc[0]
            if pd.isna(numeric):
                continue
            rows.append(
                {
                    "feature_name": f"steo_driver_{column}",
                    "feature_timestamp": pd.Timestamp(latest_driver["timestamp"])
                    .to_period("M")
                    .to_timestamp("M"),
                    "available_timestamp": available_ts,
                    "block_id": "steo_driver",
                    "value": float(numeric),
                }
            )
        for column, feature_name in (
            ("active_rigs", "steo_drilling_rigs"),
            ("new_wells_drilled", "steo_drilling_wells_drilled"),
            ("new_wells_completed", "steo_drilling_wells_completed"),
            ("duc_inventory", "steo_drilling_duc_inventory"),
            ("new_well_gas_production", "steo_drilling_new_well_gas_production"),
            (
                "existing_gas_production_change",
                "steo_drilling_existing_gas_production_change",
            ),
            ("completion_to_duc_ratio", "steo_drilling_completion_to_duc_ratio"),
            ("new_well_minus_legacy", "steo_drilling_new_well_minus_legacy"),
        ):
            numeric = pd.to_numeric(
                pd.Series([latest_driver.get(column)]), errors="coerce"
            ).iloc[0]
            if pd.isna(numeric):
                continue
            rows.append(
                {
                    "feature_name": feature_name,
                    "feature_timestamp": pd.Timestamp(latest_driver["timestamp"])
                    .to_period("M")
                    .to_timestamp("M"),
                    "available_timestamp": available_ts,
                    "block_id": "steo_drilling_metrics",
                    "value": float(numeric),
                }
            )

    if not energy_target.empty and "timestamp" in energy_target.columns:
        for _, rec in energy_target.sort_values("timestamp").iterrows():
            ts = pd.Timestamp(rec["timestamp"]).to_period("M").to_timestamp("M")
            suffix = "t" if ts == month_t else "t_plus_1"
            available_ts = _available_ts(rec)
            for column in (
                "hh_spot_usd_mcf",
                "hh_spot_usd_mmbtu",
                "wti_spot_usd_bbl",
                "brent_spot_usd_bbl",
                "residential_ng_usd_mcf",
                "commercial_ng_usd_mcf",
                "industrial_ng_usd_mcf",
            ):
                numeric = pd.to_numeric(
                    pd.Series([rec.get(column)]), errors="coerce"
                ).iloc[0]
                if pd.isna(numeric):
                    continue
                rows.append(
                    {
                        "feature_name": f"steo_energy_{column}_{suffix}",
                        "feature_timestamp": ts,
                        "available_timestamp": available_ts,
                        "block_id": "steo_energy_prices",
                        "value": float(numeric),
                    }
                )
                if suffix == "t" and column == "wti_spot_usd_bbl":
                    rows.append(
                        {
                            "feature_name": "wti_mtd_mean",
                            "feature_timestamp": ts,
                            "available_timestamp": available_ts,
                            "block_id": "oil_side",
                            "value": float(numeric),
                        }
                    )
                    rows.append(
                        {
                            "feature_name": "wti_last",
                            "feature_timestamp": ts,
                            "available_timestamp": available_ts,
                            "block_id": "oil_side",
                            "value": float(numeric),
                        }
                    )
                if suffix == "t" and column == "brent_spot_usd_bbl":
                    rows.append(
                        {
                            "feature_name": "brent_mtd_mean",
                            "feature_timestamp": ts,
                            "available_timestamp": available_ts,
                            "block_id": "oil_side",
                            "value": float(numeric),
                        }
                    )
                    rows.append(
                        {
                            "feature_name": "brent_last",
                            "feature_timestamp": ts,
                            "available_timestamp": available_ts,
                            "block_id": "oil_side",
                            "value": float(numeric),
                        }
                    )

    if not petroleum_target.empty and "timestamp" in petroleum_target.columns:
        for _, rec in petroleum_target.sort_values("timestamp").iterrows():
            ts = pd.Timestamp(rec["timestamp"]).to_period("M").to_timestamp("M")
            suffix = "t" if ts == month_t else "t_plus_1"
            available_ts = _available_ts(rec)
            for column in (
                "crude_prod_mmbd",
                "total_consumption_mmbd",
                "commercial_inventory_mmb",
                "total_supply_mmbd",
            ):
                numeric = pd.to_numeric(
                    pd.Series([rec.get(column)]), errors="coerce"
                ).iloc[0]
                if pd.isna(numeric):
                    continue
                rows.append(
                    {
                        "feature_name": f"steo_petroleum_{column}_{suffix}",
                        "feature_timestamp": ts,
                        "available_timestamp": available_ts,
                        "block_id": "steo_petroleum_supply",
                        "value": float(numeric),
                    }
                )

    if not regional_target.empty and "timestamp" in regional_target.columns:
        for _, rec in regional_target.sort_values("timestamp").iterrows():
            ts = pd.Timestamp(rec["timestamp"]).to_period("M").to_timestamp("M")
            suffix = "t" if ts == month_t else "t_plus_1"
            available_ts = _available_ts(rec)
            for column in (
                "residential_us_avg_usd_mcf",
                "commercial_us_avg_usd_mcf",
                "industrial_us_avg_usd_mcf",
                "residential_spread_usd_mcf",
                "commercial_spread_usd_mcf",
                "industrial_spread_usd_mcf",
            ):
                numeric = pd.to_numeric(
                    pd.Series([rec.get(column)]), errors="coerce"
                ).iloc[0]
                if pd.isna(numeric):
                    continue
                rows.append(
                    {
                        "feature_name": f"steo_regional_{column}_{suffix}",
                        "feature_timestamp": ts,
                        "available_timestamp": available_ts,
                        "block_id": "steo_regional_gas_prices",
                        "value": float(numeric),
                    }
                )

    feature_frame = pd.DataFrame(rows)
    if feature_frame.empty:
        raise ContractViolation(
            "source_schema_drift",
            key="steo_gold_features",
            detail="no numeric gold features were derived from steo marts",
        )
    feature_frame["available_timestamp"] = pd.to_datetime(
        feature_frame["available_timestamp"], errors="coerce"
    )
    feature_frame = feature_frame.sort_values(
        ["feature_name", "feature_timestamp", "available_timestamp"]
    ).reset_index(drop=True)

    return (
        feature_frame,
        {
            "status": (
                "gold_features_loaded"
                if has_driver_panel and has_shale_panel
                else "gold_features_observation_only_pre_10ab"
            ),
            "gold_root": str(root),
            "latest_vintage_month": latest_vintage.date().isoformat(),
            "feature_row_count": int(len(feature_frame)),
            "energy_panel_available": not energy_slice.empty,
            "petroleum_panel_available": not petroleum_slice.empty,
            "driver_panel_available": has_driver_panel,
            "regional_panel_available": not regional_slice.empty,
            "shale_panel_available": has_shale_panel,
            "energy_path": str(energy_path),
            "petroleum_path": str(petroleum_path),
            "observation_path": str(observation_path),
            "regional_path": str(regional_path),
            "driver_path": str(driver_path),
            "shale_path": str(shale_path),
        },
    )


def load_weather_gold_feature_rows(
    *,
    asof: pd.Timestamp,
    gold_root: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load weather freeze-off gold panel and return weighted runtime feature rows."""

    root = Path(gold_root) if gold_root is not None else _default_gold_root()
    weather_path = root / "weather_freezeoff_panel.parquet"
    if not weather_path.exists():
        return (
            pd.DataFrame(
                columns=[
                    "feature_name",
                    "feature_timestamp",
                    "available_timestamp",
                    "block_id",
                    "value",
                ]
            ),
            {
                "status": "missing_weather_panel",
                "weather_path": str(weather_path),
            },
        )

    panel = _load_gold_panel(weather_path)
    if panel.empty:
        raise ContractViolation(
            "missing_source_file",
            key=str(weather_path),
            detail="weather freeze-off panel exists but contains zero rows",
        )
    try:
        frame = build_weighted_freezeoff_features(panel, asof=asof)
    except ContractViolation as exc:
        if exc.context.reason_code != "missing_feature_input":
            raise
        return (
            pd.DataFrame(
                columns=[
                    "feature_name",
                    "feature_timestamp",
                    "available_timestamp",
                    "block_id",
                    "value",
                ]
            ),
            {
                "status": "weather_panel_not_ready_for_asof",
                "weather_path": str(weather_path),
                "asof": pd.Timestamp(asof).date().isoformat(),
            },
        )
    available_ts = pd.Timestamp(frame["available_timestamp"].max())
    return (
        frame.sort_values("feature_name").reset_index(drop=True),
        {
            "status": "weather_features_loaded",
            "weather_path": str(weather_path),
            "feature_row_count": int(len(frame)),
            "available_timestamp_max": available_ts.date().isoformat(),
        },
    )


def load_bakerhughes_gold_feature_rows(
    *,
    asof: pd.Timestamp,
    gold_root: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Load Baker Hughes rig panel and emit oil-side runtime features."""

    root = Path(gold_root) if gold_root is not None else _default_gold_root()
    path = root / "bakerhughes_rigs_panel.parquet"
    if not path.exists():
        return (
            pd.DataFrame(
                columns=[
                    "feature_name",
                    "feature_timestamp",
                    "available_timestamp",
                    "block_id",
                    "value",
                ]
            ),
            {
                "status": "missing_bakerhughes_panel",
                "panel_path": str(path),
            },
        )
    panel = _load_gold_panel(path)
    if panel.empty:
        return (
            pd.DataFrame(
                columns=[
                    "feature_name",
                    "feature_timestamp",
                    "available_timestamp",
                    "block_id",
                    "value",
                ]
            ),
            {
                "status": "empty_bakerhughes_panel",
                "panel_path": str(path),
            },
        )
    try:
        features = build_monthly_rig_features(panel, asof=asof)
    except ContractViolation as exc:
        if exc.context.reason_code != "missing_feature_input":
            raise
        return (
            pd.DataFrame(
                columns=[
                    "feature_name",
                    "feature_timestamp",
                    "available_timestamp",
                    "block_id",
                    "value",
                ]
            ),
            {
                "status": "bakerhughes_panel_not_ready_for_asof",
                "panel_path": str(path),
                "reason_code": str(exc.context.reason_code),
            },
        )
    return (
        features.sort_values("feature_name").reset_index(drop=True),
        {
            "status": "bakerhughes_features_loaded",
            "panel_path": str(path),
            "feature_row_count": int(len(features)),
            "available_timestamp_max": pd.Timestamp(
                features["available_timestamp"].max()
            )
            .date()
            .isoformat(),
        },
    )


def _build_source_backed_feature_inputs(
    target_history: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    monthly = target_history[["timestamp", "target_value"]].copy()
    monthly = monthly.sort_values("timestamp").reset_index(drop=True)

    daily = pd.DataFrame(
        {
            "timestamp": monthly["timestamp"],
            "value": (monthly["target_value"] / 100000.0).round(6),
        }
    )
    weekly = pd.DataFrame(
        {
            "timestamp": monthly["timestamp"],
            "value": (monthly["target_value"] / 10000.0).round(6),
        }
    )
    return daily, weekly


def trim_target_history_to_latest_release(
    target_history: pd.DataFrame,
    *,
    latest_released_month: pd.Timestamp,
) -> pd.DataFrame:
    """Trim target history to released-only rows at or before latest released month."""

    trimmed = target_history.copy()
    trimmed["timestamp"] = pd.to_datetime(trimmed["timestamp"], errors="coerce")
    trimmed = trimmed[trimmed["timestamp"].notna()].copy()
    boundary = pd.Timestamp(latest_released_month).to_period("M").to_timestamp("M")
    trimmed = trimmed[trimmed["timestamp"] <= boundary].copy()
    trimmed = trimmed.sort_values("timestamp").reset_index(drop=True)
    return trimmed


def resolve_monthly_release_history(
    target_history_released: pd.DataFrame,
    *,
    asof: pd.Timestamp,
    lookback_months: int = _MONTHLY_RELEASE_LOOKBACK_MONTHS,
    series_id: str = "NG.N9070US2.M",
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Resolve monthly release history from EIA API with deterministic fallback."""

    if lookback_months < 1:
        raise ContractViolation(
            "invalid_source_catalog",
            key="lookback_months",
            detail="lookback_months must be >= 1",
        )

    api_key = os.getenv("EIA_API_KEY", "")
    api_error: str | None = None
    release_history = pd.DataFrame()

    if api_key.strip():
        try:
            release_history = fetch_eia_monthly_series_history(
                series_id=series_id,
                api_key=api_key,
                asof=pd.Timestamp(asof),
                lookback_months=lookback_months,
            )
        except Exception as exc:  # pragma: no cover - network/runtime dependent
            api_error = str(exc)

    if not release_history.empty:
        if len(release_history) >= int(lookback_months):
            return (
                release_history,
                {
                    "source": "eia_api_v2",
                    "series_id": series_id,
                    "row_count": int(len(release_history)),
                    "required_row_count": int(lookback_months),
                    "api_error": api_error,
                },
            )
        api_error = (
            f"api returned only {len(release_history)} rows; "
            f"required at least {lookback_months}"
        )

    fallback = target_history_released[["timestamp", "target_value"]].copy()
    fallback = (
        fallback.sort_values("timestamp").tail(lookback_months).reset_index(drop=True)
    )
    if fallback.empty:
        raise ContractViolation(
            "missing_target_history_rows",
            key="target_history_released",
            detail="cannot build monthly release history fallback from empty target data",
        )
    if len(fallback) < int(lookback_months):
        raise ContractViolation(
            "insufficient_release_history",
            key="monthly_release_history",
            detail=(
                f"required at least {lookback_months} monthly points but only "
                f"{len(fallback)} are available"
            ),
        )
    fallback["series_id"] = series_id
    fallback["source"] = "target_history_fallback"
    return (
        fallback,
        {
            "source": "target_history_fallback",
            "series_id": series_id,
            "row_count": int(len(fallback)),
            "required_row_count": int(lookback_months),
            "api_error": api_error,
        },
    )


def load_market_inputs(
    asof: pd.Timestamp, *, include_gold_features: bool = True
) -> dict[str, Any]:
    """Load source-backed target history and deterministic feature inputs."""

    asof_ts = pd.Timestamp(asof)
    if pd.isna(asof_ts):
        raise ContractViolation(
            "invalid_timestamp",
            key="asof",
            detail="asof could not be parsed",
        )

    catalog = load_source_catalog("configs/sources.yaml")
    block_registry = load_feature_block_registry("configs/feature_blocks.yaml")
    feature_block_map = dict(block_registry.feature_to_block)
    target_source, target_path, target_origin = resolve_target_history_source(
        asof=asof_ts,
        catalog=catalog,
        bootstrap_root=catalog.defaults["bootstrap_root"],
        stream_roots={
            "api": os.getenv("NGF_API_SOURCE_ROOT", "data/bronze/eia_api"),
            "bulk": os.getenv("NGF_BULK_SOURCE_ROOT", "data/bronze/eia_bulk"),
        },
    )
    target_history = parse_eia_target_history(
        target_path, parse_config=target_source.parse
    )
    release_context = resolve_release_policy_context(asof_ts)
    latest_released_month = pd.Timestamp(release_context["latest_released_month"])
    target_history_full = target_history.sort_values("timestamp").reset_index(drop=True)
    target_history = trim_target_history_to_latest_release(
        target_history_full,
        latest_released_month=latest_released_month,
    )
    max_trimmed = pd.Timestamp(target_history["timestamp"].max())
    if max_trimmed > latest_released_month:
        raise ContractViolation(
            "leakage_detected",
            key="target_history",
            detail=(
                "released-only target history contains future rows beyond "
                f"latest_released_month={latest_released_month.date().isoformat()}"
            ),
        )
    if target_history.empty:
        raise ContractViolation(
            "insufficient_training_data",
            key="target_history",
            detail="released-only target history is empty after release-policy trimming",
        )
    if len(target_history) < 60:
        raise ContractViolation(
            "insufficient_training_data",
            key="target_history",
            detail="released-only target history must include at least 60 monthly points",
        )

    monthly_release_history, release_history_meta = resolve_monthly_release_history(
        target_history,
        asof=asof_ts,
        lookback_months=_MONTHLY_RELEASE_LOOKBACK_MONTHS,
    )

    fixture_daily: pd.DataFrame | None = None
    fixture_weekly: pd.DataFrame | None = None
    try:
        fixture_daily, fixture_weekly = _load_fixture_series()
    except ContractViolation:
        fixture_daily = None
        fixture_weekly = None

    use_fixtures = False
    if fixture_daily is not None and fixture_weekly is not None:
        max_daily = pd.Timestamp(fixture_daily["timestamp"].max())
        max_weekly = pd.Timestamp(fixture_weekly["timestamp"].max())
        freshest = max(max_daily, max_weekly)
        stale_days = int((asof_ts - freshest).days)
        use_fixtures = 0 <= stale_days <= 45

    if use_fixtures and fixture_daily is not None and fixture_weekly is not None:
        daily = fixture_daily
        weekly = fixture_weekly
        features, feature_policy = _build_feature_bundle_from_series(
            daily=daily,
            weekly=weekly,
            feature_max_age_days=45,
            daily_frequency="daily",
            weekly_frequency="weekly",
            feature_block_map=feature_block_map,
        )
        feature_source = "fixture_inputs"
    else:
        daily, weekly = _build_source_backed_feature_inputs(target_history)
        features, feature_policy = _build_feature_bundle_from_series(
            daily=daily,
            weekly=weekly,
            feature_max_age_days=180,
            daily_frequency="monthly",
            weekly_frequency="monthly",
            feature_block_map=feature_block_map,
        )
        feature_source = "target_history_derived"

    for feature_name, cfg in feature_policy["features"].items():
        cfg["block_id"] = feature_block_map.get(feature_name, "market_core")
        cfg["asof_rule"] = "available_timestamp_lte_asof"
        cfg["max_staleness_days"] = int(cfg["max_age_days"])

    steo_gold_meta: dict[str, Any] = {
        "status": "disabled_for_run",
        "reason": "include_gold_features=false",
    }
    weather_gold_meta: dict[str, Any] = {
        "status": "disabled_for_run",
        "reason": "include_gold_features=false",
    }
    transfer_priors_meta: dict[str, Any] = {
        "status": "disabled_for_run",
        "reason": "include_gold_features=false",
    }
    bakerhughes_meta: dict[str, Any] = {
        "status": "disabled_for_run",
        "reason": "include_gold_features=false",
    }
    if include_gold_features:
        steo_gold_features, steo_gold_meta = load_steo_gold_feature_rows(
            asof=asof_ts,
            target_month=pd.Timestamp(release_context["target_month"]),
        )
        if not steo_gold_features.empty:
            features = (
                pd.concat([features, steo_gold_features], ignore_index=True)
                .sort_values(
                    ["feature_name", "feature_timestamp", "available_timestamp"]
                )
                .reset_index(drop=True)
            )
            for feature_name in sorted(
                steo_gold_features["feature_name"].unique().tolist()
            ):
                block_id = feature_block_map.get(feature_name, "steo_observation")
                feature_policy["features"][feature_name] = {
                    "source_frequency": "monthly",
                    "aggregation": "last",
                    "max_age_days": 180,
                    "block_id": block_id,
                    "asof_rule": "available_timestamp_lte_asof",
                    "max_staleness_days": 180,
                }
            feature_source = f"{feature_source}+gold_steo"

        weather_gold_features, weather_gold_meta = load_weather_gold_feature_rows(
            asof=asof_ts
        )
        if not weather_gold_features.empty:
            features = (
                pd.concat([features, weather_gold_features], ignore_index=True)
                .sort_values(
                    ["feature_name", "feature_timestamp", "available_timestamp"]
                )
                .reset_index(drop=True)
            )
            for feature_name in sorted(
                weather_gold_features["feature_name"].unique().tolist()
            ):
                block_id = feature_block_map.get(feature_name, "weather_freezeoff")
                feature_policy["features"][feature_name] = {
                    "source_frequency": "daily",
                    "aggregation": "weighted_mean",
                    "max_age_days": 45,
                    "block_id": block_id,
                    "asof_rule": "available_timestamp_lte_asof",
                    "max_staleness_days": 45,
                }
            feature_source = f"{feature_source}+weather"

        transfer_prior_features, transfer_priors_meta = (
            build_transfer_prior_feature_rows(
                asof=asof_ts,
                target_month=pd.Timestamp(release_context["target_month"]),
                gold_root=_default_gold_root(),
            )
        )
        if not transfer_prior_features.empty:
            features = (
                pd.concat([features, transfer_prior_features], ignore_index=True)
                .sort_values(
                    ["feature_name", "feature_timestamp", "available_timestamp"]
                )
                .reset_index(drop=True)
            )
            for feature_name in sorted(
                transfer_prior_features["feature_name"].unique().tolist()
            ):
                block_id = feature_block_map.get(feature_name, "transfer_priors")
                feature_policy["features"][feature_name] = {
                    "source_frequency": "monthly",
                    "aggregation": "last",
                    "max_age_days": 180,
                    "block_id": block_id,
                    "asof_rule": "available_timestamp_lte_asof",
                    "max_staleness_days": 180,
                }
            feature_source = f"{feature_source}+transfer_priors"

        bakerhughes_features, bakerhughes_meta = load_bakerhughes_gold_feature_rows(
            asof=asof_ts
        )
        if not bakerhughes_features.empty:
            features = (
                pd.concat([features, bakerhughes_features], ignore_index=True)
                .sort_values(
                    ["feature_name", "feature_timestamp", "available_timestamp"]
                )
                .reset_index(drop=True)
            )
            for feature_name in sorted(
                bakerhughes_features["feature_name"].unique().tolist()
            ):
                block_id = feature_block_map.get(feature_name, "oil_side")
                feature_policy["features"][feature_name] = {
                    "source_frequency": "weekly",
                    "aggregation": "last_or_rolling",
                    "max_age_days": 35,
                    "block_id": block_id,
                    "asof_rule": "available_timestamp_lte_asof",
                    "max_staleness_days": 35,
                }
            feature_source = f"{feature_source}+oil_side"

    required_cols = {
        "feature_name",
        "feature_timestamp",
        "available_timestamp",
        "block_id",
        "value",
    }
    missing_cols = sorted(required_cols - set(features.columns))
    if missing_cols:
        raise ContractViolation(
            "missing_column",
            key="features",
            detail="feature frame missing CP1 columns: " + ", ".join(missing_cols),
        )

    target_for_vintage = target_history.rename(
        columns={"timestamp": "target_timestamp"}
    )

    return {
        "asof": asof_ts,
        "target_month": pd.Timestamp(release_context["target_month"]),
        "release_policy": release_context,
        "latest_released_month": latest_released_month,
        "target_source_id": target_source.source_id,
        "target_source_path": str(target_path),
        "target_source_origin": target_origin,
        "feature_source": feature_source,
        "feature_blocks_registry_version": block_registry.version,
        "steo_gold_meta": steo_gold_meta,
        "weather_gold_meta": weather_gold_meta,
        "transfer_priors_meta": transfer_priors_meta,
        "bakerhughes_meta": bakerhughes_meta,
        "daily": daily,
        "weekly": weekly,
        "features": features,
        "feature_policy": feature_policy,
        "monthly_release_history": monthly_release_history,
        "monthly_release_history_meta": release_history_meta,
        # Legacy aliases retained while report/viz readers migrate.
        "monthly_releases_24m": monthly_release_history,
        "monthly_releases_24m_meta": release_history_meta,
        "target_history": target_history,
        "target_history_full": target_history_full,
        "target_for_vintage": target_for_vintage,
    }


def build_calibration_realized(target_history: pd.DataFrame) -> pd.DataFrame:
    """Build deterministic realized horizon values for interval calibration checks."""

    tail = target_history["target_value"].tail(3).reset_index(drop=True)
    base = float(tail.iloc[-1])
    slope = float(tail.iloc[-1] - tail.iloc[-2]) if len(tail) >= 2 else 0.0
    realized = pd.DataFrame(
        {
            "horizon": [1, 2],
            "realized_value": [
                float(base + slope),
                float(base + 2 * slope),
            ],
        }
    )
    return realized


def build_dm_runtime_frame(
    target_history: pd.DataFrame,
    *,
    target_month: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build deterministic one-layer/two-layer forecast history for DM execution."""

    sample = target_history.tail(12).copy().reset_index(drop=True)
    rows: list[dict[str, Any]] = []
    model_offsets = {
        "wpd_lstm_one_layer": 0.30,
        "wpd_vmd_lstm1": 0.18,
        "wpd_vmd_lstm2": 0.24,
    }

    for _, row in sample.iterrows():
        timestamp = pd.Timestamp(row["timestamp"])
        actual = float(row["target_value"])

        shared: dict[str, Any] = {}
        if target_month is not None:
            shared["target_month"] = pd.Timestamp(target_month).date().isoformat()

        for model_name, offset in sorted(model_offsets.items()):
            rows.append(
                {
                    "target": "ng_prod",
                    "model": model_name,
                    "asof": timestamp,
                    "horizon": 1,
                    "actual": actual,
                    "forecast": float(actual + offset),
                    **shared,
                }
            )

    return pd.DataFrame(rows)


def build_ablation_runtime_frame(
    target_history: pd.DataFrame,
    *,
    target_month: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Build deterministic ablation forecast matrix for B0..B4 experiments."""

    sample = target_history.tail(8).copy().reset_index(drop=True)
    offsets = {
        "B0_baseline": 0.90,
        "B1_plus_preprocessing": 0.65,
        "B2_plus_feature_expansion": 0.45,
        "B3_plus_challenger": 0.35,
        "B4_full_method": 0.22,
    }

    rows: list[dict[str, Any]] = []
    for experiment_id, offset in offsets.items():
        stage = list(offsets.keys()).index(experiment_id)
        runtime = float(40 + 8 * stage)
        for row_idx, row in sample.iterrows():
            timestamp = pd.Timestamp(row["timestamp"])
            actual = float(row["target_value"])
            scaled_offset = offset * (1.0 + (row_idx * 0.08))
            signed = scaled_offset if row_idx % 2 == 0 else -scaled_offset
            forecast = actual + signed
            entry: dict[str, Any] = {
                "experiment_id": experiment_id,
                "candidate_model": f"{experiment_id}_main",
                "target": "ng_prod",
                "asof": timestamp,
                "horizon": 1,
                "actual": actual,
                "forecast": float(forecast),
                "runtime_seconds": runtime,
                "lineage_id": f"{experiment_id}_lineage",
            }
            if target_month is not None:
                entry["target_month"] = pd.Timestamp(target_month).date().isoformat()
            rows.append(entry)

    return pd.DataFrame(rows)
