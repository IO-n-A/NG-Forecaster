"""NASA POWER retrieval adapter for CP3 freeze-off weather features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import requests

from ng_forecaster.errors import ContractViolation

_NASA_POWER_ENDPOINT = "https://power.larc.nasa.gov/api/temporal/daily/point"
_MISSING_VALUE_SENTINEL = -999.0


def _polygon_centroid(geometry: dict[str, Any]) -> tuple[float, float]:
    geom_type = str(geometry.get("type", "")).strip()
    if geom_type != "Polygon":
        raise ContractViolation(
            "source_schema_drift",
            key="dpr_basin_polygons.geometry.type",
            detail="only Polygon geometries are supported for NASA basin retrieval",
        )
    coordinates = geometry.get("coordinates")
    if not isinstance(coordinates, list) or not coordinates:
        raise ContractViolation(
            "source_schema_drift",
            key="dpr_basin_polygons.geometry.coordinates",
            detail="polygon coordinates are required",
        )
    ring = coordinates[0]
    if not isinstance(ring, list) or len(ring) < 3:
        raise ContractViolation(
            "source_schema_drift",
            key="dpr_basin_polygons.geometry.coordinates",
            detail="polygon must contain at least three points",
        )
    lons: list[float] = []
    lats: list[float] = []
    for point in ring:
        if not isinstance(point, list) or len(point) < 2:
            raise ContractViolation(
                "source_schema_drift",
                key="dpr_basin_polygons.geometry.coordinates",
                detail="invalid polygon point structure",
            )
        lons.append(float(point[0]))
        lats.append(float(point[1]))
    return float(sum(lats) / len(lats)), float(sum(lons) / len(lons))


def load_basin_centroids(path: str | Path) -> pd.DataFrame:
    """Load basin polygons and derive centroid coordinates."""

    source = Path(path)
    if not source.exists():
        raise ContractViolation(
            "missing_source_file",
            key=str(source),
            detail="basin polygon file does not exist",
        )
    payload = json.loads(source.read_text(encoding="utf-8"))
    features = payload.get("features", [])
    if not isinstance(features, list) or not features:
        raise ContractViolation(
            "source_schema_drift",
            key=str(source),
            detail="geojson must contain a non-empty features list",
        )

    rows: list[dict[str, Any]] = []
    for entry in features:
        props = dict(entry.get("properties", {}))
        geometry = dict(entry.get("geometry", {}))
        basin_id = str(props.get("basin_id", "")).strip()
        basin_name = str(props.get("basin_name", "")).strip() or basin_id
        if not basin_id:
            raise ContractViolation(
                "source_schema_drift",
                key=str(source),
                detail="every basin polygon must include properties.basin_id",
            )
        lat, lon = _polygon_centroid(geometry)
        rows.append(
            {
                "basin_id": basin_id,
                "basin_name": basin_name,
                "latitude": lat,
                "longitude": lon,
            }
        )
    return pd.DataFrame(rows).sort_values("basin_id").reset_index(drop=True)


def _fetch_point_daily_t2m_min(
    *,
    latitude: float,
    longitude: float,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    token: str,
    timeout_seconds: int,
    session: requests.Session | None = None,
) -> dict[str, float]:
    client = session or requests.Session()
    headers = {"User-Agent": "ng-forecaster-cp3/1.0"}
    if token.strip():
        headers["Authorization"] = f"Bearer {token.strip()}"

    params = {
        "parameters": "T2M_MIN",
        "community": "AG",
        "longitude": f"{longitude:.4f}",
        "latitude": f"{latitude:.4f}",
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON",
    }
    response = client.get(
        _NASA_POWER_ENDPOINT,
        params=params,
        timeout=timeout_seconds,
        headers=headers,
    )
    if response.status_code >= 400:
        raise ContractViolation(
            "http_error",
            key=_NASA_POWER_ENDPOINT,
            detail=f"status_code={response.status_code}",
        )

    payload = response.json()
    parameter = payload.get("properties", {}).get("parameter", {}).get("T2M_MIN", {})
    if not isinstance(parameter, dict) or not parameter:
        raise ContractViolation(
            "source_schema_drift",
            key="NASA_POWER.T2M_MIN",
            detail="response missing T2M_MIN payload",
        )
    return {str(key): float(value) for key, value in parameter.items()}


def retrieve_nasa_power_t2m_min(
    *,
    asof: object,
    start_date: object,
    basin_geojson_path: str | Path,
    token: str,
    timeout_seconds: int = 30,
    session: requests.Session | None = None,
) -> pd.DataFrame:
    """Retrieve NASA POWER T2M_MIN for all configured basins."""

    asof_ts = pd.Timestamp(asof)
    start_ts = pd.Timestamp(start_date)
    if pd.isna(asof_ts) or pd.isna(start_ts):
        raise ContractViolation(
            "invalid_timestamp",
            key="asof/start_date",
            detail="asof and start_date must be parseable timestamps",
        )
    if start_ts > asof_ts:
        raise ContractViolation(
            "invalid_timestamp",
            key="start_date",
            detail="start_date cannot be after asof",
        )
    if not token.strip():
        raise ContractViolation(
            "missing_secret",
            key="NASA_EARTHDATA_TOKEN",
            detail="NASA_EARTHDATA_TOKEN must be set for retrieval",
        )

    basins = load_basin_centroids(basin_geojson_path)
    rows: list[dict[str, Any]] = []
    for _, basin in basins.iterrows():
        series = _fetch_point_daily_t2m_min(
            latitude=float(basin["latitude"]),
            longitude=float(basin["longitude"]),
            start_date=start_ts,
            end_date=asof_ts,
            token=token,
            timeout_seconds=timeout_seconds,
            session=session,
        )
        if not series:
            raise ContractViolation(
                "missing_source_file",
                key=f"nasa_power:{basin['basin_id']}",
                detail="NASA POWER response had no daily temperature values",
            )
        for yyyymmdd, raw in sorted(series.items()):
            if float(raw) <= _MISSING_VALUE_SENTINEL:
                continue
            date_value = pd.to_datetime(yyyymmdd, format="%Y%m%d", errors="coerce")
            if pd.isna(date_value):
                continue
            rows.append(
                {
                    "basin_id": str(basin["basin_id"]),
                    "basin_name": str(basin["basin_name"]),
                    "latitude": float(basin["latitude"]),
                    "longitude": float(basin["longitude"]),
                    "date": pd.Timestamp(date_value).normalize(),
                    "t2m_min_c": float(raw),
                    "source_id": "nasa_power_t2m_min",
                    "available_timestamp": asof_ts.normalize(),
                    "lineage_id": (
                        "nasa_power_t2m_min:"
                        f"{basin['basin_id']}:{start_ts.date().isoformat()}:{asof_ts.date().isoformat()}"
                    ),
                }
            )
    frame = pd.DataFrame(rows)
    if frame.empty:
        raise ContractViolation(
            "missing_source_file",
            key="nasa_power_t2m_min",
            detail="retrieval produced no valid weather rows",
        )
    frame = frame.sort_values(["basin_id", "date"]).reset_index(drop=True)
    return frame
