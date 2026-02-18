from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.ingest.adapters.nasa_power import (
    load_basin_centroids,
    retrieve_nasa_power_t2m_min,
)


class _FakeResponse:
    def __init__(self, payload: dict[str, object], status_code: int = 200) -> None:
        self._payload = payload
        self.status_code = status_code

    def json(self) -> dict[str, object]:
        return dict(self._payload)


class _FakeSession:
    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload
        self.calls: list[dict[str, object]] = []

    def get(self, url: str, **kwargs: object) -> _FakeResponse:
        self.calls.append({"url": url, **kwargs})
        return _FakeResponse(self._payload)


def _write_geojson(path: Path) -> None:
    payload = {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "properties": {"basin_id": "appalachia", "basin_name": "Appalachia"},
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [
                        [
                            [-82.0, 37.0],
                            [-80.0, 37.0],
                            [-80.0, 39.0],
                            [-82.0, 39.0],
                            [-82.0, 37.0],
                        ]
                    ],
                },
            }
        ],
    }
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_load_basin_centroids_from_geojson(tmp_path: Path) -> None:
    geojson_path = tmp_path / "basins.geojson"
    _write_geojson(geojson_path)

    centroids = load_basin_centroids(geojson_path)

    assert list(centroids.columns) == [
        "basin_id",
        "basin_name",
        "latitude",
        "longitude",
    ]
    assert centroids.loc[0, "basin_id"] == "appalachia"
    assert float(centroids.loc[0, "latitude"]) > 37.0


def test_retrieve_nasa_power_t2m_min_uses_token_and_parses_daily_payload(
    tmp_path: Path,
) -> None:
    geojson_path = tmp_path / "basins.geojson"
    _write_geojson(geojson_path)

    payload = {
        "properties": {
            "parameter": {
                "T2M_MIN": {
                    "20250101": -4.2,
                    "20250102": -3.1,
                }
            }
        }
    }
    session = _FakeSession(payload)
    frame = retrieve_nasa_power_t2m_min(
        asof="2025-01-31",
        start_date="2025-01-01",
        basin_geojson_path=geojson_path,
        token="token",
        session=session,
    )

    assert len(frame) == 2
    assert frame["source_id"].nunique() == 1
    assert frame["source_id"].iloc[0] == "nasa_power_t2m_min"
    assert frame["date"].min() == pd.Timestamp("2025-01-01")
    assert "Authorization" in session.calls[0]["headers"]  # type: ignore[index]


def test_retrieve_nasa_power_requires_token(tmp_path: Path) -> None:
    geojson_path = tmp_path / "basins.geojson"
    _write_geojson(geojson_path)

    with pytest.raises(ContractViolation, match="reason_code=missing_secret"):
        retrieve_nasa_power_t2m_min(
            asof="2025-01-31",
            start_date="2025-01-01",
            basin_geojson_path=geojson_path,
            token="",
        )
