from __future__ import annotations

import pandas as pd
import pytest

import ng_forecaster.orchestration.airflow.workflow_support as workflow_support
from ng_forecaster.errors import ContractViolation


def test_trim_target_history_to_latest_release_enforces_month_cutoff() -> None:
    history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2025-08-31", periods=6, freq="ME"),
            "target_value": [100.0, 101.0, 102.0, 103.0, 104.0, 105.0],
        }
    )

    trimmed = workflow_support.trim_target_history_to_latest_release(
        history,
        latest_released_month=pd.Timestamp("2025-11-30"),
    )
    assert trimmed["timestamp"].max().date().isoformat() == "2025-11-30"
    assert len(trimmed) == 4


def test_resolve_monthly_release_history_falls_back_to_released_target_history(
    monkeypatch,
) -> None:
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-31", periods=48, freq="ME"),
            "target_value": [3000.0 + idx for idx in range(48)],
        }
    )

    release_history, meta = workflow_support.resolve_monthly_release_history(
        history,
        asof=pd.Timestamp("2026-02-14"),
        lookback_months=36,
    )

    assert len(release_history) == 36
    assert meta["source"] == "target_history_fallback"
    assert meta["required_row_count"] == 36
    assert release_history["source"].eq("target_history_fallback").all()


def test_resolve_monthly_release_history_prefers_api_payload(
    monkeypatch,
) -> None:
    monkeypatch.setenv("EIA_API_KEY", "dummy")

    api_frame = pd.DataFrame(
        {
            "timestamp": pd.date_range("2023-01-31", periods=36, freq="ME"),
            "target_value": [3100.0 + idx for idx in range(36)],
            "series_id": ["NG.N9070US2.M"] * 36,
            "source": ["eia_api_v2"] * 36,
        }
    )
    monkeypatch.setattr(
        workflow_support,
        "fetch_eia_monthly_series_history",
        lambda **kwargs: api_frame,
    )

    history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2020-01-31", periods=72, freq="ME"),
            "target_value": [2800.0 + idx for idx in range(72)],
        }
    )

    release_history, meta = workflow_support.resolve_monthly_release_history(
        history,
        asof=pd.Timestamp("2026-02-14"),
        lookback_months=36,
    )

    assert len(release_history) == 36
    assert meta["source"] == "eia_api_v2"
    assert meta["required_row_count"] == 36
    assert release_history["source"].eq("eia_api_v2").all()


def test_resolve_monthly_release_history_requires_minimum_rows(
    monkeypatch,
) -> None:
    monkeypatch.delenv("EIA_API_KEY", raising=False)
    history = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-31", periods=12, freq="ME"),
            "target_value": [3000.0 + idx for idx in range(12)],
        }
    )

    with pytest.raises(ContractViolation) as exc:
        workflow_support.resolve_monthly_release_history(
            history,
            asof=pd.Timestamp("2026-02-14"),
            lookback_months=36,
        )

    message = str(exc.value)
    assert "insufficient_release_history" in message
    assert "required at least 36 monthly points" in message


def test_load_market_inputs_avoids_future_dated_fixture_features() -> None:
    payload = workflow_support.load_market_inputs(pd.Timestamp("2024-02-14"))
    assert str(payload["feature_source"]).startswith("target_history_derived")
