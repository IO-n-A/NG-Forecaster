from __future__ import annotations

from pathlib import Path

import pandas as pd

import ng_forecaster.evaluation.lookback_sweep as lookback_sweep


def _value_for_month(month: pd.Timestamp) -> float:
    month_end = pd.Timestamp(month).to_period("M").to_timestamp("M")
    serial = (month_end.year - 2024) * 12 + month_end.month
    return float(3000.0 + serial)


def test_run_lookback_sweep_exports_scorecard_dm_and_winner(
    monkeypatch,
    tmp_path: Path,
) -> None:
    def _fake_load_market_inputs(asof: object) -> dict[str, pd.DataFrame]:
        _ = asof
        months = pd.date_range("2023-01-31", periods=72, freq="ME")
        history = pd.DataFrame(
            {
                "timestamp": months,
                "target_value": [_value_for_month(ts) for ts in months],
            }
        )
        return {
            "target_history_full": history,
        }

    def _fake_run_nowcast_pipeline_weekly(
        *,
        asof: str | None = None,
        champion_config_override: dict[str, object] | None = None,
        fusion_config_override: dict[str, object] | None = None,
        idempotency_token: str | None = None,
    ) -> dict[str, object]:
        _ = (fusion_config_override, idempotency_token)
        asof_value = pd.Timestamp(asof or "2026-01-14")
        target_month = (asof_value.to_period("M") - 2).to_timestamp("M")

        config = champion_config_override or {}
        lstm_cfg = config.get("lstm", {})
        lookback = int(lstm_cfg.get("lookback", 36)) if isinstance(lstm_cfg, dict) else 36
        bias = abs(lookback - 36) * 0.4

        rows = []
        for horizon in (1, 2):
            month = (target_month.to_period("M") + horizon - 1).to_timestamp("M")
            actual = _value_for_month(month)
            rows.append(
                {
                    "horizon": int(horizon),
                    "fused_point": float(actual + bias + horizon * 0.05),
                }
            )

        artifact_dir = Path("data/artifacts/nowcast") / asof_value.date().isoformat()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_csv(artifact_dir / "fusion_inputs.csv", index=False)
        return {"target_month": target_month.date().isoformat()}

    monkeypatch.setattr(lookback_sweep, "load_market_inputs", _fake_load_market_inputs)
    monkeypatch.setattr(
        lookback_sweep,
        "run_nowcast_pipeline_weekly",
        _fake_run_nowcast_pipeline_weekly,
    )

    report_root = tmp_path / "reports"
    result = lookback_sweep.run_lookback_sweep(
        end_target_month="2025-11-30",
        lookbacks=[24, 36],
        replay_months=2,
        horizons=[1, 2],
        n_trials=2,
        dm_baseline_lookback=36,
        report_root=report_root,
        promote_winner=False,
    )

    assert not result.scorecard.empty
    assert not result.errors.empty
    assert not result.dm_results.empty
    assert result.winner["selected_lookback"] == 36
    assert "hac_lag_used" in result.dm_results.columns
    assert {"ape_pct", "alr"}.issubset(set(result.errors.columns))

    assert (report_root / "lookback_sweep_scorecard.csv").exists()
    assert (report_root / "lookback_sweep_errors.parquet").exists()
    assert (report_root / "lookback_sweep_dm.csv").exists()
    assert (report_root / "lookback_winner.json").exists()

