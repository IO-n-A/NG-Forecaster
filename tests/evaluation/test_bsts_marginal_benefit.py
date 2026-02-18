from __future__ import annotations

from pathlib import Path

import pandas as pd

from ng_forecaster.evaluation.bsts_marginal_benefit import compute_bsts_marginal_benefit


def test_compute_bsts_marginal_benefit_exports_tables(tmp_path: Path) -> None:
    enabled_root = tmp_path / "enabled"
    forced_root = tmp_path / "forced"
    out_root = tmp_path / "out"
    enabled_root.mkdir(parents=True)
    forced_root.mkdir(parents=True)

    rows_enabled = []
    rows_forced = []
    for idx, target_month in enumerate(
        ["2025-08-31", "2025-09-30", "2025-10-31", "2025-11-30"], start=1
    ):
        actual = 100.0 + idx
        rows_enabled.append(
            {
                "model_variant": "wpd_vmd_lstm2",
                "asof": f"2026-0{idx}-14",
                "target_month": target_month,
                "fused_point": actual + 1.0,
                "actual_released": actual,
                "regime_label": "normal" if idx % 2 else "multi_shock",
            }
        )
        rows_forced.append(
            {
                "model_variant": "wpd_vmd_lstm2",
                "asof": f"2026-0{idx}-14",
                "target_month": target_month,
                "fused_point": actual + 2.0,
                "actual_released": actual,
                "regime_label": "normal" if idx % 2 else "multi_shock",
            }
        )

    pd.DataFrame(rows_enabled).to_csv(
        enabled_root / "validation_24m_point_estimates.csv",
        index=False,
    )
    pd.DataFrame(rows_forced).to_csv(
        forced_root / "validation_24m_point_estimates.csv",
        index=False,
    )

    result = compute_bsts_marginal_benefit(
        enabled_root=enabled_root,
        forced_off_root=forced_root,
        out_root=out_root,
    )
    overall_path = Path(result["overall_path"])
    by_regime_path = Path(result["by_regime_path"])
    assert overall_path.exists()
    assert by_regime_path.exists()

    overall = pd.read_csv(overall_path)
    assert len(overall) == 1
    row = overall.iloc[0]
    assert row["model_variant"] == "wpd_vmd_lstm2"
    assert float(row["delta_mape_pct_enabled_minus_forced_off"]) < 0.0
