from __future__ import annotations

import json
from pathlib import Path

import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.qa.target_month_gate import check_target_month_gate


def _write(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _valid_payload() -> dict[str, object]:
    return {
        "asof": "2026-02-14",
        "target_month": "2025-12-31",
        "nowcasts": [
            {
                "horizon_label": "T-1",
                "target_month_label": "target_month",
                "target_month": "2025-12-31",
                "fused_point": 102.4,
                "fused_lower_95": 100.1,
                "fused_upper_95": 104.8,
            },
            {
                "horizon_label": "T",
                "target_month_label": "target_month_plus_1",
                "target_month": "2026-01-31",
                "fused_point": 103.2,
                "fused_lower_95": 100.9,
                "fused_upper_95": 105.7,
            },
        ],
    }


def test_target_month_gate_passes_for_valid_penultimate_month_payload(
    tmp_path: Path,
) -> None:
    nowcast_json = _write(tmp_path / "nowcast.json", _valid_payload())
    result = check_target_month_gate(nowcast_json)

    assert result.passed
    assert result.target_month == "2025-12-31"
    assert result.expected_target_month == "2025-12-31"
    assert result.nowcast_count == 2


def test_target_month_gate_fails_when_root_target_month_is_wrong(
    tmp_path: Path,
) -> None:
    payload = _valid_payload()
    payload["target_month"] = "2026-01-31"
    nowcast_json = _write(tmp_path / "nowcast.json", payload)

    with pytest.raises(ContractViolation, match="reason_code=target_month_mismatch"):
        check_target_month_gate(nowcast_json)


def test_target_month_gate_fails_interval_order_violations(tmp_path: Path) -> None:
    payload = _valid_payload()
    payload["nowcasts"][1]["fused_point"] = 99.0  # type: ignore[index]
    nowcast_json = _write(tmp_path / "nowcast.json", payload)

    with pytest.raises(ContractViolation, match="reason_code=interval_order_violation"):
        check_target_month_gate(nowcast_json)
