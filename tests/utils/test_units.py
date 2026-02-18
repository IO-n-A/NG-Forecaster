from __future__ import annotations

import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.utils.units import (
    bcfd_to_mmcf_per_day,
    bcfd_to_mmcf_per_month,
    mmcf_per_month_to_bcfd,
)


def test_bcfd_to_mmcf_conversions_use_calendar_month_length() -> None:
    assert bcfd_to_mmcf_per_day(100.0) == 100000.0
    assert bcfd_to_mmcf_per_month(value_bcfd=100.0, month_end="2025-02-28") == 2800000.0
    assert bcfd_to_mmcf_per_month(value_bcfd=100.0, month_end="2024-02-29") == 2900000.0


def test_mmcf_per_month_to_bcfd_roundtrip() -> None:
    monthly = bcfd_to_mmcf_per_month(value_bcfd=95.5, month_end="2025-12-31")
    recovered = mmcf_per_month_to_bcfd(
        value_mmcf_per_month=monthly,
        month_end="2025-12-31",
    )
    assert recovered == pytest.approx(95.5)


def test_units_reject_invalid_inputs() -> None:
    with pytest.raises(ContractViolation, match="reason_code=invalid_timestamp"):
        bcfd_to_mmcf_per_month(value_bcfd=100.0, month_end="not-a-date")
    with pytest.raises(ContractViolation, match="reason_code=invalid_model_policy"):
        bcfd_to_mmcf_per_day(float("nan"))

