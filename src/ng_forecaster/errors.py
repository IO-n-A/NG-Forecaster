"""Shared contract error types."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Optional


@dataclass(frozen=True)
class ContractContext:
    """Structured context carried by contract violations."""

    reason_code: str
    asof: Optional[datetime]
    key: str
    detail: str


class ContractViolation(ValueError):
    """Raised when a hard data or policy contract fails."""

    def __init__(
        self,
        reason_code: str,
        *,
        asof: date | datetime | None = None,
        key: str = "<none>",
        detail: str = "",
    ) -> None:
        resolved_asof: Optional[datetime]
        if isinstance(asof, datetime):
            resolved_asof = asof
        elif isinstance(asof, date):
            resolved_asof = datetime(asof.year, asof.month, asof.day)
        else:
            resolved_asof = None

        self.context = ContractContext(
            reason_code=reason_code,
            asof=resolved_asof,
            key=key,
            detail=detail,
        )
        asof_str = "<none>" if resolved_asof is None else resolved_asof.isoformat()
        message = (
            f"reason_code={reason_code}; asof={asof_str}; key={key}; detail={detail}"
        )
        super().__init__(message)
