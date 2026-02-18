"""Kernel utilities for the prototype cohort forecaster."""

from __future__ import annotations

import numpy as np

from ng_forecaster.errors import ContractViolation


def exponential_decay_kernel(*, horizon: int, half_life_months: float) -> np.ndarray:
    """Return monotone, positive decay weights for steps 1..horizon."""

    parsed_horizon = int(horizon)
    parsed_half_life = float(half_life_months)
    if parsed_horizon < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="horizon",
            detail="horizon must be >= 1",
        )
    if parsed_half_life <= 0.0:
        raise ContractViolation(
            "invalid_model_policy",
            key="half_life_months",
            detail="half_life_months must be > 0",
        )

    steps = np.arange(parsed_horizon, dtype=float)
    weights = np.power(0.5, steps / parsed_half_life)
    weights[0] = 1.0
    return weights


def cumulative_kernel_weight(*, horizon: int, half_life_months: float) -> float:
    """Return cumulative impulse impact up to a horizon."""

    return float(
        exponential_decay_kernel(
            horizon=int(horizon),
            half_life_months=float(half_life_months),
        ).sum()
    )


def bounded_value(value: float, *, lower: float, upper: float, key: str) -> float:
    """Clamp value to [lower, upper] with policy validation."""

    lower_value = float(lower)
    upper_value = float(upper)
    if lower_value > upper_value:
        raise ContractViolation(
            "invalid_model_policy",
            key=key,
            detail="lower bound cannot exceed upper bound",
        )
    return float(min(max(float(value), lower_value), upper_value))
