"""Fuzzy entropy utilities aligned to Sprint 3A paper contract."""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np
import pandas as pd

from ng_forecaster.errors import ContractViolation


def _as_array(values: Sequence[float] | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ContractViolation(
            "insufficient_training_data",
            key="fuzzy_entropy.signal",
            detail="signal must be non-empty",
        )
    if not np.isfinite(array).all():
        raise ContractViolation(
            "invalid_model_policy",
            key="fuzzy_entropy.signal",
            detail="signal contains non-finite values",
        )
    return array


def _resolve_r(
    signal: np.ndarray,
    *,
    r: float | None,
    r_policy: str,
) -> float:
    if r is not None:
        resolved = float(r)
    else:
        policy = str(r_policy).strip().lower()
        if policy.startswith("std_"):
            try:
                factor = float(policy.split("_", 1)[1])
            except Exception as exc:  # pragma: no cover - parse guard
                raise ContractViolation(
                    "invalid_model_policy",
                    key="fuzzy_entropy.r_policy",
                    detail=f"unable to parse r_policy={r_policy}",
                ) from exc
            resolved = float(np.std(signal, ddof=0) * factor)
        elif policy == "std":
            resolved = float(np.std(signal, ddof=0) * 0.2)
        else:
            raise ContractViolation(
                "invalid_model_policy",
                key="fuzzy_entropy.r_policy",
                detail="supported r_policy values: std or std_<factor>",
            )

    if resolved <= 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="fuzzy_entropy.r",
            detail=f"r must be positive; received={resolved}",
        )
    return resolved


def _build_embedding(signal: np.ndarray, m: int) -> np.ndarray:
    n_obs = int(signal.size)
    count = n_obs - m + 1
    if count < 2:
        raise ContractViolation(
            "insufficient_training_data",
            key="fuzzy_entropy.embedding",
            detail=f"need at least m+1 points for m={m}; received={n_obs}",
        )
    return np.vstack([signal[idx : idx + m] for idx in range(count)])


def _phi(signal: np.ndarray, *, m: int, n: float, r: float) -> float:
    vectors = _build_embedding(signal, m)
    n_vec = int(vectors.shape[0])
    # Pairwise max absolute distance d_ij^m
    max_abs_distance = np.max(
        np.abs(vectors[:, None, :] - vectors[None, :, :]),
        axis=2,
    )
    similarity = np.exp(-math.log(2.0) * np.power(max_abs_distance / r, n))
    # Exclude self-similarity on diagonal.
    np.fill_diagonal(similarity, 0.0)
    denom = max(1, n_vec - 1)
    phi_i = similarity.sum(axis=1) / float(denom)
    return float(phi_i.mean())


def fuzzy_entropy(
    values: Sequence[float] | pd.Series,
    *,
    m: int = 1,
    n: float = 2.0,
    r: float | None = None,
    r_policy: str = "std_0.2",
) -> float:
    """Compute fuzzy entropy SFE = ln(phi^m) - ln(phi^(m+1))."""

    signal = _as_array(values)
    if int(m) < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="fuzzy_entropy.m",
            detail="m must be >= 1",
        )
    if float(n) <= 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="fuzzy_entropy.n",
            detail="n must be > 0",
        )

    resolved_r = _resolve_r(signal, r=r, r_policy=r_policy)
    phi_m = _phi(signal, m=int(m), n=float(n), r=resolved_r)
    phi_m1 = _phi(signal, m=int(m) + 1, n=float(n), r=resolved_r)

    eps = 1e-12
    if phi_m <= eps or phi_m1 <= eps:
        raise ContractViolation(
            "invalid_model_policy",
            key="fuzzy_entropy.phi",
            detail="phi values must be positive for log computation",
        )
    return float(math.log(phi_m) - math.log(phi_m1))


def rank_components_by_entropy(
    component_frame: pd.DataFrame,
    *,
    m: int = 1,
    n: float = 2.0,
    r_policy: str = "std_0.2",
) -> pd.DataFrame:
    """Rank component columns by fuzzy entropy in deterministic order."""

    if component_frame.empty:
        raise ContractViolation(
            "insufficient_training_data",
            key="fuzzy_entropy.components",
            detail="component_frame must be non-empty",
        )

    rows: list[dict[str, float | int | str]] = []
    for name in sorted(component_frame.columns):
        score = fuzzy_entropy(
            component_frame[name],
            m=int(m),
            n=float(n),
            r_policy=r_policy,
        )
        rows.append({"component_name": str(name), "fuzzy_entropy": float(score)})

    ranked = (
        pd.DataFrame(rows)
        .sort_values(["fuzzy_entropy", "component_name"], ascending=[False, True])
        .reset_index(drop=True)
    )
    ranked["entropy_rank"] = ranked.index + 1
    mean_score = float(ranked["fuzzy_entropy"].mean())
    ranked["complexity_class"] = np.where(
        ranked["fuzzy_entropy"] > mean_score,
        "high_complexity",
        "low_complexity",
    )
    return ranked
