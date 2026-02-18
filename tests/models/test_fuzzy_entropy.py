from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from ng_forecaster.errors import ContractViolation
from ng_forecaster.models.decomposition.fuzzy_entropy import (
    fuzzy_entropy,
    rank_components_by_entropy,
)


def test_fuzzy_entropy_scores_higher_for_more_irregular_series() -> None:
    rng = np.random.default_rng(42)
    smooth = np.sin(np.linspace(0.0, 2.0 * np.pi, 64))
    irregular = rng.normal(0.0, 1.0, size=64)

    smooth_score = fuzzy_entropy(smooth, m=1, n=2.0, r_policy="std_0.2")
    irregular_score = fuzzy_entropy(irregular, m=1, n=2.0, r_policy="std_0.2")

    assert irregular_score > smooth_score


def test_rank_components_by_entropy_is_deterministic() -> None:
    frame = pd.DataFrame(
        {
            "PF1": np.sin(np.linspace(0.0, np.pi, 48)),
            "PF2": np.cos(np.linspace(0.0, 2.0 * np.pi, 48)),
            "PF3": np.linspace(0.0, 1.0, 48),
        }
    )
    first = rank_components_by_entropy(frame, m=1, n=2.0, r_policy="std_0.2")
    second = rank_components_by_entropy(frame, m=1, n=2.0, r_policy="std_0.2")

    assert first.equals(second)
    assert set(first.columns) == {
        "component_name",
        "fuzzy_entropy",
        "entropy_rank",
        "complexity_class",
    }


def test_fuzzy_entropy_rejects_invalid_embedding_dimension() -> None:
    with pytest.raises(ContractViolation, match="reason_code=invalid_model_policy"):
        fuzzy_entropy([1.0, 2.0, 3.0], m=0)
