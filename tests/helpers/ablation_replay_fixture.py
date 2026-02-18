"""Fixture loaders and canonicalizers for Parallel_Lead Phase F ablation regression tests."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


def load_ablation_replay_frame(path: Path) -> pd.DataFrame:
    """Load deterministic ablation replay matrix fixture."""
    frame = pd.read_csv(path)
    frame["asof"] = pd.to_datetime(frame["asof"], errors="raise")
    frame["horizon"] = pd.to_numeric(frame["horizon"], errors="raise")
    frame["actual"] = pd.to_numeric(frame["actual"], errors="raise")
    frame["forecast"] = pd.to_numeric(frame["forecast"], errors="raise")
    frame["runtime_seconds"] = pd.to_numeric(frame["runtime_seconds"], errors="raise")
    ordered = frame.sort_values(
        ["experiment_id", "candidate_model", "target", "asof"],
        ascending=[True, True, True, True],
    ).reset_index(drop=True)
    return ordered


def load_expected_frame(path: Path) -> pd.DataFrame:
    """Load expected ablation output fixture from CSV."""
    return pd.read_csv(path)


def canonicalize_ablation_scorecard(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize scorecard for deterministic fixture comparison."""
    copy = frame.copy()
    numeric_cols = [
        "mae",
        "rmse",
        "mape",
        "dm_vs_baseline_p_value",
        "dm_vs_baseline_stat",
        "dm_vs_baseline_d_bar",
        "runtime_seconds",
        "selected_metric",
    ]
    for col in numeric_cols:
        if col in copy.columns:
            copy[col] = pd.to_numeric(copy[col], errors="coerce").round(10)

    for col in ("n_obs", "n_candidates"):
        if col in copy.columns:
            copy[col] = pd.to_numeric(copy[col], errors="coerce").astype(int)

    ordered = copy.sort_values(["experiment_id", "target"]).reset_index(drop=True)
    return ordered


def canonicalize_dm_results(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize DM result output for deterministic fixture comparison."""
    copy = frame.copy()
    for col in ("d_bar", "dm_stat", "p_value", "adjusted_p_value"):
        if col in copy.columns:
            copy[col] = pd.to_numeric(copy[col], errors="coerce").round(10)

    for col in ("significant_0_05", "significant_0_01"):
        if col in copy.columns:
            copy[col] = copy[col].map(
                lambda value: str(value).strip().lower() in {"1", "true", "yes"}
            )

    if "n_obs" in copy.columns:
        copy["n_obs"] = pd.to_numeric(copy["n_obs"], errors="coerce").astype(int)

    ordered = copy.sort_values(["target", "candidate_model", "benchmark_model"])
    ordered = ordered.reset_index(drop=True)
    return ordered


def canonicalize_selection_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Canonicalize ablation selection summary output."""
    copy = frame.copy()
    if "selected_metric" in copy.columns:
        copy["selected_metric"] = pd.to_numeric(
            copy["selected_metric"], errors="coerce"
        ).round(10)
    if "n_candidates" in copy.columns:
        copy["n_candidates"] = pd.to_numeric(copy["n_candidates"], errors="coerce")
        copy["n_candidates"] = copy["n_candidates"].astype(int)

    ordered = copy.sort_values(["experiment_id", "target"]).reset_index(drop=True)
    return ordered
