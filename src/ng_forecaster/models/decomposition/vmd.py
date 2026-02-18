"""Deterministic VMD-style decomposition wrappers and k-selection policy."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ng_forecaster.errors import ContractViolation


@dataclass(frozen=True)
class VMDDecompositionResult:
    """VMD-style decomposition payload for a single signal."""

    modes: pd.DataFrame
    mode_energy: pd.DataFrame


@dataclass(frozen=True)
class VMDKSelectionResult:
    """Energy-change search payload for selecting k."""

    selected_k: int
    search_table: pd.DataFrame


def _as_array(values: Sequence[float] | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size < 2:
        raise ContractViolation(
            "insufficient_training_data",
            key="vmd.signal",
            detail="signal must contain at least two points",
        )
    if not np.isfinite(array).all():
        raise ContractViolation(
            "invalid_model_policy",
            key="vmd.signal",
            detail="signal contains non-finite values",
        )
    return array


def _fft_band_modes(signal: np.ndarray, *, k: int) -> list[np.ndarray]:
    spectrum = np.fft.rfft(signal)
    n_freq = int(len(spectrum))
    boundaries = np.linspace(0, n_freq, num=int(k) + 1, dtype=int)

    modes: list[np.ndarray] = []
    for idx in range(int(k)):
        start = int(boundaries[idx])
        stop = int(boundaries[idx + 1])
        band = np.zeros_like(spectrum)
        band[start:stop] = spectrum[start:stop]
        mode = np.fft.irfft(band, n=len(signal)).real
        modes.append(mode)
    return modes


def decompose_vmd_signal(
    values: Sequence[float] | pd.Series,
    *,
    k: int,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-7,
) -> VMDDecompositionResult:
    """Decompose a signal into k frequency-band modes."""

    _ = alpha
    _ = tau
    _ = tol
    if int(k) < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="vmd.k",
            detail="k must be >= 1",
        )

    signal = _as_array(values)
    modes = _fft_band_modes(signal, k=int(k))
    if len(modes) != int(k):
        raise ContractViolation(
            "invalid_model_policy",
            key="vmd.k",
            detail=f"expected {k} modes but computed {len(modes)}",
        )

    mode_columns = {f"mode_{idx + 1}": mode for idx, mode in enumerate(modes)}
    mode_frame = pd.DataFrame(mode_columns)
    summed = mode_frame.sum(axis=1).to_numpy(dtype=float)
    residual = signal - summed
    relative_l2 = float(
        np.linalg.norm(residual)
        / (np.linalg.norm(signal) if np.linalg.norm(signal) > 1e-12 else 1.0)
    )
    if relative_l2 > 1e-4:
        raise ContractViolation(
            "vmd_reconstruction_error",
            key="vmd.k",
            detail=f"relative reconstruction error too high: {relative_l2:.6f}",
        )

    energy_rows = [
        {
            "mode_name": name,
            "energy": float(np.sum(np.square(mode_frame[name].to_numpy(dtype=float)))),
        }
        for name in mode_frame.columns
    ]
    energy_frame = (
        pd.DataFrame(energy_rows).sort_values("mode_name").reset_index(drop=True)
    )
    return VMDDecompositionResult(modes=mode_frame, mode_energy=energy_frame)


def select_vmd_k_by_energy_change(
    values: Sequence[float] | pd.Series,
    *,
    k_min: int,
    k_max: int,
    alpha: float = 2000.0,
    tau: float = 0.0,
    tol: float = 1e-7,
    theta_threshold: float = 0.05,
) -> VMDKSelectionResult:
    """Select k by theta_{k,k-1}=|E_k-E_{k-1}|/E_{k-1} criterion."""

    if int(k_min) < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="vmd.k_min",
            detail="k_min must be >= 1",
        )
    if int(k_max) < int(k_min):
        raise ContractViolation(
            "invalid_model_policy",
            key="vmd.k_max",
            detail="k_max must be >= k_min",
        )
    if float(theta_threshold) <= 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="vmd.theta_threshold",
            detail="theta_threshold must be > 0",
        )

    signal = _as_array(values)
    rows: list[dict[str, float | int]] = []
    prev_energy: float | None = None

    for k in range(int(k_min), int(k_max) + 1):
        decomposition = decompose_vmd_signal(
            signal,
            k=int(k),
            alpha=float(alpha),
            tau=float(tau),
            tol=float(tol),
        )
        # Use highest-frequency mode energy as E_k proxy for change tracking.
        energy_values = decomposition.mode_energy["energy"].to_numpy(dtype=float)
        e_k = float(energy_values[-1])
        theta = (
            np.nan
            if prev_energy is None
            else abs(e_k - prev_energy) / max(abs(prev_energy), 1e-12)
        )
        rows.append(
            {
                "k": int(k),
                "energy": e_k,
                "theta_k_k_minus_1": float(theta) if np.isfinite(theta) else np.nan,
            }
        )
        prev_energy = e_k

    table = pd.DataFrame(rows).sort_values("k").reset_index(drop=True)
    finite_theta = table["theta_k_k_minus_1"].replace([np.inf, -np.inf], np.nan)
    candidate = table[
        finite_theta.notna() & (finite_theta.astype(float) <= float(theta_threshold))
    ]
    if not candidate.empty:
        selected_k = int(candidate.iloc[0]["k"])
    else:
        fallback = table[finite_theta.notna()].copy()
        if fallback.empty:
            selected_k = int(k_min)
        else:
            selected_k = int(
                fallback.sort_values(["theta_k_k_minus_1", "k"]).iloc[0]["k"]
            )

    return VMDKSelectionResult(selected_k=selected_k, search_table=table)
