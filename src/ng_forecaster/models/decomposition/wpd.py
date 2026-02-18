"""Wavelet packet decomposition helpers for monthly champion modeling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import pandas as pd

from ng_forecaster.errors import ContractViolation

try:
    import pywt  # type: ignore[import-untyped]
except Exception:  # pragma: no cover - optional dependency in constrained envs
    pywt = None


@dataclass(frozen=True)
class WPDResult:
    """WPD decomposition payload with reconstructed component diagnostics."""

    components: pd.DataFrame
    node_paths: tuple[str, ...]
    reconstruction_rmse: float
    reconstruction_relative_l2: float
    energy: pd.DataFrame


def _resolve_wavelet_name(*, family: str, order: int) -> str:
    family_norm = str(family).strip().lower()
    if family_norm != "db":
        raise ContractViolation(
            "invalid_model_policy",
            key="wpd.wavelet_family",
            detail="only Daubechies family ('db') is supported",
        )
    if int(order) < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="wpd.wavelet_order",
            detail="wavelet_order must be >= 1",
        )
    return f"db{int(order)}"


def _series_to_array(values: Sequence[float] | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ContractViolation(
            "insufficient_training_data",
            key="wpd.signal",
            detail="signal must contain at least one observation",
        )
    if not np.isfinite(array).all():
        raise ContractViolation(
            "invalid_model_policy",
            key="wpd.signal",
            detail="signal contains non-finite values",
        )
    return array


def _fallback_packet_components(
    signal: np.ndarray,
    *,
    levels: int,
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    # Deterministic fallback when PyWavelets is unavailable.
    components = [signal.copy()]
    paths = [""]
    for _ in range(int(levels)):
        next_components: list[np.ndarray] = []
        next_paths: list[str] = []
        for component, path in zip(components, paths):
            trend = (
                pd.Series(component)
                .rolling(window=2, min_periods=1)
                .mean()
                .to_numpy(dtype=float)
            )
            detail = component - trend
            next_components.extend([trend, detail])
            next_paths.extend([path + "a", path + "d"])
        components = next_components
        paths = next_paths
    named = {
        f"PF{idx + 1}": values.copy() for idx, values in enumerate(components, start=0)
    }
    return named, tuple(paths)


def decompose_wpd_signal(
    values: Sequence[float] | pd.Series,
    *,
    wavelet_family: str = "db",
    wavelet_order: int = 5,
    levels: int = 3,
    mode: str = "symmetric",
    component_prefix: str = "PF",
    reconstruction_tolerance: float = 1e-4,
) -> WPDResult:
    """Decompose a 1D signal with WPD and reconstruct PF components."""

    if int(levels) < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="wpd.levels",
            detail="levels must be >= 1",
        )

    signal = _series_to_array(values)
    wavelet_name = _resolve_wavelet_name(family=wavelet_family, order=wavelet_order)
    component_data: dict[str, np.ndarray]
    node_paths: tuple[str, ...]
    if pywt is None:
        component_data, node_paths = _fallback_packet_components(
            signal,
            levels=int(levels),
        )
    else:
        try:
            wavelet = pywt.Wavelet(wavelet_name)
        except Exception as exc:  # pragma: no cover - pywt validation internals
            raise ContractViolation(
                "invalid_model_policy",
                key="wpd.wavelet_order",
                detail=f"invalid Daubechies wavelet selection: {wavelet_name}",
            ) from exc

        packet = pywt.WaveletPacket(
            data=signal,
            wavelet=wavelet,
            mode=str(mode),
            maxlevel=int(levels),
        )
        level_nodes = packet.get_level(int(levels), order="freq")
        expected_nodes = 2 ** int(levels)
        if len(level_nodes) != expected_nodes:
            raise ContractViolation(
                "wpd_node_count_mismatch",
                key="wpd.levels",
                detail=(
                    f"expected {expected_nodes} terminal nodes at level={levels}, "
                    f"received={len(level_nodes)}"
                ),
            )

        component_data = {}
        node_paths_list: list[str] = []
        for idx, node in enumerate(level_nodes, start=1):
            isolated = pywt.WaveletPacket(
                data=None,
                wavelet=wavelet,
                mode=str(mode),
                maxlevel=int(levels),
            )
            isolated[node.path] = node.data
            reconstructed = np.asarray(isolated.reconstruct(update=False), dtype=float)
            reconstructed = reconstructed[: len(signal)]
            if len(reconstructed) < len(signal):
                reconstructed = np.pad(
                    reconstructed,
                    (0, len(signal) - len(reconstructed)),
                    mode="constant",
                )
            component_name = f"PF{idx}"
            component_data[component_name] = reconstructed
            node_paths_list.append(str(node.path))
        node_paths = tuple(node_paths_list)

    if component_prefix != "PF":
        component_data = {
            f"{component_prefix}{idx}": component_data[f"PF{idx}"]
            for idx in range(1, len(component_data) + 1)
        }

    components = pd.DataFrame(component_data)
    reconstructed_sum = components.sum(axis=1).to_numpy(dtype=float)
    residual = signal - reconstructed_sum
    rmse = float(np.sqrt(np.mean(np.square(residual))))
    denom = float(np.linalg.norm(signal))
    relative_l2 = float(np.linalg.norm(residual) / (denom if denom > 1e-12 else 1.0))

    if relative_l2 > float(reconstruction_tolerance):
        raise ContractViolation(
            "wpd_reconstruction_error",
            key="wpd.reconstruction_tolerance",
            detail=(
                f"relative_l2={relative_l2:.6f} exceeds "
                f"tolerance={float(reconstruction_tolerance):.6f}"
            ),
        )

    energy_rows = [
        {
            "component_name": name,
            "energy": float(np.sum(np.square(series.to_numpy(dtype=float)))),
        }
        for name, series in components.items()
    ]
    energy = (
        pd.DataFrame(energy_rows).sort_values("component_name").reset_index(drop=True)
    )

    return WPDResult(
        components=components,
        node_paths=node_paths,
        reconstruction_rmse=rmse,
        reconstruction_relative_l2=relative_l2,
        energy=energy,
    )
