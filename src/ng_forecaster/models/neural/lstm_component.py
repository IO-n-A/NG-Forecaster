"""Deterministic component-level LSTM forecaster with repeat-run diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ng_forecaster.errors import ContractViolation


def _sigmoid(values: np.ndarray) -> np.ndarray:
    return np.asarray(1.0 / (1.0 + np.exp(-values)), dtype=float)


def _as_numeric_series(values: Sequence[float] | pd.Series) -> np.ndarray:
    array = np.asarray(values, dtype=float).reshape(-1)
    if array.size == 0:
        raise ContractViolation(
            "insufficient_training_data",
            key="component_values",
            detail="component series must contain at least one value",
        )
    if not np.isfinite(array).all():
        raise ContractViolation(
            "invalid_model_policy",
            key="component_values",
            detail="component series contains non-finite values",
        )
    return array


def _validate_lstm_config(config: Mapping[str, Any]) -> dict[str, int | float]:
    lookback = int(config.get("lookback", 36))
    hidden_units = int(config.get("hidden_units", 32))
    batch_size = int(config.get("batch_size", 16))
    learning_rate = float(config.get("learning_rate", 0.001))
    repeat_runs = int(config.get("repeat_runs", 5))

    if lookback < 8:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm.lookback",
            detail="lookback must be >= 8",
        )
    if hidden_units < 2:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm.hidden_units",
            detail="hidden_units must be >= 2",
        )
    if batch_size < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm.batch_size",
            detail="batch_size must be >= 1",
        )
    if learning_rate <= 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm.learning_rate",
            detail="learning_rate must be > 0",
        )
    if repeat_runs < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm.repeat_runs",
            detail="repeat_runs must be >= 1",
        )

    return {
        "lookback": lookback,
        "hidden_units": hidden_units,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "repeat_runs": repeat_runs,
    }


def _normalize(array: np.ndarray) -> tuple[np.ndarray, float, float]:
    mean = float(np.mean(array))
    std = float(np.std(array))
    if std <= 1e-12:
        std = 1.0
    normalized = (array - mean) / std
    return normalized, mean, std


def _build_supervised_windows(
    normalized: np.ndarray,
    *,
    lookback: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_samples = int(len(normalized) - lookback)
    if n_samples < 2:
        raise ContractViolation(
            "insufficient_training_data",
            key="lstm.lookback",
            detail=(
                "component LSTM requires at least lookback+2 points; "
                f"received={len(normalized)} lookback={lookback}"
            ),
        )
    x = np.stack(
        [normalized[idx : idx + lookback] for idx in range(n_samples)],
        axis=0,
    )
    y = np.asarray(
        [normalized[idx + lookback] for idx in range(n_samples)],
        dtype=float,
    )
    return x[:, :, np.newaxis], y


@dataclass(frozen=True)
class _LSTMParams:
    w_i: np.ndarray
    u_i: np.ndarray
    b_i: np.ndarray
    w_f: np.ndarray
    u_f: np.ndarray
    b_f: np.ndarray
    w_o: np.ndarray
    u_o: np.ndarray
    b_o: np.ndarray
    w_g: np.ndarray
    u_g: np.ndarray
    b_g: np.ndarray


def _init_lstm_params(
    *,
    hidden_units: int,
    rng: np.random.Generator,
    scale: float = 0.2,
) -> _LSTMParams:
    def _weights() -> np.ndarray:
        return rng.normal(0.0, scale, size=(hidden_units, 1))

    def _recurrent() -> np.ndarray:
        return rng.normal(0.0, scale, size=(hidden_units, hidden_units))

    def _bias() -> np.ndarray:
        return np.zeros(hidden_units, dtype=float)

    return _LSTMParams(
        w_i=_weights(),
        u_i=_recurrent(),
        b_i=_bias(),
        w_f=_weights(),
        u_f=_recurrent(),
        b_f=_bias(),
        w_o=_weights(),
        u_o=_recurrent(),
        b_o=_bias(),
        w_g=_weights(),
        u_g=_recurrent(),
        b_g=_bias(),
    )


def _lstm_encode_window(window: np.ndarray, params: _LSTMParams) -> np.ndarray:
    h_state = np.zeros(params.b_i.shape[0], dtype=float)
    c_state = np.zeros(params.b_i.shape[0], dtype=float)

    for x_value in window.reshape(-1):
        x_vec = np.array([float(x_value)], dtype=float)
        i_t = _sigmoid(params.w_i @ x_vec + params.u_i @ h_state + params.b_i)
        f_t = _sigmoid(params.w_f @ x_vec + params.u_f @ h_state + params.b_f)
        o_t = _sigmoid(params.w_o @ x_vec + params.u_o @ h_state + params.b_o)
        g_t = np.tanh(params.w_g @ x_vec + params.u_g @ h_state + params.b_g)
        c_state = f_t * c_state + i_t * g_t
        h_state = o_t * np.tanh(c_state)

    return h_state


def _encode_batch(windows: np.ndarray, params: _LSTMParams) -> np.ndarray:
    states = [_lstm_encode_window(window, params) for window in windows]
    return np.vstack(states)


def _fit_ridge_readout(
    encoded: np.ndarray,
    targets: np.ndarray,
    *,
    ridge: float = 1e-2,
) -> tuple[np.ndarray, float]:
    design = np.column_stack([encoded, np.ones(len(encoded), dtype=float)])
    gram = design.T @ design
    penalty = np.eye(gram.shape[0], dtype=float)
    penalty[-1, -1] = 0.0  # do not penalize intercept
    weights = np.linalg.pinv(gram + ridge * penalty) @ (design.T @ targets)
    return weights[:-1], float(weights[-1])


def _recursive_forecast(
    history: np.ndarray,
    *,
    horizons: Sequence[int],
    lookback: int,
    params: _LSTMParams,
    readout_weights: np.ndarray,
    readout_bias: float,
) -> dict[int, float]:
    if not horizons:
        raise ContractViolation(
            "invalid_model_policy",
            key="forecast.horizons",
            detail="at least one horizon is required",
        )

    horizon_set = sorted({int(h) for h in horizons})
    if horizon_set[0] < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="forecast.horizons",
            detail="horizons must be positive integers",
        )

    state_history = history.tolist()
    predictions: dict[int, float] = {}
    for step in range(1, horizon_set[-1] + 1):
        window = np.asarray(state_history[-lookback:], dtype=float).reshape(lookback, 1)
        encoded = _lstm_encode_window(window, params)
        pred = float(encoded @ readout_weights + readout_bias)
        state_history.append(pred)
        if step in horizon_set:
            predictions[step] = pred

    return predictions


@dataclass(frozen=True)
class ComponentLSTMForecast:
    """Component forecast payload with repeat-run diagnostics."""

    point_forecast: pd.DataFrame
    run_forecasts: pd.DataFrame
    diagnostics: dict[str, Any]


def forecast_component_with_lstm(
    values: Sequence[float] | pd.Series,
    *,
    horizons: Sequence[int],
    lstm_config: Mapping[str, Any],
    component_name: str,
    base_seed: int = 42,
) -> ComponentLSTMForecast:
    """Forecast one decomposed component using a deterministic LSTM encoder."""

    cfg = _validate_lstm_config(lstm_config)
    raw = _as_numeric_series(values)
    normalized, mean, std = _normalize(raw)

    lookback = int(cfg["lookback"])
    windows, targets = _build_supervised_windows(normalized, lookback=lookback)

    run_rows: list[dict[str, Any]] = []
    readout_errors: list[float] = []

    for run_index in range(int(cfg["repeat_runs"])):
        run_seed = int(base_seed) + run_index
        seed_material = f"{component_name}|{run_index}".encode("utf-8")
        param_seed = int(hashlib.sha256(seed_material).hexdigest()[:8], 16)
        rng = np.random.default_rng(param_seed)
        params = _init_lstm_params(hidden_units=int(cfg["hidden_units"]), rng=rng)
        encoded = _encode_batch(windows, params)

        readout_weights, readout_bias = _fit_ridge_readout(encoded, targets)
        fitted = encoded @ readout_weights + readout_bias
        readout_rmse = float(np.sqrt(np.mean(np.square(targets - fitted))))
        readout_errors.append(readout_rmse)

        preds_norm = _recursive_forecast(
            normalized,
            horizons=horizons,
            lookback=lookback,
            params=params,
            readout_weights=readout_weights,
            readout_bias=readout_bias,
        )
        for horizon, pred_norm in sorted(preds_norm.items()):
            run_rows.append(
                {
                    "component_name": component_name,
                    "horizon": int(horizon),
                    "run_index": run_index,
                    "seed": run_seed,
                    "point_forecast": float(pred_norm * std + mean),
                }
            )

    run_forecasts = (
        pd.DataFrame(run_rows)
        .sort_values(["horizon", "run_index"])
        .reset_index(drop=True)
    )
    point_forecast = (
        run_forecasts.groupby("horizon", sort=True)["point_forecast"]
        .mean()
        .reset_index()
    )

    dispersion = (
        run_forecasts.groupby("horizon", sort=True)["point_forecast"]
        .agg(["std", "min", "max"])
        .reset_index()
    )
    dispersion["std"] = dispersion["std"].fillna(0.0)
    dispersion["spread"] = dispersion["max"] - dispersion["min"]

    diagnostics: dict[str, Any] = {
        "component_name": component_name,
        "input_shape": [lookback, 1],
        "training_windows": int(len(windows)),
        "normalization": {
            "mean": mean,
            "std": std,
        },
        "repeat_runs": int(cfg["repeat_runs"]),
        "seed_start": int(base_seed),
        "readout_rmse_mean": float(np.mean(readout_errors)),
        "dispersion_by_horizon": {
            str(int(row["horizon"])): {
                "std": float(row["std"]),
                "spread": float(row["spread"]),
            }
            for _, row in dispersion.iterrows()
        },
    }

    return ComponentLSTMForecast(
        point_forecast=point_forecast,
        run_forecasts=run_forecasts,
        diagnostics=diagnostics,
    )
