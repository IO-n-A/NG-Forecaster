"""Real PyTorch LSTM training loop with dropout and early stopping."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from ng_forecaster.errors import ContractViolation
from ng_forecaster.models.neural.dataset import build_lstm_supervised_dataset
from ng_forecaster.models.neural.lstm_component import ComponentLSTMForecast
from ng_forecaster.models.neural.lstm_model import ComponentLSTMRegressor

try:
    import torch
    from torch.utils.data import DataLoader, TensorDataset
except Exception:  # pragma: no cover - optional dependency
    torch = None
    DataLoader = None
    TensorDataset = None


def _ensure_torch_available() -> None:
    if torch is None or DataLoader is None or TensorDataset is None:
        raise ContractViolation(
            "missing_dependency",
            key="torch",
            detail=(
                "PyTorch is required for real LSTM training path; "
                "install torch or switch engine to deterministic"
            ),
        )


def _validate_config(config: Mapping[str, Any]) -> dict[str, Any]:
    lookback = int(config.get("lookback", 36))
    hidden_units = int(config.get("hidden_units", 32))
    batch_size = int(config.get("batch_size", 16))
    learning_rate = float(config.get("learning_rate", 0.001))
    repeat_runs = int(config.get("repeat_runs", 5))
    dropout = float(config.get("dropout", 0.1))
    max_epochs = int(config.get("max_epochs", 250))
    patience = int(config.get("early_stopping_patience", 10))
    val_fraction = float(config.get("val_fraction", 0.2))
    min_delta = float(config.get("min_delta", 1e-4))

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
    if dropout < 0 or dropout >= 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm.dropout",
            detail="dropout must be in [0, 1)",
        )
    if max_epochs < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm.max_epochs",
            detail="max_epochs must be >= 1",
        )
    if patience < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm.early_stopping_patience",
            detail="early_stopping_patience must be >= 1",
        )
    if val_fraction <= 0 or val_fraction >= 0.5:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm.val_fraction",
            detail="val_fraction must be in (0, 0.5)",
        )
    if min_delta < 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="lstm.min_delta",
            detail="min_delta must be >= 0",
        )
    return {
        "lookback": lookback,
        "hidden_units": hidden_units,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "repeat_runs": repeat_runs,
        "dropout": dropout,
        "max_epochs": max_epochs,
        "early_stopping_patience": patience,
        "val_fraction": val_fraction,
        "min_delta": min_delta,
    }


def _split_train_val(
    x: np.ndarray,
    y: np.ndarray,
    *,
    val_fraction: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    n_samples = int(len(x))
    val_size = max(1, int(round(n_samples * float(val_fraction))))
    train_size = max(1, n_samples - val_size)
    val_size = n_samples - train_size
    return (
        x[:train_size],
        y[:train_size],
        x[train_size:],
        y[train_size:],
    )


def _recursive_forecast(
    model: ComponentLSTMRegressor,
    *,
    history: np.ndarray,
    lookback: int,
    horizons: Sequence[int],
) -> dict[int, float]:
    horizon_set = sorted({int(value) for value in horizons})
    if not horizon_set or horizon_set[0] < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="horizons",
            detail="horizons must include positive integers",
        )
    state = history.tolist()
    predictions: dict[int, float] = {}
    model.eval()
    with torch.no_grad():
        for step in range(1, horizon_set[-1] + 1):
            window = np.asarray(state[-lookback:], dtype=np.float32).reshape(
                1, lookback, 1
            )
            tensor = torch.tensor(window, dtype=torch.float32)
            pred = float(model(tensor).item())
            state.append(pred)
            if step in horizon_set:
                predictions[step] = pred
    return predictions


def _train_single_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    *,
    cfg: Mapping[str, Any],
    seed: int,
) -> tuple[ComponentLSTMRegressor, dict[str, Any]]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed))

    model = ComponentLSTMRegressor(
        hidden_units=int(cfg["hidden_units"]),
        dropout=float(cfg["dropout"]),
    )
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(cfg["learning_rate"]),
    )
    loss_fn = torch.nn.L1Loss()

    train_ds = TensorDataset(
        torch.tensor(x_train, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=min(int(cfg["batch_size"]), len(train_ds)),
        shuffle=True,
    )
    val_x = torch.tensor(x_val, dtype=torch.float32)
    val_y = torch.tensor(y_val, dtype=torch.float32)

    best_state = None
    best_val = float("inf")
    best_epoch = 0
    epochs_no_improve = 0
    max_epochs = int(cfg["max_epochs"])
    patience = int(cfg["early_stopping_patience"])
    min_delta = float(cfg["min_delta"])

    for epoch in range(1, max_epochs + 1):
        model.train()
        for batch_x, batch_y in train_loader:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(batch_x), batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        with torch.no_grad():
            val_loss = float(loss_fn(model(val_x), val_y).item())
        if val_loss + min_delta < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                break

    if best_state is None:
        best_state = {
            key: value.detach().cpu().clone() for key, value in model.state_dict().items()
        }
    model.load_state_dict(best_state)
    return model, {
        "best_val_loss": float(best_val),
        "best_epoch": int(best_epoch),
    }


def forecast_component_with_pytorch_lstm(
    values: Sequence[float] | pd.Series,
    *,
    horizons: Sequence[int],
    lstm_config: Mapping[str, Any],
    component_name: str,
    base_seed: int = 42,
    artifact_dir: str | Path | None = None,
) -> ComponentLSTMForecast:
    """Train real PyTorch LSTM and forecast one decomposed component."""

    _ensure_torch_available()
    cfg = _validate_config(lstm_config)
    dataset = build_lstm_supervised_dataset(values, lookback=int(cfg["lookback"]))
    x_train, y_train, x_val, y_val = _split_train_val(
        dataset.x,
        dataset.y,
        val_fraction=float(cfg["val_fraction"]),
    )

    run_rows: list[dict[str, Any]] = []
    run_metrics: list[dict[str, Any]] = []
    model_states: dict[str, dict[str, Any]] = {}
    for run_index in range(int(cfg["repeat_runs"])):
        seed = int(base_seed) + run_index
        model, metrics = _train_single_model(
            x_train,
            y_train,
            x_val,
            y_val,
            cfg=cfg,
            seed=seed,
        )
        run_metrics.append({"run_index": run_index, "seed": seed, **metrics})
        predictions = _recursive_forecast(
            model,
            history=((np.asarray(values, dtype=float) - dataset.mean) / dataset.std),
            lookback=int(cfg["lookback"]),
            horizons=horizons,
        )
        for horizon, pred_norm in sorted(predictions.items()):
            run_rows.append(
                {
                    "component_name": component_name,
                    "horizon": int(horizon),
                    "run_index": int(run_index),
                    "seed": int(seed),
                    "point_forecast": float(pred_norm * dataset.std + dataset.mean),
                }
            )
        if run_index == 0:
            model_states = {
                key: value.detach().cpu().numpy().tolist()
                for key, value in model.state_dict().items()
            }

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

    artifact_paths: dict[str, str] = {}
    if artifact_dir is not None:
        artifact_root = Path(artifact_dir)
        artifact_root.mkdir(parents=True, exist_ok=True)
        model_path = artifact_root / "model.pt"
        manifest_path = artifact_root / "manifest.json"

        torch_payload = {
            "component_name": str(component_name),
            "state_dict": {
                key: torch.tensor(value, dtype=torch.float32)
                for key, value in model_states.items()
            },
            "config": dict(cfg),
        }
        torch.save(torch_payload, model_path)

        hash_payload = np.asarray(values, dtype=float).tobytes()
        manifest = {
            "component_name": str(component_name),
            "model_family": "pytorch_lstm",
            "data_hash": hashlib.sha256(hash_payload).hexdigest(),
            "feature_list": [str(component_name)],
            "lookback": int(cfg["lookback"]),
            "seed_schedule": [int(base_seed) + idx for idx in range(int(cfg["repeat_runs"]))],
            "train_val_split": {
                "train_windows": int(len(x_train)),
                "val_windows": int(len(x_val)),
                "val_fraction": float(cfg["val_fraction"]),
            },
            "dropout": float(cfg["dropout"]),
            "early_stopping_patience": int(cfg["early_stopping_patience"]),
            "max_epochs": int(cfg["max_epochs"]),
            "model_path": str(model_path),
        }
        manifest_path.write_text(
            json.dumps(manifest, sort_keys=True, indent=2),
            encoding="utf-8",
        )
        artifact_paths = {
            "model_path": str(model_path),
            "manifest_path": str(manifest_path),
        }

    diagnostics: dict[str, Any] = {
        "component_name": component_name,
        "engine": "pytorch",
        "input_shape": [int(cfg["lookback"]), 1],
        "training_windows": int(len(dataset.x)),
        "train_windows": int(len(x_train)),
        "val_windows": int(len(x_val)),
        "normalization": {
            "mean": float(dataset.mean),
            "std": float(dataset.std),
        },
        "repeat_runs": int(cfg["repeat_runs"]),
        "seed_start": int(base_seed),
        "run_metrics": run_metrics,
        "dispersion_by_horizon": {
            str(int(row["horizon"])): {
                "std": float(row["std"]),
                "spread": float(row["spread"]),
            }
            for _, row in dispersion.iterrows()
        },
        **artifact_paths,
    }
    return ComponentLSTMForecast(
        point_forecast=point_forecast,
        run_forecasts=run_forecasts,
        diagnostics=diagnostics,
    )
