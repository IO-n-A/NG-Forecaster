"""Small NumPy DNN for basin transfer-learning priors."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ng_forecaster.errors import ContractViolation


@dataclass(frozen=True)
class TLTrainConfig:
    """Training policy for transfer encoder/head optimization."""

    hidden_units: int = 24
    learning_rate: float = 0.01
    dropout_rate: float = 0.10
    l2_penalty: float = 1e-4
    max_epochs: int = 600
    patience: int = 40
    min_delta: float = 1e-6
    random_seed: int = 42


@dataclass(frozen=True)
class EncoderState:
    """Frozen source encoder state used for transfer learning."""

    weights: np.ndarray
    bias: np.ndarray
    feature_mean: np.ndarray
    feature_std: np.ndarray
    dropout_rate: float


@dataclass(frozen=True)
class HeadState:
    """Target-basin head state attached to a frozen encoder."""

    weights: np.ndarray
    bias: float
    target_mean: float
    target_std: float


@dataclass(frozen=True)
class TrainingHistory:
    """Training traces for deterministic diagnostics/export."""

    train_loss: list[float]
    val_loss: list[float]
    best_epoch: int
    best_val_loss: float


@dataclass(frozen=True)
class SourceEncoderResult:
    """Output bundle for source-side encoder training."""

    encoder: EncoderState
    source_head: HeadState
    history: TrainingHistory
    train_rmse: float
    val_rmse: float


@dataclass(frozen=True)
class TargetHeadResult:
    """Output bundle for target-head transfer training."""

    head: HeadState
    history: TrainingHistory
    train_rmse: float
    eval_rmse: float


def validate_train_config(config: Mapping[str, Any] | None) -> TLTrainConfig:
    """Validate transfer-training policy."""

    merged = {**TLTrainConfig().__dict__, **dict(config or {})}
    cfg = TLTrainConfig(
        hidden_units=int(merged["hidden_units"]),
        learning_rate=float(merged["learning_rate"]),
        dropout_rate=float(merged["dropout_rate"]),
        l2_penalty=float(merged["l2_penalty"]),
        max_epochs=int(merged["max_epochs"]),
        patience=int(merged["patience"]),
        min_delta=float(merged["min_delta"]),
        random_seed=int(merged["random_seed"]),
    )
    if cfg.hidden_units < 4:
        raise ContractViolation(
            "invalid_model_policy",
            key="hidden_units",
            detail="hidden_units must be >= 4",
        )
    if cfg.learning_rate <= 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="learning_rate",
            detail="learning_rate must be > 0",
        )
    if not 0 <= cfg.dropout_rate < 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="dropout_rate",
            detail="dropout_rate must be in [0, 1)",
        )
    if cfg.l2_penalty < 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="l2_penalty",
            detail="l2_penalty must be >= 0",
        )
    if cfg.max_epochs < 20:
        raise ContractViolation(
            "invalid_model_policy",
            key="max_epochs",
            detail="max_epochs must be >= 20",
        )
    if cfg.patience < 5:
        raise ContractViolation(
            "invalid_model_policy",
            key="patience",
            detail="patience must be >= 5",
        )
    if cfg.min_delta <= 0:
        raise ContractViolation(
            "invalid_model_policy",
            key="min_delta",
            detail="min_delta must be > 0",
        )
    return cfg


def _as_2d(values: np.ndarray) -> np.ndarray:
    if values.ndim != 2:
        raise ContractViolation(
            "invalid_model_policy",
            key="input_shape",
            detail="input matrix must be rank-2",
        )
    return values.astype(float, copy=False)


def _as_1d(values: np.ndarray) -> np.ndarray:
    if values.ndim != 1:
        raise ContractViolation(
            "invalid_model_policy",
            key="target_shape",
            detail="target array must be rank-1",
        )
    return values.astype(float, copy=False)


def _validate_xy(x: np.ndarray, y: np.ndarray, *, key: str, min_rows: int) -> None:
    if len(x) != len(y):
        raise ContractViolation(
            "invalid_model_policy",
            key=key,
            detail=f"feature/target row mismatch: {len(x)} vs {len(y)}",
        )
    if len(x) < min_rows:
        raise ContractViolation(
            "insufficient_training_data",
            key=key,
            detail=f"requires at least {min_rows} rows, received={len(x)}",
        )


def _safe_std(values: np.ndarray) -> np.ndarray:
    std = np.std(values, axis=0, ddof=0)
    if std.ndim == 0:
        return np.array([float(std) if float(std) > 1e-8 else 1.0], dtype=float)
    std = np.asarray(std, dtype=float)
    std[np.abs(std) < 1e-8] = 1.0
    return np.asarray(std, dtype=float)


def _split_train_val(
    x: np.ndarray, y: np.ndarray, *, min_val_rows: int = 8
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows = len(x)
    if rows <= min_val_rows + 1:
        return x, y, x, y
    val_rows = max(min_val_rows, int(round(rows * 0.2)))
    val_rows = min(val_rows, rows - 1)
    train_rows = rows - val_rows
    return x[:train_rows], y[:train_rows], x[train_rows:], y[train_rows:]


def _relu(values: np.ndarray) -> np.ndarray:
    return np.asarray(np.maximum(values, 0.0), dtype=float)


def _encode_standardized(
    x_std: np.ndarray, *, weights: np.ndarray, bias: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    hidden_linear = x_std @ weights + bias
    hidden = _relu(hidden_linear)
    return hidden_linear, hidden


def _dropout(
    hidden: np.ndarray, rate: float, *, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if rate <= 0:
        mask = np.ones_like(hidden)
        return hidden, mask
    keep_prob = 1.0 - rate
    mask = (rng.random(hidden.shape) < keep_prob).astype(float) / keep_prob
    return hidden * mask, mask


def _rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(y_true - y_pred))))


def fit_source_encoder(
    source_x: np.ndarray,
    source_y: np.ndarray,
    *,
    config: Mapping[str, Any] | None = None,
    random_seed: int | None = None,
) -> SourceEncoderResult:
    """Train source encoder + source head on pooled non-target basins."""

    cfg = validate_train_config(config)
    x = _as_2d(np.asarray(source_x))
    y = _as_1d(np.asarray(source_y))
    _validate_xy(x, y, key="source_dataset", min_rows=32)

    x_train, y_train, x_val, y_val = _split_train_val(x, y)
    feature_mean = x_train.mean(axis=0).astype(float)
    feature_std = _safe_std(x_train)
    x_train_std = (x_train - feature_mean) / feature_std
    x_val_std = (x_val - feature_mean) / feature_std

    y_mean = float(y_train.mean())
    y_std = float(_safe_std(y_train.reshape(-1, 1))[0])
    y_train_std = (y_train - y_mean) / y_std
    y_val_std = (y_val - y_mean) / y_std

    seed = cfg.random_seed if random_seed is None else int(random_seed)
    rng = np.random.default_rng(seed)
    input_dim = int(x.shape[1])
    hidden_dim = int(cfg.hidden_units)
    w_enc = rng.normal(loc=0.0, scale=0.05, size=(input_dim, hidden_dim))
    b_enc = np.zeros(hidden_dim, dtype=float)
    w_head = rng.normal(loc=0.0, scale=0.05, size=(hidden_dim, 1))
    b_head = np.zeros(1, dtype=float)

    best_state: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None = None
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0
    train_trace: list[float] = []
    val_trace: list[float] = []

    y_train_matrix = y_train_std.reshape(-1, 1)
    y_val_matrix = y_val_std.reshape(-1, 1)
    lr = float(cfg.learning_rate)
    l2 = float(cfg.l2_penalty)

    for epoch in range(cfg.max_epochs):
        hidden_linear, hidden = _encode_standardized(
            x_train_std, weights=w_enc, bias=b_enc
        )
        hidden_drop, mask = _dropout(hidden, cfg.dropout_rate, rng=rng)
        preds = hidden_drop @ w_head + b_head
        err = preds - y_train_matrix
        train_loss = float(np.mean(np.square(err)) + l2 * np.sum(np.square(w_head)))

        grad_preds = 2.0 * err / len(x_train_std)
        grad_w_head = hidden_drop.T @ grad_preds + l2 * w_head
        grad_b_head = grad_preds.sum(axis=0)
        grad_hidden = grad_preds @ w_head.T
        grad_hidden = grad_hidden * mask
        grad_hidden_linear = grad_hidden * (hidden_linear > 0).astype(float)
        grad_w_enc = x_train_std.T @ grad_hidden_linear + l2 * w_enc
        grad_b_enc = grad_hidden_linear.sum(axis=0)

        w_head = w_head - lr * grad_w_head
        b_head = b_head - lr * grad_b_head
        w_enc = w_enc - lr * grad_w_enc
        b_enc = b_enc - lr * grad_b_enc

        _, val_hidden = _encode_standardized(x_val_std, weights=w_enc, bias=b_enc)
        val_preds = val_hidden @ w_head + b_head
        val_err = val_preds - y_val_matrix
        val_loss = float(np.mean(np.square(val_err)))

        train_trace.append(train_loss)
        val_trace.append(val_loss)
        if val_loss + cfg.min_delta < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch
            best_state = (
                w_enc.copy(),
                b_enc.copy(),
                w_head.copy(),
                b_head.copy(),
            )
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    if best_state is None:
        raise ContractViolation(
            "invalid_model_policy",
            key="source_training",
            detail="source training did not produce a best state",
        )
    w_enc, b_enc, w_head, b_head = best_state

    _, train_hidden = _encode_standardized(x_train_std, weights=w_enc, bias=b_enc)
    train_pred_std = (train_hidden @ w_head + b_head).reshape(-1)
    _, val_hidden = _encode_standardized(x_val_std, weights=w_enc, bias=b_enc)
    val_pred_std = (val_hidden @ w_head + b_head).reshape(-1)
    train_rmse = _rmse(y_train, train_pred_std * y_std + y_mean)
    val_rmse = _rmse(y_val, val_pred_std * y_std + y_mean)

    encoder = EncoderState(
        weights=w_enc,
        bias=b_enc,
        feature_mean=feature_mean,
        feature_std=feature_std,
        dropout_rate=float(cfg.dropout_rate),
    )
    source_head = HeadState(
        weights=w_head,
        bias=float(b_head.reshape(-1)[0]),
        target_mean=y_mean,
        target_std=y_std,
    )
    history = TrainingHistory(
        train_loss=train_trace,
        val_loss=val_trace,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
    )
    return SourceEncoderResult(
        encoder=encoder,
        source_head=source_head,
        history=history,
        train_rmse=train_rmse,
        val_rmse=val_rmse,
    )


def fit_target_head(
    encoder: EncoderState,
    *,
    target_train_x: np.ndarray,
    target_train_y: np.ndarray,
    target_eval_x: np.ndarray,
    target_eval_y: np.ndarray,
    config: Mapping[str, Any] | None = None,
    random_seed: int | None = None,
) -> TargetHeadResult:
    """Train a frozen-encoder target head for one basin/horizon."""

    cfg = validate_train_config(config)
    x_train = _as_2d(np.asarray(target_train_x))
    y_train = _as_1d(np.asarray(target_train_y))
    x_eval = _as_2d(np.asarray(target_eval_x))
    y_eval = _as_1d(np.asarray(target_eval_y))
    _validate_xy(x_train, y_train, key="target_train_dataset", min_rows=16)
    _validate_xy(x_eval, y_eval, key="target_eval_dataset", min_rows=1)

    x_train_std = (x_train - encoder.feature_mean) / encoder.feature_std
    x_eval_std = (x_eval - encoder.feature_mean) / encoder.feature_std
    _, train_hidden = _encode_standardized(
        x_train_std, weights=encoder.weights, bias=encoder.bias
    )
    _, eval_hidden = _encode_standardized(
        x_eval_std, weights=encoder.weights, bias=encoder.bias
    )

    y_mean = float(y_train.mean())
    y_std = float(_safe_std(y_train.reshape(-1, 1))[0])
    y_train_std = (y_train - y_mean) / y_std
    y_eval_std = (y_eval - y_mean) / y_std

    seed = cfg.random_seed + 997 if random_seed is None else int(random_seed)
    rng = np.random.default_rng(seed)
    hidden_dim = int(train_hidden.shape[1])
    w_head = rng.normal(loc=0.0, scale=0.05, size=(hidden_dim, 1))
    b_head = np.zeros(1, dtype=float)

    best_state: tuple[np.ndarray, np.ndarray] | None = None
    best_val_loss = float("inf")
    best_epoch = 0
    wait = 0
    train_trace: list[float] = []
    val_trace: list[float] = []

    y_train_matrix = y_train_std.reshape(-1, 1)
    y_eval_matrix = y_eval_std.reshape(-1, 1)
    lr = float(cfg.learning_rate)
    l2 = float(cfg.l2_penalty)

    for epoch in range(cfg.max_epochs):
        hidden_drop, _ = _dropout(train_hidden, cfg.dropout_rate, rng=rng)
        preds = hidden_drop @ w_head + b_head
        err = preds - y_train_matrix
        train_loss = float(np.mean(np.square(err)) + l2 * np.sum(np.square(w_head)))

        grad_preds = 2.0 * err / len(train_hidden)
        grad_w_head = hidden_drop.T @ grad_preds + l2 * w_head
        grad_b_head = grad_preds.sum(axis=0)
        w_head = w_head - lr * grad_w_head
        b_head = b_head - lr * grad_b_head

        eval_preds = eval_hidden @ w_head + b_head
        eval_err = eval_preds - y_eval_matrix
        eval_loss = float(np.mean(np.square(eval_err)))

        train_trace.append(train_loss)
        val_trace.append(eval_loss)
        if eval_loss + cfg.min_delta < best_val_loss:
            best_val_loss = eval_loss
            best_epoch = epoch
            best_state = (w_head.copy(), b_head.copy())
            wait = 0
        else:
            wait += 1
            if wait >= cfg.patience:
                break

    if best_state is None:
        raise ContractViolation(
            "invalid_model_policy",
            key="target_training",
            detail="target head training did not produce a best state",
        )
    w_head, b_head = best_state

    train_pred_std = (train_hidden @ w_head + b_head).reshape(-1)
    eval_pred_std = (eval_hidden @ w_head + b_head).reshape(-1)
    train_rmse = _rmse(y_train, train_pred_std * y_std + y_mean)
    eval_rmse = _rmse(y_eval, eval_pred_std * y_std + y_mean)

    head = HeadState(
        weights=w_head,
        bias=float(b_head.reshape(-1)[0]),
        target_mean=y_mean,
        target_std=y_std,
    )
    history = TrainingHistory(
        train_loss=train_trace,
        val_loss=val_trace,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
    )
    return TargetHeadResult(
        head=head,
        history=history,
        train_rmse=train_rmse,
        eval_rmse=eval_rmse,
    )


def predict_with_transfer(
    encoder: EncoderState,
    head: HeadState,
    x: np.ndarray,
) -> np.ndarray:
    """Run frozen-encoder inference for target head outputs."""

    values = _as_2d(np.asarray(x))
    x_std = (values - encoder.feature_mean) / encoder.feature_std
    _, hidden = _encode_standardized(x_std, weights=encoder.weights, bias=encoder.bias)
    pred_std = (hidden @ head.weights + head.bias).reshape(-1)
    preds = pred_std * head.target_std + head.target_mean
    return np.asarray(preds, dtype=float)


def save_encoder_state(path: str | Path, encoder: EncoderState) -> Path:
    """Persist encoder state as compressed NPZ."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        weights=encoder.weights,
        bias=encoder.bias,
        feature_mean=encoder.feature_mean,
        feature_std=encoder.feature_std,
        dropout_rate=np.array([encoder.dropout_rate], dtype=float),
    )
    return target


def load_encoder_state(path: str | Path) -> EncoderState:
    """Load encoder state from compressed NPZ."""

    payload = np.load(Path(path), allow_pickle=False)
    return EncoderState(
        weights=np.asarray(payload["weights"], dtype=float),
        bias=np.asarray(payload["bias"], dtype=float),
        feature_mean=np.asarray(payload["feature_mean"], dtype=float),
        feature_std=np.asarray(payload["feature_std"], dtype=float),
        dropout_rate=float(
            np.asarray(payload["dropout_rate"], dtype=float).reshape(-1)[0]
        ),
    )


def save_head_state(path: str | Path, head: HeadState) -> Path:
    """Persist target head state as compressed NPZ."""

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        target,
        weights=head.weights,
        bias=np.array([head.bias], dtype=float),
        target_mean=np.array([head.target_mean], dtype=float),
        target_std=np.array([head.target_std], dtype=float),
    )
    return target


def load_head_state(path: str | Path) -> HeadState:
    """Load target head state from compressed NPZ."""

    payload = np.load(Path(path), allow_pickle=False)
    return HeadState(
        weights=np.asarray(payload["weights"], dtype=float),
        bias=float(np.asarray(payload["bias"], dtype=float).reshape(-1)[0]),
        target_mean=float(
            np.asarray(payload["target_mean"], dtype=float).reshape(-1)[0]
        ),
        target_std=float(np.asarray(payload["target_std"], dtype=float).reshape(-1)[0]),
    )
