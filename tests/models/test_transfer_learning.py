from __future__ import annotations

import numpy as np

from ng_forecaster.models.transfer_learning.tl_basin_dnn import (
    load_encoder_state,
    load_head_state,
    fit_source_encoder,
    fit_target_head,
    predict_with_transfer,
    save_encoder_state,
    save_head_state,
)


def test_transfer_encoder_and_head_train_and_predict(tmp_path) -> None:
    rng = np.random.default_rng(42)

    source_x = rng.normal(size=(96, 5))
    source_y = (
        1.2 * source_x[:, 0]
        - 0.7 * source_x[:, 1]
        + 0.4 * source_x[:, 2]
        + rng.normal(scale=0.05, size=96)
    )
    target_train_x = rng.normal(size=(48, 5))
    target_train_y = (
        0.9 * target_train_x[:, 0]
        - 0.5 * target_train_x[:, 1]
        + 0.3 * target_train_x[:, 2]
        + rng.normal(scale=0.08, size=48)
    )
    target_eval_x = rng.normal(size=(12, 5))
    target_eval_y = (
        0.9 * target_eval_x[:, 0]
        - 0.5 * target_eval_x[:, 1]
        + 0.3 * target_eval_x[:, 2]
        + rng.normal(scale=0.08, size=12)
    )

    source = fit_source_encoder(
        source_x,
        source_y,
        config={
            "max_epochs": 120,
            "patience": 20,
            "hidden_units": 16,
            "learning_rate": 0.02,
        },
    )
    target = fit_target_head(
        source.encoder,
        target_train_x=target_train_x,
        target_train_y=target_train_y,
        target_eval_x=target_eval_x,
        target_eval_y=target_eval_y,
        config={
            "max_epochs": 120,
            "patience": 20,
            "hidden_units": 16,
            "learning_rate": 0.02,
        },
    )

    preds = predict_with_transfer(source.encoder, target.head, target_eval_x)
    assert preds.shape == (12,)
    assert float(target.eval_rmse) < 1.0

    encoder_path = save_encoder_state(tmp_path / "encoder.npz", source.encoder)
    head_path = save_head_state(tmp_path / "head.npz", target.head)
    loaded_encoder = load_encoder_state(encoder_path)
    loaded_head = load_head_state(head_path)
    loaded_preds = predict_with_transfer(loaded_encoder, loaded_head, target_eval_x)
    np.testing.assert_allclose(preds, loaded_preds)
