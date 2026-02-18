"""Transfer-learning utilities for basin-level priors."""

from .dataset import (
    BasinTransferDataset,
    TransferDatasetBundle,
    build_basin_driver_panel,
    build_transfer_datasets,
)
from .tl_basin_dnn import (
    EncoderState,
    HeadState,
    TLTrainConfig,
    TargetHeadResult,
    fit_source_encoder,
    fit_target_head,
    load_encoder_state,
    load_head_state,
    predict_with_transfer,
    save_encoder_state,
    save_head_state,
)

__all__ = [
    "BasinTransferDataset",
    "TransferDatasetBundle",
    "build_basin_driver_panel",
    "build_transfer_datasets",
    "EncoderState",
    "HeadState",
    "TLTrainConfig",
    "TargetHeadResult",
    "fit_source_encoder",
    "fit_target_head",
    "save_encoder_state",
    "save_head_state",
    "load_encoder_state",
    "load_head_state",
    "predict_with_transfer",
]
