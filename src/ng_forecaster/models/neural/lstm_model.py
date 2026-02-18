"""PyTorch LSTM model definition for component forecasts."""

from __future__ import annotations

from ng_forecaster.errors import ContractViolation

try:
    import torch
    from torch import nn
except Exception:  # pragma: no cover - optional dependency
    torch = None
    nn = None


if nn is not None:

    class ComponentLSTMRegressor(nn.Module):
        """Single-layer LSTM regressor with dropout head."""

        def __init__(
            self,
            *,
            input_size: int = 1,
            hidden_units: int = 32,
            dropout: float = 0.1,
        ) -> None:
            super().__init__()
            self.lstm = nn.LSTM(
                input_size=input_size,
                hidden_size=int(hidden_units),
                num_layers=1,
                batch_first=True,
            )
            self.dropout = nn.Dropout(float(dropout))
            self.head = nn.Linear(int(hidden_units), 1)

        def forward(self, x):
            sequence, _ = self.lstm(x)
            last_hidden = sequence[:, -1, :]
            dropped = self.dropout(last_hidden)
            return self.head(dropped).squeeze(-1)

else:

    class ComponentLSTMRegressor:  # type: ignore[no-redef]  # pragma: no cover - dependency guard
        """Placeholder when PyTorch is unavailable.

        Keeps import contracts stable so deterministic engine paths still load,
        while failing loudly if PyTorch training is requested.
        """

        def __init__(self, *args, **kwargs) -> None:
            raise ContractViolation(
                "missing_dependency",
                key="torch",
                detail=(
                    "PyTorch is required for real LSTM training path; "
                    "install torch or switch engine to deterministic"
                ),
            )
