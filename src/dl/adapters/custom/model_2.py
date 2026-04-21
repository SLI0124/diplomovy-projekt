from __future__ import annotations

from typing import Any

import torch
from adapters.base import ModelContext
from adapters.custom.common import BaseCustomSequenceAdapter


class Model2Adapter(BaseCustomSequenceAdapter):
    model_id = "custom/model_2"
    slug = "model_2"

    def __init__(self, model_ctx: ModelContext, device: torch.device) -> None:
        super().__init__(model_ctx, device)
        self._hidden_dim: int = 128
        self._dropout: float = 0.15

    def _model_kwargs(self) -> dict[str, Any]:
        return {
            "hidden_dim": int(self._hidden_dim),
            "dropout": float(self._dropout),
            "use_temporal_conv": True,
            "use_attention": False,
            "baseline_mode": "seasonal_fixed",
        }

    def _load_checkpoint_hparams(self, payload: dict[str, Any]) -> None:
        self._hidden_dim = int(payload.get("hidden_dim", self._hidden_dim))
        self._dropout = float(payload.get("dropout", self._dropout))
