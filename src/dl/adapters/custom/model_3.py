from __future__ import annotations

from typing import Any

import torch
from adapters.base import ModelContext
from adapters.custom.common import BaseCustomSequenceAdapter


class Model3Adapter(BaseCustomSequenceAdapter):
    model_id = "custom/model_3"
    slug = "model_3"

    def __init__(self, model_ctx: ModelContext, device: torch.device) -> None:
        super().__init__(model_ctx, device)
        self._hidden_dim: int = 192
        self._dropout: float = 0.10
        self._attention_heads: int = 4
        self._ff_hidden_dim: int = 384

    def _model_kwargs(self) -> dict[str, Any]:
        return {
            "hidden_dim": int(self._hidden_dim),
            "dropout": float(self._dropout),
            "use_temporal_conv": True,
            "use_attention": True,
            "baseline_mode": "seasonal_learned",
            "attention_heads": int(self._attention_heads),
            "ff_hidden_dim": int(self._ff_hidden_dim),
        }

    def _load_checkpoint_hparams(self, payload: dict[str, Any]) -> None:
        self._hidden_dim = int(payload.get("hidden_dim", self._hidden_dim))
        self._dropout = float(payload.get("dropout", self._dropout))
        self._attention_heads = int(
            payload.get("attention_heads", self._attention_heads)
        )
        self._ff_hidden_dim = int(payload.get("ff_hidden_dim", self._ff_hidden_dim))
