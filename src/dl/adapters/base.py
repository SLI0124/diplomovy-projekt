from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch


@dataclass(frozen=True)
class ModelContext:
    prediction_length: int
    context_length: int
    num_samples: int
    lag_llama_num_parallel_samples: int


@dataclass(frozen=True)
class ForecastResult:
    y_pred: np.ndarray


@dataclass(frozen=True)
class TrainingLossPoint:
    epoch: int
    loss: float


class BaseFoundationModelAdapter:
    model_id: str
    slug: str
    model_family: str = "foundation"
    supports_finetune: bool = False

    def __init__(self, model_ctx: ModelContext, device: torch.device) -> None:
        self.model_ctx = model_ctx
        self.device = device

    def load_pretrained(self) -> None:
        raise NotImplementedError

    def finetune(
        self,
        train_series: np.ndarray,
        train_epochs: int,
        train_batch_size: int,
        train_steps_per_epoch: int,
        train_lr: float,
        train_weight_decay: float,
        train_loss: str | None,
        train_optimizer: str | None,
        artifact_dir: Path,
    ) -> list[TrainingLossPoint]:
        raise NotImplementedError

    def forecast(
        self, context: np.ndarray, context_start: pd.Timestamp
    ) -> ForecastResult:
        raise NotImplementedError

    def save_finetuned(self, artifact_dir: Path) -> None:
        raise NotImplementedError

    def load_finetuned(self, artifact_dir: Path) -> None:
        raise NotImplementedError
