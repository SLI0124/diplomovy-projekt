from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from adapters.base import (
    BaseFoundationModelAdapter,
    ForecastResult,
    ModelContext,
    TrainingLossPoint,
)


class Model1Adapter(BaseFoundationModelAdapter):
    model_id = "custom/model_1"
    slug = "model_1"
    model_family = "custom"
    supports_finetune = True

    def __init__(self, model_ctx: ModelContext, device: torch.device) -> None:
        super().__init__(model_ctx, device)

    def _not_available(self) -> None:
        raise NotImplementedError(
            "Model_1 has been reset to a clean slate. Re-implement this adapter before using it."
        )

    def load_pretrained(self) -> None:
        self._not_available()

    def finetune(
        self,
        train_series: np.ndarray,
        train_epochs: int,
        train_batch_size: int,
        train_steps_per_epoch: int,
        train_lr: float,
        train_weight_decay: float,
        checkpoint_selection: str,
        train_loss: str | None,
        train_optimizer: str | None,
        artifact_dir: Path,
        train_covariates: np.ndarray | None = None,
        train_future_covariates: np.ndarray | None = None,
    ) -> list[TrainingLossPoint]:
        del (
            train_series,
            train_epochs,
            train_batch_size,
            train_steps_per_epoch,
            train_lr,
            train_weight_decay,
            checkpoint_selection,
            train_loss,
            train_optimizer,
            artifact_dir,
            train_covariates,
            train_future_covariates,
        )
        self._not_available()

    def forecast(
        self,
        context: np.ndarray,
        context_start: pd.Timestamp,
        context_covariates: np.ndarray | None = None,
        future_covariates: np.ndarray | None = None,
    ) -> ForecastResult:
        del context, context_start, context_covariates, future_covariates
        self._not_available()

    def save_finetuned(self, artifact_dir: Path) -> None:
        del artifact_dir
        self._not_available()

    def load_finetuned(self, artifact_dir: Path) -> None:
        del artifact_dir
        self._not_available()
