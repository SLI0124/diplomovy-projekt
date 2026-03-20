from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from adapters.base import (
    BaseFoundationModelAdapter,
    ForecastResult,
    ModelContext,
    TrainingLossPoint,
)


class _Model1LSTM(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        dense_size: int,
        dropout: float,
        prediction_length: int,
    ) -> None:
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.norm = nn.BatchNorm1d(hidden_size)
        self.head = nn.Sequential(
            nn.Linear(hidden_size, dense_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dense_size, prediction_length),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x shape: [batch, time]
        seq = x.unsqueeze(-1)
        out, _ = self.lstm(seq)
        last = out[:, -1, :]
        last = self.norm(last)
        return self.head(last)


class _RMSELoss(nn.Module):
    def __init__(self, eps: float = 1e-8) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        mse = torch.mean((preds - targets) ** 2)
        return torch.sqrt(mse + self.eps)


class _MAPELoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(targets.abs(), min=self.eps)
        return torch.mean((preds - targets).abs() / denom)


class _SMAPELoss(nn.Module):
    def __init__(self, eps: float = 1e-6) -> None:
        super().__init__()
        self.eps = eps

    def forward(self, preds: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        denom = torch.clamp(preds.abs() + targets.abs(), min=self.eps)
        return torch.mean(2.0 * (preds - targets).abs() / denom)


class Model1Adapter(BaseFoundationModelAdapter):
    model_id = "custom/model_1"
    slug = "model_1"
    model_family = "custom"
    supports_finetune = True

    def __init__(self, model_ctx: ModelContext, device: torch.device) -> None:
        super().__init__(model_ctx, device)
        self._arch_version: int = 2
        self._hidden_size: int = 64
        self._num_layers: int = 2
        self._dense_size: int = 64
        self._dropout: float = 0.15
        self._model: nn.Module | None = None
        self._mean: float = 0.0
        self._std: float = 1.0
        self._loaded: bool = False

    def _build_model(self) -> _Model1LSTM:
        model = _Model1LSTM(
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            dense_size=self._dense_size,
            dropout=self._dropout,
            prediction_length=int(self.model_ctx.prediction_length),
        )
        model.to(self.device)
        return model

    def load_pretrained(self) -> None:
        self._model = self._build_model()
        self._model.eval()
        self._mean = 0.0
        self._std = 1.0
        self._loaded = True

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
        artifact_dir.mkdir(parents=True, exist_ok=True)

        series = np.asarray(train_series, dtype=np.float32)
        if series.size == 0:
            raise ValueError("Model1Adapter requires non-empty train_series.")

        self._mean = float(series.mean())
        std = float(series.std())
        self._std = std if std > 1e-6 else 1.0
        series_norm = (series - self._mean) / self._std

        prediction_length = int(max(1, self.model_ctx.prediction_length))
        context_length = int(
            min(
                max(1, self.model_ctx.context_length),
                max(1, series_norm.size - prediction_length),
            )
        )

        start_min = context_length
        start_max = int(series_norm.size - prediction_length)
        if start_max < start_min:
            self._model = self._build_model()
            self._model.eval()
            self._loaded = True
            return []

        batch_size = int(max(1, train_batch_size))
        steps_per_epoch = int(max(1, train_steps_per_epoch))
        epochs = int(max(1, train_epochs))

        self._model = self._build_model()
        self._model.train()

        optimizer = self._build_optimizer(
            train_optimizer=train_optimizer,
            params=self._model.parameters(),
            train_lr=train_lr,
            train_weight_decay=train_weight_decay,
        )
        loss_fn = self._build_loss_fn(train_loss)
        rng = np.random.default_rng()
        history: list[TrainingLossPoint] = []

        for ep in range(epochs):
            epoch_losses: list[float] = []
            for _ in range(steps_per_epoch):
                starts = rng.integers(start_min, start_max + 1, size=batch_size)
                contexts_np = np.stack(
                    [series_norm[s - context_length : s] for s in starts],
                    axis=0,
                )
                targets_np = np.stack(
                    [series_norm[s : s + prediction_length] for s in starts],
                    axis=0,
                )

                contexts = torch.from_numpy(contexts_np).to(self.device).float()
                targets = torch.from_numpy(targets_np).to(self.device).float()

                preds = self._model(contexts)
                loss = loss_fn(preds, targets)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))

            if epoch_losses:
                history.append(
                    TrainingLossPoint(
                        epoch=ep,
                        loss=float(np.mean(epoch_losses)),
                    )
                )

        self._model.eval()
        self._loaded = True
        return history

    def _build_loss_fn(self, train_loss: str | None) -> nn.Module:
        loss_name = (train_loss or "mse").strip().lower()
        if loss_name == "mse":
            return nn.MSELoss()
        if loss_name == "mae":
            return nn.L1Loss()
        if loss_name == "rmse":
            return _RMSELoss()
        if loss_name == "mape":
            return _MAPELoss()
        if loss_name == "smape":
            return _SMAPELoss()
        raise ValueError(
            f"Unsupported train loss '{train_loss}' for {self.slug}. Supported: mae, mse, rmse, mape, smape"
        )

    def _build_optimizer(
        self,
        train_optimizer: str | None,
        params,
        train_lr: float,
        train_weight_decay: float,
    ) -> torch.optim.Optimizer:
        optimizer_name = (train_optimizer or "adamw").strip().lower()
        lr = float(train_lr)
        weight_decay = float(train_weight_decay)

        if optimizer_name == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        if optimizer_name == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        if optimizer_name == "sgd":
            return torch.optim.SGD(params, lr=lr, weight_decay=weight_decay)
        raise ValueError(
            f"Unsupported train optimizer '{train_optimizer}' for {self.slug}. Supported: adamw, adam, sgd"
        )

    def forecast(
        self, context: np.ndarray, context_start: pd.Timestamp
    ) -> ForecastResult:
        if not self._loaded:
            raise RuntimeError(
                "Model_1 is not loaded. Call load_pretrained/load_finetuned."
            )

        if self._model is None:
            raise RuntimeError("Model_1 internal model is not initialized.")

        x = np.asarray(context, dtype=np.float32)
        context_length = int(max(1, self.model_ctx.context_length))

        if x.size == 0:
            x = np.zeros((context_length,), dtype=np.float32)
        elif x.size < context_length:
            pad = np.full((context_length - x.size,), float(x[-1]), dtype=np.float32)
            x = np.concatenate([pad, x], axis=0)
        else:
            x = x[-context_length:]

        x_norm = (x - self._mean) / self._std
        x_t = torch.from_numpy(x_norm).to(self.device).float().view(1, -1)

        with torch.no_grad():
            y_norm = self._model(x_t)[0].detach().cpu().numpy()

        y_pred = np.asarray(y_norm * self._std + self._mean, dtype=np.float32)
        return ForecastResult(y_pred=y_pred)

    def save_finetuned(self, artifact_dir: Path) -> None:
        if not self._loaded:
            raise RuntimeError("Model_1 is not available to save.")
        if self._model is None:
            raise RuntimeError("Model_1 internal model is not initialized.")

        artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_id": self.model_id,
                "slug": self.slug,
                "hidden_size": self._hidden_size,
                "num_layers": self._num_layers,
                "dense_size": self._dense_size,
                "dropout": self._dropout,
                "prediction_length": int(self.model_ctx.prediction_length),
                "mean": self._mean,
                "std": self._std,
                "state_dict": self._model.state_dict(),
            },
            artifact_dir / "model.pt",
        )

        (artifact_dir / "model.json").write_text(
            json.dumps(
                {
                    "model_id": self.model_id,
                    "slug": self.slug,
                    "hidden_size": self._hidden_size,
                    "num_layers": self._num_layers,
                    "dense_size": self._dense_size,
                    "dropout": self._dropout,
                    "prediction_length": int(self.model_ctx.prediction_length),
                    "mean": self._mean,
                    "std": self._std,
                    "weights_file": "model.pt",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def load_finetuned(self, artifact_dir: Path) -> None:
        model_path = artifact_dir / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing Model_1 artifact: {model_path}")

        payload = torch.load(model_path, map_location="cpu")
        self._hidden_size = int(payload.get("hidden_size", 64))
        self._num_layers = int(payload.get("num_layers", 2))
        self._dense_size = int(payload.get("dense_size", 64))
        self._dropout = float(payload.get("dropout", 0.15))
        self._mean = float(payload.get("mean", 0.0))
        std = float(payload.get("std", 1.0))
        self._std = std if std > 1e-6 else 1.0

        state_dict = payload.get("state_dict")
        if state_dict is None:
            raise KeyError("Model_1 checkpoint missing state_dict")
        self._model = self._build_model()
        self._model.load_state_dict(state_dict)
        self._model.eval()

        self._loaded = True
