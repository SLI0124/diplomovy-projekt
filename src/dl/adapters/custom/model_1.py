from __future__ import annotations

from collections.abc import Callable
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
from adapters.shared import RandomWindowDataset, RandomWindowDatasetWithCovariates
from torch.utils.data import DataLoader


class _Model1Forecaster(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        future_covariate_dim: int,
        prediction_length: int,
        hidden_dim: int,
        dropout: float,
        attention_heads: int,
        ff_hidden_dim: int,
    ) -> None:
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even for bidirectional GRU.")
        if hidden_dim % attention_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by attention_heads for MultiheadAttention."
            )

        self.feature_dim = int(feature_dim)
        self.future_covariate_dim = int(future_covariate_dim)
        self.prediction_length = int(prediction_length)

        # Layer block 1: feature normalization over target+covariate channels.
        self.input_norm = nn.LayerNorm(self.feature_dim)

        # Layer block 2: dense projection to hidden representation.
        self.input_projection = nn.Linear(self.feature_dim, hidden_dim)

        # Layer block 3: temporal convolution with normalization and dropout regularization.
        self.temporal_conv = nn.Conv1d(
            in_channels=hidden_dim,
            out_channels=hidden_dim,
            kernel_size=3,
            padding=1,
        )
        self.conv_norm = nn.BatchNorm1d(hidden_dim)
        self.dropout = nn.Dropout(dropout)

        # Layer block 4: bidirectional GRU sequence encoder.
        self.sequence_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # Layer block 5: temporal self-attention refinement.
        self.temporal_attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=attention_heads,
            dropout=dropout,
            batch_first=True,
        )

        # Layer block 6: position-wise feed-forward refinement.
        self.feed_forward = nn.Sequential(
            nn.Linear(hidden_dim, ff_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_hidden_dim, hidden_dim),
        )

        # Layer block 7: mean+max pooling and dense prediction head.
        self.prediction_head = nn.Sequential(
            nn.Linear((2 * hidden_dim) + self.future_covariate_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.prediction_length),
        )

    @staticmethod
    def _sanitize(tensor: torch.Tensor) -> torch.Tensor:
        return torch.nan_to_num(tensor, nan=0.0, posinf=0.0, neginf=0.0)

    def _build_future_summary(
        self,
        future_covariates: torch.Tensor | None,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        if self.future_covariate_dim == 0:
            return torch.zeros((batch_size, 0), dtype=torch.float32, device=device)

        if future_covariates is None:
            return torch.zeros(
                (batch_size, self.future_covariate_dim),
                dtype=torch.float32,
                device=device,
            )

        if future_covariates.ndim != 3:
            raise ValueError(
                "future_covariates must be [batch, horizon, features]. "
                f"Got shape={tuple(future_covariates.shape)}."
            )
        if future_covariates.shape[2] != self.future_covariate_dim:
            raise ValueError(
                "future covariate feature count mismatch. "
                f"Expected {self.future_covariate_dim}, got {future_covariates.shape[2]}."
            )

        finite_mask = torch.isfinite(future_covariates)
        safe_values = torch.where(
            finite_mask, future_covariates, torch.zeros_like(future_covariates)
        )
        counts = finite_mask.sum(dim=1).clamp(min=1)
        return safe_values.sum(dim=1) / counts

    def forward(
        self,
        *,
        context: torch.Tensor,
        context_covariates: torch.Tensor | None,
        future_covariates: torch.Tensor | None,
    ) -> torch.Tensor:
        if context.ndim != 2:
            raise ValueError(
                f"context must be [batch, time]. Got shape={tuple(context.shape)}."
            )

        x = context.unsqueeze(-1)
        if context_covariates is not None:
            if context_covariates.ndim != 3:
                raise ValueError(
                    "context_covariates must be [batch, time, features]. "
                    f"Got shape={tuple(context_covariates.shape)}."
                )
            if context_covariates.shape[:2] != x.shape[:2]:
                raise ValueError(
                    "context and context_covariates batch/time dimensions must match. "
                    f"context={tuple(x.shape)} context_covariates={tuple(context_covariates.shape)}"
                )
            x = torch.cat((x, context_covariates), dim=2)

        if x.shape[2] != self.feature_dim:
            raise ValueError(
                "Input feature count mismatch. "
                f"Expected {self.feature_dim}, got {x.shape[2]}."
            )

        x = self._sanitize(x)
        x = self.input_norm(x)
        x = self.input_projection(x)
        x = torch.nn.functional.gelu(x)

        x = x.transpose(1, 2)
        x = self.temporal_conv(x)
        x = self.conv_norm(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        x = x.transpose(1, 2)

        x, _ = self.sequence_encoder(x)
        x_attn, _ = self.temporal_attention(x, x, x, need_weights=False)
        x = x + x_attn
        x = x + self.feed_forward(x)

        pooled_mean = torch.mean(x, dim=1)
        pooled_max = torch.max(x, dim=1).values
        pooled = torch.cat((pooled_mean, pooled_max), dim=1)

        future_summary = self._build_future_summary(
            future_covariates=future_covariates,
            batch_size=pooled.shape[0],
            device=pooled.device,
        )
        features = torch.cat((pooled, future_summary), dim=1)
        return self.prediction_head(features)


class Model1Adapter(BaseFoundationModelAdapter):
    model_id = "custom/model_1"
    slug = "model_1"
    model_family = "custom"
    supports_finetune = True

    def __init__(self, model_ctx: ModelContext, device: torch.device) -> None:
        super().__init__(model_ctx, device)
        self._model: _Model1Forecaster | None = None
        self._feature_dim: int | None = None
        self._future_covariate_dim: int | None = None
        self._hidden_dim: int = 128
        self._ff_hidden_dim: int = 256
        self._dropout: float = 0.1
        self._attention_heads: int = 4

    @staticmethod
    def _as_float2d(
        array: np.ndarray,
        *,
        name: str,
        expected_length: int | None = None,
    ) -> np.ndarray:
        out = np.asarray(array, dtype=np.float32)
        if out.ndim != 2:
            raise ValueError(f"{name} must be a 2D array. Got shape={out.shape}.")
        if expected_length is not None and out.shape[0] != expected_length:
            raise ValueError(
                f"{name} length mismatch. Expected {expected_length}, got {out.shape[0]}."
            )
        return out

    @staticmethod
    def _select_loss(name: str) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
        key = name.strip().lower()
        if key == "mse":
            return lambda y_pred, y_true: torch.mean((y_pred - y_true) ** 2)
        if key == "mae":
            return lambda y_pred, y_true: torch.mean(torch.abs(y_pred - y_true))
        if key == "rmse":
            return lambda y_pred, y_true: torch.sqrt(
                torch.mean((y_pred - y_true) ** 2) + 1e-8
            )
        if key == "mape":
            return (
                lambda y_pred, y_true: torch.mean(
                    torch.abs(y_pred - y_true)
                    / torch.clamp(torch.abs(y_true), min=1e-8)
                )
                * 100.0
            )
        if key == "smape":
            return lambda y_pred, y_true: torch.mean(
                200.0
                * torch.abs(y_pred - y_true)
                / torch.clamp(torch.abs(y_true) + torch.abs(y_pred), min=1e-8)
            )
        raise ValueError(
            f"Unsupported loss '{name}'. Supported: mae, mse, rmse, mape, smape"
        )

    @staticmethod
    def _build_optimizer(
        *,
        model: nn.Module,
        name: str,
        lr: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        key = name.strip().lower()
        params = (p for p in model.parameters() if p.requires_grad)
        if key == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        if key == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        if key == "sgd":
            return torch.optim.SGD(
                params, lr=lr, momentum=0.9, weight_decay=weight_decay
            )
        raise ValueError(f"Unsupported optimizer '{name}'. Supported: adamw, adam, sgd")

    @staticmethod
    def _normalize_covariates(
        context_covariates: torch.Tensor,
        future_covariates: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        finite_mask = torch.isfinite(context_covariates)
        safe_context = torch.where(
            finite_mask,
            context_covariates,
            torch.zeros_like(context_covariates),
        )
        count = finite_mask.sum(dim=1, keepdim=True).clamp(min=1)
        mean = safe_context.sum(dim=1, keepdim=True) / count
        var = ((safe_context - mean) ** 2 * finite_mask).sum(
            dim=1, keepdim=True
        ) / count
        std = torch.sqrt(var + 1e-6).clamp(min=0.1)

        context_norm = torch.where(
            finite_mask,
            torch.clamp((context_covariates - mean) / std, min=-5.0, max=5.0),
            torch.zeros_like(context_covariates),
        )

        future_norm: torch.Tensor | None = None
        if future_covariates is not None:
            future_mask = torch.isfinite(future_covariates)
            future_norm = torch.where(
                future_mask,
                torch.clamp((future_covariates - mean) / std, min=-5.0, max=5.0),
                torch.zeros_like(future_covariates),
            )

        return context_norm, future_norm

    @staticmethod
    def _repeat_pattern(block: torch.Tensor, prediction_length: int) -> torch.Tensor:
        if block.shape[1] <= 0:
            raise ValueError("Pattern block must have positive time length.")
        repeats = (prediction_length + block.shape[1] - 1) // block.shape[1]
        return block.repeat(1, repeats)[:, :prediction_length]

    @staticmethod
    def _build_reference_baseline(
        context_tensor: torch.Tensor,
        prediction_length: int,
    ) -> torch.Tensor:
        # Basic seasonal baseline using recent day, previous-week day, and persistence.
        time_len = int(context_tensor.shape[1])
        last_value = context_tensor[:, -1:].expand(-1, prediction_length)

        daily_block = context_tensor[:, max(0, time_len - 24) : time_len]
        daily_pattern = Model1Adapter._repeat_pattern(daily_block, prediction_length)

        if time_len >= 192:
            weekly_block = context_tensor[:, time_len - 168 : time_len - 144]
        else:
            weekly_block = daily_block
        weekly_pattern = Model1Adapter._repeat_pattern(weekly_block, prediction_length)

        return (0.55 * daily_pattern) + (0.35 * weekly_pattern) + (0.10 * last_value)

    def _init_model(
        self,
        *,
        feature_dim: int,
        future_covariate_dim: int,
    ) -> None:
        self._model = _Model1Forecaster(
            feature_dim=feature_dim,
            future_covariate_dim=future_covariate_dim,
            prediction_length=self.model_ctx.prediction_length,
            hidden_dim=self._hidden_dim,
            dropout=self._dropout,
            attention_heads=self._attention_heads,
            ff_hidden_dim=self._ff_hidden_dim,
        )
        self._model.to(self.device)
        self._model.eval()
        self._feature_dim = int(feature_dim)
        self._future_covariate_dim = int(future_covariate_dim)

    def _ensure_model_initialized(
        self,
        *,
        feature_dim: int,
        future_covariate_dim: int,
    ) -> None:
        if self._model is None:
            self._init_model(
                feature_dim=feature_dim,
                future_covariate_dim=future_covariate_dim,
            )
            return

        if (
            self._feature_dim != feature_dim
            or self._future_covariate_dim != future_covariate_dim
        ):
            raise ValueError(
                "Model input dimensions do not match current data. "
                f"expected feature_dim={self._feature_dim}, future_covariate_dim={self._future_covariate_dim}; "
                f"got feature_dim={feature_dim}, future_covariate_dim={future_covariate_dim}."
            )

    def load_pretrained(self) -> None:
        self._model = None
        self._feature_dim = None
        self._future_covariate_dim = None

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
        del artifact_dir
        series = np.asarray(train_series, dtype=np.float32)
        if series.ndim != 1:
            raise ValueError(f"train_series must be 1D. Got shape={series.shape}.")

        context_length = int(self.model_ctx.context_length)
        prediction_length = int(self.model_ctx.prediction_length)
        min_required = context_length + prediction_length
        if series.shape[0] < min_required:
            raise ValueError(
                f"Need at least {min_required} training points, got {series.shape[0]}."
            )

        cov_train: np.ndarray | None = None
        fut_cov_train: np.ndarray | None = None
        if train_covariates is not None:
            cov_train = self._as_float2d(
                train_covariates,
                name="train_covariates",
                expected_length=series.shape[0],
            )
        if train_future_covariates is not None:
            fut_cov_train = self._as_float2d(
                train_future_covariates,
                name="train_future_covariates",
                expected_length=series.shape[0],
            )

        if cov_train is None and fut_cov_train is not None:
            raise ValueError(
                "train_future_covariates requires train_covariates in model_1."
            )

        if (
            cov_train is not None
            and fut_cov_train is not None
            and cov_train.shape[1] != fut_cov_train.shape[1]
        ):
            raise ValueError(
                "train_covariates and train_future_covariates feature-count mismatch. "
                f"train_covariates={cov_train.shape[1]} train_future_covariates={fut_cov_train.shape[1]}"
            )

        cov_dim = int(cov_train.shape[1]) if cov_train is not None else 0
        feature_dim = 1 + cov_dim
        self._init_model(feature_dim=feature_dim, future_covariate_dim=cov_dim)

        if self._model is None:
            raise RuntimeError("Model1Adapter failed to initialize model.")

        if cov_train is not None:
            ds = RandomWindowDatasetWithCovariates(
                series=series,
                covariates=cov_train,
                future_covariates=fut_cov_train,
                context_length=context_length,
                prediction_length=prediction_length,
                n_samples=max(1, train_batch_size * train_steps_per_epoch),
            )
        else:
            ds = RandomWindowDataset(
                series=series,
                context_length=context_length,
                prediction_length=prediction_length,
                n_samples=max(1, train_batch_size * train_steps_per_epoch),
            )

        dl = DataLoader(ds, batch_size=max(1, train_batch_size), shuffle=False)

        optimizer_name = train_optimizer or "adamw"
        optimizer = self._build_optimizer(
            model=self._model,
            name=optimizer_name,
            lr=float(train_lr),
            weight_decay=float(train_weight_decay),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, int(train_epochs)),
        )

        loss_name = train_loss or "mse"
        loss_fn = self._select_loss(loss_name)

        if checkpoint_selection not in {"best-train-loss", "last"}:
            raise ValueError(
                f"Unsupported checkpoint selection '{checkpoint_selection}'. "
                "Supported: best-train-loss, last"
            )

        history: list[TrainingLossPoint] = []
        best_loss = float("inf")
        best_state_dict: dict[str, torch.Tensor] | None = None

        self._model.train()
        for epoch in range(max(1, int(train_epochs))):
            epoch_losses: list[float] = []
            for batch in dl:
                context_tensor = batch["context"].to(self.device).float()
                target_tensor = batch["future_target"].to(self.device).float()
                context_cov_tensor = None
                future_cov_tensor = None
                if "context_covariates" in batch:
                    context_cov_tensor = (
                        batch["context_covariates"].to(self.device).float()
                    )
                if "future_covariates" in batch:
                    future_cov_tensor = (
                        batch["future_covariates"].to(self.device).float()
                    )

                baseline = self._build_reference_baseline(
                    context_tensor=context_tensor,
                    prediction_length=prediction_length,
                )
                scale = torch.std(context_tensor, dim=1, keepdim=True).clamp(min=1.0)
                context_norm = (
                    context_tensor - context_tensor[:, -1:].detach()
                ) / scale
                target_norm = (target_tensor - baseline) / scale

                if context_cov_tensor is not None:
                    context_cov_tensor, future_cov_tensor = self._normalize_covariates(
                        context_cov_tensor,
                        future_cov_tensor,
                    )

                y_pred = self._model(
                    context=context_norm,
                    context_covariates=context_cov_tensor,
                    future_covariates=future_cov_tensor,
                )
                loss = loss_fn(y_pred, target_norm)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))

            scheduler.step()

            if epoch_losses:
                mean_loss = float(np.mean(epoch_losses))
                history.append(TrainingLossPoint(epoch=epoch, loss=mean_loss))
                if checkpoint_selection == "best-train-loss" and mean_loss < best_loss:
                    best_loss = mean_loss
                    best_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in self._model.state_dict().items()
                    }

        if checkpoint_selection == "best-train-loss" and best_state_dict is not None:
            self._model.load_state_dict(best_state_dict)

        self._model.eval()
        return history

    def forecast(
        self,
        context: np.ndarray,
        context_start: pd.Timestamp,
        context_covariates: np.ndarray | None = None,
        future_covariates: np.ndarray | None = None,
    ) -> ForecastResult:
        del context_start
        context_array = np.asarray(context, dtype=np.float32)
        if context_array.ndim != 1:
            raise ValueError(f"context must be 1D. Got shape={context_array.shape}.")
        if context_array.size == 0:
            raise ValueError("context must not be empty.")

        effective_context_length = int(
            min(self.model_ctx.context_length, context_array.shape[0])
        )
        context_tail = context_array[-effective_context_length:]

        context_cov_tail: np.ndarray | None = None
        cov_dim = 0
        if context_covariates is not None:
            full_context_cov = self._as_float2d(
                context_covariates,
                name="context_covariates",
                expected_length=context_array.shape[0],
            )
            context_cov_tail = full_context_cov[-effective_context_length:]
            cov_dim = int(context_cov_tail.shape[1])

        future_cov_view: np.ndarray | None = None
        if future_covariates is not None:
            future_cov = self._as_float2d(
                future_covariates,
                name="future_covariates",
            )
            if future_cov.shape[0] < self.model_ctx.prediction_length:
                raise ValueError(
                    "future_covariates does not contain enough rows for prediction horizon. "
                    f"Required {self.model_ctx.prediction_length}, got {future_cov.shape[0]}."
                )
            future_cov_view = future_cov[: self.model_ctx.prediction_length]
            if cov_dim > 0 and future_cov_view.shape[1] != cov_dim:
                raise ValueError(
                    "future covariate feature count mismatch between context and future arrays. "
                    f"context={cov_dim} future={future_cov_view.shape[1]}"
                )
            cov_dim = int(future_cov_view.shape[1])

        feature_dim = 1 + cov_dim
        self._ensure_model_initialized(
            feature_dim=feature_dim,
            future_covariate_dim=cov_dim,
        )

        if self._model is None:
            raise RuntimeError("Model1Adapter model is not initialized.")

        context_tensor = torch.from_numpy(context_tail).unsqueeze(0).to(self.device)
        context_cov_tensor = (
            torch.from_numpy(context_cov_tail).unsqueeze(0).to(self.device)
            if context_cov_tail is not None
            else None
        )
        future_cov_tensor = (
            torch.from_numpy(future_cov_view).unsqueeze(0).to(self.device)
            if future_cov_view is not None
            else None
        )

        self._model.eval()
        with torch.no_grad():
            baseline = self._build_reference_baseline(
                context_tensor=context_tensor,
                prediction_length=self.model_ctx.prediction_length,
            )
            scale = torch.std(context_tensor, dim=1, keepdim=True).clamp(min=1.0)
            context_norm = (context_tensor - context_tensor[:, -1:]) / scale
            if context_cov_tensor is not None:
                context_cov_tensor, future_cov_tensor = self._normalize_covariates(
                    context_cov_tensor,
                    future_cov_tensor,
                )

            y_pred_residual = self._model(
                context=context_norm.float(),
                context_covariates=(
                    context_cov_tensor.float()
                    if context_cov_tensor is not None
                    else None
                ),
                future_covariates=(
                    future_cov_tensor.float() if future_cov_tensor is not None else None
                ),
            )

        y_pred = (y_pred_residual * scale) + baseline

        output = y_pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if output.shape[0] != self.model_ctx.prediction_length:
            raise RuntimeError(
                f"Bad forecast length for {self.slug}: got {output.shape[0]} expected {self.model_ctx.prediction_length}"
            )
        return ForecastResult(y_pred=output)

    def save_finetuned(self, artifact_dir: Path) -> None:
        if (
            self._model is None
            or self._feature_dim is None
            or self._future_covariate_dim is None
        ):
            raise RuntimeError("Model1Adapter model is not loaded.")

        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self._model.state_dict().items()
            },
            "feature_dim": int(self._feature_dim),
            "future_covariate_dim": int(self._future_covariate_dim),
            "prediction_length": int(self.model_ctx.prediction_length),
            "hidden_dim": int(self._hidden_dim),
            "ff_hidden_dim": int(self._ff_hidden_dim),
            "dropout": float(self._dropout),
            "attention_heads": int(self._attention_heads),
        }
        torch.save(payload, artifact_dir / "model.pt")

    def load_finetuned(self, artifact_dir: Path) -> None:
        checkpoint_path = artifact_dir / "model.pt"
        payload = torch.load(checkpoint_path, map_location="cpu")

        if int(
            payload.get("prediction_length", self.model_ctx.prediction_length)
        ) != int(self.model_ctx.prediction_length):
            raise ValueError(
                "Checkpoint prediction_length does not match runtime config. "
                f"checkpoint={payload.get('prediction_length')} runtime={self.model_ctx.prediction_length}"
            )

        self._hidden_dim = int(payload.get("hidden_dim", self._hidden_dim))
        self._ff_hidden_dim = int(payload.get("ff_hidden_dim", self._ff_hidden_dim))
        self._dropout = float(payload.get("dropout", self._dropout))
        self._attention_heads = int(
            payload.get("attention_heads", self._attention_heads)
        )

        feature_dim = int(payload["feature_dim"])
        future_covariate_dim = int(payload["future_covariate_dim"])
        self._init_model(
            feature_dim=feature_dim,
            future_covariate_dim=future_covariate_dim,
        )

        if self._model is None:
            raise RuntimeError(
                "Model1Adapter failed to initialize while loading checkpoint."
            )

        self._model.load_state_dict(payload["state_dict"])
        self._model.to(self.device)
        self._model.eval()
