from __future__ import annotations

from abc import abstractmethod
from collections.abc import Callable
from pathlib import Path
from typing import Any

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


class SequenceForecaster(nn.Module):
    def __init__(
        self,
        *,
        feature_dim: int,
        future_covariate_dim: int,
        prediction_length: int,
        hidden_dim: int,
        dropout: float,
        use_temporal_conv: bool,
        use_attention: bool,
        baseline_mode: str,
        attention_heads: int = 4,
        ff_hidden_dim: int | None = None,
    ) -> None:
        super().__init__()
        if hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even for bidirectional GRU.")
        if use_attention and hidden_dim % attention_heads != 0:
            raise ValueError(
                "hidden_dim must be divisible by attention_heads when attention is enabled."
            )

        self.feature_dim = int(feature_dim)
        self.future_covariate_dim = int(future_covariate_dim)
        self.prediction_length = int(prediction_length)
        self.hidden_dim = int(hidden_dim)
        self.use_temporal_conv = bool(use_temporal_conv)
        self.use_attention = bool(use_attention)
        self.baseline_mode = str(baseline_mode)
        self.attention_heads = int(attention_heads)
        self.ff_hidden_dim = (
            int(ff_hidden_dim) if ff_hidden_dim is not None else int(hidden_dim * 2)
        )
        if self.baseline_mode not in {
            "persistence",
            "seasonal_fixed",
            "seasonal_learned",
        }:
            raise ValueError(
                "Unsupported baseline_mode. "
                "Expected one of: persistence, seasonal_fixed, seasonal_learned."
            )

        self.input_norm = nn.LayerNorm(self.feature_dim)
        self.input_projection = nn.Linear(self.feature_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        if self.use_temporal_conv:
            self.temporal_conv = nn.Conv1d(
                in_channels=hidden_dim,
                out_channels=hidden_dim,
                kernel_size=3,
                padding=1,
            )
            self.conv_norm = nn.GroupNorm(1, hidden_dim)

        self.sequence_encoder = nn.GRU(
            input_size=hidden_dim,
            hidden_size=hidden_dim // 2,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        if self.use_attention:
            self.temporal_attention = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=attention_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attention_norm = nn.LayerNorm(hidden_dim)
            self.feed_forward = nn.Sequential(
                nn.Linear(hidden_dim, self.ff_hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(self.ff_hidden_dim, hidden_dim),
            )
            self.feed_forward_norm = nn.LayerNorm(hidden_dim)

        self.prediction_head = nn.Sequential(
            nn.Linear((2 * hidden_dim) + self.future_covariate_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, self.prediction_length),
        )

        if self.baseline_mode == "seasonal_learned":
            self.baseline_gate = nn.Sequential(
                nn.Linear((2 * hidden_dim) + self.future_covariate_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 3),
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

    @staticmethod
    def _repeat_pattern(block: torch.Tensor, prediction_length: int) -> torch.Tensor:
        if block.ndim != 2:
            raise ValueError(
                f"Pattern block must be 2D [batch, time]. Got shape={tuple(block.shape)}."
            )
        if block.shape[1] <= 0:
            raise ValueError("Pattern block must have positive time length.")
        repeats = (prediction_length + block.shape[1] - 1) // block.shape[1]
        return block.repeat(1, repeats)[:, :prediction_length]

    def build_baseline(
        self,
        *,
        raw_context: torch.Tensor,
        future_covariates: torch.Tensor | None,
        pooled_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if raw_context.ndim != 2:
            raise ValueError(
                f"raw_context must be [batch, time]. Got shape={tuple(raw_context.shape)}."
            )

        prediction_length = self.prediction_length
        time_len = int(raw_context.shape[1])
        last_value = raw_context[:, -1:].expand(-1, prediction_length)

        if self.baseline_mode == "persistence":
            return last_value

        daily_block = raw_context[:, max(0, time_len - 24) : time_len]
        daily_pattern = self._repeat_pattern(daily_block, prediction_length)

        if time_len >= 192:
            weekly_block = raw_context[:, time_len - 168 : time_len - 144]
        else:
            weekly_block = daily_block
        weekly_pattern = self._repeat_pattern(weekly_block, prediction_length)

        if self.baseline_mode == "seasonal_fixed":
            return (0.15 * last_value) + (0.55 * daily_pattern) + (0.30 * weekly_pattern)

        if pooled_features is None:
            raise ValueError(
                "pooled_features are required when baseline_mode='seasonal_learned'."
            )

        gate_logits = self.baseline_gate(pooled_features)
        weights = torch.softmax(gate_logits, dim=1)
        components = torch.stack((last_value, daily_pattern, weekly_pattern), dim=1)
        return torch.sum(components * weights.unsqueeze(-1), dim=1)

    def forward(
        self,
        *,
        context: torch.Tensor,
        context_covariates: torch.Tensor | None,
        future_covariates: torch.Tensor | None,
        raw_context: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if context.ndim != 2:
            raise ValueError(
                f"context must be [batch, time]. Got shape={tuple(context.shape)}."
            )
        if raw_context is None:
            raw_context = context
        if raw_context.shape != context.shape:
            raise ValueError(
                "raw_context and context must have the same batch/time dimensions. "
                f"context={tuple(context.shape)} raw_context={tuple(raw_context.shape)}"
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
        x = self.dropout(x)

        if self.use_temporal_conv:
            residual = x
            x = x.transpose(1, 2)
            x = self.temporal_conv(x)
            x = self.conv_norm(x)
            x = torch.nn.functional.gelu(x)
            x = self.dropout(x)
            x = x.transpose(1, 2)
            x = x + residual

        x, _ = self.sequence_encoder(x)

        if self.use_attention:
            x_attn, _ = self.temporal_attention(x, x, x, need_weights=False)
            x = self.attention_norm(x + x_attn)
            x = self.feed_forward_norm(x + self.feed_forward(x))

        pooled_mean = torch.mean(x, dim=1)
        pooled_max = torch.max(x, dim=1).values
        pooled = torch.cat((pooled_mean, pooled_max), dim=1)

        future_summary = self._build_future_summary(
            future_covariates=future_covariates,
            batch_size=pooled.shape[0],
            device=pooled.device,
        )
        features = torch.cat((pooled, future_summary), dim=1)
        baseline = self.build_baseline(
            raw_context=raw_context,
            future_covariates=future_covariates,
            pooled_features=features,
        )
        residual = self.prediction_head(features)
        return residual, baseline


class BaseCustomSequenceAdapter(BaseFoundationModelAdapter):
    model_family = "custom"
    supports_finetune = True

    def __init__(self, model_ctx: ModelContext, device: torch.device) -> None:
        super().__init__(model_ctx, device)
        self._model: SequenceForecaster | None = None
        self._feature_dim: int | None = None
        self._future_covariate_dim: int | None = None

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
    def _normalize_target(
        context_tensor: torch.Tensor,
        target_tensor: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        center = context_tensor[:, -1:].detach()
        scale = torch.std(context_tensor, dim=1, keepdim=True).clamp(min=1.0)
        context_norm = (context_tensor - center) / scale
        target_norm = None
        if target_tensor is not None:
            target_norm = (target_tensor - center) / scale
        return context_norm, scale, target_norm

    @staticmethod
    def _normalize_residual_target(
        *,
        context_tensor: torch.Tensor,
        target_tensor: torch.Tensor,
        baseline_tensor: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        scale = torch.std(context_tensor, dim=1, keepdim=True).clamp(min=1.0)
        target_norm = (target_tensor - baseline_tensor) / scale
        return scale, target_norm

    @abstractmethod
    def _model_kwargs(self) -> dict[str, Any]:
        raise NotImplementedError

    def _init_model(
        self,
        *,
        feature_dim: int,
        future_covariate_dim: int,
    ) -> None:
        self._model = SequenceForecaster(
            feature_dim=feature_dim,
            future_covariate_dim=future_covariate_dim,
            prediction_length=self.model_ctx.prediction_length,
            **self._model_kwargs(),
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

    def checkpoint_signature(self) -> dict[str, object]:
        return {
            "model_id": self.model_id,
            "adapter_class": self.__class__.__name__,
            **self._model_kwargs(),
        }

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
                f"train_future_covariates requires train_covariates in {self.slug}."
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
            raise RuntimeError(f"{self.__class__.__name__} failed to initialize model.")

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
        total_epochs = max(1, int(train_epochs))
        steps_per_epoch = max(1, len(dl))
        use_one_cycle = total_epochs >= 20
        if use_one_cycle:
            scheduler = torch.optim.lr_scheduler.OneCycleLR(
                optimizer,
                max_lr=float(train_lr),
                epochs=total_epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.1,
                div_factor=10.0,
                final_div_factor=100.0,
            )
        else:
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=total_epochs,
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
        for epoch in range(total_epochs):
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

                context_norm, _, target_norm = self._normalize_target(
                    context_tensor=context_tensor,
                    target_tensor=target_tensor,
                )

                if context_cov_tensor is not None:
                    context_cov_tensor, future_cov_tensor = self._normalize_covariates(
                        context_cov_tensor,
                        future_cov_tensor,
                    )

                y_pred_residual, baseline = self._model(
                    context=context_norm,
                    context_covariates=context_cov_tensor,
                    future_covariates=future_cov_tensor,
                    raw_context=context_tensor,
                )
                if target_norm is None:
                    raise RuntimeError("Normalized target is unexpectedly missing.")
                _, residual_target = self._normalize_residual_target(
                    context_tensor=context_tensor,
                    target_tensor=target_tensor,
                    baseline_tensor=baseline,
                )
                loss = loss_fn(y_pred_residual, residual_target)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                if use_one_cycle:
                    scheduler.step()
                epoch_losses.append(float(loss.detach().cpu().item()))

            if not use_one_cycle:
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
            raise RuntimeError(f"{self.slug} model is not initialized.")

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
            context_norm, scale, _ = self._normalize_target(
                context_tensor=context_tensor,
                target_tensor=None,
            )
            if context_cov_tensor is not None:
                context_cov_tensor, future_cov_tensor = self._normalize_covariates(
                    context_cov_tensor,
                    future_cov_tensor,
                )

            y_pred_norm, baseline = self._model(
                context=context_norm.float(),
                context_covariates=(
                    context_cov_tensor.float()
                    if context_cov_tensor is not None
                    else None
                ),
                future_covariates=(
                    future_cov_tensor.float() if future_cov_tensor is not None else None
                ),
                raw_context=context_tensor.float(),
            )

        y_pred = (y_pred_norm * scale) + baseline

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
            raise RuntimeError(f"{self.__class__.__name__} model is not loaded.")

        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "state_dict": {
                key: value.detach().cpu().clone()
                for key, value in self._model.state_dict().items()
            },
            "feature_dim": int(self._feature_dim),
            "future_covariate_dim": int(self._future_covariate_dim),
            "prediction_length": int(self.model_ctx.prediction_length),
            **self._model_kwargs(),
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

        self._load_checkpoint_hparams(payload)

        feature_dim = int(payload["feature_dim"])
        future_covariate_dim = int(payload["future_covariate_dim"])
        self._init_model(
            feature_dim=feature_dim,
            future_covariate_dim=future_covariate_dim,
        )

        if self._model is None:
            raise RuntimeError(
                f"{self.__class__.__name__} failed to initialize while loading checkpoint."
            )

        self._model.load_state_dict(payload["state_dict"])
        self._model.to(self.device)
        self._model.eval()

    def _load_checkpoint_hparams(self, payload: dict[str, Any]) -> None:
        del payload
