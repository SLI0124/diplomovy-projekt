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


class _Model2Transformer(nn.Module):
    def __init__(
        self,
        input_size: int,
        future_covariate_size: int,
        hidden_size: int,
        num_layers: int,
        num_heads: int,
        ffn_size: int,
        dropout: float,
        prediction_length: int,
        context_length: int,
    ) -> None:
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError(
                f"hidden_size must be divisible by num_heads. Got {hidden_size} and {num_heads}."
            )

        self.prediction_length = int(prediction_length)
        self.context_length = int(context_length)
        self.future_covariate_size = int(max(0, future_covariate_size))

        self.input_proj = nn.Linear(int(max(1, input_size)), hidden_size)
        self.positional_embedding = nn.Parameter(
            torch.zeros(1, self.context_length, hidden_size)
        )

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=num_heads,
            dim_feedforward=ffn_size,
            dropout=dropout,
            batch_first=True,
            norm_first=False,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer=encoder_layer,
            num_layers=num_layers,
        )

        self.context_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        head_input_dim = hidden_size
        self.future_covariate_head: nn.Sequential | None = None
        if self.future_covariate_size > 0:
            self.future_covariate_head = nn.Sequential(
                nn.Linear(
                    self.future_covariate_size * self.prediction_length, hidden_size
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            head_input_dim += hidden_size

        self.head = nn.Sequential(
            nn.Linear(head_input_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, self.prediction_length),
        )

    def forward(
        self,
        x: torch.Tensor,
        future_covariates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # x shape: [batch, time, features]
        batch_size, seq_len, _ = x.shape
        if seq_len > self.context_length:
            raise ValueError(
                f"Input context length exceeds configured model context_length. "
                f"Got {seq_len}, max {self.context_length}."
            )

        pos = self.positional_embedding[:, :seq_len, :]
        encoded = self.encoder(self.input_proj(x) + pos)
        pooled = torch.cat((encoded[:, -1, :], encoded.mean(dim=1)), dim=-1)
        features = self.context_head(pooled)

        if self.future_covariate_head is not None:
            if future_covariates is None:
                future_covariates = torch.zeros(
                    (batch_size, self.prediction_length, self.future_covariate_size),
                    dtype=x.dtype,
                    device=x.device,
                )
            future_cov_flat = future_covariates.reshape(batch_size, -1)
            future_feats = self.future_covariate_head(future_cov_flat)
            features = torch.cat((features, future_feats), dim=-1)

        return self.head(features)


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


class Model2Adapter(BaseFoundationModelAdapter):
    model_id = "custom/model_2"
    slug = "model_2"
    model_family = "custom"
    supports_finetune = True

    def __init__(self, model_ctx: ModelContext, device: torch.device) -> None:
        super().__init__(model_ctx, device)
        self._arch_version: int = 2
        self._hidden_size: int = 64
        self._num_layers: int = 2
        self._num_heads: int = 4
        self._ffn_size: int = 128
        self._dropout: float = 0.2
        self._model: nn.Module | None = None
        self._num_covariates: int = 0
        self._target_mean: float = 0.0
        self._target_std: float = 1.0
        self._target_clip_low: float = float("-inf")
        self._target_clip_high: float = float("inf")
        self._covariate_mean: np.ndarray = np.empty((0,), dtype=np.float32)
        self._covariate_std: np.ndarray = np.empty((0,), dtype=np.float32)
        self._covariate_clip_low: np.ndarray = np.empty((0,), dtype=np.float32)
        self._covariate_clip_high: np.ndarray = np.empty((0,), dtype=np.float32)
        self._loaded: bool = False

    @staticmethod
    def _as_float2d(
        array: np.ndarray | None,
        *,
        name: str,
        expected_length: int | None = None,
    ) -> np.ndarray | None:
        if array is None:
            return None
        out = np.asarray(array, dtype=np.float32)
        if out.ndim != 2:
            raise ValueError(f"{name} must be a 2D array. Got shape={out.shape}.")
        if expected_length is not None and out.shape[0] != expected_length:
            raise ValueError(
                f"{name} length mismatch. Expected {expected_length}, got {out.shape[0]}."
            )
        return out

    @staticmethod
    def _compute_feature_stats(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        with np.errstate(invalid="ignore"):
            mean = np.nanmean(values, axis=0).astype(np.float32)
            std = np.nanstd(values, axis=0).astype(np.float32)
        mean = np.where(np.isfinite(mean), mean, 0.0).astype(np.float32)
        std = np.where(np.isfinite(std) & (std > 1e-6), std, 1.0).astype(np.float32)
        return mean, std

    @staticmethod
    def _normalize_features(
        values: np.ndarray,
        mean: np.ndarray,
        std: np.ndarray,
    ) -> np.ndarray:
        out = (values - mean) / std
        return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    @staticmethod
    def _compute_scalar_clip_bounds(values: np.ndarray) -> tuple[float, float]:
        valid = values[np.isfinite(values)]
        if valid.size == 0:
            return float("-inf"), float("inf")
        low = float(np.quantile(valid, 0.005))
        high = float(np.quantile(valid, 0.995))
        if high <= low:
            return float("-inf"), float("inf")
        return low, high

    @staticmethod
    def _compute_feature_clip_bounds(
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        with np.errstate(invalid="ignore"):
            low = np.nanquantile(values, 0.005, axis=0).astype(np.float32)
            high = np.nanquantile(values, 0.995, axis=0).astype(np.float32)
        invalid = (~np.isfinite(low)) | (~np.isfinite(high)) | (high <= low)
        low = np.where(invalid, -np.inf, low).astype(np.float32)
        high = np.where(invalid, np.inf, high).astype(np.float32)
        return low, high

    @staticmethod
    def _clip_features(
        values: np.ndarray, low: np.ndarray, high: np.ndarray
    ) -> np.ndarray:
        return np.clip(values, low, high).astype(np.float32)

    @staticmethod
    def _local_normalize_target(
        target_context: np.ndarray,
        target_future: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        loc_mean = float(np.mean(target_context))
        loc_std = float(np.std(target_context))
        if not np.isfinite(loc_std) or loc_std <= 1e-6:
            loc_std = 1.0
        context_norm = (target_context - loc_mean) / loc_std
        future_norm = (target_future - loc_mean) / loc_std
        return context_norm.astype(np.float32), future_norm.astype(np.float32)

    @staticmethod
    def _pad_left_1d(values: np.ndarray, target_len: int) -> np.ndarray:
        if values.size >= target_len:
            return values[-target_len:]
        if values.size == 0:
            return np.zeros((target_len,), dtype=np.float32)
        pad_value = float(values[-1])
        pad = np.full((target_len - values.size,), pad_value, dtype=np.float32)
        return np.concatenate((pad, values), axis=0)

    @staticmethod
    def _pad_left_2d(values: np.ndarray, target_len: int) -> np.ndarray:
        if values.shape[0] >= target_len:
            return values[-target_len:, :]
        if values.shape[0] == 0:
            return np.zeros((target_len, values.shape[1]), dtype=np.float32)
        pad_row = values[-1:, :]
        pad = np.repeat(pad_row, target_len - values.shape[0], axis=0)
        return np.concatenate((pad, values), axis=0)

    @staticmethod
    def _align_future_covariates(
        values: np.ndarray,
        prediction_length: int,
    ) -> np.ndarray:
        if values.shape[0] >= prediction_length:
            return values[:prediction_length, :]
        if values.shape[0] == 0:
            return np.zeros((prediction_length, values.shape[1]), dtype=np.float32)
        pad_row = values[-1:, :]
        pad = np.repeat(pad_row, prediction_length - values.shape[0], axis=0)
        return np.concatenate((values, pad), axis=0)

    def _build_model(self) -> _Model2Transformer:
        model = _Model2Transformer(
            input_size=1 + self._num_covariates,
            future_covariate_size=self._num_covariates,
            hidden_size=self._hidden_size,
            num_layers=self._num_layers,
            num_heads=self._num_heads,
            ffn_size=self._ffn_size,
            dropout=self._dropout,
            prediction_length=int(self.model_ctx.prediction_length),
            context_length=int(self.model_ctx.context_length),
        )
        model.to(self.device)
        return model

    def load_pretrained(self) -> None:
        self._num_covariates = 0
        self._target_mean = 0.0
        self._target_std = 1.0
        self._target_clip_low = float("-inf")
        self._target_clip_high = float("inf")
        self._covariate_mean = np.empty((0,), dtype=np.float32)
        self._covariate_std = np.empty((0,), dtype=np.float32)
        self._covariate_clip_low = np.empty((0,), dtype=np.float32)
        self._covariate_clip_high = np.empty((0,), dtype=np.float32)
        self._model = self._build_model()
        self._model.eval()
        self._loaded = True

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
        artifact_dir.mkdir(parents=True, exist_ok=True)

        series = np.asarray(train_series, dtype=np.float32)
        if series.size == 0:
            raise ValueError("Model2Adapter requires non-empty train_series.")

        covariates = self._as_float2d(
            train_covariates,
            name="train_covariates",
            expected_length=series.size,
        )
        future_covariates = self._as_float2d(
            train_future_covariates,
            name="train_future_covariates",
            expected_length=series.size,
        )

        if covariates is None and future_covariates is not None:
            raise ValueError(
                "train_future_covariates were provided, but train_covariates is None. "
                "Provide both arrays in covariate mode."
            )

        self._num_covariates = 0 if covariates is None else int(covariates.shape[1])
        if (
            self._num_covariates > 0
            and future_covariates is not None
            and future_covariates.shape[1] != self._num_covariates
        ):
            raise ValueError(
                "Covariate feature mismatch between train_covariates and "
                f"train_future_covariates: {self._num_covariates} vs {future_covariates.shape[1]}."
            )

        self._target_clip_low, self._target_clip_high = (
            self._compute_scalar_clip_bounds(series)
        )
        series = np.clip(series, self._target_clip_low, self._target_clip_high).astype(
            np.float32
        )

        self._target_mean = float(series.mean())
        std = float(series.std())
        self._target_std = std if std > 1e-6 else 1.0

        covariates_norm: np.ndarray | None = None
        future_covariates_norm: np.ndarray | None = None
        if self._num_covariates > 0 and covariates is not None:
            self._covariate_clip_low, self._covariate_clip_high = (
                self._compute_feature_clip_bounds(covariates)
            )
            covariates = self._clip_features(
                covariates,
                self._covariate_clip_low,
                self._covariate_clip_high,
            )
            if future_covariates is not None:
                future_covariates = self._clip_features(
                    future_covariates,
                    self._covariate_clip_low,
                    self._covariate_clip_high,
                )

            self._covariate_mean, self._covariate_std = self._compute_feature_stats(
                covariates
            )
            covariates_norm = self._normalize_features(
                covariates,
                self._covariate_mean,
                self._covariate_std,
            )
            if future_covariates is not None:
                future_covariates_norm = self._normalize_features(
                    future_covariates,
                    self._covariate_mean,
                    self._covariate_std,
                )
        else:
            self._covariate_mean = np.empty((0,), dtype=np.float32)
            self._covariate_std = np.empty((0,), dtype=np.float32)
            self._covariate_clip_low = np.empty((0,), dtype=np.float32)
            self._covariate_clip_high = np.empty((0,), dtype=np.float32)

        prediction_length = int(max(1, self.model_ctx.prediction_length))
        context_length = int(
            min(
                max(1, self.model_ctx.context_length),
                max(1, series.size - prediction_length),
            )
        )

        start_min = context_length
        start_max = int(series.size - prediction_length)
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
        if checkpoint_selection not in {"best-train-loss", "last"}:
            raise ValueError(
                f"Unsupported checkpoint selection '{checkpoint_selection}'. "
                "Supported: best-train-loss, last"
            )
        loss_fn = self._build_loss_fn(train_loss)
        rng = np.random.default_rng()
        history: list[TrainingLossPoint] = []
        best_loss = float("inf")
        best_state_dict: dict[str, torch.Tensor] | None = None

        for ep in range(epochs):
            epoch_losses: list[float] = []
            for _ in range(steps_per_epoch):
                starts = rng.integers(start_min, start_max + 1, size=batch_size)

                contexts_np_list: list[np.ndarray] = []
                targets_np_list: list[np.ndarray] = []
                future_covariates_np_list: list[np.ndarray] = []
                for start in starts:
                    target_context_raw = series[start - context_length : start]
                    target_future_raw = series[start : start + prediction_length]
                    target_context, target_future = self._local_normalize_target(
                        target_context_raw,
                        target_future_raw,
                    )
                    if covariates_norm is None:
                        context_window = target_context[:, None]
                    else:
                        cov_context = covariates_norm[start - context_length : start, :]
                        context_window = np.concatenate(
                            (target_context[:, None], cov_context),
                            axis=1,
                        )
                    contexts_np_list.append(context_window.astype(np.float32))
                    targets_np_list.append(target_future.astype(np.float32))

                    if covariates_norm is not None:
                        if future_covariates_norm is None:
                            future_window = np.zeros(
                                (prediction_length, self._num_covariates),
                                dtype=np.float32,
                            )
                        else:
                            future_window = future_covariates_norm[
                                start : start + prediction_length,
                                :,
                            ]
                        future_covariates_np_list.append(
                            future_window.astype(np.float32)
                        )

                contexts_np = np.stack(contexts_np_list, axis=0)
                targets_np = np.stack(targets_np_list, axis=0)

                contexts = torch.from_numpy(contexts_np).to(self.device).float()
                targets = torch.from_numpy(targets_np).to(self.device).float()
                future_cov_batch: torch.Tensor | None = None
                if future_covariates_np_list:
                    future_cov_batch = (
                        torch.from_numpy(np.stack(future_covariates_np_list, axis=0))
                        .to(self.device)
                        .float()
                    )

                preds = self._model(contexts, future_covariates=future_cov_batch)
                loss = loss_fn(preds, targets)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))

            if epoch_losses:
                mean_epoch_loss = float(np.mean(epoch_losses))
                history.append(
                    TrainingLossPoint(
                        epoch=ep,
                        loss=mean_epoch_loss,
                    )
                )
                if (
                    checkpoint_selection == "best-train-loss"
                    and mean_epoch_loss < best_loss
                ):
                    best_loss = mean_epoch_loss
                    best_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in self._model.state_dict().items()
                    }

        if checkpoint_selection == "best-train-loss" and best_state_dict is not None:
            self._model.load_state_dict(best_state_dict)

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
        self,
        context: np.ndarray,
        context_start: pd.Timestamp,
        context_covariates: np.ndarray | None = None,
        future_covariates: np.ndarray | None = None,
    ) -> ForecastResult:
        del context_start
        if not self._loaded:
            raise RuntimeError(
                "Model_2 is not loaded. Call load_pretrained/load_finetuned."
            )

        if self._model is None:
            raise RuntimeError("Model_2 internal model is not initialized.")

        x = np.asarray(context, dtype=np.float32)
        context_length = int(max(1, self.model_ctx.context_length))
        prediction_length = int(max(1, self.model_ctx.prediction_length))

        x = np.clip(x, self._target_clip_low, self._target_clip_high).astype(np.float32)
        x = self._pad_left_1d(x, context_length)

        context_cov: np.ndarray | None = None
        if self._num_covariates > 0:
            if context_covariates is None:
                context_cov = np.zeros(
                    (context_length, self._num_covariates),
                    dtype=np.float32,
                )
            else:
                context_cov = self._as_float2d(
                    context_covariates,
                    name="context_covariates",
                )
                if context_cov is None:
                    raise RuntimeError("Unexpected None covariates.")
                if context_cov.shape[1] != self._num_covariates:
                    raise ValueError(
                        "context_covariates feature mismatch for Model_2 forecast. "
                        f"Expected {self._num_covariates}, got {context_cov.shape[1]}."
                    )
                context_cov = self._pad_left_2d(context_cov, context_length)

        loc_mean = float(np.mean(x))
        loc_std = float(np.std(x))
        if not np.isfinite(loc_std) or loc_std <= 1e-6:
            loc_std = 1.0

        x_norm = (x - loc_mean) / loc_std
        if self._num_covariates > 0:
            if context_cov is None:
                context_cov_norm = np.zeros(
                    (context_length, self._num_covariates),
                    dtype=np.float32,
                )
            else:
                context_cov = self._clip_features(
                    context_cov,
                    self._covariate_clip_low,
                    self._covariate_clip_high,
                )
                context_cov_norm = self._normalize_features(
                    context_cov,
                    self._covariate_mean,
                    self._covariate_std,
                )
            x_input = np.concatenate((x_norm[:, None], context_cov_norm), axis=1)
        else:
            x_input = x_norm[:, None]

        x_t = (
            torch.from_numpy(x_input)
            .to(self.device)
            .float()
            .view(1, context_length, -1)
        )

        future_cov_t: torch.Tensor | None = None
        if self._num_covariates > 0:
            if future_covariates is None:
                future_cov = np.zeros(
                    (prediction_length, self._num_covariates),
                    dtype=np.float32,
                )
            else:
                future_cov = self._as_float2d(
                    future_covariates,
                    name="future_covariates",
                )
                if future_cov is None:
                    raise RuntimeError("Unexpected None future covariates.")
                if future_cov.shape[1] != self._num_covariates:
                    raise ValueError(
                        "future_covariates feature mismatch for Model_2 forecast. "
                        f"Expected {self._num_covariates}, got {future_cov.shape[1]}."
                    )
                future_cov = self._align_future_covariates(
                    future_cov, prediction_length
                )
                future_cov = self._clip_features(
                    future_cov,
                    self._covariate_clip_low,
                    self._covariate_clip_high,
                )

            future_cov_norm = self._normalize_features(
                future_cov,
                self._covariate_mean,
                self._covariate_std,
            )
            future_cov_t = (
                torch.from_numpy(future_cov_norm)
                .to(self.device)
                .float()
                .view(1, prediction_length, self._num_covariates)
            )

        with torch.no_grad():
            y_norm = (
                self._model(x_t, future_covariates=future_cov_t)[0]
                .detach()
                .cpu()
                .numpy()
            )

        y_pred = np.asarray(y_norm * loc_std + loc_mean, dtype=np.float32)
        y_pred = np.clip(y_pred, self._target_clip_low, self._target_clip_high).astype(
            np.float32
        )
        return ForecastResult(y_pred=y_pred)

    def save_finetuned(self, artifact_dir: Path) -> None:
        if not self._loaded:
            raise RuntimeError("Model_2 is not available to save.")
        if self._model is None:
            raise RuntimeError("Model_2 internal model is not initialized.")

        artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_id": self.model_id,
                "slug": self.slug,
                "arch_version": self._arch_version,
                "hidden_size": self._hidden_size,
                "num_layers": self._num_layers,
                "num_heads": self._num_heads,
                "ffn_size": self._ffn_size,
                "dropout": self._dropout,
                "num_covariates": self._num_covariates,
                "prediction_length": int(self.model_ctx.prediction_length),
                "context_length": int(self.model_ctx.context_length),
                "target_mean": self._target_mean,
                "target_std": self._target_std,
                "target_clip_low": self._target_clip_low,
                "target_clip_high": self._target_clip_high,
                "covariate_mean": self._covariate_mean.tolist(),
                "covariate_std": self._covariate_std.tolist(),
                "covariate_clip_low": self._covariate_clip_low.tolist(),
                "covariate_clip_high": self._covariate_clip_high.tolist(),
                "state_dict": self._model.state_dict(),
            },
            artifact_dir / "model.pt",
        )

        (artifact_dir / "model.json").write_text(
            json.dumps(
                {
                    "model_id": self.model_id,
                    "slug": self.slug,
                    "arch_version": self._arch_version,
                    "hidden_size": self._hidden_size,
                    "num_layers": self._num_layers,
                    "num_heads": self._num_heads,
                    "ffn_size": self._ffn_size,
                    "dropout": self._dropout,
                    "num_covariates": self._num_covariates,
                    "prediction_length": int(self.model_ctx.prediction_length),
                    "context_length": int(self.model_ctx.context_length),
                    "target_mean": self._target_mean,
                    "target_std": self._target_std,
                    "target_clip_low": self._target_clip_low,
                    "target_clip_high": self._target_clip_high,
                    "covariate_mean": self._covariate_mean.tolist(),
                    "covariate_std": self._covariate_std.tolist(),
                    "covariate_clip_low": self._covariate_clip_low.tolist(),
                    "covariate_clip_high": self._covariate_clip_high.tolist(),
                    "weights_file": "model.pt",
                },
                indent=2,
            ),
            encoding="utf-8",
        )

    def load_finetuned(self, artifact_dir: Path) -> None:
        model_path = artifact_dir / "model.pt"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing Model_2 artifact: {model_path}")

        payload = torch.load(model_path, map_location="cpu")
        self._hidden_size = int(payload.get("hidden_size", 64))
        self._num_layers = int(payload.get("num_layers", 2))
        self._num_heads = int(payload.get("num_heads", 4))
        self._ffn_size = int(payload.get("ffn_size", 128))
        self._dropout = float(payload.get("dropout", 0.2))
        self._num_covariates = int(payload.get("num_covariates", 0))

        self._target_mean = float(payload.get("target_mean", 0.0))
        std = float(payload.get("target_std", 1.0))
        self._target_std = std if std > 1e-6 else 1.0
        self._target_clip_low = float(payload.get("target_clip_low", float("-inf")))
        self._target_clip_high = float(payload.get("target_clip_high", float("inf")))

        covariate_mean = payload.get("covariate_mean", [])
        covariate_std = payload.get("covariate_std", [])
        covariate_clip_low = payload.get("covariate_clip_low", [])
        covariate_clip_high = payload.get("covariate_clip_high", [])
        self._covariate_mean = np.asarray(covariate_mean, dtype=np.float32)
        self._covariate_std = np.asarray(covariate_std, dtype=np.float32)
        self._covariate_clip_low = np.asarray(covariate_clip_low, dtype=np.float32)
        self._covariate_clip_high = np.asarray(covariate_clip_high, dtype=np.float32)
        if self._num_covariates == 0:
            self._covariate_mean = np.empty((0,), dtype=np.float32)
            self._covariate_std = np.empty((0,), dtype=np.float32)
            self._covariate_clip_low = np.empty((0,), dtype=np.float32)
            self._covariate_clip_high = np.empty((0,), dtype=np.float32)
        elif self._covariate_mean.shape != (
            self._num_covariates,
        ) or self._covariate_std.shape != (self._num_covariates,):
            raise ValueError(
                "Model_2 checkpoint has inconsistent covariate normalization metadata."
            )

        if self._num_covariates > 0 and (
            self._covariate_clip_low.shape != (self._num_covariates,)
            or self._covariate_clip_high.shape != (self._num_covariates,)
        ):
            self._covariate_clip_low = np.full(
                (self._num_covariates,),
                -np.inf,
                dtype=np.float32,
            )
            self._covariate_clip_high = np.full(
                (self._num_covariates,),
                np.inf,
                dtype=np.float32,
            )

        self._covariate_std = np.where(
            (self._covariate_std > 1e-6) & np.isfinite(self._covariate_std),
            self._covariate_std,
            1.0,
        ).astype(np.float32)
        self._covariate_mean = np.where(
            np.isfinite(self._covariate_mean),
            self._covariate_mean,
            0.0,
        ).astype(np.float32)

        state_dict = payload.get("state_dict")
        if state_dict is None:
            raise KeyError("Model_2 checkpoint missing state_dict")

        self._model = self._build_model()
        self._model.load_state_dict(state_dict)
        self._model.eval()
        self._loaded = True
