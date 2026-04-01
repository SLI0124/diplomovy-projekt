from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from adapters.base import (
    BaseFoundationModelAdapter,
    ForecastResult,
    ModelContext,
    TrainingLossPoint,
)
from adapters.shared import RandomWindowDataset, RandomWindowDatasetWithCovariates
from torch.utils.data import DataLoader


def _nanmean(values: torch.Tensor, dim: int, keepdim: bool) -> torch.Tensor:
    mask = torch.isfinite(values)
    safe = torch.where(mask, values, torch.zeros_like(values))
    denom = mask.sum(dim=dim, keepdim=keepdim).clamp_min(1)
    return safe.sum(dim=dim, keepdim=keepdim) / denom


def _nanstd(
    values: torch.Tensor,
    *,
    dim: int,
    keepdim: bool,
    eps: float,
) -> torch.Tensor:
    mean = _nanmean(values, dim=dim, keepdim=True)
    mask = torch.isfinite(values)
    safe = torch.where(mask, values, mean)
    centered = torch.where(mask, safe - mean, torch.zeros_like(safe))
    denom = mask.sum(dim=dim, keepdim=True).clamp_min(1)
    var = (centered * centered).sum(dim=dim, keepdim=True) / denom
    std = torch.sqrt(var + eps)
    if not keepdim:
        std = std.squeeze(dim)
    return std


def _smape_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    eps = 1e-5
    numer = torch.abs(y_pred - y_true)
    denom = torch.abs(y_true) + torch.abs(y_pred) + eps
    return (200.0 * numer / denom).mean()


def _mape_loss(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    eps = 1e-5
    return (100.0 * torch.abs(y_pred - y_true) / (torch.abs(y_true) + eps)).mean()


class _TemporalResBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.conv_1 = nn.Conv1d(
            channels,
            channels,
            kernel_size=3,
            dilation=dilation,
            padding=dilation,
        )
        self.conv_2 = nn.Conv1d(channels, channels, kernel_size=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        y = self.norm(x)
        y = y.transpose(1, 2)
        y = self.conv_1(y)
        y = F.gelu(y)
        y = self.dropout(y)
        y = self.conv_2(y)
        y = y.transpose(1, 2)
        return residual + self.dropout(y)


class _HybridSeqModel(nn.Module):
    def __init__(
        self,
        *,
        input_dim: int,
        context_length: int,
        horizon: int,
        recurrent_cell: str,
        hidden_size: int,
        recurrent_layers: int,
        bidirectional: bool,
        dropout: float,
        model_dim: int,
        use_attention: bool,
        attention_heads: int,
        future_covariate_dim: int,
    ) -> None:
        super().__init__()
        self.context_length = int(context_length)
        self.horizon = int(horizon)
        self.future_covariate_dim = int(future_covariate_dim)

        # [1] Input conditioning: normalize feature scales and project to model width.
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_proj = nn.Linear(input_dim, model_dim)
        self.dropout = nn.Dropout(dropout)

        # [2] Temporal convolution stack: captures short/medium local patterns with dilation.
        self.temporal_blocks = nn.ModuleList(
            [
                _TemporalResBlock(model_dim, dilation=1, dropout=dropout),
                _TemporalResBlock(model_dim, dilation=2, dropout=dropout),
                _TemporalResBlock(model_dim, dilation=4, dropout=dropout),
            ]
        )

        rec_name = recurrent_cell.strip().lower()
        rnn_cls: type[nn.RNNBase]
        if rec_name == "gru":
            rnn_cls = nn.GRU
        elif rec_name == "rnn":
            rnn_cls = nn.RNN
        else:
            rnn_cls = nn.LSTM

        rec_dropout = dropout if recurrent_layers > 1 else 0.0
        # [3] Recurrent encoder: models sequential dependencies over long context.
        self.recurrent = rnn_cls(
            input_size=model_dim,
            hidden_size=hidden_size,
            num_layers=recurrent_layers,
            dropout=rec_dropout,
            batch_first=True,
            bidirectional=bidirectional,
        )

        recurrent_dim = hidden_size * (2 if bidirectional else 1)
        self.use_attention = bool(use_attention)
        self.sequence_norm = nn.LayerNorm(recurrent_dim)

        safe_heads = max(1, int(attention_heads))
        while recurrent_dim % safe_heads != 0 and safe_heads > 1:
            safe_heads -= 1

        self.self_attention: nn.MultiheadAttention | None = None
        self.query_pool: nn.MultiheadAttention | None = None
        if self.use_attention:
            # [4] Self-attention: refines time-step interactions beyond recurrent state.
            self.self_attention = nn.MultiheadAttention(
                embed_dim=recurrent_dim,
                num_heads=safe_heads,
                dropout=dropout,
                batch_first=True,
            )
            # [5] Query pooling: learns a compact sequence summary for the forecast head.
            self.query_pool = nn.MultiheadAttention(
                embed_dim=recurrent_dim,
                num_heads=safe_heads,
                dropout=dropout,
                batch_first=True,
            )

        self.pool_query = nn.Parameter(torch.zeros(1, 1, recurrent_dim))

        pool_dim = recurrent_dim * 4
        self.pool_norm = nn.LayerNorm(pool_dim)

        self.future_encoder: nn.Module | None = None
        future_embed_dim = max(32, model_dim)
        if self.future_covariate_dim > 0:
            # [6] Future-covariate encoder: compresses known-horizon drivers.
            self.future_encoder = nn.Sequential(
                nn.LayerNorm(self.future_covariate_dim),
                nn.Linear(self.future_covariate_dim, future_embed_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(future_embed_dim, future_embed_dim),
                nn.GELU(),
            )

        decomp_kernel = 25
        self.trend_pool = nn.AvgPool1d(
            kernel_size=decomp_kernel,
            stride=1,
            padding=decomp_kernel // 2,
        )
        self.linear_trend = nn.Linear(self.context_length, self.horizon)
        self.linear_season = nn.Linear(self.context_length, self.horizon)

        with torch.no_grad():
            # [7] Decomposition init: start close to moving-average extrapolation.
            self.linear_trend.weight.fill_(1.0 / self.context_length)
            self.linear_season.weight.fill_(0.0)
            self.linear_trend.bias.zero_()
            self.linear_season.bias.zero_()

        deep_head_in = pool_dim + (
            future_embed_dim if self.future_encoder is not None else 0
        )
        # [8] Deep forecast head: maps pooled latent representation to horizon outputs.
        self.deep_head = nn.Sequential(
            nn.Linear(deep_head_in, max(256, recurrent_dim)),
            nn.GELU(),
            nn.LayerNorm(max(256, recurrent_dim)),
            nn.Dropout(dropout),
            nn.Linear(max(256, recurrent_dim), self.horizon),
        )

        # [9] Mixture gate: blends deep and decomposition forecasts per sample.
        self.mix_gate = nn.Sequential(
            nn.Linear(deep_head_in, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )

    def _decomposition_forecast(self, target_context: torch.Tensor) -> torch.Tensor:
        trend = self.trend_pool(target_context.unsqueeze(1)).squeeze(1)
        season = target_context - trend
        return self.linear_trend(trend) + self.linear_season(season)

    def _pool_sequence(self, seq: torch.Tensor) -> torch.Tensor:
        last = seq[:, -1, :]
        avg = seq.mean(dim=1)
        mx = seq.max(dim=1).values
        if self.query_pool is None:
            attn_pool = avg
        else:
            query = self.pool_query.expand(seq.size(0), 1, seq.size(-1))
            attn_pool, _ = self.query_pool(query, seq, seq, need_weights=False)
            attn_pool = attn_pool.squeeze(1)
        pooled = torch.cat([last, avg, mx, attn_pool], dim=-1)
        return self.pool_norm(pooled)

    def forward(
        self,
        context_inputs: torch.Tensor,
        future_covariates: torch.Tensor | None = None,
    ) -> torch.Tensor:
        # [10] Shared encoder trunk for both univariate and covariate modes.
        x = self.input_norm(context_inputs)
        x = F.gelu(self.input_proj(x))
        x = self.dropout(x)

        for block in self.temporal_blocks:
            x = block(x)

        seq, _ = self.recurrent(x)
        seq = self.sequence_norm(seq)

        if self.self_attention is not None:
            attn_out, _ = self.self_attention(seq, seq, seq, need_weights=False)
            seq = self.sequence_norm(seq + self.dropout(attn_out))

        pooled = self._pool_sequence(seq)

        if self.future_encoder is not None:
            if future_covariates is None:
                raise ValueError(
                    "Model expects future covariates but none were provided for forward pass."
                )
            if future_covariates.shape[-1] != self.future_covariate_dim:
                raise ValueError(
                    "Future covariate feature mismatch. "
                    f"Expected {self.future_covariate_dim}, got {future_covariates.shape[-1]}."
                )
            future_summary = future_covariates.mean(dim=1)
            future_embed = self.future_encoder(future_summary)
            pooled = torch.cat([pooled, future_embed], dim=-1)

        # [11] Nonlinear deep branch forecast.
        deep_forecast = self.deep_head(pooled)

        # [12] Linear decomposition branch forecast (trend + seasonal components).
        target_context = context_inputs[:, :, 0]
        dlinear_forecast = self._decomposition_forecast(target_context)

        # [13] Adaptive fusion between decomposition and deep branches.
        gate = torch.sigmoid(self.mix_gate(pooled))
        forecast = gate * deep_forecast + (1.0 - gate) * dlinear_forecast
        return forecast


class Model1Adapter(BaseFoundationModelAdapter):
    model_id = "custom/model_1"
    slug = "model_1"
    model_family = "custom"
    supports_finetune = True

    RECURRENT_CELL = "gru"
    HIDDEN_SIZE = 224
    RECURRENT_LAYERS = 2
    BIDIRECTIONAL = True
    DROPOUT = 0.15
    MODEL_DIM = 160
    USE_ATTENTION = True
    ATTENTION_HEADS = 8

    def __init__(self, model_ctx: ModelContext, device: torch.device) -> None:
        super().__init__(model_ctx, device)
        self._model: _HybridSeqModel | None = None
        self._input_dim = 1
        self._future_cov_dim = 0

    @staticmethod
    def _as_float1d(values: np.ndarray, *, name: str) -> np.ndarray:
        out = np.asarray(values, dtype=np.float32)
        if out.ndim != 1:
            raise ValueError(f"{name} must be a 1D array. Got shape={out.shape}.")
        return out

    @staticmethod
    def _as_float2d(
        values: np.ndarray,
        *,
        name: str,
        expected_length: int | None = None,
    ) -> np.ndarray:
        out = np.asarray(values, dtype=np.float32)
        if out.ndim != 2:
            raise ValueError(f"{name} must be a 2D array. Got shape={out.shape}.")
        if expected_length is not None and out.shape[0] != expected_length:
            raise ValueError(
                f"{name} length mismatch. Expected {expected_length}, got {out.shape[0]}."
            )
        return out

    @staticmethod
    def _resolve_loss(name: str | None):
        key = "hybrid" if name is None else name.strip().lower()
        if key == "hybrid":

            def _hybrid(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
                mse = F.mse_loss(y_pred, y_true)
                mae = F.l1_loss(y_pred, y_true)
                smape = _smape_loss(y_pred, y_true) / 100.0
                return 0.55 * mse + 0.25 * mae + 0.20 * smape

            return _hybrid
        if key == "mse":
            return lambda y_pred, y_true: F.mse_loss(y_pred, y_true)
        if key == "mae":
            return lambda y_pred, y_true: F.l1_loss(y_pred, y_true)
        if key == "rmse":
            return lambda y_pred, y_true: torch.sqrt(F.mse_loss(y_pred, y_true) + 1e-8)
        if key == "mape":
            return _mape_loss
        if key == "smape":
            return _smape_loss
        raise ValueError(
            f"Unsupported train_loss '{name}'. Supported: mae, mse, rmse, mape, smape"
        )

    @staticmethod
    def _build_optimizer(
        name: str | None,
        *,
        params,
        lr: float,
        weight_decay: float,
    ) -> torch.optim.Optimizer:
        key = "adamw" if name is None else name.strip().lower()
        if key == "adamw":
            return torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        if key == "adam":
            return torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)
        if key == "sgd":
            return torch.optim.SGD(
                params,
                lr=lr,
                weight_decay=weight_decay,
                momentum=0.9,
                nesterov=True,
            )
        raise ValueError(
            f"Unsupported train_optimizer '{name}'. Supported: adamw, adam, sgd"
        )

    def _build_model(self, *, input_dim: int, future_cov_dim: int) -> None:
        self._input_dim = int(input_dim)
        self._future_cov_dim = int(future_cov_dim)
        self._model = _HybridSeqModel(
            input_dim=self._input_dim,
            context_length=self.model_ctx.context_length,
            horizon=self.model_ctx.prediction_length,
            recurrent_cell=self.RECURRENT_CELL,
            hidden_size=self.HIDDEN_SIZE,
            recurrent_layers=self.RECURRENT_LAYERS,
            bidirectional=self.BIDIRECTIONAL,
            dropout=self.DROPOUT,
            model_dim=self.MODEL_DIM,
            use_attention=self.USE_ATTENTION,
            attention_heads=self.ATTENTION_HEADS,
            future_covariate_dim=self._future_cov_dim,
        )
        self._model.to(self.device)

    def _ensure_model(self, *, input_dim: int, future_cov_dim: int) -> None:
        if self._model is None:
            self._build_model(input_dim=input_dim, future_cov_dim=future_cov_dim)
            return
        if self._input_dim != input_dim or self._future_cov_dim != future_cov_dim:
            raise ValueError(
                "Model input dimensions do not match adapter state. "
                f"expected(input={self._input_dim}, future={self._future_cov_dim}) "
                f"got(input={input_dim}, future={future_cov_dim})."
            )

    def _prepare_batch(
        self,
        *,
        context: torch.Tensor,
        future_target: torch.Tensor,
        context_covariates: torch.Tensor | None,
        future_covariates: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor,
        torch.Tensor,
    ]:
        if context.ndim != 2 or future_target.ndim != 2:
            raise ValueError(
                "Expected batched 2D context/future_target tensors. "
                f"Got context={tuple(context.shape)} future_target={tuple(future_target.shape)}"
            )

        context = context.to(self.device, dtype=torch.float32)
        future_target = future_target.to(self.device, dtype=torch.float32)

        if context.shape[1] != self.model_ctx.context_length:
            raise ValueError(
                "Context length mismatch. "
                f"Expected {self.model_ctx.context_length}, got {context.shape[1]}."
            )
        if future_target.shape[1] != self.model_ctx.prediction_length:
            raise ValueError(
                "Future target horizon mismatch. "
                f"Expected {self.model_ctx.prediction_length}, got {future_target.shape[1]}."
            )

        target_loc = context.mean(dim=1, keepdim=True)
        target_scale = context.std(dim=1, keepdim=True, unbiased=False).clamp_min(1e-4)
        context_norm = (context - target_loc) / target_scale
        future_target_norm = (future_target - target_loc) / target_scale

        context_features = [context_norm.unsqueeze(-1)]
        prepared_future_cov: torch.Tensor | None = None

        if context_covariates is not None:
            if context_covariates.ndim != 3:
                raise ValueError(
                    f"context_covariates must be 3D [B,T,F]. Got shape={tuple(context_covariates.shape)}"
                )
            context_cov = context_covariates.to(self.device, dtype=torch.float32)
            cov_mean = _nanmean(context_cov, dim=1, keepdim=True)
            cov_std = _nanstd(context_cov, dim=1, keepdim=True, eps=1e-5)
            context_cov_norm = (context_cov - cov_mean) / cov_std
            context_cov_obs = torch.isfinite(context_cov_norm).to(torch.float32)
            context_cov_norm = torch.where(
                torch.isfinite(context_cov_norm),
                context_cov_norm,
                torch.zeros_like(context_cov_norm),
            )
            context_features.append(context_cov_norm)
            context_features.append(context_cov_obs)

            if future_covariates is not None:
                if future_covariates.ndim != 3:
                    raise ValueError(
                        "future_covariates must be 3D [B,H,F]. "
                        f"Got shape={tuple(future_covariates.shape)}"
                    )
                future_cov = future_covariates.to(self.device, dtype=torch.float32)
                future_cov_norm = (future_cov - cov_mean) / cov_std
                future_cov_obs = torch.isfinite(future_cov_norm).to(torch.float32)
                future_cov_norm = torch.where(
                    torch.isfinite(future_cov_norm),
                    future_cov_norm,
                    torch.zeros_like(future_cov_norm),
                )
                prepared_future_cov = torch.cat(
                    [future_cov_norm, future_cov_obs],
                    dim=-1,
                )

        context_inputs = torch.cat(context_features, dim=-1)
        return (
            context_inputs,
            future_target_norm,
            prepared_future_cov,
            target_loc,
            target_scale,
        )

    def load_pretrained(self) -> None:
        self._model = None
        self._input_dim = 1
        self._future_cov_dim = 0

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

        y_train = self._as_float1d(train_series, name="train_series")
        if (
            y_train.shape[0]
            < self.model_ctx.context_length + self.model_ctx.prediction_length
        ):
            required = self.model_ctx.context_length + self.model_ctx.prediction_length
            raise ValueError(
                f"Not enough train points for context+horizon. Required {required}, got {y_train.shape[0]}."
            )

        use_covariates = train_covariates is not None
        covariates_train_np: np.ndarray | None = None
        future_covariates_train_np: np.ndarray | None = None
        if use_covariates:
            covariates_train_np = self._as_float2d(
                train_covariates,
                name="train_covariates",
                expected_length=y_train.shape[0],
            )
            if train_future_covariates is not None:
                future_covariates_train_np = self._as_float2d(
                    train_future_covariates,
                    name="train_future_covariates",
                    expected_length=y_train.shape[0],
                )
                if covariates_train_np.shape[1] != future_covariates_train_np.shape[1]:
                    raise ValueError(
                        "train_covariates and train_future_covariates feature mismatch. "
                        f"Got {covariates_train_np.shape[1]} and {future_covariates_train_np.shape[1]}."
                    )
            else:
                future_covariates_train_np = covariates_train_np

        samples_per_epoch = max(
            train_batch_size,
            train_batch_size * train_steps_per_epoch,
        )
        if covariates_train_np is None:
            dataset = RandomWindowDataset(
                series=y_train,
                context_length=self.model_ctx.context_length,
                prediction_length=self.model_ctx.prediction_length,
                n_samples=samples_per_epoch,
            )
        else:
            dataset = RandomWindowDatasetWithCovariates(
                series=y_train,
                covariates=covariates_train_np,
                future_covariates=future_covariates_train_np,
                context_length=self.model_ctx.context_length,
                prediction_length=self.model_ctx.prediction_length,
                n_samples=samples_per_epoch,
            )

        loader = DataLoader(
            dataset,
            batch_size=train_batch_size,
            shuffle=False,
            drop_last=True,
            num_workers=0,
            pin_memory=torch.cuda.is_available(),
        )

        first_batch = next(iter(loader))
        context_cov = first_batch.get("context_covariates")
        future_cov = first_batch.get("future_covariates")
        first_context_inputs, _, first_future_cov, _, _ = self._prepare_batch(
            context=first_batch["context"],
            future_target=first_batch["future_target"],
            context_covariates=context_cov,
            future_covariates=future_cov,
        )

        input_dim = int(first_context_inputs.shape[-1])
        future_cov_dim = (
            int(first_future_cov.shape[-1]) if first_future_cov is not None else 0
        )

        self._build_model(input_dim=input_dim, future_cov_dim=future_cov_dim)
        self._ensure_model(input_dim=input_dim, future_cov_dim=future_cov_dim)
        if self._model is None:
            raise RuntimeError("Internal error: model was not initialized.")

        objective = self._resolve_loss(train_loss)
        optimizer = self._build_optimizer(
            train_optimizer,
            params=self._model.parameters(),
            lr=train_lr,
            weight_decay=train_weight_decay,
        )

        total_steps = max(1, int(train_epochs) * int(train_steps_per_epoch))
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=total_steps
        )

        history: list[TrainingLossPoint] = []
        best_loss = float("inf")
        best_state: dict[str, torch.Tensor] | None = None

        self._model.train()
        global_step = 0

        for epoch in range(int(train_epochs)):
            running_loss = 0.0
            batch_count = 0

            for step, batch in enumerate(loader):
                if step >= int(train_steps_per_epoch):
                    break

                context_cov = batch.get("context_covariates")
                future_cov = batch.get("future_covariates")
                context_inputs, y_true_norm, future_cov_inputs, _, _ = (
                    self._prepare_batch(
                        context=batch["context"],
                        future_target=batch["future_target"],
                        context_covariates=context_cov,
                        future_covariates=future_cov,
                    )
                )

                optimizer.zero_grad(set_to_none=True)
                y_pred_norm = self._model(
                    context_inputs=context_inputs,
                    future_covariates=future_cov_inputs,
                )

                loss = objective(y_pred_norm, y_true_norm)
                if not torch.isfinite(loss):
                    raise ValueError(
                        f"Non-finite training loss encountered at epoch={epoch} step={step}."
                    )

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.25)
                optimizer.step()

                global_step += 1
                if global_step <= total_steps:
                    scheduler.step()

                running_loss += float(loss.detach().cpu().item())
                batch_count += 1

            if batch_count == 0:
                raise RuntimeError(
                    "No batches were processed in training epoch. "
                    "Check train_batch_size and train_steps_per_epoch settings."
                )

            epoch_loss = running_loss / batch_count
            history.append(TrainingLossPoint(epoch=epoch, loss=float(epoch_loss)))

            if epoch_loss < best_loss:
                best_loss = epoch_loss
                best_state = copy.deepcopy(self._model.state_dict())

        if checkpoint_selection not in {"best-train-loss", "last"}:
            raise ValueError(
                f"Unsupported checkpoint selection '{checkpoint_selection}'. "
                "Supported: best-train-loss, last"
            )
        if checkpoint_selection == "best-train-loss" and best_state is not None:
            self._model.load_state_dict(best_state)

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

        context_full = self._as_float1d(context, name="context")
        context_tail = context_full[-self.model_ctx.context_length :]

        context_batch = torch.from_numpy(context_tail[None, :])
        horizon = int(self.model_ctx.prediction_length)
        dummy_future_target = torch.zeros((1, horizon), dtype=torch.float32)

        context_cov_batch: torch.Tensor | None = None
        future_cov_batch: torch.Tensor | None = None

        if context_covariates is not None:
            cov_full = self._as_float2d(
                context_covariates,
                name="context_covariates",
                expected_length=context_full.shape[0],
            )
            cov_tail = cov_full[-context_tail.shape[0] :]
            context_cov_batch = torch.from_numpy(cov_tail[None, :, :])

        if future_covariates is not None:
            fut_all = self._as_float2d(
                future_covariates,
                name="future_covariates",
            )
            if fut_all.shape[0] < horizon:
                raise ValueError(
                    "future_covariates does not contain enough rows for prediction horizon. "
                    f"Required {horizon}, got {fut_all.shape[0]}."
                )
            fut_tail = fut_all[:horizon]
            future_cov_batch = torch.from_numpy(fut_tail[None, :, :])

        context_inputs, _, future_cov_inputs, target_loc, target_scale = (
            self._prepare_batch(
                context=context_batch,
                future_target=dummy_future_target,
                context_covariates=context_cov_batch,
                future_covariates=future_cov_batch,
            )
        )

        input_dim = int(context_inputs.shape[-1])
        future_cov_dim = (
            int(future_cov_inputs.shape[-1]) if future_cov_inputs is not None else 0
        )

        if self._model is None:
            self._build_model(input_dim=input_dim, future_cov_dim=future_cov_dim)
        self._ensure_model(input_dim=input_dim, future_cov_dim=future_cov_dim)

        if self._model is None:
            raise RuntimeError(
                "Internal error: model is not initialized for inference."
            )

        self._model.eval()
        with torch.no_grad():
            y_pred_norm = self._model(
                context_inputs=context_inputs.to(self.device),
                future_covariates=(
                    future_cov_inputs.to(self.device)
                    if future_cov_inputs is not None
                    else None
                ),
            )
            y_pred = y_pred_norm * target_scale.to(self.device) + target_loc.to(
                self.device
            )

        y_pred_np = y_pred.squeeze(0).detach().cpu().numpy().astype(np.float32)
        if y_pred_np.shape[0] != horizon:
            raise ValueError(
                f"Bad forecast length: expected {horizon}, got {y_pred_np.shape[0]}."
            )
        return ForecastResult(y_pred=y_pred_np)

    def save_finetuned(self, artifact_dir: Path) -> None:
        if self._model is None:
            raise RuntimeError("Model is not initialized; nothing to save.")
        artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "model_id": self.model_id,
                "input_dim": int(self._input_dim),
                "future_cov_dim": int(self._future_cov_dim),
                "hparams": {
                    "recurrent_cell": self.RECURRENT_CELL,
                    "hidden_size": int(self.HIDDEN_SIZE),
                    "recurrent_layers": int(self.RECURRENT_LAYERS),
                    "bidirectional": bool(self.BIDIRECTIONAL),
                    "dropout": float(self.DROPOUT),
                    "model_dim": int(self.MODEL_DIM),
                    "use_attention": bool(self.USE_ATTENTION),
                    "attention_heads": int(self.ATTENTION_HEADS),
                    "horizon": int(self.model_ctx.prediction_length),
                    "context_length": int(self.model_ctx.context_length),
                },
            },
            artifact_dir / "model.pt",
        )

    def load_finetuned(self, artifact_dir: Path) -> None:
        payload = torch.load(artifact_dir / "model.pt", map_location="cpu")
        input_dim = int(payload["input_dim"])
        future_cov_dim = int(payload.get("future_cov_dim", 0))

        saved_hparams = payload.get("hparams", {})
        if int(saved_hparams.get("horizon", self.model_ctx.prediction_length)) != int(
            self.model_ctx.prediction_length
        ):
            raise ValueError(
                "Checkpoint horizon is incompatible with current runtime config. "
                f"checkpoint={saved_hparams.get('horizon')} runtime={self.model_ctx.prediction_length}"
            )

        if int(
            saved_hparams.get("context_length", self.model_ctx.context_length)
        ) != int(self.model_ctx.context_length):
            raise ValueError(
                "Checkpoint context length is incompatible with current runtime config. "
                f"checkpoint={saved_hparams.get('context_length')} runtime={self.model_ctx.context_length}"
            )

        self._build_model(input_dim=input_dim, future_cov_dim=future_cov_dim)
        if self._model is None:
            raise RuntimeError("Internal error: model was not initialized on load.")
        self._model.load_state_dict(payload["state_dict"])
        self._model.to(self.device)
        self._model.eval()
