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
from adapters.shared import RandomWindowDataset, RandomWindowDatasetWithCovariates
from torch.utils.data import DataLoader


class GraniteTTMAdapter(BaseFoundationModelAdapter):
    model_id = "ibm-granite/granite-timeseries-ttm-r2"
    slug = "granite_ttm"
    model_family = "foundation"
    supports_finetune = True

    def __init__(self, model_ctx: ModelContext, device: torch.device) -> None:
        super().__init__(model_ctx, device)
        self._model = None
        self._freq_token_value = (
            7  # hourly token in tsfm_public DEFAULT_FREQUENCY_MAPPING
        )

    @staticmethod
    def _as_float2d(
        array: np.ndarray, *, name: str, expected_length: int | None = None
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
    def _sanitize_with_observed_mask(
        values: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        observed = torch.isfinite(values).to(dtype=torch.float32)
        sanitized = torch.where(observed > 0, values, torch.zeros_like(values))
        return sanitized, observed

    def _load_model(self, model_path: str | Path) -> None:
        from tsfm_public.toolkit.get_model import get_model

        self._model = get_model(
            str(model_path),
            context_length=self.model_ctx.context_length,
            prediction_length=self.model_ctx.prediction_length,
            freq_prefix_tuning=True,
            freq="h",
            prefer_longer_context=True,
        )
        self._model.to(self.device)  # type: ignore
        self._model.eval()  # type: ignore

    def _build_past_values(
        self,
        *,
        context: np.ndarray,
        context_covariates: np.ndarray | None,
        effective_context_length: int,
    ) -> torch.Tensor:
        target_ctx = np.asarray(context, dtype=np.float32)[-effective_context_length:]
        if target_ctx.ndim != 1:
            raise ValueError(
                f"context must be a 1D array. Got shape={target_ctx.shape}."
            )

        target_channel = target_ctx[:, np.newaxis]
        if context_covariates is None:
            stacked = target_channel
        else:
            cov_full = self._as_float2d(
                context_covariates,
                name="context_covariates",
                expected_length=len(np.asarray(context)),
            )
            cov_view = cov_full[-effective_context_length:]
            stacked = np.concatenate((target_channel, cov_view), axis=1)

        past = torch.from_numpy(stacked).unsqueeze(0).to(self.device)
        return past

    def _build_future_values_and_mask(
        self,
        *,
        future_target: torch.Tensor | None,
        future_covariates: torch.Tensor | None,
        batch_size: int,
        prediction_length: int,
        covariate_dim: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if future_target is None:
            target = torch.full(
                (batch_size, prediction_length, 1),
                float("nan"),
                device=self.device,
                dtype=torch.float32,
            )
        else:
            target = future_target.to(self.device).unsqueeze(-1)

        if covariate_dim == 0:
            future_values = target
        else:
            if future_covariates is None:
                cov = torch.full(
                    (batch_size, prediction_length, covariate_dim),
                    float("nan"),
                    device=self.device,
                    dtype=target.dtype,
                )
            else:
                cov = future_covariates.to(self.device)
                if cov.ndim != 3:
                    raise ValueError(
                        "future_covariates must be a 3D tensor [B, prediction_length, num_covariates]. "
                        f"Got shape={tuple(cov.shape)}."
                    )
                if cov.shape[2] != covariate_dim:
                    raise ValueError(
                        "Covariate dimension mismatch between context and future covariates. "
                        f"Expected {covariate_dim}, got {cov.shape[2]}."
                    )
            future_values = torch.cat((target, cov), dim=2)

        sanitized, observed = self._sanitize_with_observed_mask(future_values)
        return sanitized, observed

    def load_pretrained(self) -> None:
        self._load_model(self.model_id)

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
        if train_loss is not None or train_optimizer is not None:
            raise ValueError(
                f"--train-loss/--train-optimizer are only supported for custom models. '{self.slug}' is a foundation model."
            )

        self._load_model(self.model_id)
        if self._model is None:
            raise RuntimeError("Granite TTM model failed to load.")

        model_context_length = int(getattr(self._model.config, "context_length", 0))  # type: ignore
        if model_context_length <= 0:
            raise RuntimeError("Granite TTM config has invalid context_length.")

        context_length = int(min(self.model_ctx.context_length, model_context_length))

        use_covariates = train_covariates is not None
        if use_covariates:
            cov_train = self._as_float2d(
                train_covariates,
                name="train_covariates",
                expected_length=len(train_series),
            )
            fut_cov_train = (
                self._as_float2d(
                    train_future_covariates,
                    name="train_future_covariates",
                    expected_length=len(train_series),
                )
                if train_future_covariates is not None
                else None
            )
            ds = RandomWindowDatasetWithCovariates(
                series=train_series,
                covariates=cov_train,
                future_covariates=fut_cov_train,
                context_length=context_length,
                prediction_length=self.model_ctx.prediction_length,
                n_samples=max(1, train_batch_size * train_steps_per_epoch),
            )
        else:
            ds = RandomWindowDataset(
                series=train_series,
                context_length=context_length,
                prediction_length=self.model_ctx.prediction_length,
                n_samples=max(1, train_batch_size * train_steps_per_epoch),
            )

        dl = DataLoader(ds, batch_size=max(1, train_batch_size), shuffle=False)

        for param in self._model.parameters():  # type: ignore
            param.requires_grad = True

        optimizer = torch.optim.AdamW(
            (p for p in self._model.parameters() if p.requires_grad),  # type: ignore
            lr=float(train_lr),
            weight_decay=float(train_weight_decay),
        )

        if checkpoint_selection not in {"best-train-loss", "last"}:
            raise ValueError(
                f"Unsupported checkpoint selection '{checkpoint_selection}'. "
                "Supported: best-train-loss, last"
            )

        self._model.train()  # type: ignore
        history: list[TrainingLossPoint] = []
        best_loss = float("inf")
        best_state_dict: dict[str, torch.Tensor] | None = None

        for ep in range(max(1, train_epochs)):
            epoch_losses: list[float] = []

            for batch in dl:
                context = batch["context"].to(self.device).float()
                future_target = batch["future_target"].to(self.device).float()

                batch_size = int(context.shape[0])
                context_cov = batch.get("context_covariates")
                future_cov = batch.get("future_covariates")

                target_channel = context.unsqueeze(-1)
                if context_cov is not None:
                    context_cov = context_cov.to(self.device).float()
                    past_values = torch.cat((target_channel, context_cov), dim=2)
                else:
                    past_values = target_channel

                covariate_dim = int(past_values.shape[2] - 1)

                future_cov_tensor: torch.Tensor | None = None
                if future_cov is not None:
                    future_cov_tensor = future_cov.to(self.device).float()

                future_values, future_observed_mask = (
                    self._build_future_values_and_mask(
                        future_target=future_target,
                        future_covariates=future_cov_tensor,
                        batch_size=batch_size,
                        prediction_length=self.model_ctx.prediction_length,
                        covariate_dim=covariate_dim,
                    )
                )

                freq_token = torch.full(
                    (batch_size,),
                    self._freq_token_value,
                    dtype=torch.long,
                    device=self.device,
                )

                out = self._model(  # type: ignore
                    past_values=past_values,
                    future_values=future_values,
                    future_observed_mask=future_observed_mask,
                    freq_token=freq_token,
                    return_loss=True,
                )
                loss = out.loss
                if loss is None:
                    raise RuntimeError("Granite TTM did not return a loss.")

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))

            if epoch_losses:
                mean_epoch_loss = float(np.mean(epoch_losses))
                history.append(TrainingLossPoint(epoch=ep, loss=mean_epoch_loss))
                if (
                    checkpoint_selection == "best-train-loss"
                    and mean_epoch_loss < best_loss
                ):
                    best_loss = mean_epoch_loss
                    best_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in self._model.state_dict().items()  # type: ignore
                    }

        if checkpoint_selection == "best-train-loss" and best_state_dict is not None:
            self._model.load_state_dict(best_state_dict)  # type: ignore

        self._model.eval()  # type: ignore
        return history

    def forecast(
        self,
        context: np.ndarray,
        context_start: pd.Timestamp,
        context_covariates: np.ndarray | None = None,
        future_covariates: np.ndarray | None = None,
    ) -> ForecastResult:
        del context_start
        if self._model is None:
            raise RuntimeError("Granite TTM model is not loaded.")

        model_context_length = int(getattr(self._model.config, "context_length", 0))  # type: ignore
        if model_context_length <= 0:
            raise RuntimeError("Granite TTM config has invalid context_length.")

        effective_context_length = int(
            min(self.model_ctx.context_length, model_context_length)
        )
        past_values = self._build_past_values(
            context=context,
            context_covariates=context_covariates,
            effective_context_length=effective_context_length,
        )

        covariate_dim = int(past_values.shape[2] - 1)
        future_values: torch.Tensor | None = None
        future_observed_mask: torch.Tensor | None = None

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
            if covariate_dim <= 0:
                raise ValueError(
                    "future_covariates were provided but context_covariates are missing."
                )
            if future_cov.shape[1] != covariate_dim:
                raise ValueError(
                    "Covariate feature count mismatch between context and future covariates. "
                    f"Expected {covariate_dim}, got {future_cov.shape[1]}."
                )

            future_cov_tensor = (
                torch.from_numpy(future_cov[: self.model_ctx.prediction_length])
                .unsqueeze(0)
                .to(self.device)
                .float()
            )

            future_values, future_observed_mask = self._build_future_values_and_mask(
                future_target=None,
                future_covariates=future_cov_tensor,
                batch_size=1,
                prediction_length=self.model_ctx.prediction_length,
                covariate_dim=covariate_dim,
            )

        freq_token = torch.full(
            (1,),
            self._freq_token_value,
            dtype=torch.long,
            device=self.device,
        )

        with torch.no_grad():
            out = self._model(  # type: ignore
                past_values=past_values,
                future_values=future_values,
                future_observed_mask=future_observed_mask,
                freq_token=freq_token,
                return_loss=False,
            )

        prediction_outputs = out.prediction_outputs
        if prediction_outputs is None:
            raise RuntimeError("Granite TTM did not return prediction_outputs.")

        if prediction_outputs.shape[1] < self.model_ctx.prediction_length:
            raise RuntimeError(
                "Granite TTM returned fewer forecast steps than requested. "
                f"requested={self.model_ctx.prediction_length} got={prediction_outputs.shape[1]}"
            )

        y_pred = (
            prediction_outputs[0, : self.model_ctx.prediction_length, 0]
            .detach()
            .cpu()
            .numpy()
            .astype(np.float32)
        )
        return ForecastResult(y_pred=y_pred)

    def save_finetuned(self, artifact_dir: Path) -> None:
        if self._model is None:
            raise RuntimeError("Granite TTM model is not loaded.")

        artifact_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(artifact_dir)  # type: ignore

    def load_finetuned(self, artifact_dir: Path) -> None:
        from tsfm_public.models.tinytimemixer import TinyTimeMixerForPrediction

        model = TinyTimeMixerForPrediction.from_pretrained(str(artifact_dir))
        model.to(self.device)  # type: ignore
        model.eval()
        self._model = model
