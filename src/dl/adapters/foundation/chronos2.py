from __future__ import annotations

import inspect
import math
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


class Chronos2Adapter(BaseFoundationModelAdapter):
    model_id = "amazon/chronos-2"
    slug = "chronos2"
    model_family = "foundation"
    supports_finetune = True

    def __init__(self, model_ctx: ModelContext, device: torch.device) -> None:
        super().__init__(model_ctx, device)
        self._pipe = None
        self._model = None
        self._q_index = None

    def load_pretrained(self) -> None:
        from chronos import Chronos2Pipeline

        self._pipe = Chronos2Pipeline.from_pretrained(self.model_id)
        self._pipe.model.to(self.device)  # type: ignore
        self._model = None
        qs = self._pipe.model.chronos_config.quantiles  # type: ignore
        self._q_index = int(np.argmin(np.abs(np.asarray(qs) - 0.5)))

    @staticmethod
    def _supports_grouped_covariates(callable_obj) -> bool:
        try:
            signature = inspect.signature(callable_obj)
        except (TypeError, ValueError):
            return False

        params = signature.parameters
        return "future_covariates" in params and "group_ids" in params

    def _build_grouped_covariate_batch(
        self,
        *,
        context: torch.Tensor,
        context_covariates: torch.Tensor,
        future_target: torch.Tensor | None,
        future_covariates: torch.Tensor | None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor | None,
        torch.Tensor | None,
    ]:
        batch_size, context_len = context.shape
        _, pred_len, cov_count = (
            future_covariates.shape
            if future_covariates is not None
            else (
                context_covariates.shape[0],
                self.model_ctx.prediction_length,
                context_covariates.shape[2],
            )
        )

        target_context = context.unsqueeze(1)
        cov_context = context_covariates.transpose(1, 2)
        grouped_context = torch.cat((target_context, cov_context), dim=1).reshape(
            batch_size * (cov_count + 1), context_len
        )

        if future_covariates is None:
            cov_future = torch.full(
                (batch_size, cov_count, pred_len),
                float("nan"),
                device=context.device,
                dtype=context.dtype,
            )
        else:
            cov_future = future_covariates.transpose(1, 2)

        target_future_cov = torch.full(
            (batch_size, 1, pred_len),
            float("nan"),
            device=context.device,
            dtype=context.dtype,
        )
        grouped_future_cov = torch.cat((target_future_cov, cov_future), dim=1).reshape(
            batch_size * (cov_count + 1), pred_len
        )

        group_ids = torch.arange(batch_size, device=context.device).repeat_interleave(
            cov_count + 1
        )

        grouped_future_target: torch.Tensor | None = None
        grouped_future_target_mask: torch.Tensor | None = None
        if future_target is not None:
            target_future = future_target.unsqueeze(1)
            cov_future_target = torch.full(
                (batch_size, cov_count, pred_len),
                float("nan"),
                device=context.device,
                dtype=future_target.dtype,
            )
            grouped_future_target = torch.cat(
                (target_future, cov_future_target), dim=1
            ).reshape(batch_size * (cov_count + 1), pred_len)

            target_mask = torch.ones(
                (batch_size, 1, pred_len),
                device=context.device,
                dtype=future_target.dtype,
            )
            cov_mask = torch.zeros(
                (batch_size, cov_count, pred_len),
                device=context.device,
                dtype=future_target.dtype,
            )
            grouped_future_target_mask = torch.cat(
                (target_mask, cov_mask), dim=1
            ).reshape(batch_size * (cov_count + 1), pred_len)

        return (
            grouped_context,
            grouped_future_cov,
            group_ids,
            grouped_future_target,
            grouped_future_target_mask,
        )

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

        from chronos.chronos2 import Chronos2Model

        model = Chronos2Model.from_pretrained(self.model_id).to(self.device)  # type: ignore

        for param in model.parameters():
            param.requires_grad = False
        for param in model.output_patch_embedding.parameters():
            param.requires_grad = True

        context_length = int(
            min(self.model_ctx.context_length, model.chronos_config.context_length)
        )
        num_output_patches = int(
            math.ceil(
                self.model_ctx.prediction_length
                / model.chronos_config.output_patch_size
            )
        )

        supports_covariates = self._supports_grouped_covariates(model.forward)
        use_covariates = train_covariates is not None

        if use_covariates and not supports_covariates:
            raise ValueError(
                "Covariate mode requested, but installed Chronos2Model does not expose "
                "grouped covariate arguments ('group_ids' and 'future_covariates'). "
                "Upgrade the chronos package to a covariate-capable version."
            )

        if use_covariates:
            ds = RandomWindowDatasetWithCovariates(
                series=train_series,
                covariates=train_covariates,
                future_covariates=train_future_covariates,
                context_length=context_length,
                prediction_length=self.model_ctx.prediction_length,
                n_samples=train_batch_size * train_steps_per_epoch,
            )
        else:
            ds = RandomWindowDataset(
                series=train_series,
                context_length=context_length,
                prediction_length=self.model_ctx.prediction_length,
                n_samples=train_batch_size * train_steps_per_epoch,
            )
        dl = DataLoader(ds, batch_size=train_batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(
            (p for p in model.parameters() if p.requires_grad),
            lr=train_lr,
            weight_decay=train_weight_decay,
        )

        model.train()
        history: list[TrainingLossPoint] = []
        best_loss = float("inf")
        best_state_dict: dict[str, torch.Tensor] | None = None
        if checkpoint_selection not in {"best-train-loss", "last"}:
            raise ValueError(
                f"Unsupported checkpoint selection '{checkpoint_selection}'. "
                "Supported: best-train-loss, last"
            )
        for ep in range(train_epochs):
            epoch_losses: list[float] = []
            for batch in dl:
                ctx = batch["context"].to(self.device)
                fut = batch["future_target"].to(self.device)
                model_kwargs: dict[str, torch.Tensor | int] = {
                    "context": ctx,
                    "future_target": fut,
                    "num_output_patches": num_output_patches,
                }

                if use_covariates:
                    context_cov = batch["context_covariates"].to(self.device)
                    future_cov = batch["future_covariates"].to(self.device)
                    (
                        grouped_context,
                        grouped_future_cov,
                        group_ids,
                        grouped_future_target,
                        grouped_future_target_mask,
                    ) = self._build_grouped_covariate_batch(
                        context=ctx,
                        context_covariates=context_cov,
                        future_target=fut,
                        future_covariates=future_cov,
                    )
                    model_kwargs["context"] = grouped_context
                    model_kwargs["group_ids"] = group_ids
                    model_kwargs["future_covariates"] = grouped_future_cov
                    if grouped_future_target is not None:
                        model_kwargs["future_target"] = grouped_future_target
                    if grouped_future_target_mask is not None:
                        model_kwargs["future_target_mask"] = grouped_future_target_mask

                out = model(**model_kwargs)
                loss = out.loss
                if loss is None:
                    raise RuntimeError("Chronos2Model did not return a loss.")

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
                        for key, value in model.state_dict().items()
                    }

        if checkpoint_selection == "best-train-loss" and best_state_dict is not None:
            model.load_state_dict(best_state_dict)

        model.eval()
        self._model = model
        self._pipe = None

        qs = model.chronos_config.quantiles
        self._q_index = int(np.argmin(np.abs(np.asarray(qs) - 0.5)))
        return history

    def forecast(
        self,
        context: np.ndarray,
        context_start: pd.Timestamp,
        context_covariates: np.ndarray | None = None,
        future_covariates: np.ndarray | None = None,
    ) -> ForecastResult:
        del context_start
        if self._model is not None:
            effective_context_length = int(
                min(
                    self.model_ctx.context_length,
                    self._model.chronos_config.context_length,
                )
            )
            x = np.asarray(context, dtype=np.float32)[-effective_context_length:]
            ctx = torch.from_numpy(x)[None, :].to(self.device)
            supports_covariates = self._supports_grouped_covariates(self._model.forward)
            use_covariates = (
                context_covariates is not None or future_covariates is not None
            )

            if use_covariates and not supports_covariates:
                raise ValueError(
                    "Covariate mode requested, but loaded Chronos2 model does not expose "
                    "grouped covariate arguments ('group_ids' and 'future_covariates')."
                )

            context_cov_tensor: torch.Tensor | None = None
            if context_covariates is not None:
                context_cov_array = np.asarray(context_covariates, dtype=np.float32)
                context_cov_array = context_cov_array[-effective_context_length:]
                context_cov_tensor = torch.from_numpy(context_cov_array)[None, :, :].to(
                    self.device
                )

            future_cov_tensor: torch.Tensor | None = None
            if future_covariates is not None:
                future_cov_array = np.asarray(future_covariates, dtype=np.float32)
                future_cov_array = future_cov_array[: self.model_ctx.prediction_length]
                future_cov_tensor = torch.from_numpy(future_cov_array)[None, :, :].to(
                    self.device
                )

            output_patch_size = int(self._model.chronos_config.output_patch_size)
            num_output_patches = int(
                (self.model_ctx.prediction_length + output_patch_size - 1)
                // output_patch_size
            )
            model_kwargs: dict[str, torch.Tensor | int] = {
                "context": ctx,
                "num_output_patches": num_output_patches,
            }
            if use_covariates:
                if context_cov_tensor is None:
                    raise ValueError(
                        "Covariate mode requires context_covariates for Chronos2 forecast."
                    )
                (
                    grouped_context,
                    grouped_future_cov,
                    group_ids,
                    _,
                    _,
                ) = self._build_grouped_covariate_batch(
                    context=ctx,
                    context_covariates=context_cov_tensor,
                    future_target=None,
                    future_covariates=future_cov_tensor,
                )
                model_kwargs["context"] = grouped_context
                model_kwargs["group_ids"] = group_ids
                model_kwargs["future_covariates"] = grouped_future_cov

            with torch.no_grad():
                out = self._model(**model_kwargs)

            q = out.quantile_preds[0, self._q_index, : self.model_ctx.prediction_length]
            return ForecastResult(y_pred=q.detach().cpu().numpy())

        if self._pipe is None:
            raise RuntimeError("Chronos2 model is not loaded.")

        x = np.asarray(context, dtype=np.float32)[-self.model_ctx.context_length :]
        x3 = x[None, None, :]
        supports_covariates = self._supports_grouped_covariates(self._pipe.model.forward)  # type: ignore
        use_covariates = context_covariates is not None or future_covariates is not None
        if use_covariates and not supports_covariates:
            raise ValueError(
                "Covariate mode requested, but loaded Chronos2 model does not expose "
                "grouped covariate arguments ('group_ids' and 'future_covariates')."
            )

        if use_covariates:
            effective_context_length = int(
                min(
                    self.model_ctx.context_length,
                    self._pipe.model.chronos_config.context_length,  # type: ignore
                )
            )
            ctx_np = np.asarray(context, dtype=np.float32)[-effective_context_length:]
            ctx_tensor = torch.from_numpy(ctx_np)[None, :].to(self.device)

            if context_covariates is None:
                raise ValueError(
                    "Covariate mode requires context_covariates for Chronos2 forecast."
                )

            context_cov_array = np.asarray(context_covariates, dtype=np.float32)[
                -effective_context_length:
            ]
            context_cov_tensor = torch.from_numpy(context_cov_array)[None, :, :].to(
                self.device
            )

            future_cov_tensor: torch.Tensor | None = None
            if future_covariates is not None:
                future_cov_array = np.asarray(future_covariates, dtype=np.float32)[
                    : self.model_ctx.prediction_length
                ]
                future_cov_tensor = torch.from_numpy(future_cov_array)[None, :, :].to(
                    self.device
                )

            output_patch_size = int(self._pipe.model.chronos_config.output_patch_size)  # type: ignore
            num_output_patches = int(
                (self.model_ctx.prediction_length + output_patch_size - 1)
                // output_patch_size
            )

            (
                grouped_context,
                grouped_future_cov,
                group_ids,
                _,
                _,
            ) = self._build_grouped_covariate_batch(
                context=ctx_tensor,
                context_covariates=context_cov_tensor,
                future_target=None,
                future_covariates=future_cov_tensor,
            )

            with torch.no_grad():
                out = self._pipe.model(  # type: ignore
                    context=grouped_context,
                    group_ids=group_ids,
                    future_covariates=grouped_future_cov,
                    num_output_patches=num_output_patches,
                )

            q = out.quantile_preds[0, self._q_index, : self.model_ctx.prediction_length]
            return ForecastResult(y_pred=q.detach().cpu().numpy())

        predict_kwargs: dict[str, object] = {
            "prediction_length": self.model_ctx.prediction_length,
            "batch_size": 1,
            "context_length": min(len(x), self.model_ctx.context_length),
        }

        with torch.no_grad():
            out = self._pipe.predict(x3, **predict_kwargs)  # type: ignore

        samples = out[0][0]
        y_pred = samples.float().mean(dim=0).cpu().numpy()
        return ForecastResult(y_pred=y_pred)

    def save_finetuned(self, artifact_dir: Path) -> None:
        if self._model is None:
            raise RuntimeError("Chronos2 fine-tuned model is not available to save.")

        artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._model.state_dict(),
                "repo_id": self.model_id,
            },
            artifact_dir / "model.pt",
        )

    def load_finetuned(self, artifact_dir: Path) -> None:
        from chronos.chronos2 import Chronos2Model

        payload = torch.load(artifact_dir / "model.pt", map_location="cpu")
        repo_id = str(payload["repo_id"])
        state_dict = payload["state_dict"]

        model = Chronos2Model.from_pretrained(repo_id).to(self.device)  # type: ignore
        model.load_state_dict(state_dict)
        model.eval()

        self._model = model
        self._pipe = None

        qs = model.chronos_config.quantiles
        self._q_index = int(np.argmin(np.abs(np.asarray(qs) - 0.5)))
