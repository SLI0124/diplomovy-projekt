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
from adapters.shared import RandomWindowDataset
from torch.utils.data import DataLoader


class TimesFM25Adapter(BaseFoundationModelAdapter):
    model_id = "google/timesfm-2.5-200m-pytorch"
    slug = "timesfm25"
    model_family = "foundation"
    supports_finetune = True

    def __init__(self, model_ctx: ModelContext, device: torch.device) -> None:
        super().__init__(model_ctx, device)
        self._model = None

    def _compile_model(self, model) -> None:
        import timesfm

        model.compile(
            timesfm.ForecastConfig(
                max_context=self.model_ctx.context_length,
                max_horizon=max(256, self.model_ctx.prediction_length),
                normalize_inputs=True,
                use_continuous_quantile_head=True,
                force_flip_invariance=True,
                infer_is_positive=False,
                fix_quantile_crossing=True,
            )
        )

    def load_pretrained(self) -> None:
        try:
            import timesfm  # type: ignore
        except Exception as exc:
            raise ImportError(
                'TimesFM is not installed. Install with uv (latest): uv add "timesfm[torch]"'
            ) from exc

        if not hasattr(timesfm, "TimesFM_2p5_200M_torch"):
            raise ImportError(
                "Installed timesfm package does not expose TimesFM_2p5_200M_torch. "
                "Please upgrade timesfm package."
            )

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            self.model_id, torch_compile=False
        )
        self._compile_model(model)
        self._model = model

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
        del train_covariates, train_future_covariates
        if train_loss is not None or train_optimizer is not None:
            raise ValueError(
                f"--train-loss/--train-optimizer are only supported for custom models. '{self.slug}' is a foundation model."
            )

        if self._model is None:
            self.load_pretrained()

        module = self._model.model  # type: ignore
        module.to(self.device)

        for param in module.parameters():
            param.requires_grad = False
        for param in module.output_projection_point.parameters():  # type: ignore
            param.requires_grad = True

        patch_size = int(module.p)  # type: ignore
        output_patch = int(module.o)  # type: ignore
        mean_index = int(module.aridx)  # type: ignore

        context_length = self.model_ctx.context_length

        ds = RandomWindowDataset(
            series=train_series,
            context_length=context_length,
            prediction_length=self.model_ctx.prediction_length,
            n_samples=max(1, train_batch_size * train_steps_per_epoch),
        )
        dl = DataLoader(ds, batch_size=max(1, train_batch_size), shuffle=False)

        optimizer = torch.optim.AdamW(
            (p for p in module.parameters() if p.requires_grad),
            lr=float(train_lr),
            weight_decay=float(train_weight_decay),
        )
        if checkpoint_selection not in {"best-train-loss", "last"}:
            raise ValueError(
                f"Unsupported checkpoint selection '{checkpoint_selection}'. "
                "Supported: best-train-loss, last"
            )
        loss_fn = torch.nn.MSELoss()

        module.train()
        history: list[TrainingLossPoint] = []
        best_loss = float("inf")
        best_state_dict: dict[str, torch.Tensor] | None = None
        for ep in range(max(1, train_epochs)):
            epoch_losses: list[float] = []
            for batch in dl:
                ctx_raw = batch["context"].to(self.device).float()
                fut_raw = batch["future_target"].to(self.device).float()

                mu = ctx_raw.mean(dim=1, keepdim=True)
                sigma = ctx_raw.std(dim=1, keepdim=True)
                sigma = torch.clamp(sigma, min=1e-6)

                ctx = (ctx_raw - mu) / sigma
                fut = (fut_raw - mu) / sigma

                remainder = int(ctx.shape[1]) % patch_size
                if remainder != 0:
                    pad = patch_size - remainder
                    ctx = torch.nn.functional.pad(ctx, (pad, 0), value=0.0)

                masks = torch.zeros_like(ctx, dtype=torch.bool, device=self.device)
                patched_ctx = ctx.reshape(ctx.shape[0], -1, patch_size)
                patched_masks = masks.reshape(masks.shape[0], -1, patch_size)

                (_, _, output_ts, _), _ = module(patched_ctx, patched_masks)

                output_ts = output_ts.reshape(ctx.shape[0], -1, output_patch, module.q)
                pred = output_ts[:, -1, : self.model_ctx.prediction_length, mean_index]

                loss = loss_fn(pred, fut[:, : self.model_ctx.prediction_length])
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))

            if epoch_losses:
                mean_epoch_loss = float(np.mean(epoch_losses))
                history.append(
                    TrainingLossPoint(epoch=ep, loss=mean_epoch_loss)
                )
                if (
                    checkpoint_selection == "best-train-loss"
                    and mean_epoch_loss < best_loss
                ):
                    best_loss = mean_epoch_loss
                    best_state_dict = {
                        key: value.detach().cpu().clone()
                        for key, value in module.state_dict().items()
                    }

        if (
            checkpoint_selection == "best-train-loss"
            and best_state_dict is not None
        ):
            module.load_state_dict(best_state_dict)

        module.eval()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(artifact_dir)  # type: ignore
        return history

    def forecast(
        self,
        context: np.ndarray,
        context_start: pd.Timestamp,
        context_covariates: np.ndarray | None = None,
        future_covariates: np.ndarray | None = None,
    ) -> ForecastResult:
        del context_covariates, future_covariates
        if self._model is None:
            raise RuntimeError("TimesFM-2.5 model is not loaded.")

        x = np.asarray(context, dtype=np.float32)[-self.model_ctx.context_length :]
        point_forecast, _ = self._model.forecast(
            horizon=self.model_ctx.prediction_length,
            inputs=[x],
        )
        y_pred = np.asarray(point_forecast[0], dtype=np.float32)
        return ForecastResult(y_pred=y_pred)

    def save_finetuned(self, artifact_dir: Path) -> None:
        if self._model is None:
            raise RuntimeError("TimesFM-2.5 model is not loaded.")

        artifact_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(artifact_dir)

    def load_finetuned(self, artifact_dir: Path) -> None:
        try:
            import timesfm  # type: ignore
        except Exception as exc:
            raise ImportError(
                'TimesFM is not installed. Install with uv (latest): uv add "timesfm[torch]"'
            ) from exc

        model = timesfm.TimesFM_2p5_200M_torch.from_pretrained(
            str(artifact_dir),
            torch_compile=False,
        )
        self._compile_model(model)
        self._model = model
