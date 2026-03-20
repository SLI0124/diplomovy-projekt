from __future__ import annotations

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
from adapters.shared import RandomWindowDataset
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
        self._q_index = None

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
        for ep in range(train_epochs):
            epoch_losses: list[float] = []
            for batch in dl:
                ctx = batch["context"].to(self.device)
                fut = batch["future_target"].to(self.device)

                out = model(
                    context=ctx,
                    future_target=fut,
                    num_output_patches=num_output_patches,
                )
                loss = out.loss
                if loss is None:
                    raise RuntimeError("Chronos2Model did not return a loss.")

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                epoch_losses.append(float(loss.detach().cpu().item()))

            if epoch_losses:
                history.append(
                    TrainingLossPoint(epoch=ep, loss=float(np.mean(epoch_losses)))
                )

        model.eval()
        self._model = model
        self._pipe = None

        qs = model.chronos_config.quantiles
        self._q_index = int(np.argmin(np.abs(np.asarray(qs) - 0.5)))
        return history

    def forecast(
        self, context: np.ndarray, context_start: pd.Timestamp
    ) -> ForecastResult:
        if self._model is not None:
            effective_context_length = int(
                min(
                    self.model_ctx.context_length,
                    self._model.chronos_config.context_length,
                )
            )
            x = np.asarray(context, dtype=np.float32)[-effective_context_length:]
            ctx = torch.from_numpy(x)[None, :].to(self.device)
            output_patch_size = int(self._model.chronos_config.output_patch_size)
            num_output_patches = int(
                (self.model_ctx.prediction_length + output_patch_size - 1)
                // output_patch_size
            )
            with torch.no_grad():
                out = self._model(context=ctx, num_output_patches=num_output_patches)

            q = out.quantile_preds[0, self._q_index, : self.model_ctx.prediction_length]
            return ForecastResult(y_pred=q.detach().cpu().numpy())

        if self._pipe is None:
            raise RuntimeError("Chronos2 model is not loaded.")

        x = np.asarray(context, dtype=np.float32)[-self.model_ctx.context_length :]
        x3 = x[None, None, :]
        with torch.no_grad():
            out = self._pipe.predict(
                x3,
                prediction_length=self.model_ctx.prediction_length,
                batch_size=1,
                context_length=min(len(x), self.model_ctx.context_length),
            )

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
