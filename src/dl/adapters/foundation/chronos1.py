from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from adapters.base import BaseFoundationModelAdapter, ForecastResult, ModelContext
from adapters.shared import RandomWindowDataset
from torch.utils.data import DataLoader


class Chronos1Adapter(BaseFoundationModelAdapter):
    model_id = "amazon/chronos-t5-base"
    slug = "chronos1"
    model_family = "foundation"
    supports_finetune = True

    def __init__(self, model_ctx: ModelContext, device: torch.device) -> None:
        super().__init__(model_ctx, device)
        self._pipe = None

    def load_pretrained(self) -> None:
        from chronos import ChronosPipeline

        self._pipe = ChronosPipeline.from_pretrained(self.model_id)
        self._pipe.model.to(self.device)  # type: ignore

    def finetune(
        self,
        train_series: np.ndarray,
        train_epochs: int,
        train_batch_size: int,
        train_steps_per_epoch: int,
        train_lr: float,
        train_weight_decay: float,
        artifact_dir: Path,
    ) -> None:
        pipe = self._ensure_pipeline_loaded()
        model = pipe.model.model

        for param in model.parameters():
            param.requires_grad = False

        trainable_params: list[torch.nn.Parameter] = []
        lm_head = getattr(model, "lm_head", None)
        if isinstance(lm_head, torch.nn.Module):
            for param in lm_head.parameters():
                param.requires_grad = True
                trainable_params.append(param)

        if not trainable_params:
            for param in model.parameters():
                param.requires_grad = True
            trainable_params = list(model.parameters())

        context_length = int(
            min(self.model_ctx.context_length, pipe.model.config.context_length)
        )
        train_prediction_length = int(pipe.model.config.prediction_length)

        ds = RandomWindowDataset(
            series=train_series,
            context_length=context_length,
            prediction_length=train_prediction_length,
            n_samples=train_batch_size * train_steps_per_epoch,
        )
        dl = DataLoader(ds, batch_size=train_batch_size, shuffle=False)

        optimizer = torch.optim.AdamW(
            trainable_params,
            lr=train_lr,
            weight_decay=train_weight_decay,
        )

        model.train()
        for _ in range(train_epochs):
            for batch in dl:
                ctx = batch["context"]
                fut = batch["future_target"]

                input_ids, attention_mask, tokenizer_state = (
                    pipe.tokenizer.context_input_transform(ctx)
                )
                labels, labels_mask = pipe.tokenizer.label_input_transform(
                    fut, tokenizer_state
                )

                input_ids = input_ids.to(self.device)
                attention_mask = attention_mask.to(self.device)
                labels = labels.to(self.device)
                labels_mask = labels_mask.to(self.device)
                labels = labels.masked_fill(~labels_mask.bool(), -100)

                out = model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = out.loss
                if loss is None:
                    raise RuntimeError("Chronos1 model did not return a loss.")

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        model.eval()

    def forecast(
        self, context: np.ndarray, context_start: pd.Timestamp
    ) -> ForecastResult:
        del context_start

        pipe = self._ensure_pipeline_loaded()
        effective_context_length = int(
            min(self.model_ctx.context_length, pipe.model.config.context_length)
        )
        x = np.asarray(context, dtype=np.float32)[-effective_context_length:]
        ctx = torch.from_numpy(x)[None, :]

        with torch.no_grad():
            samples = pipe.predict(
                ctx,
                prediction_length=self.model_ctx.prediction_length,
                num_samples=self.model_ctx.num_samples,
            )

        y_pred = samples[0].float().mean(dim=0).cpu().numpy()
        return ForecastResult(y_pred=y_pred)

    def save_finetuned(self, artifact_dir: Path) -> None:
        pipe = self._ensure_pipeline_loaded()

        artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": pipe.model.model.state_dict(),
                "repo_id": self.model_id,
            },
            artifact_dir / "model.pt",
        )

    def load_finetuned(self, artifact_dir: Path) -> None:
        from chronos import ChronosPipeline

        payload = torch.load(artifact_dir / "model.pt", map_location="cpu")
        repo_id = str(payload.get("repo_id", self.model_id))
        state_dict = payload["state_dict"]

        pipe = ChronosPipeline.from_pretrained(repo_id)
        pipe.model.model.load_state_dict(state_dict)
        pipe.model.to(self.device)  # type: ignore
        pipe.model.model.eval()
        self._pipe = pipe

    def _ensure_pipeline_loaded(self):
        if self._pipe is None:
            self.load_pretrained()
        if self._pipe is None:
            raise RuntimeError("Chronos1 model is not loaded.")
        return self._pipe
