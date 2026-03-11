from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from adapters.base import BaseFoundationModelAdapter, ForecastResult, ModelContext


class Model1Adapter(BaseFoundationModelAdapter):
    model_id = "custom/model_1"
    slug = "model_1"
    model_family = "custom"
    supports_finetune = True

    def __init__(self, model_ctx: ModelContext, device: torch.device) -> None:
        super().__init__(model_ctx, device)
        self._last_value: float = 0.0
        self._slope_per_step: float = 0.0
        self._loaded: bool = False

    def load_pretrained(self) -> None:
        # Baseline one-shot behavior: persistence forecast (last value repeated).
        self._last_value = 0.0
        self._slope_per_step = 0.0
        self._loaded = True

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
        artifact_dir.mkdir(parents=True, exist_ok=True)

        series = np.asarray(train_series, dtype=np.float32)
        if series.size == 0:
            raise ValueError("Model1Adapter requires non-empty train_series.")

        self._last_value = float(series[-1])

        prediction_length = int(max(1, self.model_ctx.prediction_length))
        start_min = 1
        start_max = int(series.size - prediction_length)
        if start_max < start_min:
            self._slope_per_step = 0.0
            self._loaded = True
            return

        # Warm-start slope from recent deltas before optimizer updates.
        lookback = int(min(series.size - 1, 24 * 7))
        recent = series[-(lookback + 1) :]
        initial_slope = float(np.diff(recent).mean()) if recent.size >= 2 else 0.0

        batch_size = int(max(1, train_batch_size))
        steps_per_epoch = int(max(1, train_steps_per_epoch))
        epochs = int(max(1, train_epochs))

        slope = torch.nn.Parameter(
            torch.tensor(initial_slope, dtype=torch.float32, device=self.device)
        )
        optimizer = torch.optim.AdamW(
            [slope],
            lr=float(train_lr),
            weight_decay=float(train_weight_decay),
        )
        step_offsets = torch.arange(
            1,
            prediction_length + 1,
            device=self.device,
            dtype=torch.float32,
        ).view(1, -1)
        rng = np.random.default_rng()

        for _ in range(epochs):
            for _ in range(steps_per_epoch):
                starts = rng.integers(start_min, start_max + 1, size=batch_size)
                bases = torch.from_numpy(series[starts - 1]).to(self.device).float()
                bases = bases.view(-1, 1)
                targets_np = np.stack(
                    [series[s : s + prediction_length] for s in starts],
                    axis=0,
                )
                targets = torch.from_numpy(targets_np).to(self.device).float()

                preds = bases + slope * step_offsets
                loss = torch.mean((preds - targets) ** 2)

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

        self._slope_per_step = float(slope.detach().cpu().item())

        self._loaded = True

    def forecast(
        self, context: np.ndarray, context_start: pd.Timestamp
    ) -> ForecastResult:
        if not self._loaded:
            raise RuntimeError(
                "Model_1 is not loaded. Call load_pretrained/load_finetuned."
            )

        x = np.asarray(context, dtype=np.float32)
        base = float(x[-1]) if x.size > 0 else self._last_value

        horizon = int(self.model_ctx.prediction_length)
        steps = np.arange(1, horizon + 1, dtype=np.float32)
        y_pred = np.asarray(base + self._slope_per_step * steps, dtype=np.float32)
        return ForecastResult(y_pred=y_pred)

    def save_finetuned(self, artifact_dir: Path) -> None:
        if not self._loaded:
            raise RuntimeError("Model_1 is not available to save.")

        artifact_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "model_id": self.model_id,
            "slug": self.slug,
            "last_value": self._last_value,
            "slope_per_step": self._slope_per_step,
        }
        (artifact_dir / "model.json").write_text(
            json.dumps(payload, indent=2),
            encoding="utf-8",
        )

    def load_finetuned(self, artifact_dir: Path) -> None:
        model_path = artifact_dir / "model.json"
        if not model_path.exists():
            raise FileNotFoundError(f"Missing Model_1 artifact: {model_path}")

        payload = json.loads(model_path.read_text(encoding="utf-8"))
        self._last_value = float(payload.get("last_value", 0.0))
        self._slope_per_step = float(payload.get("slope_per_step", 0.0))
        self._loaded = True
