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


class MoiraiAdapter(BaseFoundationModelAdapter):
    model_id = "Salesforce/moirai-1.0-R-base"
    slug = "moirai"
    model_family = "foundation"
    supports_finetune = True

    def __init__(self, model_ctx: ModelContext, device: torch.device) -> None:
        super().__init__(model_ctx, device)
        self._module = None
        self._predictor = None

    def _refresh_predictor(self) -> None:
        from uni2ts.model.moirai import MoiraiForecast

        if self._module is None:
            raise RuntimeError("Moirai module is not loaded.")

        forecast = MoiraiForecast(
            prediction_length=self.model_ctx.prediction_length,
            target_dim=1,
            feat_dynamic_real_dim=0,
            past_feat_dynamic_real_dim=0,
            context_length=self.model_ctx.context_length,
            module=self._module,
            num_samples=self.model_ctx.num_samples,
        )
        self._predictor = forecast.create_predictor(
            batch_size=32,
            device="cuda" if self.device.type == "cuda" else "cpu",
        )

    def load_pretrained(self) -> None:
        from uni2ts.model.moirai import MoiraiModule

        self._module = MoiraiModule.from_pretrained(self.model_id)
        self._module.to(self.device)
        self._module.eval()
        self._refresh_predictor()

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

        import lightning as L
        from lightning.pytorch.callbacks import Callback
        from uni2ts.data.dataset import TimeSeriesDataset
        from uni2ts.data.indexer._base import Indexer
        from uni2ts.data.loader import DataLoader as UniDataLoader
        from uni2ts.data.loader import PackCollate
        from uni2ts.model.moirai import MoiraiModule
        from uni2ts.model.moirai.finetune import MoiraiFinetune

        class _EpochLossCallback(Callback):
            def __init__(self) -> None:
                self.history: list[TrainingLossPoint] = []

            def on_train_epoch_end(self, trainer, pl_module) -> None:
                del pl_module
                metrics = trainer.callback_metrics
                value = metrics.get("train_loss") or metrics.get("loss")
                if value is None:
                    return
                self.history.append(
                    TrainingLossPoint(
                        epoch=int(trainer.current_epoch),
                        loss=float(value.detach().cpu().item()),
                    )
                )

        class _SingleSeriesIndexer(Indexer):
            def __init__(self, series: np.ndarray, freq: str = "H"):
                super().__init__(uniform=True)
                self._series = np.asarray(series, dtype=np.float32)
                self._freq = freq

            def __len__(self) -> int:
                return 1

            def _getitem_int(self, idx: int) -> dict[str, object]:
                return {
                    "target": self._series,
                    "freq": self._freq,
                    "start": pd.Timestamp("2013-01-01 00:00:00"),
                    "item_id": "series",
                }

            def _getitem_iterable(self, idx):
                idx_list = list(idx)
                return {
                    "target": [self._series for _ in idx_list],
                    "freq": [self._freq for _ in idx_list],
                    "start": [pd.Timestamp("2013-01-01 00:00:00") for _ in idx_list],
                    "item_id": ["series" for _ in idx_list],
                }

        base_module = MoiraiModule.from_pretrained(self.model_id)
        base_module.to(self.device)

        num_training_steps = int(train_epochs * train_steps_per_epoch)
        num_warmup_steps = max(1, int(0.1 * num_training_steps))

        ft = MoiraiFinetune(
            min_patches=2,
            min_mask_ratio=0.1,
            max_mask_ratio=0.5,
            max_dim=1,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
            module=base_module,
            lr=train_lr,
            log_on_step=False,
        )

        train_transform = ft.train_transform_map["default"]()
        indexer = _SingleSeriesIndexer(train_series, freq="H")

        ds = TimeSeriesDataset(
            indexer=indexer, transform=train_transform, dataset_weight=100.0
        )

        collate = PackCollate(
            max_length=ft.module.max_seq_len,
            seq_fields=ft.seq_fields,
            pad_func_map=ft.pad_func_map,
            target_field="target",
        )

        train_loader = UniDataLoader(
            dataset=ds,
            batch_size=train_batch_size,
            cycle=True,
            num_batches_per_epoch=train_steps_per_epoch,
            shuffle=False,
            num_workers=0,
            collate_fn=collate,
            pin_memory=torch.cuda.is_available(),
            drop_last=True,
            fill_last=True,
        )

        epoch_loss_cb = _EpochLossCallback()
        trainer = L.Trainer(
            max_epochs=train_epochs,
            accelerator="gpu" if self.device.type == "cuda" else "cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            log_every_n_steps=10,
            callbacks=[epoch_loss_cb],
        )
        trainer.fit(ft, train_dataloaders=train_loader)

        self._module = ft.module
        self._module.to(self.device)
        self._module.eval()
        self._refresh_predictor()
        return epoch_loss_cb.history

    def forecast(
        self, context: np.ndarray, context_start: pd.Timestamp
    ) -> ForecastResult:
        from gluonts.dataset.common import ListDataset

        if self._predictor is None:
            raise RuntimeError("Moirai predictor is not loaded.")

        x = np.asarray(context, dtype=np.float32)[-self.model_ctx.context_length :]
        dataset = ListDataset(
            [{"start": context_start, "target": x}],
            freq="h",
            one_dim_target=True,
        )
        fcst = next(iter(self._predictor.predict(dataset)))
        y_pred = np.asarray(fcst.samples, dtype=np.float32).mean(axis=0)  # type: ignore
        return ForecastResult(y_pred=y_pred)

    def save_finetuned(self, artifact_dir: Path) -> None:
        if self._module is None:
            raise RuntimeError("Moirai fine-tuned module is not available to save.")

        artifact_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": self._module.state_dict(),
                "repo_id": self.model_id,
            },
            artifact_dir / "model.pt",
        )

    def load_finetuned(self, artifact_dir: Path) -> None:
        from uni2ts.model.moirai import MoiraiModule

        payload = torch.load(artifact_dir / "model.pt", map_location="cpu")
        repo_id = str(payload["repo_id"])
        state_dict = payload["state_dict"]

        module = MoiraiModule.from_pretrained(repo_id)
        module.load_state_dict(state_dict)
        module.to(self.device)
        module.eval()

        self._module = module
        self._refresh_predictor()
