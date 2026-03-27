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


class Moirai1Adapter(BaseFoundationModelAdapter):
    model_id = "Salesforce/moirai-1.0-R-base"
    slug = "moirai1"
    model_family = "foundation"
    supports_finetune = True

    def __init__(self, model_ctx: ModelContext, device: torch.device) -> None:
        super().__init__(model_ctx, device)
        self._module = None
        self._predictor = None
        self._feat_dynamic_real_dim = 0
        self._past_feat_dynamic_real_dim = 0

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
        values: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray]:
        observed = np.isfinite(values).astype(np.float32)
        sanitized = np.where(observed > 0, values, 0.0).astype(np.float32)
        return sanitized, observed

    def _set_covariate_dims(self, *, feat_dim: int, past_feat_dim: int) -> None:
        if feat_dim < 0 or past_feat_dim < 0:
            raise ValueError(
                f"Covariate dims must be non-negative. Got feat_dim={feat_dim}, past_feat_dim={past_feat_dim}."
            )
        self._feat_dynamic_real_dim = int(feat_dim)
        self._past_feat_dynamic_real_dim = int(past_feat_dim)

    def _refresh_predictor(self) -> None:
        from uni2ts.model.moirai import MoiraiForecast

        if self._module is None:
            raise RuntimeError("Moirai module is not loaded.")

        forecast = MoiraiForecast(
            prediction_length=self.model_ctx.prediction_length,
            target_dim=1,
            feat_dynamic_real_dim=self._feat_dynamic_real_dim,
            past_feat_dynamic_real_dim=self._past_feat_dynamic_real_dim,
            context_length=self.model_ctx.context_length,
            module=self._module,
            num_samples=self.model_ctx.num_samples,
        )
        self._predictor = forecast.create_predictor(
            batch_size=32,
            device="cuda" if self.device.type == "cuda" else "cpu",
        )

    @staticmethod
    def _ensure_patch_freq_aliases() -> None:
        """Backfill Uni2TS patch-size aliases for newer pandas offset names.

        Uni2TS 1.1.1 expects legacy uppercase aliases (e.g. "H", "T", "S"), but
        newer pandas can emit lowercase/renamed offsets (e.g. "h", "min", "s").
        """
        from uni2ts.transform.patch import DefaultPatchSizeConstraints

        ranges = DefaultPatchSizeConstraints.DEFAULT_RANGES
        alias_to_canonical = {
            "h": "H",
            "min": "T",
            "t": "T",
            "s": "S",
            "d": "D",
            "b": "B",
            "w": "W",
            "me": "M",
            "m": "M",
            "qe": "Q",
            "q": "Q",
            "ye": "Y",
            "y": "Y",
            "a": "A",
        }
        for alias, canonical in alias_to_canonical.items():
            if alias not in ranges and canonical in ranges:
                ranges[alias] = ranges[canonical]

    def load_pretrained(self) -> None:
        from uni2ts.model.moirai import MoiraiModule

        self._set_covariate_dims(feat_dim=0, past_feat_dim=0)
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
                self.best_loss = float("inf")
                self.best_module_state: dict[str, torch.Tensor] | None = None

            def on_train_epoch_end(self, trainer, pl_module) -> None:
                del pl_module
                metrics = trainer.callback_metrics
                value = metrics.get("train_loss") or metrics.get("loss")
                if value is None:
                    return
                loss_value = float(value.detach().cpu().item())
                self.history.append(
                    TrainingLossPoint(
                        epoch=int(trainer.current_epoch),
                        loss=loss_value,
                    )
                )
                if loss_value < self.best_loss:
                    self.best_loss = loss_value
                    self.best_module_state = {
                        key: weight.detach().cpu().clone()
                        for key, weight in ft.module.state_dict().items()
                    }

        class _SingleSeriesIndexer(Indexer):
            def __init__(
                self,
                series: np.ndarray,
                freq: str = "H",
                past_feat_dynamic_real: np.ndarray | None = None,
                past_observed_feat_dynamic_real: np.ndarray | None = None,
                feat_dynamic_real: np.ndarray | None = None,
                observed_feat_dynamic_real: np.ndarray | None = None,
            ):
                super().__init__(uniform=True)
                self._series = np.asarray(series, dtype=np.float32)
                self._freq = freq
                self._past_feat_dynamic_real = past_feat_dynamic_real
                self._past_observed_feat_dynamic_real = past_observed_feat_dynamic_real
                self._feat_dynamic_real = feat_dynamic_real
                self._observed_feat_dynamic_real = observed_feat_dynamic_real

            def __len__(self) -> int:
                return 1

            def _getitem_int(self, idx: int) -> dict[str, object]:
                del idx
                row: dict[str, object] = {
                    "target": self._series,
                    "freq": self._freq,
                    "start": pd.Timestamp("2013-01-01 00:00:00"),
                    "item_id": "series",
                }
                if self._past_feat_dynamic_real is not None:
                    row["past_feat_dynamic_real"] = self._past_feat_dynamic_real
                if self._past_observed_feat_dynamic_real is not None:
                    row["past_observed_feat_dynamic_real"] = (
                        self._past_observed_feat_dynamic_real
                    )
                if self._feat_dynamic_real is not None:
                    row["feat_dynamic_real"] = self._feat_dynamic_real
                if self._observed_feat_dynamic_real is not None:
                    row["observed_feat_dynamic_real"] = self._observed_feat_dynamic_real
                return row

            def _getitem_iterable(self, idx):
                idx_list = list(idx)
                row: dict[str, object] = {
                    "target": [self._series for _ in idx_list],
                    "freq": [self._freq for _ in idx_list],
                    "start": [pd.Timestamp("2013-01-01 00:00:00") for _ in idx_list],
                    "item_id": ["series" for _ in idx_list],
                }
                if self._past_feat_dynamic_real is not None:
                    row["past_feat_dynamic_real"] = [
                        self._past_feat_dynamic_real for _ in idx_list
                    ]
                if self._past_observed_feat_dynamic_real is not None:
                    row["past_observed_feat_dynamic_real"] = [
                        self._past_observed_feat_dynamic_real for _ in idx_list
                    ]
                if self._feat_dynamic_real is not None:
                    row["feat_dynamic_real"] = [
                        self._feat_dynamic_real for _ in idx_list
                    ]
                if self._observed_feat_dynamic_real is not None:
                    row["observed_feat_dynamic_real"] = [
                        self._observed_feat_dynamic_real for _ in idx_list
                    ]
                return row

        y_train = np.asarray(train_series, dtype=np.float32)
        if y_train.ndim != 1:
            raise ValueError(
                f"train_series must be a 1D array. Got shape={y_train.shape}."
            )

        train_past_covariates: np.ndarray | None = None
        train_past_observed_covariates: np.ndarray | None = None
        train_feat_dynamic_real: np.ndarray | None = None
        train_observed_feat_dynamic_real: np.ndarray | None = None
        if train_covariates is not None:
            cov_train = self._as_float2d(
                train_covariates,
                name="train_covariates",
                expected_length=len(y_train),
            )
            (
                train_past_covariates_time_major,
                train_past_observed_covariates_time_major,
            ) = self._sanitize_with_observed_mask(cov_train)
            train_past_covariates = train_past_covariates_time_major.T
            train_past_observed_covariates = train_past_observed_covariates_time_major.T

        feat_dim = 0
        if train_future_covariates is not None:
            fut_train = self._as_float2d(
                train_future_covariates,
                name="train_future_covariates",
                expected_length=len(y_train),
            )
            feat_dim = int(fut_train.shape[1])
            (
                train_feat_dynamic_real_time_major,
                train_observed_feat_dynamic_real_time_major,
            ) = self._sanitize_with_observed_mask(fut_train)
            train_feat_dynamic_real = train_feat_dynamic_real_time_major.T
            train_observed_feat_dynamic_real = (
                train_observed_feat_dynamic_real_time_major.T
            )

            if (
                train_covariates is not None
                and fut_train.shape[1] != train_covariates.shape[1]
            ):
                raise ValueError(
                    "train_future_covariates must have the same feature count as train_covariates. "
                    f"Got {fut_train.shape[1]} and {train_covariates.shape[1]}."
                )

        past_feat_dim = (
            int(train_past_covariates.shape[1])
            if train_past_covariates is not None
            else 0
        )
        self._set_covariate_dims(feat_dim=feat_dim, past_feat_dim=past_feat_dim)
        max_dim = 1 + past_feat_dim

        base_module = MoiraiModule.from_pretrained(self.model_id)
        base_module.to(self.device)

        num_training_steps = int(train_epochs * train_steps_per_epoch)
        num_warmup_steps = max(1, int(0.1 * num_training_steps))

        ft = MoiraiFinetune(
            min_patches=2,
            min_mask_ratio=0.1,
            max_mask_ratio=0.5,
            max_dim=max_dim,
            num_training_steps=num_training_steps,
            num_warmup_steps=num_warmup_steps,
            module=base_module,
            lr=train_lr,
            log_on_step=False,
        )

        self._ensure_patch_freq_aliases()

        train_transform = ft.train_transform_map["default"]()
        indexer = _SingleSeriesIndexer(
            y_train,
            freq="H",
            past_feat_dynamic_real=train_past_covariates,
            past_observed_feat_dynamic_real=train_past_observed_covariates,
            feat_dynamic_real=train_feat_dynamic_real,
            observed_feat_dynamic_real=train_observed_feat_dynamic_real,
        )

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

        if checkpoint_selection not in {"best-train-loss", "last"}:
            raise ValueError(
                f"Unsupported checkpoint selection '{checkpoint_selection}'. "
                "Supported: best-train-loss, last"
            )
        if (
            checkpoint_selection == "best-train-loss"
            and epoch_loss_cb.best_module_state is not None
        ):
            ft.module.load_state_dict(epoch_loss_cb.best_module_state)

        self._module = ft.module
        self._module.to(self.device)
        self._module.eval()
        self._refresh_predictor()
        return epoch_loss_cb.history

    def forecast(
        self,
        context: np.ndarray,
        context_start: pd.Timestamp,
        context_covariates: np.ndarray | None = None,
        future_covariates: np.ndarray | None = None,
    ) -> ForecastResult:
        from gluonts.dataset.common import ListDataset

        if self._predictor is None:
            raise RuntimeError("Moirai predictor is not loaded.")

        x = np.asarray(context, dtype=np.float32)[-self.model_ctx.context_length :]
        if x.ndim != 1:
            raise ValueError(f"context must be a 1D array. Got shape={x.shape}.")

        context_cov_view: np.ndarray | None = None
        if context_covariates is not None:
            context_cov_full = self._as_float2d(
                context_covariates,
                name="context_covariates",
                expected_length=len(np.asarray(context)),
            )
            context_cov_view = context_cov_full[-len(x) :]

        future_cov_view: np.ndarray | None = None
        if future_covariates is not None:
            future_cov_all = self._as_float2d(
                future_covariates,
                name="future_covariates",
            )
            if future_cov_all.shape[0] < self.model_ctx.prediction_length:
                raise ValueError(
                    "future_covariates does not contain enough rows for prediction horizon. "
                    f"Required {self.model_ctx.prediction_length}, got {future_cov_all.shape[0]}."
                )
            future_cov_view = future_cov_all[: self.model_ctx.prediction_length]

        if (
            context_cov_view is not None
            and future_cov_view is not None
            and context_cov_view.shape[1] != future_cov_view.shape[1]
        ):
            raise ValueError(
                "context_covariates and future_covariates must have the same feature count. "
                f"Got {context_cov_view.shape[1]} and {future_cov_view.shape[1]}."
            )

        required_past_dim = (
            int(context_cov_view.shape[1]) if context_cov_view is not None else 0
        )
        required_feat_dim = (
            int(future_cov_view.shape[1]) if future_cov_view is not None else 0
        )
        if (
            required_feat_dim != self._feat_dynamic_real_dim
            or required_past_dim != self._past_feat_dynamic_real_dim
        ):
            self._set_covariate_dims(
                feat_dim=required_feat_dim,
                past_feat_dim=required_past_dim,
            )
            self._refresh_predictor()

        record: dict[str, object] = {"start": context_start, "target": x}
        if context_cov_view is not None:
            past_values, past_observed = self._sanitize_with_observed_mask(
                context_cov_view
            )
            record["past_feat_dynamic_real"] = past_values.T
            record["past_observed_feat_dynamic_real"] = past_observed.T

        if future_cov_view is not None:
            if context_cov_view is None:
                raise ValueError(
                    "future_covariates were provided but context_covariates are missing."
                )
            context_values, context_observed = self._sanitize_with_observed_mask(
                context_cov_view
            )
            future_values, future_observed = self._sanitize_with_observed_mask(
                future_cov_view
            )
            feat_values = np.concatenate((context_values, future_values), axis=0)
            feat_observed = np.concatenate((context_observed, future_observed), axis=0)
            record["feat_dynamic_real"] = feat_values.T
            record["observed_feat_dynamic_real"] = feat_observed.T

        dataset = ListDataset(
            [record],
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
                "feat_dynamic_real_dim": int(self._feat_dynamic_real_dim),
                "past_feat_dynamic_real_dim": int(self._past_feat_dynamic_real_dim),
            },
            artifact_dir / "model.pt",
        )

    def load_finetuned(self, artifact_dir: Path) -> None:
        from uni2ts.model.moirai import MoiraiModule

        payload = torch.load(artifact_dir / "model.pt", map_location="cpu")
        repo_id = str(payload["repo_id"])
        state_dict = payload["state_dict"]
        feat_dynamic_real_dim = int(payload.get("feat_dynamic_real_dim", 0))
        past_feat_dynamic_real_dim = int(payload.get("past_feat_dynamic_real_dim", 0))

        module = MoiraiModule.from_pretrained(repo_id)
        module.load_state_dict(state_dict)
        module.to(self.device)
        module.eval()

        self._set_covariate_dims(
            feat_dim=feat_dynamic_real_dim,
            past_feat_dim=past_feat_dynamic_real_dim,
        )
        self._module = module
        self._refresh_predictor()
