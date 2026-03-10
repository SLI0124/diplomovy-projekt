from __future__ import annotations

import math
import shutil
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset


@dataclass(frozen=True)
class ModelContext:
    prediction_length: int
    context_length: int
    num_samples: int
    lag_llama_num_parallel_samples: int


@dataclass(frozen=True)
class ForecastResult:
    y_pred: np.ndarray


class RandomWindowDataset(Dataset):
    def __init__(
        self,
        series: np.ndarray,
        context_length: int,
        prediction_length: int,
        n_samples: int,
    ) -> None:
        self.series = np.asarray(series, dtype=np.float32)
        self.context_length = int(context_length)
        self.prediction_length = int(prediction_length)
        self.n_samples = int(n_samples)

        min_total = self.context_length + self.prediction_length
        if len(self.series) < min_total:
            raise ValueError(
                f"Need at least {min_total} points, got {len(self.series)}"
            )

    def __len__(self) -> int:
        return self.n_samples

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        max_start = len(self.series) - (self.context_length + self.prediction_length)
        start = int(np.random.randint(0, max_start + 1))

        past = self.series[start : start + self.context_length]
        future = self.series[
            start
            + self.context_length : start
            + self.context_length
            + self.prediction_length
        ]

        return {
            "context": torch.from_numpy(past),
            "future_target": torch.from_numpy(future),
        }


class BaseFoundationModelAdapter:
    model_id: str
    slug: str
    supports_finetune: bool = False

    def __init__(self, model_ctx: ModelContext, device: torch.device) -> None:
        self.model_ctx = model_ctx
        self.device = device

    def load_pretrained(self) -> None:
        raise NotImplementedError

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
        raise NotImplementedError

    def forecast(
        self, context: np.ndarray, context_start: pd.Timestamp
    ) -> ForecastResult:
        raise NotImplementedError

    def save_finetuned(self, artifact_dir: Path) -> None:
        raise NotImplementedError

    def load_finetuned(self, artifact_dir: Path) -> None:
        raise NotImplementedError


class Chronos2Adapter(BaseFoundationModelAdapter):
    model_id = "amazon/chronos-2"
    slug = "chronos2"
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
        artifact_dir: Path,
    ) -> None:
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
        for _ in range(train_epochs):
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

        model.eval()
        self._model = model
        self._pipe = None

        qs = model.chronos_config.quantiles
        self._q_index = int(np.argmin(np.abs(np.asarray(qs) - 0.5)))

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


def _patch_lag_llama_lags_handling() -> None:
    _ensure_lag_llama_augmentation_shims()

    import lag_llama.gluon.estimator as ll_est

    orig_fn = ll_est.get_lags_for_frequency

    def patched_get_lags_for_frequency(freq_str, *args, **kwargs):
        if isinstance(freq_str, int):
            return [freq_str + 1]
        return orig_fn(freq_str, *args, **kwargs)

    ll_est.get_lags_for_frequency = patched_get_lags_for_frequency


def _patch_lagged_sequence_values_for_lag_llama() -> None:
    _ensure_lag_llama_augmentation_shims()

    import gluonts.torch.util as gtu
    import lag_llama.model.module as ll_module
    from gluonts.torch.util import slice_along_dim

    def stable_lagged_sequence_values(
        indices: list[int],
        prior_sequence: torch.Tensor,
        sequence: torch.Tensor,
        dim: int,
    ) -> torch.Tensor:
        seq_len = int(sequence.shape[dim])
        if seq_len <= 0:
            raise ValueError(f"sequence length must be > 0, got {seq_len}")

        full_sequence = torch.cat((prior_sequence, sequence), dim=dim)
        full_len = int(full_sequence.shape[dim])

        lags_values: list[torch.Tensor] = []
        for lag_index in indices:
            lag = int(lag_index)
            if lag < 0:
                raise ValueError(f"negative lag is not supported: {lag}")

            end = full_len - lag
            start = end - seq_len
            lags_values.append(
                slice_along_dim(full_sequence, dim=dim, slice_=slice(start, end))
            )

        return torch.stack(lags_values, dim=-1)

    gtu.lagged_sequence_values = stable_lagged_sequence_values
    ll_module.lagged_sequence_values = stable_lagged_sequence_values


def _ensure_lag_llama_augmentation_shims() -> None:
    if "data.augmentations.augmentations" in sys.modules:
        return

    data_mod = sys.modules.get("data")
    if data_mod is None or not hasattr(data_mod, "__path__"):
        data_mod = types.ModuleType("data")
        data_mod.__path__ = []
        sys.modules["data"] = data_mod

    aug_pkg = types.ModuleType("data.augmentations")
    aug_pkg.__path__ = []
    sys.modules["data.augmentations"] = aug_pkg

    freq_mask_mod = types.ModuleType("data.augmentations.freq_mask")
    freq_mix_mod = types.ModuleType("data.augmentations.freq_mix")
    aug_mod = types.ModuleType("data.augmentations.augmentations")

    def freq_mask(past_target, future_target, rate=0.0):
        return past_target, future_target

    def freq_mix(past_target, future_target, rate=0.0):
        return past_target, future_target

    class _NoOpAugmentation:
        def __init__(self, *args, **kwargs):
            pass

        def __call__(self, past_target, future_target):
            return past_target, future_target

    class ApplyAugmentations:
        def __init__(self, transforms):
            self.transforms = list(transforms)

        def __call__(self, past_target, future_target):
            out_past, out_future = past_target, future_target
            for transform in self.transforms:
                out_past, out_future = transform(out_past, out_future)
            return out_past, out_future

    freq_mask_mod.freq_mask = freq_mask  # type: ignore
    freq_mix_mod.freq_mix = freq_mix  # type: ignore

    aug_mod.ApplyAugmentations = ApplyAugmentations  # type: ignore
    aug_mod.Jitter = _NoOpAugmentation  # type: ignore
    aug_mod.MagnitudeWarp = _NoOpAugmentation  # type: ignore
    aug_mod.Permutation = _NoOpAugmentation  # type: ignore
    aug_mod.Rotation = _NoOpAugmentation  # type: ignore
    aug_mod.Scaling = _NoOpAugmentation  # type: ignore
    aug_mod.TimeWarp = _NoOpAugmentation  # type: ignore
    aug_mod.WindowSlice = _NoOpAugmentation  # type: ignore
    aug_mod.WindowWarp = _NoOpAugmentation  # type: ignore

    sys.modules["data.augmentations.freq_mask"] = freq_mask_mod
    sys.modules["data.augmentations.freq_mix"] = freq_mix_mod
    sys.modules["data.augmentations.augmentations"] = aug_mod


class LagLlamaAdapter(BaseFoundationModelAdapter):
    model_id = "time-series-foundation-models/Lag-Llama"
    slug = "lag-llama"
    supports_finetune = True

    def __init__(self, model_ctx: ModelContext, device: torch.device) -> None:
        super().__init__(model_ctx, device)
        self._predictor = None
        self._ckpt_path = None
        self._model_kwargs = None
        self._finetuned_ckpt_path = None

    def _download_ckpt(self) -> str:
        from huggingface_hub import hf_hub_download

        return hf_hub_download(repo_id=self.model_id, filename="lag-llama.ckpt")

    def _build_predictor_from_ckpt(self, ckpt_path: str) -> None:
        _ensure_lag_llama_augmentation_shims()

        from gluonts.torch.distributions.studentT import StudentTOutput
        from gluonts.torch.modules.loss import NegativeLogLikelihood
        from lag_llama.gluon.estimator import LagLlamaEstimator

        torch.serialization.add_safe_globals([StudentTOutput, NegativeLogLikelihood])

        ckpt_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_kwargs = ckpt_obj["hyper_parameters"]["model_kwargs"]
        lags_seq = [int(v) for v in model_kwargs["lags_seq"]]

        _patch_lag_llama_lags_handling()
        _patch_lagged_sequence_values_for_lag_llama()

        estimator = LagLlamaEstimator(
            prediction_length=self.model_ctx.prediction_length,
            context_length=int(model_kwargs["context_length"]),
            input_size=int(model_kwargs["input_size"]),
            n_layer=int(model_kwargs["n_layer"]),
            n_head=int(model_kwargs["n_head"]),
            n_embd_per_head=int(model_kwargs["n_embd_per_head"]),
            scaling=str(model_kwargs.get("scaling", "robust")),
            time_feat=bool(model_kwargs.get("time_feat", True)),
            dropout=float(model_kwargs.get("dropout", 0.0)),
            ckpt_path=None,
            device=self.device,
            batch_size=1,
            num_parallel_samples=self.model_ctx.lag_llama_num_parallel_samples,
            lags_seq=lags_seq,
            use_single_pass_sampling=True,
        )

        transformation = estimator.create_transformation()
        module = estimator.create_lightning_module(use_kv_cache=True)

        state_dict = ckpt_obj.get("state_dict")
        if state_dict is None:
            raise KeyError("Lag-Llama checkpoint missing state_dict")

        module.load_state_dict(state_dict, strict=False)
        module.to(self.device)
        module.eval()

        self._predictor = estimator.create_predictor(transformation, module)
        self._ckpt_path = ckpt_path
        self._model_kwargs = model_kwargs

    def load_pretrained(self) -> None:
        ckpt_path = self._download_ckpt()
        self._build_predictor_from_ckpt(ckpt_path)

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
        _ensure_lag_llama_augmentation_shims()

        from gluonts.dataset.common import ListDataset
        from gluonts.torch.distributions.studentT import StudentTOutput
        from gluonts.torch.modules.loss import NegativeLogLikelihood
        from lag_llama.gluon.estimator import LagLlamaEstimator
        from lightning.pytorch.callbacks import ModelCheckpoint

        artifact_dir.mkdir(parents=True, exist_ok=True)

        ckpt_path = self._download_ckpt()

        _patch_lag_llama_lags_handling()
        _patch_lagged_sequence_values_for_lag_llama()
        torch.serialization.add_safe_globals([StudentTOutput, NegativeLogLikelihood])

        ckpt_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_kwargs = ckpt_obj["hyper_parameters"]["model_kwargs"]
        lags_seq = [int(v) for v in model_kwargs["lags_seq"]]

        checkpoint_cb = ModelCheckpoint(
            dirpath=str(artifact_dir),
            filename="model",
            save_last=True,
            save_top_k=-1,
            every_n_epochs=1,
        )

        estimator = LagLlamaEstimator(
            prediction_length=self.model_ctx.prediction_length,
            context_length=int(model_kwargs["context_length"]),
            input_size=int(model_kwargs["input_size"]),
            n_layer=int(model_kwargs["n_layer"]),
            n_head=int(model_kwargs["n_head"]),
            n_embd_per_head=int(model_kwargs["n_embd_per_head"]),
            scaling=str(model_kwargs.get("scaling", "robust")),
            time_feat=bool(model_kwargs.get("time_feat", True)),
            dropout=float(model_kwargs.get("dropout", 0.0)),
            ckpt_path=ckpt_path,
            device=self.device,
            batch_size=max(1, train_batch_size),
            num_batches_per_epoch=max(1, train_steps_per_epoch),
            lr=float(train_lr),
            weight_decay=float(train_weight_decay),
            num_parallel_samples=self.model_ctx.lag_llama_num_parallel_samples,
            lags_seq=lags_seq,
            use_single_pass_sampling=True,
            trainer_kwargs={
                "max_epochs": max(1, train_epochs),
                "logger": False,
                "enable_progress_bar": True,
                "callbacks": [checkpoint_cb],
                "default_root_dir": str(artifact_dir),
            },
        )

        train_ds = ListDataset(
            [{"start": pd.Timestamp("2013-01-01 00:00:00"), "target": train_series}],
            freq="h",
            one_dim_target=True,
        )
        self._predictor = estimator.train(train_ds)

        ckpt_candidates = sorted(artifact_dir.glob("*.ckpt"))
        if (artifact_dir / "last.ckpt").exists():
            self._finetuned_ckpt_path = artifact_dir / "last.ckpt"
        elif ckpt_candidates:
            self._finetuned_ckpt_path = ckpt_candidates[-1]
        else:
            raise FileNotFoundError(
                f"Lag-Llama fine-tuning finished but no checkpoint was saved in: {artifact_dir}"
            )

    def forecast(
        self, context: np.ndarray, context_start: pd.Timestamp
    ) -> ForecastResult:
        from gluonts.dataset.common import ListDataset

        if self._predictor is None:
            raise RuntimeError("Lag-Llama predictor is not loaded.")

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
        if self._finetuned_ckpt_path is None or not self._finetuned_ckpt_path.exists():
            raise RuntimeError(
                "Lag-Llama fine-tuned checkpoint is not available to save."
            )

        artifact_dir.mkdir(parents=True, exist_ok=True)
        target = artifact_dir / "model.ckpt"
        if self._finetuned_ckpt_path.resolve() != target.resolve():
            shutil.copy2(self._finetuned_ckpt_path, target)

        (artifact_dir / "finetune_manifest.json").write_text(
            '{"checkpoint":"model.ckpt"}',
            encoding="utf-8",
        )

    def load_finetuned(self, artifact_dir: Path) -> None:
        from gluonts.torch.distributions.studentT import StudentTOutput
        from gluonts.torch.modules.loss import NegativeLogLikelihood

        _ensure_lag_llama_augmentation_shims()

        from lag_llama.gluon.estimator import LagLlamaEstimator

        ckpt_path = artifact_dir / "model.ckpt"
        if not ckpt_path.exists():
            raise FileNotFoundError(
                f"Missing Lag-Llama checkpoint file: {ckpt_path}"
            )

        _patch_lag_llama_lags_handling()
        _patch_lagged_sequence_values_for_lag_llama()

        torch.serialization.add_safe_globals([StudentTOutput, NegativeLogLikelihood])

        ckpt_obj = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model_kwargs = ckpt_obj["hyper_parameters"]["model_kwargs"]
        lags_seq = [int(v) for v in model_kwargs["lags_seq"]]
        state_dict = ckpt_obj.get("state_dict")
        if state_dict is None:
            raise KeyError("Lag-Llama checkpoint missing state_dict")

        estimator = LagLlamaEstimator(
            prediction_length=self.model_ctx.prediction_length,
            context_length=int(model_kwargs["context_length"]),
            input_size=int(model_kwargs["input_size"]),
            n_layer=int(model_kwargs["n_layer"]),
            n_head=int(model_kwargs["n_head"]),
            n_embd_per_head=int(model_kwargs["n_embd_per_head"]),
            scaling=str(model_kwargs.get("scaling", "robust")),
            time_feat=bool(model_kwargs.get("time_feat", True)),
            dropout=float(model_kwargs.get("dropout", 0.0)),
            ckpt_path=None,
            device=self.device,
            batch_size=1,
            num_parallel_samples=self.model_ctx.lag_llama_num_parallel_samples,
            lags_seq=lags_seq,
            use_single_pass_sampling=True,
        )

        transformation = estimator.create_transformation()
        module = estimator.create_lightning_module(use_kv_cache=True)
        module.load_state_dict(state_dict, strict=False)
        module.to(self.device)
        module.eval()

        self._predictor = estimator.create_predictor(transformation, module)
        self._finetuned_ckpt_path = ckpt_path


class MoiraiAdapter(BaseFoundationModelAdapter):
    model_id = "Salesforce/moirai-1.0-R-base"
    slug = "moirai"
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
        artifact_dir: Path,
    ) -> None:
        import lightning as L
        from uni2ts.data.dataset import TimeSeriesDataset
        from uni2ts.data.indexer._base import Indexer
        from uni2ts.data.loader import DataLoader as UniDataLoader
        from uni2ts.data.loader import PackCollate
        from uni2ts.model.moirai import MoiraiModule
        from uni2ts.model.moirai.finetune import MoiraiFinetune

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

        trainer = L.Trainer(
            max_epochs=train_epochs,
            accelerator="gpu" if self.device.type == "cuda" else "cpu",
            devices=1,
            logger=False,
            enable_checkpointing=False,
            enable_progress_bar=True,
            log_every_n_steps=10,
        )
        trainer.fit(ft, train_dataloaders=train_loader)

        self._module = ft.module
        self._module.to(self.device)
        self._module.eval()
        self._refresh_predictor()

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


class TimesFM25Adapter(BaseFoundationModelAdapter):
    model_id = "google/timesfm-2.5-200m-pytorch"
    slug = "timesfm2.5"
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
        artifact_dir: Path,
    ) -> None:
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
        loss_fn = torch.nn.MSELoss()

        module.train()
        for _ in range(max(1, train_epochs)):
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

        module.eval()
        artifact_dir.mkdir(parents=True, exist_ok=True)
        self._model.save_pretrained(artifact_dir)  # type: ignore

    def forecast(
        self, context: np.ndarray, context_start: pd.Timestamp
    ) -> ForecastResult:
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


def build_model_adapter(
    model_name: str, model_ctx: ModelContext, device: torch.device
) -> BaseFoundationModelAdapter:
    model_name = model_name.lower()
    if model_name == "chronos2":
        return Chronos2Adapter(model_ctx=model_ctx, device=device)
    if model_name == "lag-llama":
        return LagLlamaAdapter(model_ctx=model_ctx, device=device)
    if model_name == "moirai":
        return MoiraiAdapter(model_ctx=model_ctx, device=device)
    if model_name == "timesfm2.5":
        return TimesFM25Adapter(model_ctx=model_ctx, device=device)

    raise ValueError(
        f"Unsupported model '{model_name}'. Supported: chronos2, lag-llama, moirai, timesfm2.5"
    )
