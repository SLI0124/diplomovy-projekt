from __future__ import annotations

import shutil
import sys
import types
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
    model_family = "foundation"
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
        train_loss: str | None,
        train_optimizer: str | None,
        artifact_dir: Path,
    ) -> list[TrainingLossPoint]:
        if train_loss is not None or train_optimizer is not None:
            raise ValueError(
                f"--train-loss/--train-optimizer are only supported for custom models. '{self.slug}' is a foundation model."
            )

        _ensure_lag_llama_augmentation_shims()

        from gluonts.dataset.common import ListDataset
        from gluonts.torch.distributions.studentT import StudentTOutput
        from gluonts.torch.modules.loss import NegativeLogLikelihood
        from lag_llama.gluon.estimator import LagLlamaEstimator
        from lightning.pytorch.callbacks import Callback, ModelCheckpoint

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
        epoch_loss_cb = _EpochLossCallback()

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
                "callbacks": [checkpoint_cb, epoch_loss_cb],
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

        return epoch_loss_cb.history

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
            raise FileNotFoundError(f"Missing Lag-Llama checkpoint file: {ckpt_path}")

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
