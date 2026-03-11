from __future__ import annotations

import hashlib
from pathlib import Path

import mlflow
import pandas as pd
import torch
from config import (
    RuntimeConfig,
    mlflow_experiment,
    mlflow_uri,
    models_root,
    results_root,
    to_serializable_dict,
)
from dataset import DatasetBundle
from folds import FoldSpec


def ensure_mlflow(config: RuntimeConfig) -> None:
    mlflow.set_tracking_uri(mlflow_uri())
    mlflow.set_experiment(mlflow_experiment())


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_profile(bundle: DatasetBundle, target_col: str) -> dict[str, object]:
    df = bundle.dataframe
    target = df[target_col].astype(float)

    profile: dict[str, object] = {
        "dataset_path": str(bundle.dataset_path),
        "dataset_file_size_bytes": int(bundle.dataset_path.stat().st_size),
        "dataset_file_sha256": sha256_file(bundle.dataset_path),
        "dataset_rows": int(len(df)),
        "dataset_columns": int(len(df.columns)),
        "timestamp_min": str(pd.Timestamp(df["timestamp"].min())),
        "timestamp_max": str(pd.Timestamp(df["timestamp"].max())),
        "target_col": target_col,
        "target_mean": float(target.mean()),
        "target_std": float(target.std(ddof=0)),
        "target_min": float(target.min()),
        "target_p50": float(target.quantile(0.5)),
        "target_max": float(target.max()),
        "target_nan_count": int(df[target_col].isna().sum()),
    }

    if bundle.run_params_path is not None:
        profile["preprocessing_run_params_path"] = str(bundle.run_params_path)

    return profile


def build_dataset_columns_preview(bundle: DatasetBundle, max_rows: int = 120) -> str:
    rows: list[str] = ["column,dtype,na_count"]

    df = bundle.dataframe
    for column in df.columns[:max_rows]:
        rows.append(f"{column},{df[column].dtype},{int(df[column].isna().sum())}")

    return "\n".join(rows) + "\n"


def safe_log_run_params(
    config: RuntimeConfig,
    bundle: DatasetBundle,
    fold: FoldSpec,
    dataset_tag: str,
) -> None:
    mlflow.log_params(
        {
            "action": config.action,
            "mode": config.mode,
            "eval_after_train": config.eval_after_train,
            "test_year": fold.test_year,
            "train_end_year": fold.train_end_year,
            "target_col": config.target_col,
            "prediction_length": config.prediction_length,
            "window_stride": config.window_stride,
            "context_length": config.context_length,
            "max_origins_per_year": config.max_origins_per_year,
            "variant_stem": config.variant_stem or "",
            "dataset_tag": dataset_tag,
        }
    )

    if config.mode == "finetuned":
        mlflow.log_params(
            {
                "train_epochs": config.train_epochs,
                "train_batch_size": config.train_batch_size,
                "train_lr": config.train_lr,
                "train_weight_decay": config.train_weight_decay,
                "train_steps_per_epoch": config.train_steps_per_epoch,
            }
        )
    else:
        mlflow.log_params(
            {
                "num_samples": config.num_samples,
                "lag_llama_num_parallel_samples": config.lag_llama_num_parallel_samples,
            }
        )


def safe_log_dataset_context(bundle: DatasetBundle, config: RuntimeConfig) -> None:
    profile = build_dataset_profile(bundle=bundle, target_col=config.target_col)

    mlflow.set_tags(
        {
            "dataset.path": str(profile["dataset_path"]),
            "dataset.sha256": str(profile["dataset_file_sha256"]),
        }
    )
    mlflow.log_dict(profile, "dataset_profile.json")


def safe_log_run_context(
    *,
    config: RuntimeConfig,
    bundle: DatasetBundle,
    fold: FoldSpec,
    adapter,
    requested_model_name: str,
    device: torch.device,
    run_root: Path,
    dataset_tag: str,
    checkpoint_dir: Path | None,
) -> None:
    mode_flags = {
        "is_one_shot": config.mode == "one-shot",
        "is_finetuned": config.mode == "finetuned",
        "is_train": config.action == "train",
        "is_test": config.action == "test",
        "is_eval": config.action == "eval",
        "eval_after_train": config.eval_after_train,
    }

    mlflow.set_tags(
        {
            "run.kind": f"{config.action}:{config.mode}",
            "run.model.requested": requested_model_name,
            "run.model.resolved": adapter.slug,
            "run.dataset.tag": dataset_tag,
            "run.fold": f"train_2013-{fold.train_end_year}__test-{fold.test_year}",
        }
    )

    context_payload = {
        "run_name": (
            f"{adapter.slug}__{config.mode}__{config.action}__"
            f"train-2013-{fold.train_end_year}__test-{fold.test_year}"
        ),
        "requested_model": requested_model_name,
        "resolved_model": {
            "slug": adapter.slug,
            "model_id": adapter.model_id,
            "supports_finetune": bool(adapter.supports_finetune),
        },
        "mode_flags": mode_flags,
        "fold": {
            "train_years": fold.train_years_label,
            "train_end_year": fold.train_end_year,
            "test_year": fold.test_year,
        },
        "runtime": to_serializable_dict(config),
        "dataset": {
            "path": str(bundle.dataset_path),
            "tag": dataset_tag,
            "run_params_path": (
                str(bundle.run_params_path) if bundle.run_params_path else None
            ),
        },
        "paths": {
            "run_root": str(run_root),
            "models_root": str(models_root()),
            "results_root": str(results_root()),
            "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        },
        "device": str(device),
    }
    mlflow.log_dict(context_payload, "run_context.json")
