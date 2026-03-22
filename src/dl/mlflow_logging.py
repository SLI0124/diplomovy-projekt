from __future__ import annotations

import hashlib
from pathlib import Path

import mlflow
import torch
from config import (
    RuntimeConfig,
    mlflow_uri,
    models_root,
    results_root,
    to_serializable_dict,
)
from dataset import DatasetBundle, FoldData
from folds import FoldSpec


def ensure_mlflow(experiment_name: str) -> None:
    mlflow.set_tracking_uri(mlflow_uri())
    mlflow.set_experiment(experiment_name)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_dataset_profile(
    bundle: DatasetBundle,
    fold_data: FoldData,
    target_col: str,
) -> dict[str, object]:
    test_df = fold_data.test_df
    target = test_df[target_col].astype(float)

    profile: dict[str, object] = {
        "split_root": str(bundle.split_root),
        "train_split_path": str(fold_data.train_path),
        "test_split_path": str(fold_data.test_path),
        "train_split_size_bytes": int(fold_data.train_path.stat().st_size),
        "test_split_size_bytes": int(fold_data.test_path.stat().st_size),
        "train_split_sha256": sha256_file(fold_data.train_path),
        "test_split_sha256": sha256_file(fold_data.test_path),
        "train_rows": int(fold_data.train_series.shape[0]),
        "test_rows": int(len(test_df)),
        "test_columns": int(len(test_df.columns)),
        "target_col": target_col,
        "test_target_mean": float(target.mean()),
        "test_target_std": float(target.std(ddof=0)),
        "test_target_min": float(target.min()),
        "test_target_p50": float(target.quantile(0.5)),
        "test_target_max": float(target.max()),
        "test_target_nan_count": int(test_df[target_col].isna().sum()),
    }

    if bundle.run_params_path is not None:
        profile["preprocessing_run_params_path"] = str(bundle.run_params_path)

    return profile


def build_dataset_columns_preview(fold_data: FoldData, max_rows: int = 120) -> str:
    rows: list[str] = ["column,dtype,na_count"]

    df = fold_data.test_df
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
            }
        )


def safe_log_dataset_context(
    bundle: DatasetBundle,
    fold_data: FoldData,
    config: RuntimeConfig,
) -> None:
    profile = build_dataset_profile(
        bundle=bundle,
        fold_data=fold_data,
        target_col=config.target_col,
    )

    mlflow.set_tags(
        {
            "dataset.path": str(profile["split_root"]),
            "dataset.sha256": str(profile["test_split_sha256"]),
        }
    )
    mlflow.log_dict(profile, "dataset_profile.json")


def safe_log_run_context(
    *,
    config: RuntimeConfig,
    bundle: DatasetBundle,
    fold_data: FoldData,
    fold: FoldSpec,
    adapter,
    requested_model_name: str,
    model_family: str,
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
            "run.model.family": model_family,
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
        "model_family": model_family,
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
            "split_root": str(bundle.split_root),
            "train_split_path": str(fold_data.train_path),
            "test_split_path": str(fold_data.test_path),
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
