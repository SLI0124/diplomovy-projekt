from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path
from urllib.parse import unquote

import mlflow
import torch
from config import (
    RuntimeConfig,
    mlflow_artifacts_root,
    mlflow_uri,
    models_root,
    results_root,
    to_serializable_dict,
)
from dataset import DatasetBundle, FoldData
from folds import FoldSpec
from mlflow.tracking import MlflowClient


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _experiment_artifact_location(experiment_name: str) -> str:
    artifact_root = mlflow_artifacts_root().resolve() / experiment_name
    artifact_root.mkdir(parents=True, exist_ok=True)
    return artifact_root.as_uri()


def _sqlite_db_path(tracking_uri: str) -> Path | None:
    if not tracking_uri.startswith("sqlite:///"):
        return None
    return Path(unquote(tracking_uri.removeprefix("sqlite:///")))


def _repair_experiment_artifact_location(
    *,
    tracking_uri: str,
    experiment_id: str,
    artifact_location: str,
) -> bool:
    db_path = _sqlite_db_path(tracking_uri)
    if db_path is None:
        return False

    with sqlite3.connect(db_path) as connection:
        connection.execute(
            "UPDATE experiments SET artifact_location = ? WHERE experiment_id = ?",
            (artifact_location, experiment_id),
        )
        connection.commit()

    return True


def ensure_mlflow(experiment_name: str) -> None:
    tracking_uri = mlflow_uri()
    desired_artifact_location = _experiment_artifact_location(experiment_name)

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri)
    experiment = client.get_experiment_by_name(experiment_name)

    if experiment is None:
        client.create_experiment(
            name=experiment_name,
            artifact_location=desired_artifact_location,
        )
    elif experiment.artifact_location != desired_artifact_location:
        repaired = _repair_experiment_artifact_location(
            tracking_uri=tracking_uri,
            experiment_id=experiment.experiment_id,
            artifact_location=desired_artifact_location,
        )
        if not repaired:
            current = experiment.artifact_location
            raise RuntimeError(
                "MLflow experiment artifact location is not portable and could not be repaired automatically. "
                f"experiment='{experiment_name}' current='{current}' desired='{desired_artifact_location}'"
            )

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


def build_covariate_columns_preview(fold_data: FoldData) -> str:
    rows: list[str] = ["group,column"]
    for column in fold_data.covariate_columns:
        rows.append(f"selected,{column}")
    for column in fold_data.future_covariate_columns:
        rows.append(f"future,{column}")
    for column in fold_data.past_covariate_columns:
        rows.append(f"past,{column}")
    return "\n".join(rows) + "\n"


def safe_log_run_params(
    config: RuntimeConfig,
    bundle: DatasetBundle,
    fold: FoldSpec,
    dataset_tag: str,
    fold_data: FoldData,
) -> None:
    del bundle
    covariate_digest = _sha256_text("|".join(fold_data.covariate_columns))

    mlflow.log_params(
        {
            "action": config.action,
            "mode": config.mode,
            "training_input_mode": config.training_input_mode,
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
            "covariate_columns_count": len(fold_data.covariate_columns),
            "future_covariate_columns_count": len(fold_data.future_covariate_columns),
            "past_covariate_columns_count": len(fold_data.past_covariate_columns),
            "covariate_column_digest": covariate_digest,
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
            "run.training_input_mode": config.training_input_mode,
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
    mlflow.log_text(build_covariate_columns_preview(fold_data), "covariate_columns.csv")
