from __future__ import annotations

import hashlib
from pathlib import Path

import mlflow
import pandas as pd
from config import (
    RuntimeConfig,
    mlflow_experiment,
    mlflow_uri,
    models_root,
    results_root,
)
from dataset import DatasetBundle
from folds import FoldSpec
from metrics import FoldMetrics


def ensure_mlflow() -> None:
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
        "feature_count": int(len(bundle.feature_columns)),
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


def safe_log_run_params(
    config: RuntimeConfig,
    bundle: DatasetBundle,
    fold: FoldSpec,
    dataset_tag: str,
    model_name: str,
) -> None:
    mlflow.log_params(
        {
            "action": config.action,
            "strategy": config.strategy,
            "model": model_name,
            "eval_after_train": config.eval_after_train,
            "test_year": fold.test_year,
            "train_end_year": fold.train_end_year,
            "target_col": config.target_col,
            "hourly_min_train_samples": config.hourly_min_train_samples,
            "hourly_min_test_samples": config.hourly_min_test_samples,
            "drop_high_missing_threshold": config.drop_high_missing_threshold,
            "variant_stem": config.variant_stem or "",
            "dataset_tag": dataset_tag,
            "feature_count": len(bundle.feature_columns),
            "seed": config.seed,
            "dt_max_depth": config.dt_max_depth,
            "dt_min_samples_split": config.dt_min_samples_split,
            "rf_n_estimators": config.rf_n_estimators,
            "rf_max_depth": config.rf_max_depth,
            "gb_n_estimators": config.gb_n_estimators,
            "gb_learning_rate": config.gb_learning_rate,
            "gb_max_depth": config.gb_max_depth,
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
    run_root: Path,
    checkpoint_dir: Path,
    model_name: str,
    dataset_tag: str,
) -> None:
    mlflow.set_tags(
        {
            "run.kind": config.action,
            "run.model": model_name,
            "run.strategy": config.strategy,
            "run.dataset.tag": dataset_tag,
            "run.fold": f"train_2013-{fold.train_end_year}__test-{fold.test_year}",
        }
    )
    context_payload = {
        "model": model_name,
        "strategy": config.strategy,
        "fold": {
            "train_years": fold.train_years_label,
            "train_end_year": fold.train_end_year,
            "test_year": fold.test_year,
        },
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
            "checkpoint_dir": str(checkpoint_dir),
        },
    }
    mlflow.log_dict(context_payload, "run_context.json")


def log_fold_metrics(rows: list[FoldMetrics]) -> None:
    for row in rows:
        prefix = f"{row.segment}."
        mlflow.log_metric(prefix + "smape", row.smape)
        mlflow.log_metric(prefix + "mape", row.mape)
        mlflow.log_metric(prefix + "mae", row.mae)
        mlflow.log_metric(prefix + "mse", row.mse)
        mlflow.log_metric(prefix + "r2", row.r2)
        mlflow.log_metric(prefix + "n_hours", row.n_hours)
        mlflow.log_metric(prefix + "n_points", row.n_points)
