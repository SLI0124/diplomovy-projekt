from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
from adapters import ModelContext, build_model_adapter, resolve_model_family
from adapters.base import TrainingLossPoint
from checkpoints import (
    build_checkpoint_dir,
    build_missing_checkpoint_error,
    dataset_tag,
    resolve_checkpoint_status,
    write_checkpoint_manifest,
)
from config import (
    RuntimeConfig,
    mlflow_experiment_for_family,
    results_root,
    save_runtime_config,
    to_serializable_dict,
)
from dataset import DatasetBundle, FoldData, iter_test_origins, load_fold_data
from folds import FoldSpec, folds_for_action
from metrics import FoldMetrics, compute_metrics
from mlflow_logging import (
    build_dataset_columns_preview,
    ensure_mlflow,
    safe_log_dataset_context,
    safe_log_run_context,
    safe_log_run_params,
)


def _log(message: str) -> None:
    print(f"[dl] {message}")


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _save_training_losses(
    run_root: Path,
    adapter_slug: str,
    config: RuntimeConfig,
    fold: FoldSpec,
    rows: list[TrainingLossPoint],
) -> None:
    if not rows:
        return

    out_dir = run_root / "training_losses"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "model": [adapter_slug for _ in rows],
            "mode": [config.mode for _ in rows],
            "train_years": [fold.train_years_label for _ in rows],
            "test_year": [fold.test_year for _ in rows],
            "epoch": [row.epoch for row in rows],
            "loss": [row.loss for row in rows],
        }
    )
    out_path = out_dir / f"{adapter_slug}__test-{fold.test_year}.csv"
    df.to_csv(out_path, index=False)


def _save_prediction_rows(
    run_root: Path,
    adapter_slug: str,
    fold: FoldSpec,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return

    out_dir = run_root / "predictions"
    out_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(rows)
    out_path = out_dir / f"{adapter_slug}__test-{fold.test_year}.csv"
    df.to_csv(out_path, index=False)


def _evaluate_fold(
    adapter,
    fold_data: FoldData,
    config: RuntimeConfig,
    fold: FoldSpec,
) -> tuple[list[FoldMetrics], list[dict[str, Any]]]:
    test_df = fold_data.test_df
    test_series = test_df[config.target_col].to_numpy(dtype=np.float32)

    seg_true: dict[str, list[np.ndarray]] = {
        "all": [],
        "pre_conflict": [],
        "post_conflict": [],
    }
    seg_pred: dict[str, list[np.ndarray]] = {
        "all": [],
        "pre_conflict": [],
        "post_conflict": [],
    }
    seg_windows: dict[str, int] = {"all": 0, "pre_conflict": 0, "post_conflict": 0}
    prediction_rows: list[dict[str, Any]] = []

    origins = list(
        iter_test_origins(
            test_df,
            prediction_length=config.prediction_length,
            stride=config.window_stride,
        )
    )
    if config.max_origins_per_year is not None:
        origins = origins[: config.max_origins_per_year]

    conflict_date = pd.Timestamp(config.conflict_date)

    def row_timestamp(index: int) -> pd.Timestamp:
        # Adapters require a datetime start value, but DL does not build or index by timestamps.
        return pd.Timestamp(year=fold.test_year, month=1, day=1) + pd.Timedelta(
            hours=index
        )

    for i in origins:
        origin_ts = row_timestamp(i)
        context = np.concatenate((fold_data.train_series, test_series[:i]))
        y_true = test_series[i : i + config.prediction_length]
        context_covariates = None
        future_covariates = None

        if config.training_input_mode == "covariate":
            if fold_data.train_covariates is None or fold_data.test_covariates is None:
                raise ValueError(
                    "Covariate mode is enabled but context covariate arrays are missing in fold data."
                )

            context_covariates = np.concatenate(
                (fold_data.train_covariates, fold_data.test_covariates[:i]), axis=0
            )
            if fold_data.test_future_covariates is not None:
                future_covariates = fold_data.test_future_covariates[
                    i : i + config.prediction_length
                ]

        context_used = context[-config.context_length :]
        context_start = origin_ts - pd.Timedelta(hours=len(context_used))
        y_pred = adapter.forecast(
            context=context,
            context_start=context_start,
            context_covariates=context_covariates,
            future_covariates=future_covariates,
        ).y_pred

        if len(y_pred) != len(y_true):
            raise ValueError(
                f"Bad forecast length for {adapter.slug}: got {len(y_pred)} expected {len(y_true)}"
            )

        segment = "post_conflict" if origin_ts >= conflict_date else "pre_conflict"

        for horizon_index, (true_value, pred_value) in enumerate(
            zip(y_true, y_pred, strict=True)
        ):
            target_ts = row_timestamp(i + horizon_index)
            prediction_rows.append(
                {
                    "model": adapter.slug,
                    "mode": config.mode,
                    "train_years": fold.train_years_label,
                    "test_year": fold.test_year,
                    "segment": segment,
                    "origin_index": int(i),
                    "origin_timestamp": origin_ts.isoformat(),
                    "context_start": context_start.isoformat(),
                    "horizon_index": int(horizon_index),
                    "target_timestamp": target_ts.isoformat(),
                    "y_true": float(true_value),
                    "y_pred": float(pred_value),
                }
            )

        seg_true["all"].append(y_true)
        seg_pred["all"].append(y_pred)
        seg_windows["all"] += 1

        seg_true[segment].append(y_true)
        seg_pred[segment].append(y_pred)
        seg_windows[segment] += 1

    rows: list[FoldMetrics] = []
    for segment in ["all", "pre_conflict", "post_conflict"]:
        if seg_windows[segment] == 0:
            continue

        y_true_all = np.concatenate(seg_true[segment])
        y_pred_all = np.concatenate(seg_pred[segment])
        metrics = compute_metrics(y_true_all, y_pred_all)
        rows.append(
            FoldMetrics(
                model=adapter.slug,
                mode=config.mode,
                train_years=fold.train_years_label,
                test_year=fold.test_year,
                segment=segment,
                n_windows=seg_windows[segment],
                n_points=int(y_true_all.size),
                **metrics,
            )
        )

    return rows, prediction_rows


def _log_rows_to_mlflow(rows: list[FoldMetrics]) -> None:
    for row in rows:
        prefix = f"{row.segment}."
        mlflow.log_metric(prefix + "smape", row.smape)
        mlflow.log_metric(prefix + "mape", row.mape)
        mlflow.log_metric(prefix + "mae", row.mae)
        mlflow.log_metric(prefix + "mse", row.mse)
        mlflow.log_metric(prefix + "r2", row.r2)
        mlflow.log_metric(prefix + "n_windows", row.n_windows)
        mlflow.log_metric(prefix + "n_points", row.n_points)


def run(config: RuntimeConfig, bundle: DatasetBundle) -> pd.DataFrame:
    run_root = results_root() / _run_id()
    run_root.mkdir(parents=True, exist_ok=True)

    save_runtime_config(config, run_root / "runtime_config.json")
    if bundle.run_params is not None:
        (run_root / "preprocessing_run_params.json").write_text(
            json.dumps(bundle.run_params, indent=2), encoding="utf-8"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _log(f"Using device: {device}")

    model_ctx = ModelContext(
        prediction_length=config.prediction_length,
        context_length=config.context_length,
        num_samples=config.num_samples,
    )

    folds = folds_for_action(config)
    fold_data_by_test_year = {
        fold.test_year: load_fold_data(bundle=bundle, config=config, fold=fold)
        for fold in folds
    }
    results: list[FoldMetrics] = []
    successful_runs = 0
    failed_runs: list[dict[str, object]] = []
    checkpoint_outcomes: dict[str, dict[str, int]] = defaultdict(
        lambda: {"trained": 0, "reused": 0, "failed": 0}
    )

    current_dataset_tag = dataset_tag(bundle)

    total_folds = len(folds)
    for model_name in config.models:
        for fold_index, fold in enumerate(folds, start=1):
            fold_data = fold_data_by_test_year[fold.test_year]
            _log(
                f"Model={model_name} mode={config.mode} action={config.action} "
                f"input_mode={config.training_input_mode} "
                f"fold={fold_index}/{total_folds} "
                f"train=2013-{fold.train_end_year} test={fold.test_year}"
            )

            adapter = build_model_adapter(
                model_name=model_name,
                model_ctx=model_ctx,
                device=device,
            )
            model_family = resolve_model_family(model_name)
            experiment_name = mlflow_experiment_for_family(model_family)
            ensure_mlflow(experiment_name)

            run_name = (
                f"{adapter.slug}__{config.mode}__{config.action}__"
                f"train-2013-{fold.train_end_year}__test-{fold.test_year}"
            )

            checkpoint_dir: Path | None = None
            if config.mode == "finetuned":
                checkpoint_dir = build_checkpoint_dir(
                    config=config,
                    model_slug=adapter.slug,
                    fold=fold,
                    current_dataset_tag=current_dataset_tag,
                )

            should_evaluate = config.action != "train" or config.eval_after_train
            checkpoint_status: str | None = None
            checkpoint_reason: str | None = None
            if config.mode == "finetuned" and checkpoint_dir is not None:
                checkpoint_status, checkpoint_reason = resolve_checkpoint_status(
                    checkpoint_dir=checkpoint_dir,
                    config=config,
                    fold=fold,
                    model_slug=adapter.slug,
                    current_dataset_tag=current_dataset_tag,
                    covariate_columns=fold_data.covariate_columns,
                    future_covariate_columns=fold_data.future_covariate_columns,
                    past_covariate_columns=fold_data.past_covariate_columns,
                )

                if (
                    config.action == "train"
                    and checkpoint_status == "compatible_exists"
                    and not should_evaluate
                ):
                    _log(
                        f"Skipping finetune for {adapter.slug} test={fold.test_year}; "
                        "compatible checkpoint already exists."
                    )
                    checkpoint_outcomes[adapter.slug]["reused"] += 1
                    successful_runs += 1
                    continue

            with mlflow.start_run(run_name=run_name):
                safe_log_run_params(
                    config=config,
                    bundle=bundle,
                    fold=fold,
                    dataset_tag=current_dataset_tag,
                    fold_data=fold_data,
                )
                safe_log_dataset_context(
                    bundle=bundle,
                    fold_data=fold_data,
                    config=config,
                )
                safe_log_run_context(
                    config=config,
                    bundle=bundle,
                    fold_data=fold_data,
                    fold=fold,
                    adapter=adapter,
                    requested_model_name=model_name,
                    model_family=model_family,
                    device=device,
                    run_root=run_root,
                    dataset_tag=current_dataset_tag,
                    checkpoint_dir=checkpoint_dir,
                )

                mlflow.log_text(
                    build_dataset_columns_preview(fold_data=fold_data),
                    "dataset_columns_preview.csv",
                )

                try:
                    if config.mode == "one-shot":
                        adapter.load_pretrained()

                    elif config.mode == "finetuned":
                        ckpt_dir = checkpoint_dir
                        if ckpt_dir is None:
                            raise RuntimeError(
                                "Internal error: checkpoint_dir was not prepared for finetuned mode."
                            )

                        if config.action == "train":
                            if checkpoint_status == "compatible_exists":
                                _log(
                                    f"Skipping finetune for {adapter.slug} test={fold.test_year}; "
                                    "compatible checkpoint already exists."
                                )
                                checkpoint_outcomes[adapter.slug]["reused"] += 1
                                mlflow.log_param("checkpoint_reused", True)
                                mlflow.log_param("checkpoint_dir", str(ckpt_dir))
                                if should_evaluate:
                                    adapter.load_finetuned(ckpt_dir)
                            elif checkpoint_status == "incompatible_manifest":
                                raise ValueError(
                                    "Existing checkpoint is incompatible with current runtime config. "
                                    f"checkpoint_dir={ckpt_dir}\n{checkpoint_reason}"
                                )
                            else:
                                if not adapter.supports_finetune:
                                    raise NotImplementedError(
                                        f"Model '{adapter.slug}' does not support finetuning in this script yet."
                                    )
                                if (
                                    config.train_loss is not None
                                    or config.train_optimizer is not None
                                ) and model_family != "custom":
                                    raise ValueError(
                                        "--train-loss/--train-optimizer are only supported for custom models. "
                                        f"Model '{adapter.slug}' is in family '{model_family}'."
                                    )

                                train_series = fold_data.train_series
                                train_covariates = None
                                train_future_covariates = None
                                if config.training_input_mode == "covariate":
                                    train_covariates = fold_data.train_covariates
                                    train_future_covariates = (
                                        fold_data.train_future_covariates
                                    )

                                train_params: dict[str, int] = {
                                    "train_series_points": int(train_series.shape[0]),
                                }
                                if train_covariates is not None:
                                    train_params["train_covariate_columns"] = int(
                                        train_covariates.shape[1]
                                    )
                                if train_future_covariates is not None:
                                    train_params["train_future_covariate_columns"] = (
                                        int(train_future_covariates.shape[1])
                                    )
                                mlflow.log_params(train_params)

                                training_history = adapter.finetune(
                                    train_series=train_series,
                                    train_epochs=config.train_epochs,
                                    train_batch_size=config.train_batch_size,
                                    train_steps_per_epoch=config.train_steps_per_epoch,
                                    train_lr=config.train_lr,
                                    train_weight_decay=config.train_weight_decay,
                                    checkpoint_selection=config.checkpoint_selection,
                                    train_loss=config.train_loss,
                                    train_optimizer=config.train_optimizer,
                                    artifact_dir=ckpt_dir,
                                    train_covariates=train_covariates,
                                    train_future_covariates=train_future_covariates,
                                )
                                _save_training_losses(
                                    run_root=run_root,
                                    adapter_slug=adapter.slug,
                                    config=config,
                                    fold=fold,
                                    rows=training_history,
                                )
                                adapter.save_finetuned(ckpt_dir)
                                mlflow.log_param("checkpoint_dir", str(ckpt_dir))

                                metadata = {
                                    "model": adapter.slug,
                                    "mode": "finetuned",
                                    "train_years": fold.train_years_label,
                                    "test_year": fold.test_year,
                                    "dataset_tag": current_dataset_tag,
                                    "runtime_config": to_serializable_dict(config),
                                }
                                (ckpt_dir / "metadata.json").write_text(
                                    json.dumps(metadata, indent=2), encoding="utf-8"
                                )
                                write_checkpoint_manifest(
                                    checkpoint_dir=ckpt_dir,
                                    config=config,
                                    fold=fold,
                                    model_slug=adapter.slug,
                                    current_dataset_tag=current_dataset_tag,
                                    covariate_columns=fold_data.covariate_columns,
                                    future_covariate_columns=fold_data.future_covariate_columns,
                                    past_covariate_columns=fold_data.past_covariate_columns,
                                )
                                checkpoint_outcomes[adapter.slug]["trained"] += 1

                        else:
                            if checkpoint_status == "missing":
                                raise FileNotFoundError(
                                    build_missing_checkpoint_error(
                                        model_slug=adapter.slug,
                                        fold=fold,
                                        ckpt_dir=ckpt_dir,
                                        training_input_mode=config.training_input_mode,
                                    )
                                )
                            if checkpoint_status == "incompatible_manifest":
                                raise ValueError(
                                    "Checkpoint exists but is incompatible with the current runtime config. "
                                    f"checkpoint_dir={ckpt_dir}\n{checkpoint_reason}"
                                )
                            adapter.load_finetuned(ckpt_dir)
                            checkpoint_outcomes[adapter.slug]["reused"] += 1
                            mlflow.log_param("checkpoint_dir", str(ckpt_dir))

                    if should_evaluate:
                        fold_rows, prediction_rows = _evaluate_fold(
                            adapter=adapter,
                            fold_data=fold_data,
                            config=config,
                            fold=fold,
                        )
                        _save_prediction_rows(
                            run_root=run_root,
                            adapter_slug=adapter.slug,
                            fold=fold,
                            rows=prediction_rows,
                        )
                        _log_rows_to_mlflow(fold_rows)
                        results.extend(fold_rows)
                    else:
                        _log(
                            f"Skipping evaluation after training for fold test={fold.test_year} "
                            "(use --eval-after-train to enable)."
                        )

                    successful_runs += 1
                    mlflow.log_param("status", "ok")

                except Exception as exc:
                    _log(
                        f"FAILED {adapter.slug} year={fold.test_year}: {type(exc).__name__}: {exc}"
                    )
                    mlflow.log_param("status", "failed")
                    mlflow.log_param("error_type", type(exc).__name__)
                    mlflow.log_param("error_message", str(exc)[:500])
                    failed_runs.append(
                        {
                            "model": adapter.slug,
                            "test_year": fold.test_year,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )
                    checkpoint_outcomes[adapter.slug]["failed"] += 1

    if config.mode == "finetuned":
        _log("Checkpoint summary by model:")
        for model_slug in sorted(checkpoint_outcomes):
            counts = checkpoint_outcomes[model_slug]
            _log(
                f"  {model_slug}: trained={counts['trained']} reused={counts['reused']} failed={counts['failed']}"
            )

    if successful_runs == 0:
        raise RuntimeError(
            "All model-fold executions failed. Check logs and MLflow runs."
        )

    if failed_runs:
        details = "\n".join(
            (
                f"- model={entry['model']} test_year={entry['test_year']} "
                f"{entry['error_type']}: {entry['error_message']}"
            )
            for entry in failed_runs
        )
        raise RuntimeError(
            "Run finished with partial failures. "
            f"successful={successful_runs} failed={len(failed_runs)}. "
            "Refusing to produce partial aggregate outputs.\n"
            f"{details}"
        )

    if not results:
        _log("Train-only run completed. No evaluation metrics were produced.")
        results_df = pd.DataFrame(
            columns=[
                "model",
                "mode",
                "train_years",
                "test_year",
                "segment",
                "n_windows",
                "n_points",
                "smape",
                "mape",
                "mae",
                "mse",
                "r2",
            ]
        )
        results_df.to_csv(run_root / "results.csv", index=False)
        summary_df = pd.DataFrame(
            columns=[
                "model",
                "mode",
                "smape",
                "mape",
                "mae",
                "mse",
                "r2",
                "n_windows",
                "n_points",
            ]
        )
        summary_df.to_csv(run_root / "summary.csv", index=False)
        _log(f"Run artifacts saved to: {run_root}")
        return results_df

    results_df = pd.DataFrame([asdict(row) for row in results])
    results_df = results_df.sort_values(
        ["model", "mode", "test_year", "segment"]
    ).reset_index(drop=True)
    results_df.to_csv(run_root / "results.csv", index=False)

    summary = (
        results_df[results_df["segment"] == "all"]
        .groupby(["model", "mode"], as_index=False)
        .agg(
            smape=("smape", "mean"),
            mape=("mape", "mean"),
            mae=("mae", "mean"),
            mse=("mse", "mean"),
            r2=("r2", "mean"),
            n_windows=("n_windows", "sum"),
            n_points=("n_points", "sum"),
        )
        .sort_values(["smape", "mae"])
    )
    summary.to_csv(run_root / "summary.csv", index=False)

    _log(f"Run artifacts saved to: {run_root}")
    return results_df
