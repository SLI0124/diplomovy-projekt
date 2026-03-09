from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch
from checkpoints import (
    build_checkpoint_dir,
    build_missing_checkpoint_error,
    dataset_tag,
    validate_checkpoint_manifest,
    write_checkpoint_manifest,
)
from config import RuntimeConfig, save_runtime_config, to_serializable_dict
from dataset import DatasetBundle, iter_test_origins, make_test_df, make_train_target
from folds import FoldSpec, folds_for_action
from metrics import FoldMetrics, compute_metrics
from mlflow_logging import (
    build_dataset_columns_preview,
    ensure_mlflow,
    safe_log_dataset_context,
    safe_log_run_context,
    safe_log_run_params,
)
from models import ModelContext, build_model_adapter


def _log(message: str) -> None:
    print(f"[dl] {message}")


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _evaluate_fold(
    adapter,
    full_df: pd.DataFrame,
    config: RuntimeConfig,
    fold: FoldSpec,
) -> list[FoldMetrics]:
    test_df = make_test_df(full_df, fold.test_year)
    if test_df.empty:
        raise ValueError(f"No rows for test year {fold.test_year}")

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

    for i in origins:
        origin_ts = pd.Timestamp(test_df["timestamp"].iloc[i])

        context = full_df[full_df["timestamp"] < origin_ts][config.target_col].to_numpy(
            dtype=np.float32
        )

        y_true = (
            test_df[config.target_col]
            .iloc[i : i + config.prediction_length]
            .to_numpy(dtype=np.float32)
        )

        context_used = context[-config.context_length :]
        context_start = origin_ts - pd.Timedelta(hours=len(context_used))
        y_pred = adapter.forecast(context=context, context_start=context_start).y_pred

        if len(y_pred) != len(y_true):
            raise ValueError(
                f"Bad forecast length for {adapter.slug}: got {len(y_pred)} expected {len(y_true)}"
            )

        segment = "post_conflict" if origin_ts >= conflict_date else "pre_conflict"

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

    return rows


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
    ensure_mlflow(config)

    run_root = config.results_root / _run_id()
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
        lag_llama_num_parallel_samples=config.lag_llama_num_parallel_samples,
    )

    folds = folds_for_action(config)
    results: list[FoldMetrics] = []
    successful_runs = 0

    current_dataset_tag = dataset_tag(bundle)

    total_folds = len(folds)
    for model_name in config.models:
        for fold_index, fold in enumerate(folds, start=1):
            _log(
                f"Model={model_name} mode={config.mode} action={config.action} "
                f"fold={fold_index}/{total_folds} "
                f"train=2013-{fold.train_end_year} test={fold.test_year}"
            )

            adapter = build_model_adapter(
                model_name=model_name,
                model_ctx=model_ctx,
                device=device,
            )

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

            with mlflow.start_run(run_name=run_name):
                safe_log_run_params(
                    config=config,
                    bundle=bundle,
                    fold=fold,
                    dataset_tag=current_dataset_tag,
                )
                safe_log_dataset_context(bundle=bundle, config=config)
                safe_log_run_context(
                    config=config,
                    bundle=bundle,
                    fold=fold,
                    adapter=adapter,
                    requested_model_name=model_name,
                    device=device,
                    run_root=run_root,
                    dataset_tag=current_dataset_tag,
                    checkpoint_dir=checkpoint_dir,
                )

                mlflow.log_text(
                    build_dataset_columns_preview(bundle),
                    "dataset_columns_preview.csv",
                )

                try:
                    should_evaluate = (
                        config.action != "train" or config.eval_after_train
                    )

                    if config.mode == "one-shot":
                        adapter.load_pretrained()

                    elif config.mode == "finetuned":
                        ckpt_dir = checkpoint_dir
                        if ckpt_dir is None:
                            raise RuntimeError(
                                "Internal error: checkpoint_dir was not prepared for finetuned mode."
                            )

                        if config.action == "train":
                            if not adapter.supports_finetune:
                                raise NotImplementedError(
                                    f"Model '{adapter.slug}' does not support finetuning in this script yet."
                                )

                            train_series = make_train_target(
                                full_df=bundle.dataframe,
                                target_col=config.target_col,
                                train_end_year=fold.train_end_year,
                            )
                            mlflow.log_params(
                                {
                                    "train_series_points": int(train_series.shape[0]),
                                }
                            )
                            adapter.finetune(
                                train_series=train_series,
                                train_epochs=config.train_epochs,
                                train_batch_size=config.train_batch_size,
                                train_steps_per_epoch=config.train_steps_per_epoch,
                                train_lr=config.train_lr,
                                train_weight_decay=config.train_weight_decay,
                                artifact_dir=ckpt_dir,
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
                            )

                        else:
                            if not ckpt_dir.exists():
                                raise FileNotFoundError(
                                    build_missing_checkpoint_error(
                                        model_slug=adapter.slug,
                                        fold=fold,
                                        ckpt_dir=ckpt_dir,
                                    )
                                )
                            validate_checkpoint_manifest(
                                checkpoint_dir=ckpt_dir,
                                config=config,
                                fold=fold,
                                model_slug=adapter.slug,
                                current_dataset_tag=current_dataset_tag,
                            )
                            adapter.load_finetuned(ckpt_dir)
                            mlflow.log_param("checkpoint_dir", str(ckpt_dir))

                    if should_evaluate:
                        fold_rows = _evaluate_fold(
                            adapter=adapter,
                            full_df=bundle.dataframe,
                            config=config,
                            fold=fold,
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

    if successful_runs == 0:
        raise RuntimeError(
            "All model-fold executions failed. Check logs and MLflow runs."
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
