from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime
from pathlib import Path

import joblib
import mlflow
import numpy as np
import pandas as pd
from checkpoints import (
    build_checkpoint_dir,
    build_missing_checkpoint_error,
    dataset_tag,
    resolve_checkpoint_status,
    write_checkpoint_manifest,
)
from config import (
    RuntimeConfig,
    results_root,
    save_runtime_config,
    to_serializable_dict,
)
from dataset import DatasetBundle, make_train_test_df
from folds import FoldSpec, folds_for_action
from metrics import FoldMetrics, HourMetrics, compute_metrics
from mlflow_logging import (
    ensure_mlflow,
    log_fold_metrics,
    safe_log_dataset_context,
    safe_log_run_context,
    safe_log_run_params,
)
from models import build_pipeline


def _log(message: str) -> None:
    print(f"[ml] {message}")


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _hour_feature_columns(feature_columns: tuple[str, ...]) -> tuple[str, ...]:
    return tuple(col for col in feature_columns if col != "hour")


def _fit_pipeline_with_checkpoint_selection(
    *,
    model_name: str,
    config: RuntimeConfig,
    train_hour: pd.DataFrame,
    hour_features: tuple[str, ...],
    target_col: str,
    warned_models: set[str],
):
    strategy = config.checkpoint_selection
    if strategy not in {"best-train-loss", "last-fit"}:
        raise ValueError(
            f"Unsupported checkpoint selection '{strategy}'. Supported: best-train-loss, last-fit"
        )

    x_train = train_hour[list(hour_features)]
    y_train = train_hour[target_col]

    pipeline = build_pipeline(model_name=model_name, config=config)
    pipeline.fit(x_train, y_train)

    if strategy == "last-fit":
        return pipeline

    if model_name != "gradient-boosting":
        if model_name not in warned_models:
            _log(
                f"model={model_name}: checkpoint-selection=best-train-loss falls back to last-fit "
                "because this estimator does not expose staged training loss."
            )
            warned_models.add(model_name)
        return pipeline

    model = pipeline.named_steps["model"]
    x_transformed = pipeline[:-1].transform(x_train)
    y_true = y_train.to_numpy(dtype=float)

    best_stage = 1
    best_loss = float("inf")
    final_stage = 0

    for stage, stage_pred in enumerate(model.staged_predict(x_transformed), start=1):
        final_stage = stage
        stage_pred_arr = np.asarray(stage_pred, dtype=float)
        mse = float(np.mean((y_true - stage_pred_arr) ** 2))
        if mse < best_loss:
            best_loss = mse
            best_stage = stage

    if final_stage <= 0:
        return pipeline

    if best_stage >= final_stage:
        return pipeline

    _log(
        f"model=gradient-boosting: selected best-train-loss stage n_estimators={best_stage} "
        f"(final={final_stage})."
    )

    best_pipeline = build_pipeline(model_name=model_name, config=config)
    best_pipeline.named_steps["model"].set_params(n_estimators=best_stage)
    best_pipeline.fit(x_train, y_train)
    return best_pipeline


def _train_hourly_models(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    *,
    feature_columns: tuple[str, ...],
    target_col: str,
    model_name: str,
    config: RuntimeConfig,
    checkpoint_dir: Path,
) -> tuple[list[int], pd.DataFrame]:
    trained_hours: list[int] = []
    skipped_rows: list[dict[str, object]] = []
    hour_features = _hour_feature_columns(feature_columns)
    warned_models: set[str] = set()

    for hour in range(24):
        train_hour = train_df[train_df["hour"] == hour]
        test_hour = test_df[test_df["hour"] == hour]

        if len(train_hour) < config.hourly_min_train_samples:
            skipped_rows.append(
                {
                    "hour": hour,
                    "status": "skipped_insufficient_train_samples",
                    "train_samples": len(train_hour),
                    "test_samples": len(test_hour),
                }
            )
            continue

        if len(test_hour) < config.hourly_min_test_samples:
            skipped_rows.append(
                {
                    "hour": hour,
                    "status": "skipped_insufficient_test_samples",
                    "train_samples": len(train_hour),
                    "test_samples": len(test_hour),
                }
            )
            continue

        pipeline = _fit_pipeline_with_checkpoint_selection(
            model_name=model_name,
            config=config,
            train_hour=train_hour,
            hour_features=hour_features,
            target_col=target_col,
            warned_models=warned_models,
        )

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        joblib.dump(pipeline, checkpoint_dir / f"hour_{hour:02d}.joblib")
        trained_hours.append(hour)

    if not trained_hours:
        raise RuntimeError(
            f"No hourly models trained for model={model_name}. "
            "Check hourly thresholds or data coverage."
        )

    skipped_df = pd.DataFrame(skipped_rows)
    return trained_hours, skipped_df


def _load_hourly_models(checkpoint_dir: Path) -> dict[int, object]:
    loaded: dict[int, object] = {}
    for hour in range(24):
        model_path = checkpoint_dir / f"hour_{hour:02d}.joblib"
        if model_path.exists():
            loaded[hour] = joblib.load(model_path)
    return loaded


def _evaluate_hourly_models(
    *,
    test_df: pd.DataFrame,
    feature_columns: tuple[str, ...],
    target_col: str,
    hourly_models: dict[int, object],
    fold: FoldSpec,
    model_name: str,
    config: RuntimeConfig,
) -> tuple[list[HourMetrics], list[FoldMetrics]]:
    hour_rows: list[HourMetrics] = []
    all_true: list[np.ndarray] = []
    all_pred: list[np.ndarray] = []
    all_ts: list[np.ndarray] = []
    hour_features = _hour_feature_columns(feature_columns)

    for hour in sorted(hourly_models):
        test_hour = test_df[test_df["hour"] == hour].copy()
        if test_hour.empty:
            continue

        model = hourly_models[hour]
        y_true = test_hour[target_col].to_numpy(dtype=float)
        y_pred = model.predict(test_hour[list(hour_features)]) # type: ignore
        y_pred = np.asarray(y_pred, dtype=float)

        core = compute_metrics(y_true, y_pred)
        hour_rows.append(
            HourMetrics(
                model=model_name,
                strategy=config.strategy,
                train_years=fold.train_years_label,
                test_year=fold.test_year,
                hour=hour,
                train_samples=0,
                test_samples=len(test_hour),
                **core,
            )
        )

        all_true.append(y_true)
        all_pred.append(y_pred)
        all_ts.append(test_hour["timestamp"].to_numpy())

    if not all_true:
        raise RuntimeError(
            f"No predictions available for evaluation in fold test_year={fold.test_year}."
        )

    y_true_all = np.concatenate(all_true)
    y_pred_all = np.concatenate(all_pred)
    ts_all = np.concatenate(all_ts)
    conflict = pd.Timestamp(config.conflict_date)
    ts_series = pd.to_datetime(ts_all)

    fold_rows: list[FoldMetrics] = []
    for segment, mask in (
        ("all", np.ones(len(ts_series), dtype=bool)),
        ("pre_conflict", ts_series < conflict),
        ("post_conflict", ts_series >= conflict),
    ):
        if not np.any(mask):
            continue
        segment_metrics = compute_metrics(y_true_all[mask], y_pred_all[mask])
        fold_rows.append(
            FoldMetrics(
                model=model_name,
                strategy=config.strategy,
                train_years=fold.train_years_label,
                test_year=fold.test_year,
                segment=segment,
                n_hours=len(hour_rows),
                n_points=int(np.sum(mask)),
                **segment_metrics,
            )
        )

    return hour_rows, fold_rows


def run(
    config: RuntimeConfig, bundle: DatasetBundle
) -> tuple[pd.DataFrame, pd.DataFrame]:
    run_root = results_root() / _run_id()
    run_root.mkdir(parents=True, exist_ok=True)

    save_runtime_config(config, run_root / "runtime_config.json")
    if bundle.run_params is not None:
        (run_root / "preprocessing_run_params.json").write_text(
            json.dumps(bundle.run_params, indent=2), encoding="utf-8"
        )

    feature_meta = {
        "feature_columns": list(bundle.feature_columns),
        "dropped_high_missing_columns": list(bundle.dropped_high_missing_columns),
        "excluded_leakage_columns": list(bundle.excluded_leakage_columns),
    }
    (run_root / "feature_metadata.json").write_text(
        json.dumps(feature_meta, indent=2), encoding="utf-8"
    )

    folds = folds_for_action(config)
    all_hour_rows: list[HourMetrics] = []
    all_fold_rows: list[FoldMetrics] = []
    failed_runs: list[dict[str, object]] = []
    successful_runs = 0
    outcomes: dict[str, dict[str, int]] = defaultdict(
        lambda: {"trained": 0, "reused": 0, "failed": 0}
    )

    current_dataset_tag = dataset_tag(bundle)
    ensure_mlflow()

    for model_name in config.models:
        for fold in folds:
            _log(
                f"model={model_name} action={config.action} fold=train_2013-{fold.train_end_year}__test-{fold.test_year}"
            )

            train_df, test_df = make_train_test_df(bundle.dataframe, fold.test_year)
            if train_df.empty:
                raise ValueError(
                    f"No train rows available before test year {fold.test_year}."
                )
            if test_df.empty:
                raise ValueError(
                    f"No test rows available for test year {fold.test_year}."
                )

            checkpoint_dir = build_checkpoint_dir(
                config=config,
                model_name=model_name,
                fold=fold,
                current_dataset_tag=current_dataset_tag,
                feature_columns=bundle.feature_columns,
            )

            run_name = (
                f"{model_name}__{config.strategy}__{config.action}__"
                f"train-2013-{fold.train_end_year}__test-{fold.test_year}"
            )

            should_evaluate = config.action != "train" or config.eval_after_train
            checkpoint_status, checkpoint_reason = resolve_checkpoint_status(
                checkpoint_dir=checkpoint_dir,
                config=config,
                fold=fold,
                model_name=model_name,
                current_dataset_tag=current_dataset_tag,
                feature_columns=bundle.feature_columns,
            )

            if config.force_retrain:
                checkpoint_status = "missing"
                checkpoint_reason = None

            with mlflow.start_run(run_name=run_name):
                safe_log_run_params(
                    config=config,
                    bundle=bundle,
                    fold=fold,
                    dataset_tag=current_dataset_tag,
                    model_name=model_name,
                )
                safe_log_dataset_context(bundle=bundle, config=config)
                safe_log_run_context(
                    config=config,
                    bundle=bundle,
                    fold=fold,
                    run_root=run_root,
                    checkpoint_dir=checkpoint_dir,
                    model_name=model_name,
                    dataset_tag=current_dataset_tag,
                )

                try:
                    hourly_models: dict[int, object]
                    skipped_df = pd.DataFrame(
                        columns=["hour", "status", "train_samples", "test_samples"]
                    )

                    if config.action == "train":
                        if checkpoint_status == "compatible_exists":
                            outcomes[model_name]["reused"] += 1
                            mlflow.log_param("checkpoint_reused", True)
                            hourly_models = _load_hourly_models(checkpoint_dir)
                            if not hourly_models:
                                raise RuntimeError(
                                    "Compatible checkpoint manifest exists, but no hourly model files found."
                                )
                        elif checkpoint_status == "incompatible_manifest":
                            raise ValueError(
                                "Existing checkpoint is incompatible with current runtime config. "
                                f"checkpoint_dir={checkpoint_dir}\n{checkpoint_reason}"
                            )
                        else:
                            trained_hours, skipped_df = _train_hourly_models(
                                train_df=train_df,
                                test_df=test_df,
                                feature_columns=bundle.feature_columns,
                                target_col=config.target_col,
                                model_name=model_name,
                                config=config,
                                checkpoint_dir=checkpoint_dir,
                            )
                            write_checkpoint_manifest(
                                checkpoint_dir=checkpoint_dir,
                                config=config,
                                fold=fold,
                                model_name=model_name,
                                current_dataset_tag=current_dataset_tag,
                                feature_columns=bundle.feature_columns,
                                trained_hours=trained_hours,
                            )
                            outcomes[model_name]["trained"] += 1
                            hourly_models = _load_hourly_models(checkpoint_dir)
                            if skipped_df.empty:
                                skipped_df = pd.DataFrame(
                                    [
                                        {
                                            "hour": -1,
                                            "status": "none_skipped",
                                            "train_samples": 0,
                                            "test_samples": 0,
                                        }
                                    ]
                                )

                    else:
                        if checkpoint_status == "missing":
                            raise FileNotFoundError(
                                build_missing_checkpoint_error(
                                    fold=fold,
                                    model_name=model_name,
                                    ckpt_dir=checkpoint_dir,
                                )
                            )
                        if checkpoint_status == "incompatible_manifest":
                            raise ValueError(
                                "Checkpoint exists but is incompatible with current runtime config. "
                                f"checkpoint_dir={checkpoint_dir}\n{checkpoint_reason}"
                            )
                        hourly_models = _load_hourly_models(checkpoint_dir)
                        if not hourly_models:
                            raise RuntimeError(
                                f"No hourly models found in checkpoint_dir={checkpoint_dir}"
                            )
                        outcomes[model_name]["reused"] += 1

                    if should_evaluate:
                        hour_rows, fold_rows = _evaluate_hourly_models(
                            test_df=test_df,
                            feature_columns=bundle.feature_columns,
                            target_col=config.target_col,
                            hourly_models=hourly_models,
                            fold=fold,
                            model_name=model_name,
                            config=config,
                        )
                        all_hour_rows.extend(hour_rows)
                        all_fold_rows.extend(fold_rows)
                        log_fold_metrics(fold_rows)
                    else:
                        _log(
                            f"Skipping evaluation after training for fold test={fold.test_year} "
                            "(use --eval-after-train to enable)."
                        )

                    skipped_artifact = run_root / (
                        f"skipped_hours__{model_name}__train_2013-{fold.train_end_year}__test-{fold.test_year}.csv"
                    )
                    skipped_df.to_csv(skipped_artifact, index=False)
                    mlflow.log_artifact(
                        str(skipped_artifact), artifact_path="skipped_hours"
                    )

                    mlflow.log_param("status", "ok")
                    mlflow.log_param("checkpoint_dir", str(checkpoint_dir))
                    successful_runs += 1

                except Exception as exc:
                    outcomes[model_name]["failed"] += 1
                    mlflow.log_param("status", "failed")
                    mlflow.log_param("error_type", type(exc).__name__)
                    mlflow.log_param("error_message", str(exc)[:500])
                    failed_runs.append(
                        {
                            "model": model_name,
                            "test_year": fold.test_year,
                            "error_type": type(exc).__name__,
                            "error_message": str(exc),
                        }
                    )

    _log("Checkpoint summary by model:")
    for model_name in sorted(outcomes):
        counts = outcomes[model_name]
        _log(
            f"  {model_name}: trained={counts['trained']} reused={counts['reused']} failed={counts['failed']}"
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

    hour_df = pd.DataFrame([asdict(row) for row in all_hour_rows])
    fold_df = pd.DataFrame([asdict(row) for row in all_fold_rows])

    if not hour_df.empty:
        hour_df = hour_df.sort_values(["model", "test_year", "hour"]).reset_index(
            drop=True
        )
    if not fold_df.empty:
        fold_df = fold_df.sort_values(["model", "test_year", "segment"]).reset_index(
            drop=True
        )

    hour_df.to_csv(run_root / "hour_metrics.csv", index=False)
    fold_df.to_csv(run_root / "fold_summary.csv", index=False)

    if not fold_df.empty:
        compact = (
            fold_df[fold_df["segment"] == "all"]
            .groupby(["model", "strategy"], as_index=False)
            .agg(
                smape=("smape", "mean"),
                mape=("mape", "mean"),
                mae=("mae", "mean"),
                mse=("mse", "mean"),
                r2=("r2", "mean"),
                n_hours=("n_hours", "mean"),
                n_points=("n_points", "sum"),
            )
            .sort_values(["smape", "mae"])
            .reset_index(drop=True)
        )
    else:
        compact = pd.DataFrame(
            columns=[
                "model",
                "strategy",
                "smape",
                "mape",
                "mae",
                "mse",
                "r2",
                "n_hours",
                "n_points",
            ]
        )

    compact.to_csv(run_root / "summary.csv", index=False)
    (run_root / "runtime_snapshot.json").write_text(
        json.dumps(to_serializable_dict(config), indent=2), encoding="utf-8"
    )
    _log(f"Run artifacts saved to: {run_root}")
    return hour_df, fold_df
