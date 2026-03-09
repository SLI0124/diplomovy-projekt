from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import torch

try:
    from .config import RuntimeConfig, save_runtime_config, to_serializable_dict
    from .dataset import (
        DatasetBundle,
        iter_test_origins,
        make_test_df,
        make_train_target,
    )
    from .models import ModelContext, build_model_adapter
except ImportError:
    from config import RuntimeConfig, save_runtime_config, to_serializable_dict
    from dataset import (
        DatasetBundle,
        iter_test_origins,
        make_test_df,
        make_train_target,
    )
    from models import ModelContext, build_model_adapter


@dataclass(frozen=True)
class FoldSpec:
    test_year: int

    @property
    def train_end_year(self) -> int:
        return self.test_year - 1

    @property
    def train_years_label(self) -> str:
        return f"2013-{self.train_end_year}"


@dataclass(frozen=True)
class FoldMetrics:
    model: str
    mode: str
    train_years: str
    test_year: int
    segment: str
    n_windows: int
    n_points: int
    smape: float
    mape: float
    mae: float
    mse: float
    r2: float


def _log(message: str) -> None:
    print(f"[dl] {message}")


def _smape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.abs(y_true) + np.abs(y_pred)
    denom = np.maximum(denom, 1e-8)
    return float(200.0 * np.mean(np.abs(y_pred - y_true) / denom))


def _mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    denom = np.maximum(np.abs(y_true), 1e-8)
    return float(100.0 * np.mean(np.abs((y_true - y_pred) / denom)))


def _compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    return {
        "smape": _smape(y_true, y_pred),
        "mape": _mape(y_true, y_pred),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "mse": float(mean_squared_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _folds_for_action(config: RuntimeConfig) -> list[FoldSpec]:
    if config.action == "test":
        return [FoldSpec(test_year=config.test_year)]

    start = 2014
    if config.test_year < start:
        raise ValueError("test-year must be >= 2014")
    return [FoldSpec(test_year=year) for year in range(start, config.test_year + 1)]


def _build_checkpoint_dir(
    config: RuntimeConfig,
    model_slug: str,
    fold: FoldSpec,
    dataset_tag: str,
) -> Path:
    params_for_hash = {
        "pred_len": config.prediction_length,
        "context_len": config.context_length,
        "epochs": config.train_epochs,
        "batch_size": config.train_batch_size,
        "lr": config.train_lr,
        "weight_decay": config.train_weight_decay,
        "steps_per_epoch": config.train_steps_per_epoch,
        "stride": config.window_stride,
        "target": config.target_col,
    }
    digest = hashlib.md5(
        json.dumps(params_for_hash, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]

    folder = (
        config.models_root
        / model_slug
        / "finetuned"
        / dataset_tag
        / f"train_2013-{fold.train_end_year}__test-{fold.test_year}__{digest}"
    )
    return folder


def _dataset_tag(bundle: DatasetBundle) -> str:
    if bundle.run_params and isinstance(bundle.run_params.get("variant_stem"), str):
        return str(bundle.run_params["variant_stem"])
    return bundle.dataset_path.stem


def _run_id() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _build_dataset_profile(
    bundle: DatasetBundle,
    target_col: str,
) -> dict[str, object]:
    df = bundle.dataframe
    target = df[target_col].astype(float)

    profile: dict[str, object] = {
        "dataset_path": str(bundle.dataset_path),
        "dataset_file_size_bytes": int(bundle.dataset_path.stat().st_size),
        "dataset_file_sha256": _sha256_file(bundle.dataset_path),
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


def _build_dataset_columns_preview(bundle: DatasetBundle, max_rows: int = 120) -> str:
    rows: list[str] = []
    rows.append("column,dtype,na_count")

    df = bundle.dataframe
    for column in df.columns[:max_rows]:
        dtype = str(df[column].dtype)
        na_count = int(df[column].isna().sum())
        rows.append(f"{column},{dtype},{na_count}")

    return "\n".join(rows) + "\n"


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
        metrics = _compute_metrics(y_true_all, y_pred_all)
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


def _ensure_mlflow(config: RuntimeConfig) -> None:
    mlflow.set_tracking_uri(config.mlflow_uri)
    mlflow.set_experiment(config.mlflow_experiment)


def _safe_log_run_params(
    config: RuntimeConfig, bundle: DatasetBundle, fold: FoldSpec
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
            "dataset_tag": _dataset_tag(bundle),
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


def _safe_log_dataset_context(bundle: DatasetBundle, config: RuntimeConfig) -> None:
    profile = _build_dataset_profile(bundle=bundle, target_col=config.target_col)

    mlflow.set_tags(
        {
            "dataset.path": str(profile["dataset_path"]),
            "dataset.sha256": str(profile["dataset_file_sha256"]),
        }
    )

    mlflow.log_dict(profile, "dataset_profile.json")


def _safe_log_run_context(
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
            "models_root": str(config.models_root),
            "results_root": str(config.results_root),
            "checkpoint_dir": str(checkpoint_dir) if checkpoint_dir else None,
        },
        "device": str(device),
    }
    mlflow.log_dict(context_payload, "run_context.json")


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


def _build_missing_checkpoint_error(
    *,
    config: RuntimeConfig,
    model_slug: str,
    fold: FoldSpec,
    ckpt_dir: Path,
) -> str:
    return (
        "Fine-tuned checkpoint folder not found.\n"
        f"Expected path: {ckpt_dir}\n"
        "This runner does not auto-load the latest checkpoint in finetuned mode.\n"
        "Create this checkpoint first, for example:\n"
        f"  cd src/dl\n"
        f"  python main.py train --mode finetuned --test-year {fold.test_year} --models {model_slug}\n"
        "Optional: add --eval-after-train if you also want immediate evaluation metrics."
    )


def run(config: RuntimeConfig, bundle: DatasetBundle) -> pd.DataFrame:
    _ensure_mlflow(config)

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

    folds = _folds_for_action(config)
    results: list[FoldMetrics] = []
    successful_runs = 0

    dataset_tag = _dataset_tag(bundle)

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
                checkpoint_dir = _build_checkpoint_dir(
                    config=config,
                    model_slug=adapter.slug,
                    fold=fold,
                    dataset_tag=dataset_tag,
                )

            with mlflow.start_run(run_name=run_name):
                _safe_log_run_params(config, bundle, fold)
                _safe_log_dataset_context(bundle=bundle, config=config)
                _safe_log_run_context(
                    config=config,
                    bundle=bundle,
                    fold=fold,
                    adapter=adapter,
                    requested_model_name=model_name,
                    device=device,
                    run_root=run_root,
                    dataset_tag=dataset_tag,
                    checkpoint_dir=checkpoint_dir,
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
                                "dataset_tag": dataset_tag,
                                "runtime_config": to_serializable_dict(config),
                            }
                            (ckpt_dir / "metadata.json").write_text(
                                json.dumps(metadata, indent=2), encoding="utf-8"
                            )

                        else:
                            if not ckpt_dir.exists():
                                raise FileNotFoundError(
                                    _build_missing_checkpoint_error(
                                        config=config,
                                        model_slug=adapter.slug,
                                        fold=fold,
                                        ckpt_dir=ckpt_dir,
                                    )
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
