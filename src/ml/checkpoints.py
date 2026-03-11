from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from config import RuntimeConfig, models_root
from dataset import DatasetBundle
from folds import FoldSpec

CHECKPOINT_MANIFEST_VERSION = "ml_checkpoint_manifest.v1"
CheckpointStatus = Literal["compatible_exists", "missing", "incompatible_manifest"]


def dataset_tag(bundle: DatasetBundle) -> str:
    if bundle.run_params and isinstance(bundle.run_params.get("variant_stem"), str):
        return str(bundle.run_params["variant_stem"])
    return bundle.dataset_path.stem


def _feature_signature(feature_columns: tuple[str, ...]) -> str:
    return hashlib.sha256("|".join(feature_columns).encode("utf-8")).hexdigest()


def build_checkpoint_dir(
    config: RuntimeConfig,
    model_name: str,
    fold: FoldSpec,
    current_dataset_tag: str,
    feature_columns: tuple[str, ...],
) -> Path:
    params_for_hash = {
        "strategy": config.strategy,
        "target": config.target_col,
        "hourly_min_train_samples": config.hourly_min_train_samples,
        "hourly_min_test_samples": config.hourly_min_test_samples,
        "dt_max_depth": config.dt_max_depth,
        "dt_min_samples_split": config.dt_min_samples_split,
        "rf_n_estimators": config.rf_n_estimators,
        "rf_max_depth": config.rf_max_depth,
        "gb_n_estimators": config.gb_n_estimators,
        "gb_learning_rate": config.gb_learning_rate,
        "gb_max_depth": config.gb_max_depth,
        "feature_signature": _feature_signature(feature_columns),
    }
    digest = hashlib.md5(
        json.dumps(params_for_hash, sort_keys=True).encode("utf-8")
    ).hexdigest()[:10]

    return (
        models_root()
        / model_name
        / config.strategy
        / current_dataset_tag
        / f"train_2013-{fold.train_end_year}__test-{fold.test_year}__{digest}"
    )


def write_checkpoint_manifest(
    *,
    checkpoint_dir: Path,
    config: RuntimeConfig,
    fold: FoldSpec,
    model_name: str,
    current_dataset_tag: str,
    feature_columns: tuple[str, ...],
    trained_hours: list[int],
) -> Path:
    payload = {
        "schema": CHECKPOINT_MANIFEST_VERSION,
        "model": model_name,
        "strategy": config.strategy,
        "dataset_tag": current_dataset_tag,
        "fold": {
            "train_years": fold.train_years_label,
            "train_end_year": fold.train_end_year,
            "test_year": fold.test_year,
        },
        "trained_hours": trained_hours,
        "compatibility": {
            "target_col": config.target_col,
            "hourly_min_train_samples": config.hourly_min_train_samples,
            "hourly_min_test_samples": config.hourly_min_test_samples,
            "feature_signature": _feature_signature(feature_columns),
            "dt_max_depth": config.dt_max_depth,
            "dt_min_samples_split": config.dt_min_samples_split,
            "rf_n_estimators": config.rf_n_estimators,
            "rf_max_depth": config.rf_max_depth,
            "gb_n_estimators": config.gb_n_estimators,
            "gb_learning_rate": config.gb_learning_rate,
            "gb_max_depth": config.gb_max_depth,
        },
    }

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return manifest_path


def validate_checkpoint_manifest(
    *,
    checkpoint_dir: Path,
    config: RuntimeConfig,
    fold: FoldSpec,
    model_name: str,
    current_dataset_tag: str,
    feature_columns: tuple[str, ...],
) -> None:
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Missing checkpoint_manifest.json for model checkpoint. "
            f"checkpoint_dir={checkpoint_dir}"
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    if payload.get("schema") != CHECKPOINT_MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported checkpoint manifest schema in {manifest_path}: "
            f"{payload.get('schema')}"
        )

    checks = {
        "model": (payload.get("model"), model_name),
        "strategy": (payload.get("strategy"), config.strategy),
        "dataset_tag": (payload.get("dataset_tag"), current_dataset_tag),
        "fold.test_year": (
            ((payload.get("fold") or {}).get("test_year")),
            fold.test_year,
        ),
        "compatibility.target_col": (
            ((payload.get("compatibility") or {}).get("target_col")),
            config.target_col,
        ),
        "compatibility.feature_signature": (
            ((payload.get("compatibility") or {}).get("feature_signature")),
            _feature_signature(feature_columns),
        ),
    }

    mismatches: list[str] = []
    for key, (observed, expected) in checks.items():
        if observed != expected:
            mismatches.append(f"{key}: expected={expected} observed={observed}")

    if mismatches:
        rendered = "\n".join(mismatches)
        raise ValueError(
            "Checkpoint manifest compatibility mismatch:\n"
            f"{rendered}\n"
            f"manifest={manifest_path}"
        )


def resolve_checkpoint_status(
    *,
    checkpoint_dir: Path,
    config: RuntimeConfig,
    fold: FoldSpec,
    model_name: str,
    current_dataset_tag: str,
    feature_columns: tuple[str, ...],
) -> tuple[CheckpointStatus, str | None]:
    if not checkpoint_dir.exists():
        return "missing", None

    try:
        validate_checkpoint_manifest(
            checkpoint_dir=checkpoint_dir,
            config=config,
            fold=fold,
            model_name=model_name,
            current_dataset_tag=current_dataset_tag,
            feature_columns=feature_columns,
        )
    except (FileNotFoundError, ValueError) as exc:
        return "incompatible_manifest", str(exc)

    return "compatible_exists", None


def build_missing_checkpoint_error(
    *, fold: FoldSpec, model_name: str, ckpt_dir: Path
) -> str:
    return (
        "Model checkpoint folder not found.\n"
        f"Expected path: {ckpt_dir}\n"
        "Create this checkpoint first, for example:\n"
        "  cd src/ml\n"
        f"  python main.py train --test-year {fold.test_year} --models {model_name} --strategy hourly\n"
    )
