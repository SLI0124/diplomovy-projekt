from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Literal

from config import RuntimeConfig
from dataset import DatasetBundle
from folds import FoldSpec

CHECKPOINT_MANIFEST_VERSION = "dl_checkpoint_manifest.v1"
CheckpointStatus = Literal["compatible_exists", "missing", "incompatible_manifest"]


def dataset_tag(bundle: DatasetBundle) -> str:
    if bundle.run_params and isinstance(bundle.run_params.get("variant_stem"), str):
        return str(bundle.run_params["variant_stem"])
    return bundle.dataset_path.stem


def build_checkpoint_dir(
    config: RuntimeConfig,
    model_slug: str,
    fold: FoldSpec,
    current_dataset_tag: str,
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

    return (
        config.models_root
        / model_slug
        / "finetuned"
        / current_dataset_tag
        / f"train_2013-{fold.train_end_year}__test-{fold.test_year}__{digest}"
    )


def write_checkpoint_manifest(
    *,
    checkpoint_dir: Path,
    config: RuntimeConfig,
    fold: FoldSpec,
    model_slug: str,
    current_dataset_tag: str,
) -> Path:
    payload = {
        "schema": CHECKPOINT_MANIFEST_VERSION,
        "model_slug": model_slug,
        "mode": "finetuned",
        "fold": {
            "train_years": fold.train_years_label,
            "train_end_year": fold.train_end_year,
            "test_year": fold.test_year,
        },
        "dataset_tag": current_dataset_tag,
        "compatibility": {
            "prediction_length": config.prediction_length,
            "context_length": config.context_length,
            "target_col": config.target_col,
            "window_stride": config.window_stride,
            "train_epochs": config.train_epochs,
            "train_batch_size": config.train_batch_size,
            "train_lr": config.train_lr,
            "train_weight_decay": config.train_weight_decay,
            "train_steps_per_epoch": config.train_steps_per_epoch,
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
    model_slug: str,
    current_dataset_tag: str,
) -> None:
    manifest_path = checkpoint_dir / "checkpoint_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(
            "Missing checkpoint_manifest.json for finetuned checkpoint. "
            "Refusing to load in strict mode. "
            f"checkpoint_dir={checkpoint_dir}"
        )

    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    if payload.get("schema") != CHECKPOINT_MANIFEST_VERSION:
        raise ValueError(
            f"Unsupported checkpoint manifest schema in {manifest_path}: "
            f"{payload.get('schema')}"
        )

    checks = {
        "model_slug": (payload.get("model_slug"), model_slug),
        "dataset_tag": (payload.get("dataset_tag"), current_dataset_tag),
        "fold.test_year": (
            ((payload.get("fold") or {}).get("test_year")),
            fold.test_year,
        ),
        "compatibility.prediction_length": (
            ((payload.get("compatibility") or {}).get("prediction_length")),
            config.prediction_length,
        ),
        "compatibility.context_length": (
            ((payload.get("compatibility") or {}).get("context_length")),
            config.context_length,
        ),
        "compatibility.target_col": (
            ((payload.get("compatibility") or {}).get("target_col")),
            config.target_col,
        ),
        "compatibility.window_stride": (
            ((payload.get("compatibility") or {}).get("window_stride")),
            config.window_stride,
        ),
        "compatibility.train_epochs": (
            ((payload.get("compatibility") or {}).get("train_epochs")),
            config.train_epochs,
        ),
        "compatibility.train_batch_size": (
            ((payload.get("compatibility") or {}).get("train_batch_size")),
            config.train_batch_size,
        ),
        "compatibility.train_lr": (
            ((payload.get("compatibility") or {}).get("train_lr")),
            config.train_lr,
        ),
        "compatibility.train_weight_decay": (
            ((payload.get("compatibility") or {}).get("train_weight_decay")),
            config.train_weight_decay,
        ),
        "compatibility.train_steps_per_epoch": (
            ((payload.get("compatibility") or {}).get("train_steps_per_epoch")),
            config.train_steps_per_epoch,
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
    model_slug: str,
    current_dataset_tag: str,
) -> tuple[CheckpointStatus, str | None]:
    if not checkpoint_dir.exists():
        return "missing", None

    try:
        validate_checkpoint_manifest(
            checkpoint_dir=checkpoint_dir,
            config=config,
            fold=fold,
            model_slug=model_slug,
            current_dataset_tag=current_dataset_tag,
        )
    except (FileNotFoundError, ValueError) as exc:
        return "incompatible_manifest", str(exc)

    return "compatible_exists", None


def build_missing_checkpoint_error(
    *,
    fold: FoldSpec,
    model_slug: str,
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
