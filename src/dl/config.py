from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RuntimeConfig:
    mode: str
    action: str
    models: tuple[str, ...]
    test_year: int
    prediction_length: int
    window_stride: int
    context_length: int
    max_origins_per_year: int | None
    target_col: str
    conflict_date: str
    seed: int

    train_epochs: int
    train_batch_size: int
    train_lr: float
    train_weight_decay: float
    train_steps_per_epoch: int

    lag_llama_num_parallel_samples: int
    num_samples: int

    dataset_path: Path
    variant_stem: str | None
    eval_after_train: bool


def _project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def preprocessed_root() -> Path:
    return _project_root_from_here() / "data" / "preprocessed"


def models_root() -> Path:
    return _project_root_from_here() / "data" / "models" / "deep_learning"


def results_root() -> Path:
    return _project_root_from_here() / "data" / "results" / "deep_learning"


def mlflow_uri() -> str:
    mlflow_db = (_project_root_from_here() / "data" / "results" / "mlflow.db").resolve()
    return f"sqlite:///{mlflow_db.as_posix()}"


def mlflow_experiment() -> str:
    return "deep-learning-foundation-expanding-window"


def _parse_models(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("At least one model must be provided.")
    return parsed


def build_parser() -> argparse.ArgumentParser:
    default_dataset = preprocessed_root() / "merged_all_years_preprocessed.csv"

    parser = argparse.ArgumentParser(
        description=(
            "Foundation-model training/testing runner for expanding-window backtests "
            "(run from src/dl)."
        )
    )

    parser.add_argument(
        "action",
        choices=["train", "test", "eval"],
        help=(
            "train: fit fold checkpoints up to --test-year (evaluation optional via --eval-after-train); "
            "test: evaluate a single year; "
            "eval: evaluate all folds up to --test-year"
        ),
    )
    parser.add_argument(
        "--mode",
        choices=["one-shot", "finetuned"],
        default="finetuned",
        help="one-shot = pretrained only, finetuned = fold-specific training/checkpoint use",
    )
    parser.add_argument(
        "--models",
        type=_parse_models,
        default=("chronos2", "lag-llama", "moirai", "timesfm2.5"),
        help="Comma-separated model ids: chronos2, lag-llama, moirai, timesfm2.5",
    )
    parser.add_argument(
        "--test-year",
        type=int,
        required=True,
        help="Target test year (e.g. 2021).",
    )

    parser.add_argument("--prediction-length", type=int, default=24)
    parser.add_argument("--window-stride", type=int, default=24)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument(
        "--max-origins-per-year",
        type=int,
        default=None,
        help="Optional cap for faster experiments.",
    )
    parser.add_argument("--target-col", type=str, default="consumption_total")
    parser.add_argument("--conflict-date", type=str, default="2022-02-24 00:00:00")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--train-batch-size", type=int, default=8)
    parser.add_argument("--train-lr", type=float, default=5e-5)
    parser.add_argument("--train-weight-decay", type=float, default=0.0)
    parser.add_argument("--train-steps-per-epoch", type=int, default=50)

    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--lag-llama-num-parallel-samples", type=int, default=20)

    parser.add_argument("--dataset-path", type=Path, default=default_dataset)
    parser.add_argument(
        "--variant-stem",
        type=str,
        default=None,
        help=(
            "Optional preprocessing split variant stem. If provided and --dataset-path is not overridden, "
            "dataset resolves to preprocessed/splits/<variant_stem>/merged_all_years_preprocessed.csv"
        ),
    )
    parser.add_argument(
        "--eval-after-train",
        action="store_true",
        help=(
            "Only for action=train. If set, run evaluation right after each fold is trained. "
            "By default train only saves checkpoints."
        ),
    )

    return parser


def parse_args() -> RuntimeConfig:
    parser = build_parser()
    args = parser.parse_args()

    if args.mode == "one-shot" and args.action == "train":
        parser.error(
            "Invalid combination: '--mode one-shot' does not support 'train'. "
            "Use 'test' (single fold) or 'eval' (all folds)."
        )
    if args.eval_after_train and args.action != "train":
        parser.error(
            "Invalid combination: '--eval-after-train' is only valid with action 'train'."
        )

    return RuntimeConfig(
        mode=args.mode,
        action=args.action,
        models=tuple(args.models),
        test_year=args.test_year,
        prediction_length=args.prediction_length,
        window_stride=args.window_stride,
        context_length=args.context_length,
        max_origins_per_year=args.max_origins_per_year,
        target_col=args.target_col,
        conflict_date=args.conflict_date,
        seed=args.seed,
        train_epochs=args.train_epochs,
        train_batch_size=args.train_batch_size,
        train_lr=args.train_lr,
        train_weight_decay=args.train_weight_decay,
        train_steps_per_epoch=args.train_steps_per_epoch,
        lag_llama_num_parallel_samples=args.lag_llama_num_parallel_samples,
        num_samples=args.num_samples,
        dataset_path=args.dataset_path.resolve(),
        variant_stem=args.variant_stem,
        eval_after_train=args.eval_after_train,
    )


def to_serializable_dict(config: RuntimeConfig) -> dict[str, object]:
    out = {
        **config.__dict__,
        "dataset_path": str(config.dataset_path),
        "preprocessed_root": str(preprocessed_root()),
        "models_root": str(models_root()),
        "results_root": str(results_root()),
        "mlflow_uri": mlflow_uri(),
        "mlflow_experiment": mlflow_experiment(),
    }
    return out


def save_runtime_config(config: RuntimeConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_serializable_dict(config), indent=2), encoding="utf-8"
    )
