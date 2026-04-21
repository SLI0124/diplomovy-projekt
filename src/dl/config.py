from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from adapters import resolve_model_family, supported_model_ids


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
    training_input_mode: str
    covariate_columns: tuple[str, ...] | None
    future_covariate_columns: tuple[str, ...] | None
    past_covariate_columns: tuple[str, ...] | None
    conflict_date: str
    seed: int

    train_epochs: int
    train_batch_size: int
    train_lr: float
    train_weight_decay: float
    train_steps_per_epoch: int
    checkpoint_selection: str
    train_loss: str | None
    train_optimizer: str | None

    num_samples: int

    variant_stem: str
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


def mlflow_custom_experiment() -> str:
    return "deep-learning-custom-expanding-window"


def mlflow_experiment_for_family(model_family: str) -> str:
    family = model_family.strip().lower()
    if family == "foundation":
        return mlflow_experiment()
    if family == "custom":
        return mlflow_custom_experiment()
    raise ValueError(f"Unsupported model family '{model_family}'.")


def _parse_models(value: str) -> tuple[str, ...]:
    parsed = tuple(part.strip().lower() for part in value.split(",") if part.strip())
    if not parsed:
        raise argparse.ArgumentTypeError("At least one model must be provided.")
    supported = set(supported_model_ids())
    unknown = sorted({name for name in parsed if name not in supported})
    if unknown:
        supported_str = ", ".join(sorted(supported))
        unknown_str = ", ".join(unknown)
        raise argparse.ArgumentTypeError(
            f"Unsupported model(s): {unknown_str}. Supported: {supported_str}"
        )
    return parsed


def _parse_column_list(value: str) -> tuple[str, ...]:
    parsed: list[str] = []
    seen: set[str] = set()
    for part in value.split(","):
        column = part.strip()
        if not column:
            continue
        if column in seen:
            continue
        seen.add(column)
        parsed.append(column)
    return tuple(parsed)


def build_parser() -> argparse.ArgumentParser:
    supported_models = ", ".join(supported_model_ids())

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
        default=("chronos2", "moirai1", "granite"),
        help=f"Comma-separated model ids: {supported_models}",
    )
    parser.add_argument(
        "--test-year",
        type=int,
        required=True,
        help="Target test year (e.g. 2021).",
    )
    parser.add_argument(
        "--training-input-mode",
        choices=["univariate", "covariate"],
        required=True,
        help=(
            "Required input mode. univariate uses only target; "
            "covariate enables model-specific multi-covariate inputs (chronos2, moirai1, granite)."
        ),
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
    parser.add_argument(
        "--covariate-columns",
        type=_parse_column_list,
        default=None,
        help=(
            "Optional comma-separated covariate columns. "
            "When omitted in covariate mode, all non-target columns are used."
        ),
    )
    parser.add_argument(
        "--future-covariate-columns",
        type=_parse_column_list,
        default=None,
        help=(
            "Optional comma-separated future covariates. "
            "When omitted in covariate mode, defaults to calendar/holiday features."
        ),
    )
    parser.add_argument(
        "--past-covariate-columns",
        type=_parse_column_list,
        default=None,
        help=(
            "Optional comma-separated past covariates. "
            "When omitted in covariate mode, defaults to selected covariates minus future covariates."
        ),
    )
    parser.add_argument("--conflict-date", type=str, default="2022-02-24 00:00:00")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--train-epochs", type=int, default=10)
    parser.add_argument("--train-batch-size", type=int, default=16)
    parser.add_argument("--train-lr", type=float, default=5e-5)
    parser.add_argument("--train-weight-decay", type=float, default=0.0)
    parser.add_argument("--train-steps-per-epoch", type=int, default=100)
    parser.add_argument(
        "--checkpoint-selection",
        choices=["best-train-loss", "last"],
        default="best-train-loss",
        help=(
            "Checkpoint selection strategy during finetuning. "
            "best-train-loss restores the epoch with lowest training loss before saving; "
            "last saves the final epoch state."
        ),
    )
    parser.add_argument(
        "--train-loss",
        choices=["mae", "mse", "rmse", "mape", "smape"],
        default=None,
        help=(
            "Optional training loss for custom models only. "
            "Foundation models reject this parameter."
        ),
    )
    parser.add_argument(
        "--train-optimizer",
        choices=["adamw", "adam", "sgd"],
        default=None,
        help=(
            "Optional optimizer for custom models only. "
            "Defaults to adamw when omitted. Foundation models reject this parameter."
        ),
    )

    parser.add_argument("--num-samples", type=int, default=20)

    parser.add_argument(
        "--variant-stem",
        type=str,
        default="base",
        help=(
            "Preprocessing split variant stem. "
            "DL resolves train/test artifacts from preprocessed/splits/<variant_stem>/"
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
    if args.training_input_mode == "covariate":
        covariate_capable_models = {
            "chronos2",
            "moirai1",
            "granite",
            "model_1",
            "model_2",
            "model_3",
        }
        unsupported_models = [
            model_name
            for model_name in args.models
            if model_name not in covariate_capable_models
        ]
        if unsupported_models:
            parser.error(
                "Invalid combination: '--training-input-mode covariate' is currently "
                "supported only for models 'chronos2', 'moirai1', 'granite', "
                "'model_1', 'model_2', and 'model_3'. "
                f"Received unsupported model(s): {', '.join(unsupported_models)}"
            )
    elif any(
        value is not None
        for value in (
            args.covariate_columns,
            args.future_covariate_columns,
            args.past_covariate_columns,
        )
    ):
        parser.error(
            "Invalid combination: '--covariate-columns', '--future-covariate-columns', "
            "and '--past-covariate-columns' require '--training-input-mode covariate'."
        )
    if args.train_loss is not None:
        non_custom_models = [
            model_name
            for model_name in args.models
            if resolve_model_family(model_name) != "custom"
        ]
        if non_custom_models:
            parser.error(
                "Invalid combination: '--train-loss' is only supported for custom models. "
                f"Received foundation model(s): {', '.join(non_custom_models)}"
            )
    if args.train_optimizer is not None:
        non_custom_models = [
            model_name
            for model_name in args.models
            if resolve_model_family(model_name) != "custom"
        ]
        if non_custom_models:
            parser.error(
                "Invalid combination: '--train-optimizer' is only supported for custom models. "
                f"Received foundation model(s): {', '.join(non_custom_models)}"
            )
    if not args.variant_stem.strip():
        parser.error("Invalid value: '--variant-stem' must not be empty.")

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
        training_input_mode=args.training_input_mode,
        covariate_columns=args.covariate_columns,
        future_covariate_columns=args.future_covariate_columns,
        past_covariate_columns=args.past_covariate_columns,
        conflict_date=args.conflict_date,
        seed=args.seed,
        train_epochs=args.train_epochs,
        train_batch_size=args.train_batch_size,
        train_lr=args.train_lr,
        train_weight_decay=args.train_weight_decay,
        train_steps_per_epoch=args.train_steps_per_epoch,
        checkpoint_selection=args.checkpoint_selection,
        train_loss=args.train_loss,
        train_optimizer=args.train_optimizer,
        num_samples=args.num_samples,
        variant_stem=args.variant_stem.strip(),
        eval_after_train=args.eval_after_train,
    )


def to_serializable_dict(config: RuntimeConfig) -> dict[str, object]:
    out = {
        **config.__dict__,
        "preprocessed_root": str(preprocessed_root()),
        "models_root": str(models_root()),
        "results_root": str(results_root()),
        "mlflow_uri": mlflow_uri(),
        "mlflow_experiment": mlflow_experiment(),
        "mlflow_experiment_foundation": mlflow_experiment(),
        "mlflow_experiment_custom": mlflow_custom_experiment(),
    }
    return out


def save_runtime_config(config: RuntimeConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_serializable_dict(config), indent=2), encoding="utf-8"
    )
