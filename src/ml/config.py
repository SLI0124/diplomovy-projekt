from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

from models import supported_model_ids


@dataclass(frozen=True)
class RuntimeConfig:
    action: str
    strategy: str
    models: tuple[str, ...]
    test_year: int

    target_col: str
    conflict_date: str
    seed: int

    hourly_min_train_samples: int
    hourly_min_test_samples: int
    drop_high_missing_threshold: float

    dt_max_depth: int
    dt_min_samples_split: int
    rf_n_estimators: int
    rf_max_depth: int
    gb_n_estimators: int
    gb_learning_rate: float
    gb_max_depth: int

    dataset_path: Path
    variant_stem: str | None
    eval_after_train: bool
    force_retrain: bool


def _project_root_from_here() -> Path:
    return Path(__file__).resolve().parents[2]


def preprocessed_root() -> Path:
    return _project_root_from_here() / "data" / "preprocessed"


def models_root() -> Path:
    return _project_root_from_here() / "data" / "models" / "ml"


def results_root() -> Path:
    return _project_root_from_here() / "data" / "results" / "ml"


def mlflow_uri() -> str:
    mlflow_db = (_project_root_from_here() / "data" / "results" / "mlflow.db").resolve()
    return f"sqlite:///{mlflow_db.as_posix()}"


def mlflow_experiment() -> str:
    return "ml-expanding-window"


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


def build_parser() -> argparse.ArgumentParser:
    default_dataset = preprocessed_root() / "merged_all_years_preprocessed.csv"
    supported_models = ", ".join(supported_model_ids())

    parser = argparse.ArgumentParser(
        description="Classical ML expanding-window runner (run from src/ml)."
    )

    parser.add_argument(
        "action",
        choices=["train", "test", "eval"],
        help=(
            "train: train folds up to --test-year; "
            "test: evaluate one fold for --test-year; "
            "eval: evaluate all folds up to --test-year"
        ),
    )
    parser.add_argument(
        "--strategy",
        choices=["hourly"],
        default="hourly",
        help="Forecast strategy. hourly=train 24 separate models per fold.",
    )
    parser.add_argument(
        "--models",
        type=_parse_models,
        default=(
            "decision-tree",
            "random-forest",
            "gradient-boosting",
            "linear-regression",
        ),
        help=f"Comma-separated model ids: {supported_models}",
    )
    parser.add_argument("--test-year", type=int, required=True)

    parser.add_argument("--target-col", type=str, default="consumption_total")
    parser.add_argument("--conflict-date", type=str, default="2022-02-24 00:00:00")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--hourly-min-train-samples", type=int, default=50)
    parser.add_argument("--hourly-min-test-samples", type=int, default=5)
    parser.add_argument("--drop-high-missing-threshold", type=float, default=0.8)

    parser.add_argument("--dt-max-depth", type=int, default=12)
    parser.add_argument("--dt-min-samples-split", type=int, default=10)
    parser.add_argument("--rf-n-estimators", type=int, default=120)
    parser.add_argument("--rf-max-depth", type=int, default=14)
    parser.add_argument("--gb-n-estimators", type=int, default=100)
    parser.add_argument("--gb-learning-rate", type=float, default=0.1)
    parser.add_argument("--gb-max-depth", type=int, default=3)

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
        help="Only for action=train. Evaluate each fold immediately after training.",
    )
    parser.add_argument(
        "--force-retrain",
        action="store_true",
        help="Ignore compatible checkpoints and retrain fold artifacts.",
    )

    return parser


def parse_args() -> RuntimeConfig:
    parser = build_parser()
    args = parser.parse_args()

    if args.eval_after_train and args.action != "train":
        parser.error(
            "Invalid combination: '--eval-after-train' is only valid with action 'train'."
        )
    if args.test_year < 2014:
        parser.error("test-year must be >= 2014")
    if not 0.0 <= args.drop_high_missing_threshold <= 1.0:
        parser.error("drop-high-missing-threshold must be in [0, 1].")

    return RuntimeConfig(
        action=args.action,
        strategy=args.strategy,
        models=tuple(args.models),
        test_year=args.test_year,
        target_col=args.target_col,
        conflict_date=args.conflict_date,
        seed=args.seed,
        hourly_min_train_samples=args.hourly_min_train_samples,
        hourly_min_test_samples=args.hourly_min_test_samples,
        drop_high_missing_threshold=args.drop_high_missing_threshold,
        dt_max_depth=args.dt_max_depth,
        dt_min_samples_split=args.dt_min_samples_split,
        rf_n_estimators=args.rf_n_estimators,
        rf_max_depth=args.rf_max_depth,
        gb_n_estimators=args.gb_n_estimators,
        gb_learning_rate=args.gb_learning_rate,
        gb_max_depth=args.gb_max_depth,
        dataset_path=args.dataset_path.resolve(),
        variant_stem=args.variant_stem,
        eval_after_train=args.eval_after_train,
        force_retrain=args.force_retrain,
    )


def to_serializable_dict(config: RuntimeConfig) -> dict[str, object]:
    return {
        **config.__dict__,
        "dataset_path": str(config.dataset_path),
        "preprocessed_root": str(preprocessed_root()),
        "models_root": str(models_root()),
        "results_root": str(results_root()),
        "mlflow_uri": mlflow_uri(),
        "mlflow_experiment": mlflow_experiment(),
    }


def save_runtime_config(config: RuntimeConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(to_serializable_dict(config), indent=2), encoding="utf-8"
    )
