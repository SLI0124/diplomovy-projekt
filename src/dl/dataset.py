from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from config import RuntimeConfig, preprocessed_root
from folds import FoldSpec


@dataclass(frozen=True)
class DatasetBundle:
    split_root: Path
    run_params_path: Path | None
    run_params: dict[str, object] | None


@dataclass(frozen=True)
class FoldData:
    train_path: Path
    test_path: Path
    train_series: np.ndarray
    test_df: pd.DataFrame


def _resolve_split_root(config: RuntimeConfig) -> Path:
    return (preprocessed_root() / "splits" / config.variant_stem).resolve()


def _resolve_run_params_path(split_root: Path) -> Path | None:
    candidate = split_root / "run_params.json"
    return candidate if candidate.exists() else None


def _resolve_train_path(split_root: Path, train_end_year: int) -> Path:
    pattern = f"ranges_from_2013_to_*/range_2013_{train_end_year}.csv"
    candidates = sorted(split_root.glob(pattern))

    if not candidates:
        raise FileNotFoundError(
            _build_missing_split_artifact_message(
                split_root=split_root,
                missing_path=split_root / pattern,
            )
        )

    if len(candidates) > 1:
        rendered = "\n".join(str(path) for path in candidates)
        raise ValueError(
            "Ambiguous training split artifact match. "
            f"Expected exactly one file for train_end_year={train_end_year}, got {len(candidates)}:\n"
            f"{rendered}"
        )

    return candidates[0].resolve()


def _resolve_test_path(split_root: Path, test_year: int) -> Path:
    path = split_root / "single_years" / f"year_{test_year}.csv"
    if not path.exists():
        raise FileNotFoundError(
            _build_missing_split_artifact_message(
                split_root=split_root,
                missing_path=path,
            )
        )
    return path.resolve()


def _load_run_params(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _warn(message: str) -> None:
    print(f"[dataset] warning: {message}")


def _validate_run_params(
    run_params: dict[str, object] | None,
    run_params_path: Path | None,
) -> None:
    if run_params is None:
        return

    schema = run_params.get("schema")
    if schema != "preprocessing.run_params.v1":
        _warn(
            "Unexpected preprocessing params schema"
            f" ({schema!r}) at {run_params_path}. Continuing with best effort."
        )

    if "variant_stem" not in run_params:
        _warn(
            f"Missing 'variant_stem' in {run_params_path}. "
            "Dataset tagging will fallback to variant stem."
        )

    features = run_params.get("features")
    splits = run_params.get("splits")

    if not isinstance(features, dict):
        _warn(f"Missing or invalid 'features' section in {run_params_path}.")
    if not isinstance(splits, dict):
        _warn(f"Missing or invalid 'splits' section in {run_params_path}.")


def _build_missing_split_artifact_message(split_root: Path, missing_path: Path) -> str:
    return (
        f"Required preprocessing split artifact does not exist: {missing_path}\n"
        f"Expected split root: {split_root}\n"
        "Create base preprocessing splits first, then rerun DL:\n"
        "  cd ../preprocessing\n"
        "  python main.py\n"
    )


def load_dataset(config: RuntimeConfig) -> DatasetBundle:
    split_root = _resolve_split_root(config)
    if not split_root.exists():
        raise FileNotFoundError(
            _build_missing_split_artifact_message(
                split_root=split_root,
                missing_path=split_root,
            )
        )

    run_params_path = _resolve_run_params_path(split_root)
    run_params = _load_run_params(run_params_path)
    _validate_run_params(run_params=run_params, run_params_path=run_params_path)

    return DatasetBundle(
        split_root=split_root,
        run_params_path=run_params_path,
        run_params=run_params,
    )


def _validate_target_column(df: pd.DataFrame, target_col: str, path: Path) -> None:
    if target_col not in df.columns:
        raise ValueError(f"Missing target column '{target_col}' in dataset: {path}")

    if df[target_col].isna().any():
        missing = int(df[target_col].isna().sum())
        raise ValueError(
            f"Target column '{target_col}' contains {missing} NaN values in {path}. "
            "No in-module imputation is performed."
        )


def load_fold_data(
    bundle: DatasetBundle,
    config: RuntimeConfig,
    fold: FoldSpec,
) -> FoldData:
    train_path = _resolve_train_path(
        split_root=bundle.split_root,
        train_end_year=fold.train_end_year,
    )
    test_path = _resolve_test_path(
        split_root=bundle.split_root,
        test_year=fold.test_year,
    )

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    _validate_target_column(train_df, config.target_col, train_path)
    _validate_target_column(test_df, config.target_col, test_path)

    if train_df.empty:
        raise ValueError(f"Training split is empty: {train_path}")
    if test_df.empty:
        raise ValueError(f"Testing split is empty: {test_path}")

    train_series = train_df[config.target_col].to_numpy(dtype=np.float32)

    return FoldData(
        train_path=train_path,
        test_path=test_path,
        train_series=train_series,
        test_df=test_df.reset_index(drop=True),
    )


def iter_test_origins(test_df: pd.DataFrame, prediction_length: int, stride: int):
    max_i = len(test_df) - prediction_length
    if max_i < 0:
        raise ValueError(
            "Not enough test points for a single forecast window: "
            f"rows={len(test_df)} prediction_length={prediction_length}"
        )

    i = 0
    while i <= max_i:
        yield i
        i += stride
