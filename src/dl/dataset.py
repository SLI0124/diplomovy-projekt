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
    train_covariates: np.ndarray | None
    test_covariates: np.ndarray | None
    train_future_covariates: np.ndarray | None
    test_future_covariates: np.ndarray | None
    covariate_columns: tuple[str, ...]
    future_covariate_columns: tuple[str, ...]
    past_covariate_columns: tuple[str, ...]


DEFAULT_FUTURE_COVARIATE_COLUMNS: tuple[str, ...] = (
    "year",
    "month",
    "day",
    "hour",
    "day_of_week",
    "holiday",
    "before_holiday",
)


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

    column_rules = run_params.get("column_rules")
    if column_rules is None:
        return
    if not isinstance(column_rules, dict):
        raise ValueError(
            f"Invalid 'column_rules' section in {run_params_path}. Expected object."
        )

    enabled = bool(column_rules.get("enabled", False))
    if not enabled:
        return

    report_file = column_rules.get("report_file")
    if not isinstance(report_file, str) or not report_file.strip():
        raise ValueError(
            "Preprocessing metadata indicates enabled column_rules but missing report_file. "
            f"run_params={run_params_path}"
        )

    if run_params_path is None:
        raise ValueError(
            "Cannot validate column_rules metadata because run_params_path is missing."
        )

    report_path = run_params_path.parent / report_file
    if not report_path.exists():
        raise FileNotFoundError(
            "Preprocessing metadata indicates enabled column_rules but report file is missing: "
            f"{report_path}"
        )


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


def _validate_columns_exist(
    *,
    requested: tuple[str, ...],
    available: set[str],
    role: str,
    train_path: Path,
    test_path: Path,
) -> None:
    missing = tuple(column for column in requested if column not in available)
    if missing:
        rendered = ", ".join(missing)
        raise ValueError(
            f"Missing {role} column(s): {rendered}. "
            f"Columns must exist in both train and test splits. "
            f"train={train_path} test={test_path}"
        )


def _resolve_covariate_layout(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: RuntimeConfig,
    train_path: Path,
    test_path: Path,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    if config.training_input_mode != "covariate":
        return (), (), ()

    train_columns = set(str(column) for column in train_df.columns)
    test_columns = set(str(column) for column in test_df.columns)
    shared_columns = train_columns.intersection(test_columns)

    if config.target_col not in shared_columns:
        raise ValueError(
            "Target column must exist in both train and test splits for covariate mode. "
            f"target={config.target_col} train={train_path} test={test_path}"
        )

    if config.covariate_columns is not None:
        selected_covariates = tuple(config.covariate_columns)
    else:
        selected_covariates = tuple(
            str(column)
            for column in train_df.columns
            if str(column) in shared_columns and str(column) != config.target_col
        )

    if config.target_col in selected_covariates:
        raise ValueError(
            f"Covariates must not contain target column '{config.target_col}'."
        )

    _validate_columns_exist(
        requested=selected_covariates,
        available=shared_columns,
        role="covariate",
        train_path=train_path,
        test_path=test_path,
    )

    if not selected_covariates:
        raise ValueError(
            "Covariate mode requires at least one covariate column. "
            "Pass --covariate-columns or provide non-target columns in the dataset."
        )

    if config.future_covariate_columns is not None:
        future_covariates = tuple(config.future_covariate_columns)
    else:
        future_covariates = tuple(
            column
            for column in DEFAULT_FUTURE_COVARIATE_COLUMNS
            if column in selected_covariates
        )

    _validate_columns_exist(
        requested=future_covariates,
        available=shared_columns,
        role="future covariate",
        train_path=train_path,
        test_path=test_path,
    )

    if config.past_covariate_columns is not None:
        past_covariates = tuple(config.past_covariate_columns)
    else:
        future_set = set(future_covariates)
        past_covariates = tuple(
            column for column in selected_covariates if column not in future_set
        )

    _validate_columns_exist(
        requested=past_covariates,
        available=shared_columns,
        role="past covariate",
        train_path=train_path,
        test_path=test_path,
    )

    selected_set = set(selected_covariates)
    if not set(future_covariates).issubset(selected_set):
        raise ValueError(
            "All future covariates must be selected covariates. "
            f"selected={selected_covariates} future={future_covariates}"
        )
    if not set(past_covariates).issubset(selected_set):
        raise ValueError(
            "All past covariates must be selected covariates. "
            f"selected={selected_covariates} past={past_covariates}"
        )

    overlap = set(future_covariates).intersection(past_covariates)
    if overlap:
        rendered = ", ".join(sorted(overlap))
        raise ValueError(
            "Future and past covariates must be disjoint. "
            f"Overlapping columns: {rendered}"
        )

    return selected_covariates, future_covariates, past_covariates


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
    covariate_columns, future_covariate_columns, past_covariate_columns = (
        _resolve_covariate_layout(
            train_df=train_df,
            test_df=test_df,
            config=config,
            train_path=train_path,
            test_path=test_path,
        )
    )

    train_covariates: np.ndarray | None = None
    test_covariates: np.ndarray | None = None
    train_future_covariates: np.ndarray | None = None
    test_future_covariates: np.ndarray | None = None

    if covariate_columns:
        train_covariates = train_df.loc[:, list(covariate_columns)].to_numpy(
            dtype=np.float32
        )
        test_covariates = test_df.loc[:, list(covariate_columns)].to_numpy(
            dtype=np.float32
        )

        # Future covariates are aligned to selected covariate columns; non-future columns remain NaN.
        train_future_covariates = np.full_like(train_covariates, np.nan)
        test_future_covariates = np.full_like(test_covariates, np.nan)
        future_set = set(future_covariate_columns)
        for column_index, column_name in enumerate(covariate_columns):
            if column_name not in future_set:
                continue
            train_future_covariates[:, column_index] = train_df[column_name].to_numpy(
                dtype=np.float32
            )
            test_future_covariates[:, column_index] = test_df[column_name].to_numpy(
                dtype=np.float32
            )

    return FoldData(
        train_path=train_path,
        test_path=test_path,
        train_series=train_series,
        test_df=test_df.reset_index(drop=True),
        train_covariates=train_covariates,
        test_covariates=test_covariates,
        train_future_covariates=train_future_covariates,
        test_future_covariates=test_future_covariates,
        covariate_columns=covariate_columns,
        future_covariate_columns=future_covariate_columns,
        past_covariate_columns=past_covariate_columns,
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
