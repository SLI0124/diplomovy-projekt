from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .config import RuntimeConfig
except ImportError:
    from config import RuntimeConfig


@dataclass(frozen=True)
class DatasetBundle:
    dataframe: pd.DataFrame
    dataset_path: Path
    run_params_path: Path | None
    run_params: dict[str, object] | None


def _resolve_dataset_path(config: RuntimeConfig) -> Path:
    if config.variant_stem is None:
        return config.dataset_path

    from_variant = (
        config.preprocessed_root
        / "splits"
        / config.variant_stem
        / "merged_all_years_preprocessed.csv"
    )

    default_root_dataset = (
        config.preprocessed_root / "merged_all_years_preprocessed.csv"
    ).resolve()

    if config.dataset_path == default_root_dataset:
        return from_variant.resolve()

    return config.dataset_path


def _resolve_run_params_path(config: RuntimeConfig, dataset_path: Path) -> Path | None:
    candidate_in_variant = dataset_path.parent / "run_params.json"
    if candidate_in_variant.exists():
        return candidate_in_variant

    if config.variant_stem is not None:
        candidate = (
            config.preprocessed_root
            / "splits"
            / config.variant_stem
            / "run_params.json"
        )
        if candidate.exists():
            return candidate.resolve()

    return None


def _load_run_params(path: Path | None) -> dict[str, object] | None:
    if path is None:
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _build_missing_dataset_message(path: Path) -> str:
    script = Path(__file__).resolve().parents[1] / "preprocessing" / "main.py"
    return (
        f"Dataset file does not exist: {path}\n"
        "Create it first using preprocessing module, for example:\n"
        f"  cd {script.parent}\n"
        "  python main.py --add-cyclical --drop-cyclical-source-columns\n"
    )


def load_dataset(config: RuntimeConfig) -> DatasetBundle:
    dataset_path = _resolve_dataset_path(config)
    if not dataset_path.exists():
        raise FileNotFoundError(_build_missing_dataset_message(dataset_path))

    df = pd.read_csv(dataset_path)

    required_time = ["year", "month", "day", "hour"]
    missing_time = [column for column in required_time if column not in df.columns]
    if missing_time:
        raise ValueError(
            f"Missing required time columns: {missing_time}. "
            "Dataset must include year/month/day/hour."
        )

    if config.target_col not in df.columns:
        raise ValueError(
            f"Missing target column '{config.target_col}' in dataset: {dataset_path}"
        )

    timestamp = pd.to_datetime(df[required_time], errors="coerce")
    if timestamp.isna().any():
        bad = int(timestamp.isna().sum())
        raise ValueError(
            f"Found {bad} invalid timestamps in dataset. Fix preprocessing."
        )

    if df[config.target_col].isna().any():
        missing = int(df[config.target_col].isna().sum())
        raise ValueError(
            f"Target column '{config.target_col}' contains {missing} NaN values. "
            "No in-module imputation is performed."
        )

    out = df.copy()
    out["timestamp"] = timestamp
    out = out.sort_values("timestamp", kind="stable").reset_index(drop=True)

    run_params_path = _resolve_run_params_path(config, dataset_path)
    run_params = _load_run_params(run_params_path)

    return DatasetBundle(
        dataframe=out,
        dataset_path=dataset_path.resolve(),
        run_params_path=run_params_path,
        run_params=run_params,
    )


def make_train_target(
    full_df: pd.DataFrame,
    target_col: str,
    train_end_year: int,
) -> np.ndarray:
    end = pd.Timestamp(year=train_end_year + 1, month=1, day=1)
    return full_df.loc[full_df["timestamp"] < end, target_col].to_numpy(
        dtype=np.float32
    )


def make_test_df(full_df: pd.DataFrame, test_year: int) -> pd.DataFrame:
    start = pd.Timestamp(year=test_year, month=1, day=1)
    end = pd.Timestamp(year=test_year + 1, month=1, day=1)
    return full_df[
        (full_df["timestamp"] >= start) & (full_df["timestamp"] < end)
    ].reset_index(drop=True)


def iter_test_origins(test_df: pd.DataFrame, prediction_length: int, stride: int):
    max_i = len(test_df) - prediction_length
    i = 0
    while i <= max_i:
        yield i
        i += stride
