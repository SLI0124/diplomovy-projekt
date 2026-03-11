from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from config import RuntimeConfig, preprocessed_root


@dataclass(frozen=True)
class DatasetBundle:
    dataframe: pd.DataFrame
    dataset_path: Path
    run_params_path: Path | None
    run_params: dict[str, object] | None
    feature_columns: tuple[str, ...]
    dropped_high_missing_columns: tuple[str, ...]
    excluded_leakage_columns: tuple[str, ...]


def _resolve_dataset_path(config: RuntimeConfig) -> Path:
    if config.variant_stem is None:
        return config.dataset_path

    from_variant = (
        preprocessed_root()
        / "splits"
        / config.variant_stem
        / "merged_all_years_preprocessed.csv"
    )

    default_root_dataset = (
        preprocessed_root() / "merged_all_years_preprocessed.csv"
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
            preprocessed_root() / "splits" / config.variant_stem / "run_params.json"
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


def _warn(message: str) -> None:
    print(f"[dataset] warning: {message}")


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
    out = df.copy()
    out["timestamp"] = timestamp
    out = out.dropna(subset=["timestamp", config.target_col]).copy()
    out = out.sort_values("timestamp", kind="stable").reset_index(drop=True)

    missing_ratio = out.isna().mean()
    protected = set(required_time + ["timestamp", config.target_col])
    dropped_high_missing_columns = tuple(
        col
        for col in out.columns
        if col not in protected
        and missing_ratio[col] > config.drop_high_missing_threshold
    )
    if dropped_high_missing_columns:
        out = out.drop(columns=list(dropped_high_missing_columns))
        _warn(
            f"Dropped {len(dropped_high_missing_columns)} high-missing columns "
            f"(threshold={config.drop_high_missing_threshold})."
        )

    excluded_leakage_columns = tuple(
        col
        for col in out.columns
        if col.startswith("consumption_") and col != config.target_col
    )

    feature_exclude = set(["timestamp", config.target_col, *excluded_leakage_columns])
    feature_columns = tuple(col for col in out.columns if col not in feature_exclude)
    if not feature_columns:
        raise ValueError("No feature columns available after filtering.")

    run_params_path = _resolve_run_params_path(config, dataset_path)
    run_params = _load_run_params(run_params_path)
    if (
        run_params is not None
        and run_params.get("schema") != "preprocessing.run_params.v1"
    ):
        _warn(
            f"Unexpected preprocessing schema in {run_params_path}: {run_params.get('schema')}"
        )

    return DatasetBundle(
        dataframe=out,
        dataset_path=dataset_path.resolve(),
        run_params_path=run_params_path,
        run_params=run_params,
        feature_columns=feature_columns,
        dropped_high_missing_columns=dropped_high_missing_columns,
        excluded_leakage_columns=excluded_leakage_columns,
    )


def make_train_test_df(
    full_df: pd.DataFrame, test_year: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = full_df[full_df["year"] < test_year].copy()
    test = full_df[full_df["year"] == test_year].copy()
    return train.reset_index(drop=True), test.reset_index(drop=True)
