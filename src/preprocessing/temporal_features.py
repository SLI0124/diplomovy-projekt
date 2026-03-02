"""Temporal feature engineering utilities used by preprocessing CLI."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class TemporalFeatureConfig:
    """Switches and parameters for optional temporal feature generation."""

    add_cyclical: bool = False
    cyclical_columns: tuple[str, ...] = ("hour", "month", "day_of_week")
    drop_cyclical_source_columns: bool = False

    add_lag_features: bool = False
    lag_columns: tuple[str, ...] = ("consumption_total",)
    lag_counts: tuple[int, ...] = (1, 24, 168)

    add_rolling_features: bool = False
    rolling_columns: tuple[str, ...] = ("consumption_total",)
    rolling_windows: tuple[int, ...] = (24, 168)
    rolling_aggregation: str = "mean"

    add_expanding_features: bool = False
    expanding_columns: tuple[str, ...] = ("consumption_total",)
    expanding_min_periods: int = 24
    expanding_aggregation: str = "mean"

    drop_columns: tuple[str, ...] = ()


_CYCLICAL_PERIODS: dict[str, int] = {
    "hour": 24,
    "month": 12,
    "day_of_week": 7,
}


def _log(message: str) -> None:
    print(f"[features] {message}")


def _existing_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _add_cyclical_features(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    for column in columns:
        period = _CYCLICAL_PERIODS.get(column)
        if period is None:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        angle = 2.0 * np.pi * (values / period)
        df[f"{column}_sin"] = np.sin(angle)
        df[f"{column}_cos"] = np.cos(angle)
    return df


def _add_lag_features(
    df: pd.DataFrame, columns: list[str], lag_counts: tuple[int, ...]
) -> pd.DataFrame:
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce")
        for lag in lag_counts:
            if lag < 1:
                continue
            df[f"{column}_lag_{lag}"] = series.shift(lag)
    return df


def _add_rolling_features(
    df: pd.DataFrame,
    columns: list[str],
    windows: tuple[int, ...],
    aggregation: str,
) -> pd.DataFrame:
    """Add rolling features from past values only (shifted by one step)."""

    use_mean = aggregation in {"mean", "both"}
    use_sum = aggregation in {"sum", "both"}

    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce")
        shifted = series.shift(1)
        for window in windows:
            if window < 1:
                continue
            rolling = shifted.rolling(window=window, min_periods=window)
            if use_mean:
                df[f"{column}_rolling_mean_{window}"] = rolling.mean()
            if use_sum:
                df[f"{column}_rolling_sum_{window}"] = rolling.sum()
    return df


def _add_expanding_features(
    df: pd.DataFrame,
    columns: list[str],
    min_periods: int,
    aggregation: str,
) -> pd.DataFrame:
    """Add expanding features from past values only (shifted by one step)."""

    use_mean = aggregation in {"mean", "both"}
    use_sum = aggregation in {"sum", "both"}

    effective_min_periods = max(int(min_periods), 1)
    for column in columns:
        series = pd.to_numeric(df[column], errors="coerce")
        shifted = series.shift(1)
        expanding = shifted.expanding(min_periods=effective_min_periods)
        if use_mean:
            df[f"{column}_expanding_mean_{effective_min_periods}"] = expanding.mean()
        if use_sum:
            df[f"{column}_expanding_sum_{effective_min_periods}"] = expanding.sum()
    return df


def apply_temporal_features(
    dataframe: pd.DataFrame,
    config: TemporalFeatureConfig,
) -> pd.DataFrame:
    """Apply enabled temporal feature transforms in a deterministic order."""

    df = dataframe.copy()
    original_columns = set(df.columns)

    if config.add_cyclical:
        _log("Adding cyclical features")
        cyclical_columns = _existing_columns(df, config.cyclical_columns)
        df = _add_cyclical_features(df, cyclical_columns)
        if config.drop_cyclical_source_columns and cyclical_columns:
            _log("Dropping cyclical source columns: " + ", ".join(cyclical_columns))
            df = df.drop(columns=cyclical_columns)

    if config.add_lag_features:
        _log("Adding lag features")
        lag_columns = _existing_columns(df, config.lag_columns)
        df = _add_lag_features(df, lag_columns, config.lag_counts)

    if config.add_rolling_features:
        _log("Adding rolling features")
        rolling_columns = _existing_columns(df, config.rolling_columns)
        df = _add_rolling_features(
            df,
            rolling_columns,
            config.rolling_windows,
            config.rolling_aggregation,
        )

    if config.add_expanding_features:
        _log("Adding expanding features")
        expanding_columns = _existing_columns(df, config.expanding_columns)
        df = _add_expanding_features(
            df,
            expanding_columns,
            config.expanding_min_periods,
            config.expanding_aggregation,
        )

    if config.drop_columns:
        removable = [column for column in config.drop_columns if column in df.columns]
        if removable:
            _log("Dropping columns: " + ", ".join(removable))
            df = df.drop(columns=removable)

    added_columns = len(set(df.columns) - original_columns)
    removed_columns = len(original_columns - set(df.columns))
    _log(
        f"Feature processing complete: +{added_columns} columns, -{removed_columns} columns"
    )

    return df
