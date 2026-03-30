"""Temporal feature engineering utilities used by preprocessing CLI."""

from __future__ import annotations

from dataclasses import dataclass
from fnmatch import fnmatch

import numpy as np
import pandas as pd
from sklearn.preprocessing import (
    MinMaxScaler,
    PowerTransformer,
    RobustScaler,
    StandardScaler,
)


@dataclass(frozen=True)
class ColumnMethodSpec:
    """Pair a method name with one or more target columns."""

    columns: tuple[str, ...]
    method: str


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

    preset: str | None = None
    clip_specs: tuple[ColumnMethodSpec, ...] = ()
    transform_specs: tuple[ColumnMethodSpec, ...] = ()
    scale_specs: tuple[ColumnMethodSpec, ...] = ()

    drop_columns: tuple[str, ...] = ()


_CYCLICAL_PERIODS: dict[str, int] = {
    "hour": 24,
    "month": 12,
    "day_of_week": 7,
}

_TRANSFORM_METHODS = {"log1p", "sqrt", "yeo-johnson", "boxcox"}
_SCALE_METHODS = {"standard", "minmax", "robust"}
_CLIP_METHODS = {"quantile", "iqr", "absolute"}

_QUANTILE_LOWER = 0.01
_QUANTILE_UPPER = 0.99
_IQR_MULTIPLIER = 1.5

_ABSOLUTE_CLIP_BOUNDS: dict[str, tuple[float | None, float | None]] = {
    "precipitation": (0.0, 15.0),
    "snowfall": (0.0, 2.5),
    "snow_depth": (0.0, 0.2),
    "relative_humidity_2m": (0.0, 100.0),
    "cloud_cover": (0.0, 100.0),
    "wind_direction_10m": (0.0, 360.0),
}

_PRESET_GROUP_PATTERNS: dict[str, tuple[str, ...]] = {
    "consumption": (
        "consumption_gasnet",
        "consumption_jmpnet",
        "consumption_smpnet",
        "consumption_total",
        "consumption_vcpnet",
    ),
    "temperature": ("temperature_2m", "apparent_temperature"),
    "pressure": ("pressure_msl", "surface_pressure"),
    "humidity": ("relative_humidity_2m", "dew_point_2m"),
    "cyclical_bounded": ("cloud_cover", "wind_direction_10m"),
    "wind": ("wind_speed_10m", "wind_gusts_10m"),
    "precip": ("precipitation", "snowfall", "snow_depth"),
    "traded_volume": ("traded_volume_mwh",),
    "prices": ("*_price_eur_mwh",),
}


def _log(message: str) -> None:
    print(f"[features] {message}")


def _existing_columns(df: pd.DataFrame, columns: tuple[str, ...]) -> list[str]:
    return [column for column in columns if column in df.columns]


def _unique_existing_columns(df: pd.DataFrame, columns: list[str]) -> list[str]:
    seen: set[str] = set()
    existing: list[str] = []
    for column in columns:
        if column in df.columns and column not in seen:
            seen.add(column)
            existing.append(column)
    return existing


def _expand_columns_by_patterns(
    df: pd.DataFrame, patterns: tuple[str, ...]
) -> list[str]:
    matched: list[str] = []
    for column in df.columns:
        if any(fnmatch(column, pattern) for pattern in patterns):
            matched.append(str(column))
    return _unique_existing_columns(df, matched)


def _fit_transform_non_null(
    series: pd.Series,
    transformer: StandardScaler | MinMaxScaler | RobustScaler | PowerTransformer,
) -> pd.Series:
    non_null = series.notna()
    if int(non_null.sum()) == 0:
        return series

    values = series.loc[non_null].to_numpy(dtype=float).reshape(-1, 1)
    transformed = transformer.fit_transform(values).reshape(-1)
    output = series.copy()
    output.loc[non_null] = transformed
    return output


def _clip_quantile(series: pd.Series) -> pd.Series:
    lower = float(series.quantile(_QUANTILE_LOWER))
    upper = float(series.quantile(_QUANTILE_UPPER))
    return series.clip(lower=lower, upper=upper)


def _clip_iqr(series: pd.Series) -> pd.Series:
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    iqr = q3 - q1
    lower = q1 - _IQR_MULTIPLIER * iqr
    upper = q3 + _IQR_MULTIPLIER * iqr
    return series.clip(lower=lower, upper=upper)


def _clip_absolute(series: pd.Series, column: str) -> pd.Series:
    bounds = _ABSOLUTE_CLIP_BOUNDS.get(column)
    if bounds is None:
        raise ValueError(
            "Absolute clipping is not defined for column "
            f"'{column}'. Use quantile/iqr or add explicit bounds in preprocessing config."
        )
    lower, upper = bounds
    return series.clip(lower=lower, upper=upper)


def _apply_clip(df: pd.DataFrame, spec: ColumnMethodSpec) -> pd.DataFrame:
    if spec.method not in _CLIP_METHODS:
        raise ValueError(f"Unsupported clip method '{spec.method}'.")

    columns = _existing_columns(df, spec.columns)
    for column in columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        if spec.method == "quantile":
            numeric = _clip_quantile(numeric)
        elif spec.method == "iqr":
            numeric = _clip_iqr(numeric)
        else:
            numeric = _clip_absolute(numeric, column)
        df[column] = numeric
    return df


def _apply_transform(df: pd.DataFrame, spec: ColumnMethodSpec) -> pd.DataFrame:
    if spec.method not in _TRANSFORM_METHODS:
        raise ValueError(f"Unsupported transform method '{spec.method}'.")

    columns = _existing_columns(df, spec.columns)
    for column in columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        values = numeric.dropna()
        if values.empty:
            df[column] = numeric
            continue

        if spec.method == "log1p":
            if bool((values <= -1.0).any()):
                raise ValueError(
                    f"Transform log1p requires values > -1 for column '{column}'."
                )
            numeric.loc[numeric.notna()] = np.log1p(values)
        elif spec.method == "sqrt":
            if bool((values < 0.0).any()):
                raise ValueError(
                    f"Transform sqrt requires values >= 0 for column '{column}'."
                )
            numeric.loc[numeric.notna()] = np.sqrt(values)
        elif spec.method == "yeo-johnson":
            numeric = _fit_transform_non_null(
                numeric,
                PowerTransformer(method="yeo-johnson", standardize=False),
            )
        else:
            if bool((values <= 0.0).any()):
                raise ValueError(
                    "Transform boxcox requires strictly positive values for "
                    f"column '{column}'."
                )
            numeric = _fit_transform_non_null(
                numeric,
                PowerTransformer(method="box-cox", standardize=False),
            )

        df[column] = numeric
    return df


def _apply_scale(df: pd.DataFrame, spec: ColumnMethodSpec) -> pd.DataFrame:
    if spec.method not in _SCALE_METHODS:
        raise ValueError(f"Unsupported scale method '{spec.method}'.")

    scaler: StandardScaler | MinMaxScaler | RobustScaler
    if spec.method == "standard":
        scaler = StandardScaler()
    elif spec.method == "minmax":
        scaler = MinMaxScaler()
    else:
        scaler = RobustScaler()

    columns = _existing_columns(df, spec.columns)
    for column in columns:
        numeric = pd.to_numeric(df[column], errors="coerce")
        numeric = _fit_transform_non_null(numeric, scaler)
        df[column] = numeric
    return df


def _group_columns(df: pd.DataFrame, group_name: str) -> tuple[str, ...]:
    patterns = _PRESET_GROUP_PATTERNS[group_name]
    return tuple(_expand_columns_by_patterns(df, patterns))


def _preset_specs(df: pd.DataFrame, preset: str) -> tuple[
    tuple[ColumnMethodSpec, ...],
    tuple[ColumnMethodSpec, ...],
    tuple[ColumnMethodSpec, ...],
]:
    if preset not in {"scale", "scale-transform", "scale-transform-clip"}:
        raise ValueError(f"Unsupported preprocessing preset '{preset}'.")

    groups = {key: _group_columns(df, key) for key in _PRESET_GROUP_PATTERNS}

    clip_specs: list[ColumnMethodSpec] = []
    transform_specs: list[ColumnMethodSpec] = []
    scale_specs: list[ColumnMethodSpec] = []

    def add_spec(
        target: list[ColumnMethodSpec], columns: tuple[str, ...], method: str
    ) -> None:
        if columns:
            target.append(ColumnMethodSpec(columns=columns, method=method))

    # Preset 1: scaling only.
    add_spec(scale_specs, groups["consumption"], "robust")
    add_spec(scale_specs, groups["temperature"], "standard")
    add_spec(scale_specs, groups["pressure"], "standard")
    add_spec(scale_specs, groups["humidity"], "minmax")
    add_spec(scale_specs, groups["cyclical_bounded"], "minmax")
    add_spec(scale_specs, groups["wind"], "robust")
    add_spec(scale_specs, groups["precip"], "robust")
    add_spec(scale_specs, groups["traded_volume"], "robust")
    add_spec(scale_specs, groups["prices"], "robust")

    if preset in {"scale-transform", "scale-transform-clip"}:
        transform_specs = []
        scale_specs = []

        # Preset 2: scaling + transform.
        add_spec(transform_specs, groups["consumption"], "yeo-johnson")
        add_spec(scale_specs, groups["consumption"], "standard")

        add_spec(transform_specs, groups["temperature"], "yeo-johnson")
        add_spec(scale_specs, groups["temperature"], "standard")

        add_spec(scale_specs, groups["pressure"], "standard")

        add_spec(transform_specs, groups["humidity"], "yeo-johnson")
        add_spec(scale_specs, groups["humidity"], "standard")

        add_spec(scale_specs, groups["cyclical_bounded"], "minmax")

        add_spec(transform_specs, groups["wind"], "sqrt")
        add_spec(scale_specs, groups["wind"], "standard")

        add_spec(transform_specs, groups["precip"], "log1p")
        add_spec(scale_specs, groups["precip"], "robust")

        add_spec(transform_specs, groups["traded_volume"], "log1p")
        add_spec(scale_specs, groups["traded_volume"], "standard")

        add_spec(transform_specs, groups["prices"], "yeo-johnson")
        add_spec(scale_specs, groups["prices"], "standard")

    if preset == "scale-transform-clip":
        # Preset 3: clipping + scaling + transform.
        add_spec(clip_specs, groups["consumption"], "iqr")
        add_spec(clip_specs, groups["temperature"], "quantile")
        add_spec(clip_specs, groups["pressure"], "quantile")
        add_spec(clip_specs, groups["humidity"], "quantile")
        add_spec(clip_specs, groups["wind"], "iqr")
        add_spec(clip_specs, groups["precip"], "absolute")
        add_spec(clip_specs, groups["traded_volume"], "iqr")
        add_spec(clip_specs, groups["prices"], "iqr")

    return tuple(clip_specs), tuple(transform_specs), tuple(scale_specs)


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

    clip_specs = list(config.clip_specs)
    transform_specs = list(config.transform_specs)
    scale_specs = list(config.scale_specs)

    if config.preset:
        _log(f"Applying preset: {config.preset}")
        preset_clip, preset_transform, preset_scale = _preset_specs(df, config.preset)
        clip_specs = [*preset_clip, *clip_specs]
        transform_specs = [*preset_transform, *transform_specs]
        scale_specs = [*preset_scale, *scale_specs]

    for spec in clip_specs:
        _log(f"Applying clip method '{spec.method}'")
        df = _apply_clip(df, spec)

    for spec in transform_specs:
        _log(f"Applying transform method '{spec.method}'")
        df = _apply_transform(df, spec)

    for spec in scale_specs:
        _log(f"Applying scale method '{spec.method}'")
        df = _apply_scale(df, spec)

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
