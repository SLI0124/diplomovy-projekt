"""Compact preprocessing pipeline for merged gas dataset.

Design goals:
- one config object as schema + thresholds,
- deterministic rule masking before model-based fill,
- DecisionTreeRegressor imputation with simple time features,
- concise QA report focused on final data validity.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor


@dataclass(frozen=True)
class PreprocessingConfig:
    """All tunable settings and schema in one place."""

    time_cols: tuple[str, str, str, str] = ("year", "month", "day", "hour")
    consumption_cols: tuple[str, ...] = (
        "consumption_gasnet",
        "consumption_jmpnet",
        "consumption_smpnet",
        "consumption_vcpnet",
    )
    consumption_total_col: str = "consumption_total"
    price_cols: tuple[str, ...] = (
        "traded_volume_mwh",
        "weighted_avg_price_eur_mwh",
        "min_price_eur_mwh",
        "max_price_eur_mwh",
    )
    weather_cols: tuple[str, ...] = (
        "temperature_2m",
        "relative_humidity_2m",
        "pressure_msl",
        "surface_pressure",
        "dew_point_2m",
        "cloud_cover",
        "wind_speed_10m",
        "wind_direction_10m",
        "apparent_temperature",
        "wind_gusts_10m",
        "precipitation",
        "snowfall",
        "snow_depth",
    )
    consumption_upper_bound: float = 800_000.0
    max_price_upper_bound: float = 500.0
    tree_max_depth: int = 14
    tree_min_samples_leaf: int = 20
    tree_min_train_rows: int = 300
    tree_random_state: int = 42


@dataclass
class PreprocessingResult:
    """Pipeline outputs: cleaned dataframe and compact report."""

    dataframe: pd.DataFrame
    report: dict[str, Any]


def _log(step: str, message: str) -> None:
    """Print one-line progress updates for long preprocessing runs."""
    print(f"[{step}] {message}")


def _build_timestamp(df: pd.DataFrame, cfg: PreprocessingConfig) -> pd.DatetimeIndex:
    """Build a validated hourly timestamp index from configured time columns.

    Data is assumed sorted already; this function only validates and constructs
    a reliable datetime index for grouping and temporal features.
    """
    timestamp = pd.to_datetime(df[list(cfg.time_cols)], errors="coerce")
    if timestamp.isna().any():
        raise ValueError("Invalid values found in time columns.")
    return pd.DatetimeIndex(timestamp)


def _calendar_features(index: pd.DatetimeIndex) -> pd.DataFrame:
    """Create compact calendar features used by tree-based imputers."""
    iso = index.isocalendar()
    return pd.DataFrame(
        {
            "_month": index.month,
            "_day": index.day,
            "_hour": index.hour,
            "_dow": index.dayofweek,
            "_doy": index.dayofyear,
            "_week": iso.week.astype(int),
        },
        index=index,
    )


def _tree_fill(
    df: pd.DataFrame,
    target: str,
    features: list[str],
    cfg: PreprocessingConfig,
) -> tuple[int, int]:
    """Fill missing values in one target column with DecisionTreeRegressor.

    Returns tuple `(missing_before_tree, missing_after_tree)`.
    If training data is too small/flat, function exits fast and leaves fallback
    handling to downstream time-aware fills.
    """
    target_series = pd.to_numeric(df[target], errors="coerce")
    missing = target_series.isna()
    before = int(missing.sum())
    if before == 0:
        return 0, 0

    usable = [column for column in features if column in df.columns]
    if not usable:
        return before, before

    x = df[usable].apply(pd.to_numeric, errors="coerce")
    x = x.replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)

    # Train only on rows where target exists and all features are valid.
    train = (~missing) & x.notna().all(axis=1)
    if int(train.sum()) < cfg.tree_min_train_rows:
        return before, before

    y = target_series.loc[train]
    if y.nunique(dropna=True) <= 1:
        return before, before

    model = DecisionTreeRegressor(
        max_depth=cfg.tree_max_depth,
        min_samples_leaf=cfg.tree_min_samples_leaf,
        random_state=cfg.tree_random_state,
    )
    model.fit(x.loc[train], y)

    predict = missing & x.notna().all(axis=1)
    if predict.any():
        df.loc[predict, target] = model.predict(x.loc[predict])

    after = int(pd.to_numeric(df[target], errors="coerce").isna().sum())
    return before, after


def _fill_time_median(series: pd.Series, index: pd.DatetimeIndex) -> pd.Series:
    """Time-aware fallback fill using month-hour median, then forward/backward fill."""
    med = series.groupby([index.month, index.hour]).transform("median")
    return series.fillna(med).ffill().bfill()


def _impute_hourly_group(
    df: pd.DataFrame,
    targets: list[str],
    base_features: list[str],
    index: pd.DatetimeIndex,
    cfg: PreprocessingConfig,
    enforce_positive: bool,
    upper_clip: float | None,
) -> dict[str, dict[str, int]]:
    """Impute hourly columns (consumption/weather) using tree + time fallback.

    Workflow per target:
    1) tree fill from sibling + calendar + optional base features,
    2) month-hour median fallback,
    3) optional positivity/upper-bound enforcement.
    """
    stats: dict[str, dict[str, int]] = {}
    time_features = _calendar_features(index)
    for column in time_features.columns:
        df[column] = time_features[column]

    for target in targets:
        sibling = [column for column in targets if column != target]
        features = sibling + base_features + list(time_features.columns)

        before_tree, after_tree = _tree_fill(df, target, features, cfg)

        # Fallback keeps continuity when tree cannot fill every row.
        col = pd.to_numeric(df[target], errors="coerce")
        col = _fill_time_median(col, index)
        if enforce_positive:
            col = col.mask(col <= 0)
            col = _fill_time_median(col, index)
        if upper_clip is not None:
            col = col.clip(upper=upper_clip)

        df[target] = col
        stats[target] = {
            "nan_before_tree": before_tree,
            "nan_after_tree": after_tree,
            "nan_after_fill": int(col.isna().sum()),
        }

    df.drop(columns=list(time_features.columns), inplace=True)
    return stats


def _enforce_price_rules(daily: pd.DataFrame, cfg: PreprocessingConfig) -> pd.DataFrame:
    """Apply business constraints on daily price columns after tree imputation."""
    min_col = "min_price_eur_mwh"
    max_col = "max_price_eur_mwh"
    avg_col = "weighted_avg_price_eur_mwh"

    if min_col in daily.columns and max_col in daily.columns:
        daily[max_col] = np.maximum(daily[max_col], daily[min_col])
    if (
        min_col in daily.columns
        and max_col in daily.columns
        and avg_col in daily.columns
    ):
        daily[avg_col] = daily[avg_col].clip(lower=daily[min_col], upper=daily[max_col])
    if max_col in daily.columns:
        daily[max_col] = daily[max_col].clip(upper=cfg.max_price_upper_bound)

    month = pd.DatetimeIndex(daily.index).month
    for column in daily.columns:
        values = pd.to_numeric(daily[column], errors="coerce").mask(lambda s: s <= 0)
        values = (
            values.fillna(values.groupby(month).transform("median")).ffill().bfill()
        )
        daily[column] = values
    return daily


def _impute_price_daily(
    df: pd.DataFrame,
    price_cols: list[str],
    index: pd.DatetimeIndex,
    cfg: PreprocessingConfig,
) -> dict[str, Any]:
    """Impute prices at daily level and broadcast back to hourly rows.

    Price source is daily and replicated 24x; this preserves that structure.
    """
    date = index.floor("D")
    # Price is conceptually daily; aggregate first, then broadcast to hours.
    daily = df[price_cols].groupby(date).median()
    daily_index = pd.DatetimeIndex(daily.index)

    calendar = _calendar_features(daily_index)
    for column in calendar.columns:
        daily[column] = calendar[column]

    stats: dict[str, Any] = {"daily_rows": int(len(daily)), "columns": {}}
    for target in price_cols:
        sibling = [column for column in price_cols if column != target]
        features = sibling + ["_month", "_day", "_dow", "_doy", "_week"]
        before_tree, after_tree = _tree_fill(daily, target, features, cfg)
        stats["columns"][target] = {
            "daily_nan_before_tree": before_tree,
            "daily_nan_after_tree": after_tree,
        }

    daily = _enforce_price_rules(daily, cfg)
    daily = daily[price_cols]

    for target in price_cols:
        lookup = daily[target].to_dict()
        # Every hour of the same day receives the same cleaned daily value.
        df[target] = pd.Series(
            [lookup.get(day) for day in date], index=df.index, dtype=float
        )
        stats["columns"][target]["hourly_nan_after_broadcast"] = int(
            df[target].isna().sum()
        )

    return stats


def _mask_invalid(
    df: pd.DataFrame,
    consumption_cols: list[str],
    total_col: str | None,
    price_cols: list[str],
    cfg: PreprocessingConfig,
) -> dict[str, int]:
    """Mask rule-breaking values to NaN before imputation starts."""
    mask_counts: dict[str, int] = {}

    for column in consumption_cols:
        series = pd.to_numeric(df[column], errors="coerce")
        bad = series.isna() | (series <= 0) | (series > cfg.consumption_upper_bound)
        df[column] = series.mask(bad)
        mask_counts[column] = int(bad.sum())

    if total_col is not None:
        series = pd.to_numeric(df[total_col], errors="coerce")
        bad = series.isna() | (series <= 0)
        df[total_col] = series.mask(bad)
        mask_counts[total_col] = int(bad.sum())

    for column in price_cols:
        series = pd.to_numeric(df[column], errors="coerce")
        bad = series.isna() | (series == 0)
        if column == "max_price_eur_mwh":
            bad = bad | (series > cfg.max_price_upper_bound)
        df[column] = series.mask(bad)
        mask_counts[column] = int(bad.sum())

    return mask_counts


def _compact_report(
    df: pd.DataFrame,
    index: pd.DatetimeIndex,
    cfg: PreprocessingConfig,
    masks: dict[str, int],
    imputation: dict[str, Any],
    consumption_cols: list[str],
    total_col: str | None,
    price_cols: list[str],
    recomputed_total_rows: int,
) -> dict[str, Any]:
    """Build a compact QA report focused on final data quality and rule checks."""
    missing = int(df.isna().sum().sum())
    violations = {
        "consumption_non_positive": (
            int((df[consumption_cols] <= 0).sum().sum()) if consumption_cols else 0
        ),
        "consumption_above_upper": (
            int((df[consumption_cols] > cfg.consumption_upper_bound).sum().sum())
            if consumption_cols
            else 0
        ),
        "price_zero": int((df[price_cols] == 0).sum().sum()) if price_cols else 0,
        "max_price_above_upper": (
            int((df["max_price_eur_mwh"] > cfg.max_price_upper_bound).sum())
            if "max_price_eur_mwh" in df.columns
            else 0
        ),
    }

    price_daily_unique_max: dict[str, int] = {}
    if price_cols:
        daily_unique = (
            df[price_cols].groupby(index.floor("D")).nunique(dropna=False).max()
        )
        price_daily_unique_max = {
            str(column): int(value) for column, value in daily_unique.items()
        }

    return {
        "meta": {
            "rows": int(len(df)),
            "start": index.min().isoformat(),
            "end": index.max().isoformat(),
        },
        "config": {
            "consumption_upper_bound": cfg.consumption_upper_bound,
            "max_price_upper_bound": cfg.max_price_upper_bound,
            "tree_max_depth": cfg.tree_max_depth,
            "tree_min_samples_leaf": cfg.tree_min_samples_leaf,
            "tree_min_train_rows": cfg.tree_min_train_rows,
            "tree_random_state": cfg.tree_random_state,
        },
        "mask_counts": masks,
        "imputation": imputation,
        "checks": {
            "remaining_nan_total": missing,
            "rule_violations": violations,
            "price_daily_unique_max": price_daily_unique_max,
            "recomputed_total_rows": int(recomputed_total_rows),
            "has_total_column": bool(total_col),
        },
    }


def preprocess_merged_dataframe(
    dataframe: pd.DataFrame,
    config: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    """Run preprocessing on merged dataframe.

    Steps:
    1) validate schema and convert numeric columns,
    2) mask invalid values,
    3) impute consumption/weather hourly and price daily,
    4) recompute consumption_total where needed,
    5) return cleaned data + compact report.
    """
    cfg = config or PreprocessingConfig()
    df = dataframe.copy()
    index = _build_timestamp(df, cfg)

    consumption_cols = [
        column for column in cfg.consumption_cols if column in df.columns
    ]
    if not consumption_cols:
        raise ValueError("No consumption columns found.")

    price_cols = [column for column in cfg.price_cols if column in df.columns]
    if not price_cols:
        raise ValueError("No price columns found.")

    weather_cols = [column for column in cfg.weather_cols if column in df.columns]
    total_col = (
        cfg.consumption_total_col if cfg.consumption_total_col in df.columns else None
    )

    numeric_cols = (
        consumption_cols
        + price_cols
        + weather_cols
        + ([total_col] if total_col else [])
    )
    for column in numeric_cols:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    _log("1/5", "Masking invalid values")
    component_before = df[consumption_cols].copy()
    total_before = df[total_col].copy() if total_col else None
    masks = _mask_invalid(df, consumption_cols, total_col, price_cols, cfg)

    _log("2/5", "Imputing consumption (DecisionTreeRegressor)")
    time_base = [
        column
        for column in ["day_of_week", "holiday", "before_holiday"]
        if column in df.columns
    ]
    consumption_stats = _impute_hourly_group(
        df,
        consumption_cols,
        base_features=time_base,
        index=index,
        cfg=cfg,
        enforce_positive=True,
        upper_clip=cfg.consumption_upper_bound,
    )

    _log("3/5", "Imputing price daily and broadcasting to hourly")
    price_stats = _impute_price_daily(df, price_cols, index, cfg)

    _log("4/5", "Filling weather NaNs")
    weather_stats = (
        _impute_hourly_group(
            df,
            weather_cols,
            base_features=time_base,
            index=index,
            cfg=cfg,
            enforce_positive=False,
            upper_clip=None,
        )
        if weather_cols
        else {}
    )

    _log("5/5", "Recomputing consumption total and final checks")
    recomputed_total_rows = 0
    if total_col is not None:
        if total_before is None:
            raise ValueError(
                "Unexpected missing total series while total column exists."
            )
        component_after = df[consumption_cols].copy()
        # Row changed if any component value differs after cleaning/imputation.
        changed = pd.Series(
            (
                ~np.isclose(
                    component_before.to_numpy(dtype=float),
                    component_after.to_numpy(dtype=float),
                    atol=1e-9,
                    rtol=1e-9,
                    equal_nan=True,
                )
            ).any(axis=1),
            index=component_before.index,
        )
        total_before_numeric = pd.to_numeric(total_before, errors="coerce")
        total_invalid_before = total_before_numeric.isna() | (total_before_numeric <= 0)
        # Recompute total only where components changed or original total was invalid.
        recompute_mask = changed | total_invalid_before
        df.loc[recompute_mask, total_col] = (
            df[consumption_cols].sum(axis=1).loc[recompute_mask]
        )
        recomputed_total_rows = int(recompute_mask.sum())

    report = _compact_report(
        df=df,
        index=index,
        cfg=cfg,
        masks=masks,
        imputation={
            "consumption": consumption_stats,
            "price": price_stats,
            "weather": weather_stats,
        },
        consumption_cols=consumption_cols,
        total_col=total_col,
        price_cols=price_cols,
        recomputed_total_rows=recomputed_total_rows,
    )

    # Keep original column order exactly as input.
    output = df[
        [column for column in dataframe.columns if column in df.columns]
    ].reset_index(drop=True)
    return PreprocessingResult(dataframe=output, report=report)


def preprocess_merged_csv(
    input_path: str | Path,
    output_path: str | Path,
    config: PreprocessingConfig | None = None,
) -> PreprocessingResult:
    """Load merged CSV, run preprocessing, and write cleaned CSV."""
    source = Path(input_path)
    destination = Path(output_path)

    result = preprocess_merged_dataframe(pd.read_csv(source), config=config)
    destination.parent.mkdir(parents=True, exist_ok=True)
    result.dataframe.to_csv(destination, index=False)
    return result
