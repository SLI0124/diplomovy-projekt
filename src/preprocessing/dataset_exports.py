"""Dataset export helpers for year-range and single-year split outputs."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class DatasetExportConfig:
    """Configuration for writing derived dataset splits."""

    output_root: Path
    base_stem: str
    year_column: str = "year"
    range_anchor_year: int = 2013
    drop_export_columns: tuple[str, ...] = ()


def _log(message: str) -> None:
    print(f"[exports] {message}")


def _dataset_root(config: DatasetExportConfig) -> Path:
    return config.output_root / config.base_stem


def _ranges_dir(config: DatasetExportConfig, anchor_year: int, max_year: int) -> Path:
    return _dataset_root(config) / f"ranges_from_{anchor_year}_to_{max_year}"


def _single_years_dir(config: DatasetExportConfig) -> Path:
    return _dataset_root(config) / "single_years"


def _available_years(df: pd.DataFrame, year_column: str) -> list[int]:
    """Return sorted unique years present in the configured year column."""

    if year_column not in df.columns:
        raise ValueError(f"Column '{year_column}' is required for export splitting.")

    years = pd.to_numeric(df[year_column], errors="coerce")
    years = years.dropna().astype(int)
    values = sorted(years.unique().tolist())
    if not values:
        raise ValueError("No valid years found in year column.")
    return values


def save_variant_snapshot(df: pd.DataFrame, config: DatasetExportConfig) -> Path:
    """Save full dataframe snapshot under variant root with stable base filename."""

    root_dir = _dataset_root(config)
    root_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = root_dir / "merged_all_years_preprocessed.csv"
    drop_columns = [
        column for column in config.drop_export_columns if column in df.columns
    ]
    output_df = df.drop(columns=drop_columns) if drop_columns else df
    _log(f"Saving variant snapshot: {snapshot_path}")
    output_df.to_csv(snapshot_path, index=False)
    return snapshot_path


def save_run_parameters(
    parameters: dict[str, Any],
    config: DatasetExportConfig,
) -> Path:
    """Save run parameter manifest for downstream modules to parse."""

    root_dir = _dataset_root(config)
    root_dir.mkdir(parents=True, exist_ok=True)
    params_path = root_dir / "run_params.json"
    _log(f"Saving run parameters: {params_path}")
    params_path.write_text(json.dumps(parameters, indent=2), encoding="utf-8")
    return params_path


def export_split_datasets(
    df: pd.DataFrame,
    config: DatasetExportConfig,
) -> dict[str, Any]:
    """Export cumulative train ranges and per-year datasets.

    Range exports intentionally stop at `(max_year - 1)` so the most recent
    year can be used as a holdout/test period.
    """

    years = _available_years(df, config.year_column)
    year_min, year_max = years[0], years[-1]

    details: dict[str, Any] = {
        "available_year_min": year_min,
        "available_year_max": year_max,
        "year_ranges": [],
        "single_years": [],
    }

    root_dir = _dataset_root(config)
    root_dir.mkdir(parents=True, exist_ok=True)
    details["dataset_root"] = str(root_dir)

    anchor = max(config.range_anchor_year, year_min)
    ranges_dir = _ranges_dir(config, anchor, year_max)
    ranges_dir.mkdir(parents=True, exist_ok=True)
    details["ranges_dir"] = str(ranges_dir)

    train_end_year = year_max - 1
    details["range_max_train_year"] = int(train_end_year)
    _log(
        f"Exporting cumulative ranges from {anchor} to {train_end_year} into {ranges_dir}"
    )
    for end_year in range(anchor, train_end_year + 1):
        mask = (df[config.year_column] >= anchor) & (df[config.year_column] <= end_year)
        sliced = df.loc[mask]
        if sliced.empty:
            continue
        drop_columns = [
            column for column in config.drop_export_columns if column in sliced.columns
        ]
        if drop_columns:
            sliced = sliced.drop(columns=drop_columns)
        filename = f"range_{anchor}_{end_year}.csv"
        path = ranges_dir / filename
        sliced.to_csv(path, index=False)
        details["year_ranges"].append(
            {
                "start_year": anchor,
                "end_year": end_year,
                "rows": int(len(sliced)),
                "path": str(path),
            }
        )

    single_years_dir = _single_years_dir(config)
    single_years_dir.mkdir(parents=True, exist_ok=True)
    details["single_years_dir"] = str(single_years_dir)
    _log(
        f"Exporting single-year files from {year_min} to {year_max} into {single_years_dir}"
    )
    for year in range(year_min, year_max + 1):
        mask = df[config.year_column] == year
        sliced = df.loc[mask]
        if sliced.empty:
            continue
        drop_columns = [
            column for column in config.drop_export_columns if column in sliced.columns
        ]
        if drop_columns:
            sliced = sliced.drop(columns=drop_columns)
        filename = f"year_{year}.csv"
        path = single_years_dir / filename
        sliced.to_csv(path, index=False)
        details["single_years"].append(
            {
                "year": year,
                "rows": int(len(sliced)),
                "path": str(path),
            }
        )

    _log(
        f"Export complete: {len(details['year_ranges'])} ranges, {len(details['single_years'])} single-year files"
    )

    return details
