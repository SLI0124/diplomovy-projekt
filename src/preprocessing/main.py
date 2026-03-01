from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd
from dataset_exports import (
    DatasetExportConfig,
    export_split_datasets,
    save_variant_snapshot,
)
from temporal_features import TemporalFeatureConfig, apply_temporal_features
from value_cleaning import preprocess_merged_csv


def _parse_csv_columns(value: str) -> tuple[str, ...]:
    """Parse comma-separated column names into a normalized tuple."""

    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def _parse_csv_ints(value: str) -> tuple[int, ...]:
    """Parse comma-separated positive integers used by lag/window options."""

    if not value:
        return ()

    parsed: list[int] = []
    for raw in value.split(","):
        token = raw.strip()
        if not token:
            continue
        number = int(token)
        if number < 1:
            raise argparse.ArgumentTypeError("All integer list values must be >= 1.")
        parsed.append(number)

    return tuple(parsed)


def _sort_by_time(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Sort rows by reconstructed timestamp when time columns are available."""

    required = ["year", "month", "day", "hour"]
    if not all(column in dataframe.columns for column in required):
        return dataframe.copy()

    timestamp = pd.to_datetime(dataframe[required], errors="coerce")
    if timestamp.isna().any():
        return dataframe.copy()

    sorted_df = dataframe.copy()
    sorted_df["__timestamp"] = timestamp
    sorted_df = sorted_df.sort_values("__timestamp").drop(columns=["__timestamp"])
    sorted_df = sorted_df.reset_index(drop=True)
    return sorted_df


def _build_temporal_config(args: argparse.Namespace) -> TemporalFeatureConfig:
    """Map CLI arguments to TemporalFeatureConfig."""

    return TemporalFeatureConfig(
        add_cyclical=args.add_cyclical,
        cyclical_columns=_parse_csv_columns(args.cyclical_columns),
        drop_cyclical_source_columns=args.drop_cyclical_source_columns,
        add_lag_features=args.add_lag_features,
        lag_columns=_parse_csv_columns(args.lag_columns),
        lag_counts=_parse_csv_ints(args.lag_counts),
        add_rolling_features=args.add_rolling_features,
        rolling_columns=_parse_csv_columns(args.rolling_columns),
        rolling_windows=_parse_csv_ints(args.rolling_windows),
        rolling_aggregation=args.rolling_aggregation,
        add_expanding_features=args.add_expanding_features,
        expanding_columns=_parse_csv_columns(args.expanding_columns),
        expanding_min_periods=args.expanding_min_periods,
        expanding_aggregation=args.expanding_aggregation,
        drop_columns=_parse_csv_columns(args.drop_columns),
    )


def _sanitize_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    token = re.sub(r"-+", "-", token).strip("-")
    return token or "x"


def _build_feature_variant_stem(args: argparse.Namespace) -> str:
    """Build readable folder stem from enabled feature parameters.

    This keeps derived dataset folders self-descriptive without additional
    metadata lookup.
    """

    tokens: list[str] = []

    drop_columns = _parse_csv_columns(args.drop_columns)
    if drop_columns:
        tokens.append(
            f"drop-{'-'.join(_sanitize_token(column) for column in drop_columns)}"
        )

    if args.add_cyclical:
        cyclical = _parse_csv_columns(args.cyclical_columns)
        cyclical_token = "-".join(_sanitize_token(column) for column in cyclical)
        source_token = (
            "src-dropped" if args.drop_cyclical_source_columns else "src-kept"
        )
        tokens.append(f"cyc-{cyclical_token}-{source_token}")

    if args.add_lag_features:
        lag_columns = _parse_csv_columns(args.lag_columns)
        lag_counts = _parse_csv_ints(args.lag_counts)
        tokens.append(
            "lag-"
            + "-".join(_sanitize_token(column) for column in lag_columns)
            + "-"
            + "-".join(str(value) for value in lag_counts)
        )

    if args.add_rolling_features:
        rolling_columns = _parse_csv_columns(args.rolling_columns)
        rolling_windows = _parse_csv_ints(args.rolling_windows)
        tokens.append(
            "roll-"
            + "-".join(_sanitize_token(column) for column in rolling_columns)
            + "-"
            + "-".join(str(value) for value in rolling_windows)
            + f"-{args.rolling_aggregation}"
        )

    if args.add_expanding_features:
        expanding_columns = _parse_csv_columns(args.expanding_columns)
        tokens.append(
            "exp-"
            + "-".join(_sanitize_token(column) for column in expanding_columns)
            + f"-min{args.expanding_min_periods}-{args.expanding_aggregation}"
        )

    return "__".join(tokens) if tokens else "base"


def _feature_flags_enabled(args: argparse.Namespace) -> bool:
    """Return True when any feature-engineering flag is enabled."""

    return any(
        [
            args.add_cyclical,
            args.add_lag_features,
            args.add_rolling_features,
            args.add_expanding_features,
            bool(_parse_csv_columns(args.drop_columns)),
        ]
    )


def parse_args() -> argparse.Namespace:
    """Define and parse preprocessing CLI arguments."""

    default_input = (
        Path(__file__).resolve().parents[2]
        / "data/processed/merged/merged_all_years.csv"
    ).resolve()
    default_output = (
        Path(__file__).resolve().parents[2]
        / "data/preprocessed/merged_all_years_preprocessed.csv"
    ).resolve()

    parser = argparse.ArgumentParser(
        description="Run preprocessing for merged gas dataset."
    )
    parser.add_argument("--input", type=Path, default=default_input)
    parser.add_argument("--output", type=Path, default=default_output)
    parser.add_argument(
        "--report",
        type=Path,
        default=None,
        help="Optional path to save JSON report. Defaults to output path with .report.json suffix.",
    )

    parser.add_argument(
        "--drop-columns",
        type=str,
        default="",
        help="Comma-separated list of columns to drop after feature generation.",
    )

    parser.add_argument(
        "--add-cyclical",
        action="store_true",
        help="Add cyclical sin/cos features for datetime columns.",
    )
    parser.add_argument(
        "--cyclical-columns",
        type=str,
        default="hour,month,day_of_week",
        help="Comma-separated source columns for cyclical features.",
    )
    parser.add_argument(
        "--drop-cyclical-source-columns",
        action="store_true",
        help="Drop source datetime columns after creating cyclical features.",
    )

    parser.add_argument(
        "--add-lag-features",
        action="store_true",
        help="Add lag features for selected columns.",
    )
    parser.add_argument(
        "--lag-columns",
        type=str,
        default="consumption_total",
        help="Comma-separated columns for lag features.",
    )
    parser.add_argument(
        "--lag-counts",
        type=str,
        default="1,24,168",
        help="Comma-separated lag offsets (hours) used for lag features.",
    )

    parser.add_argument(
        "--add-rolling-features",
        action="store_true",
        help="Add rolling window features using past values only.",
    )
    parser.add_argument(
        "--rolling-columns",
        type=str,
        default="consumption_total",
        help="Comma-separated columns for rolling features.",
    )
    parser.add_argument(
        "--rolling-windows",
        type=str,
        default="24,168",
        help="Comma-separated rolling window sizes (hours).",
    )
    parser.add_argument(
        "--rolling-aggregation",
        choices=["mean", "sum", "both"],
        default="mean",
        help="Aggregation for rolling features.",
    )

    parser.add_argument(
        "--add-expanding-features",
        action="store_true",
        help="Add expanding window features using past values only.",
    )
    parser.add_argument(
        "--expanding-columns",
        type=str,
        default="consumption_total",
        help="Comma-separated columns for expanding features.",
    )
    parser.add_argument(
        "--expanding-min-periods",
        type=int,
        default=24,
        help="Minimum history length for expanding features.",
    )
    parser.add_argument(
        "--expanding-aggregation",
        choices=["mean", "sum", "both"],
        default="mean",
        help="Aggregation for expanding features.",
    )

    parser.add_argument(
        "--export-year-ranges",
        action="store_true",
        help="Export cumulative range files (anchor..year).",
    )
    parser.add_argument(
        "--range-anchor-year",
        type=int,
        default=2014,
        help="Anchor year for cumulative ranges.",
    )
    parser.add_argument(
        "--export-single-years",
        action="store_true",
        help="Export one file per year.",
    )
    parser.add_argument(
        "--single-year-start",
        type=int,
        default=2014,
        help="First year for single-year exports.",
    )
    parser.add_argument(
        "--single-year-end",
        type=int,
        default=None,
        help="Last year for single-year exports. Defaults to max available year.",
    )
    parser.add_argument(
        "--export-all-splits",
        action="store_true",
        help="Shortcut to enable both --export-year-ranges and --export-single-years.",
    )
    parser.add_argument(
        "--exports-subdir",
        type=str,
        default="splits",
        help="Subdirectory under output folder where additional datasets are saved in nested folders.",
    )

    return parser.parse_args()


def main() -> None:
    """Run preprocessing, optional feature engineering, and dataset exports."""

    args = parse_args()
    report_path = args.report or args.output.with_suffix(".report.json")
    export_year_ranges = args.export_year_ranges or args.export_all_splits
    export_single_years = args.export_single_years or args.export_all_splits

    result = preprocess_merged_csv(
        args.input,
        args.output,
    )

    working_df = _sort_by_time(result.dataframe)
    feature_flags_enabled = _feature_flags_enabled(args)
    export_base_stem = "base"
    if feature_flags_enabled:
        temporal_config = _build_temporal_config(args)
        working_df = apply_temporal_features(working_df, temporal_config)
        export_base_stem = _build_feature_variant_stem(args)

    exports_root = args.output.parent / args.exports_subdir
    export_details: dict[str, Any] = {
        "exports_root": str(exports_root.resolve()),
        "variant_stem": export_base_stem,
        "variant_snapshot_saved": False,
        "year_ranges": [],
        "single_years": [],
    }

    should_export_derived = (
        export_year_ranges or export_single_years or feature_flags_enabled
    )
    if should_export_derived:
        export_config = DatasetExportConfig(
            output_root=exports_root,
            base_stem=export_base_stem,
            range_anchor_year=args.range_anchor_year,
            export_year_ranges=export_year_ranges,
            export_single_years=export_single_years,
            single_year_start=args.single_year_start,
            single_year_end=args.single_year_end,
        )

        if feature_flags_enabled:
            snapshot_path = save_variant_snapshot(working_df, export_config)
            export_details["variant_snapshot_saved"] = True
            export_details["variant_snapshot_path"] = str(snapshot_path)

        if export_year_ranges or export_single_years:
            split_stats = export_split_datasets(working_df, export_config)
            export_details.update(split_stats)

    result.report["derived_exports"] = export_details

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result.report, indent=2), encoding="utf-8")

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Report: {report_path}")
    print(f"Derived exports root: {export_details.get('exports_root')}")
    if export_details.get("variant_snapshot_saved"):
        print(f"Variant snapshot: {export_details.get('variant_snapshot_path')}")
    year_ranges_count = len(export_details.get("year_ranges", []))
    single_years_count = len(export_details.get("single_years", []))
    if year_ranges_count or single_years_count:
        print(f"Year ranges exported: {year_ranges_count}")
        print(f"Single-year files exported: {single_years_count}")
    print(json.dumps(result.report, indent=2))


if __name__ == "__main__":
    main()
