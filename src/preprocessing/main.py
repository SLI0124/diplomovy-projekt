from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from column_rules import apply_column_rules, load_column_rules
from dataset_exports import (
    DatasetExportConfig,
    export_split_datasets,
    save_run_parameters,
    save_variant_metadata,
    save_variant_snapshot,
)
from run_helpers import (
    build_column_rules_from_cli,
    build_feature_variant_stem,
    build_run_parameters_payload,
    build_temporal_config,
    feature_flags_enabled,
    sort_by_time,
)
from temporal_features import apply_temporal_features
from value_cleaning import preprocess_merged_csv


def _log(step: str, message: str) -> None:
    print(f"[{step}] {message}")


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
        "--print-report",
        action="store_true",
        help="Print full JSON report to console (default: off).",
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
        "--exports-subdir",
        type=str,
        default="splits",
        help="Subdirectory under output folder where additional datasets are saved in nested folders.",
    )
    parser.add_argument(
        "--column-rules-preset",
        choices=["none", "scale_only", "scale_transform", "scale_transform_clip"],
        default="none",
        help=(
            "Optional built-in preset for distribution-aware preprocessing rules. "
            "Can be combined with explicit rule flags; explicit rules override preset columns."
        ),
    )
    parser.add_argument(
        "--clip-rule",
        action="append",
        default=[],
        help=(
            "Repeatable clipping rule in format <columns>:<method>[:k=v,...]. "
            "Example: consumption_total:quantile:lower_q=0.005,upper_q=0.995"
        ),
    )
    parser.add_argument(
        "--transform-rule",
        action="append",
        default=[],
        help=(
            "Repeatable transform rule in format <columns>:<method>[:k=v,...]. "
            "Example: consumption_total,temperature_2m:box-cox"
        ),
    )
    parser.add_argument(
        "--scale-rule",
        action="append",
        default=[],
        help=(
            "Repeatable scaling rule in format <columns>:<method>[:k=v,...]. "
            "Example: consumption_total:robust"
        ),
    )

    return parser.parse_args()


def main() -> None:
    """Run preprocessing, optional feature engineering, and dataset exports."""

    args = parse_args()
    report_path = args.report or args.output.with_suffix(".report.json")
    _log("main", "Running base preprocessing")
    result = preprocess_merged_csv(
        args.input,
        args.output,
    )

    working_df = sort_by_time(result.dataframe)
    split_reference_df = sort_by_time(result.dataframe)
    enabled_features = feature_flags_enabled(args)
    column_rules = load_column_rules(
        build_column_rules_from_cli(args),
        preset=args.column_rules_preset,
    )
    column_rules_enabled = any(
        column_rules.get(stage) for stage in ("clip", "transform", "scale")
    )
    has_explicit_column_rules = bool(
        args.clip_rule or args.transform_rule or args.scale_rule
    )
    export_base_stem = "base"
    if enabled_features:
        _log("main", "Applying temporal feature engineering")
        temporal_config = build_temporal_config(args)
        working_df = apply_temporal_features(working_df, temporal_config)
        export_base_stem = build_feature_variant_stem(args)
    else:
        _log("main", "No temporal feature flags enabled; using base variant")

    if column_rules_enabled:
        if args.column_rules_preset != "none":
            column_rules_stem = args.column_rules_preset
        elif has_explicit_column_rules:
            column_rules_stem = "rules_custom"
        else:
            column_rules_stem = "rules"

        if export_base_stem == "base":
            export_base_stem = column_rules_stem
        else:
            export_base_stem = f"{export_base_stem}__{column_rules_stem}"

    column_rules_report: dict[str, Any] = {
        "enabled": False,
        "order": ["clip", "transform", "scale"],
        "clip": {},
        "transform": {},
        "scale": {},
        "skipped": [],
    }
    if column_rules_enabled:
        _log("main", "Applying column clip/transform/scale rules")
        column_rules_result = apply_column_rules(working_df, column_rules)
        working_df = column_rules_result.dataframe
        column_rules_report = column_rules_result.report

    exports_root = args.output.parent / args.exports_subdir
    export_year_column = "year"
    drop_export_columns: list[str] = []
    helper_year_column = "__split_year"
    if (
        export_year_column not in working_df.columns
        and "year" in split_reference_df.columns
    ):
        working_df[helper_year_column] = split_reference_df["year"].to_numpy()
        export_year_column = helper_year_column
        drop_export_columns.append(helper_year_column)

    export_details: dict[str, Any] = {
        "exports_root": str(exports_root.resolve()),
        "variant_stem": export_base_stem,
        "variant_snapshot_saved": False,
        "year_ranges": [],
        "single_years": [],
    }

    export_config = DatasetExportConfig(
        output_root=exports_root,
        base_stem=export_base_stem,
        year_column=export_year_column,
        range_anchor_year=2013,
        drop_export_columns=tuple(drop_export_columns),
    )

    column_rules_report_filename = (
        "column_rules_report.json" if column_rules_enabled else None
    )

    params_payload = build_run_parameters_payload(
        args=args,
        enabled_features=enabled_features,
        export_config=export_config,
        column_rules=column_rules,
        column_rules_enabled=column_rules_enabled,
        column_rules_report_filename=column_rules_report_filename,
    )
    params_path = save_run_parameters(params_payload, export_config)
    export_details["params_path"] = str(params_path)

    if column_rules_enabled and column_rules_report_filename is not None:
        column_rules_report_path = save_variant_metadata(
            payload=column_rules_report,
            config=export_config,
            filename=column_rules_report_filename,
        )
        export_details["column_rules_report_path"] = str(column_rules_report_path)

    if enabled_features:
        _log("main", "Saving feature-variant snapshot")
        snapshot_path = save_variant_snapshot(working_df, export_config)
        export_details["variant_snapshot_saved"] = True
        export_details["variant_snapshot_path"] = str(snapshot_path)

    _log("main", "Exporting year ranges and single-year datasets")
    split_stats = export_split_datasets(working_df, export_config)
    export_details.update(split_stats)

    result.report["derived_exports"] = export_details
    result.report["column_rules"] = column_rules_report

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
    if args.print_report:
        print(json.dumps(result.report, indent=2))


if __name__ == "__main__":
    main()
