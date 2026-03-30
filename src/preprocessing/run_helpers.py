from __future__ import annotations

import argparse
import re
from typing import Any

import pandas as pd
from dataset_exports import DatasetExportConfig
from temporal_features import TemporalFeatureConfig


def parse_csv_columns(value: str) -> tuple[str, ...]:
    """Parse comma-separated column names into a normalized tuple."""
    if not value:
        return ()
    return tuple(part.strip() for part in value.split(",") if part.strip())


def parse_csv_ints(value: str) -> tuple[int, ...]:
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


def _parse_scalar(value: str) -> Any:
    token = value.strip()
    lower = token.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    try:
        return int(token)
    except ValueError:
        pass
    try:
        return float(token)
    except ValueError:
        return token


def parse_column_rule_entry(
    value: str,
    *,
    stage_name: str,
) -> tuple[tuple[str, ...], str, dict[str, Any]]:
    """Parse one rule entry: <col1,col2>:<method>[:k=v,k2=v2]."""

    parts = value.split(":", 2)
    if len(parts) < 2:
        raise argparse.ArgumentTypeError(
            f"Invalid {stage_name} rule '{value}'. Use <columns>:<method>[:k=v,...]."
        )

    columns = parse_csv_columns(parts[0])
    if not columns:
        raise argparse.ArgumentTypeError(
            f"Invalid {stage_name} rule '{value}'. At least one column is required."
        )

    method = parts[1].strip().lower()
    if not method:
        raise argparse.ArgumentTypeError(
            f"Invalid {stage_name} rule '{value}'. Method cannot be empty."
        )

    params: dict[str, Any] = {}
    if len(parts) == 3 and parts[2].strip():
        for token in parts[2].split(","):
            item = token.strip()
            if not item:
                continue
            if "=" not in item:
                raise argparse.ArgumentTypeError(
                    f"Invalid parameter '{item}' in {stage_name} rule '{value}'. Use key=value."
                )
            key, raw_val = item.split("=", 1)
            key = key.strip()
            if not key:
                raise argparse.ArgumentTypeError(
                    f"Invalid parameter '{item}' in {stage_name} rule '{value}'."
                )
            params[key] = _parse_scalar(raw_val)

    return columns, method, params


def build_column_rules_from_cli(args: argparse.Namespace) -> dict[str, Any]:
    """Build normalized column-rules payload from CLI repeated rule options."""

    rules: dict[str, dict[str, dict[str, Any]]] = {
        "clip": {},
        "transform": {},
        "scale": {},
    }

    for entry in args.clip_rule:
        columns, method, params = parse_column_rule_entry(entry, stage_name="clip")
        for column in columns:
            rules["clip"][column] = {"method": method, **params}

    for entry in args.transform_rule:
        columns, method, params = parse_column_rule_entry(
            entry,
            stage_name="transform",
        )
        for column in columns:
            rules["transform"][column] = {"method": method, **params}

    for entry in args.scale_rule:
        columns, method, params = parse_column_rule_entry(entry, stage_name="scale")
        for column in columns:
            rules["scale"][column] = {"method": method, **params}

    return rules


def sort_by_time(dataframe: pd.DataFrame) -> pd.DataFrame:
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


def build_temporal_config(args: argparse.Namespace) -> TemporalFeatureConfig:
    """Map CLI arguments to TemporalFeatureConfig."""
    return TemporalFeatureConfig(
        add_cyclical=args.add_cyclical,
        cyclical_columns=parse_csv_columns(args.cyclical_columns),
        drop_cyclical_source_columns=args.drop_cyclical_source_columns,
        add_lag_features=args.add_lag_features,
        lag_columns=parse_csv_columns(args.lag_columns),
        lag_counts=parse_csv_ints(args.lag_counts),
        add_rolling_features=args.add_rolling_features,
        rolling_columns=parse_csv_columns(args.rolling_columns),
        rolling_windows=parse_csv_ints(args.rolling_windows),
        rolling_aggregation=args.rolling_aggregation,
        add_expanding_features=args.add_expanding_features,
        expanding_columns=parse_csv_columns(args.expanding_columns),
        expanding_min_periods=args.expanding_min_periods,
        expanding_aggregation=args.expanding_aggregation,
        drop_columns=parse_csv_columns(args.drop_columns),
    )


def _sanitize_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    token = re.sub(r"-+", "-", token).strip("-")
    return token or "x"


def build_feature_variant_stem(args: argparse.Namespace) -> str:
    """Build readable folder stem from enabled feature parameters."""
    tokens: list[str] = []

    drop_columns = parse_csv_columns(args.drop_columns)
    if drop_columns:
        tokens.append(
            f"drop-{'-'.join(_sanitize_token(column) for column in drop_columns)}"
        )

    if args.add_cyclical:
        cyclical = parse_csv_columns(args.cyclical_columns)
        cyclical_token = "-".join(_sanitize_token(column) for column in cyclical)
        source_token = (
            "src-dropped" if args.drop_cyclical_source_columns else "src-kept"
        )
        tokens.append(f"cyc-{cyclical_token}-{source_token}")

    if args.add_lag_features:
        lag_columns = parse_csv_columns(args.lag_columns)
        lag_counts = parse_csv_ints(args.lag_counts)
        tokens.append(
            "lag-"
            + "-".join(_sanitize_token(column) for column in lag_columns)
            + "-"
            + "-".join(str(value) for value in lag_counts)
        )

    if args.add_rolling_features:
        rolling_columns = parse_csv_columns(args.rolling_columns)
        rolling_windows = parse_csv_ints(args.rolling_windows)
        tokens.append(
            "roll-"
            + "-".join(_sanitize_token(column) for column in rolling_columns)
            + "-"
            + "-".join(str(value) for value in rolling_windows)
            + f"-{args.rolling_aggregation}"
        )

    if args.add_expanding_features:
        expanding_columns = parse_csv_columns(args.expanding_columns)
        tokens.append(
            "exp-"
            + "-".join(_sanitize_token(column) for column in expanding_columns)
            + f"-min{args.expanding_min_periods}-{args.expanding_aggregation}"
        )

    return "__".join(tokens) if tokens else "base"


def feature_flags_enabled(args: argparse.Namespace) -> bool:
    """Return True when any feature-engineering flag is enabled."""
    return any(
        [
            args.add_cyclical,
            args.add_lag_features,
            args.add_rolling_features,
            args.add_expanding_features,
            bool(parse_csv_columns(args.drop_columns)),
        ]
    )


def build_run_parameters_payload(
    args: argparse.Namespace,
    enabled_features: bool,
    export_config: DatasetExportConfig,
    column_rules: dict[str, Any],
    column_rules_enabled: bool,
    column_rules_report_filename: str | None,
) -> dict[str, Any]:
    """Build parse-friendly manifest of run parameters for downstream modules."""
    return {
        "schema": "preprocessing.run_params.v2",
        "variant_stem": export_config.base_stem,
        "features": {
            "enabled": bool(enabled_features),
            "drop_columns": list(parse_csv_columns(args.drop_columns)),
            "cyclical": {
                "enabled": bool(args.add_cyclical),
                "columns": list(parse_csv_columns(args.cyclical_columns)),
                "drop_source_columns": bool(args.drop_cyclical_source_columns),
            },
            "lag": {
                "enabled": bool(args.add_lag_features),
                "columns": list(parse_csv_columns(args.lag_columns)),
                "counts": list(parse_csv_ints(args.lag_counts)),
            },
            "rolling": {
                "enabled": bool(args.add_rolling_features),
                "columns": list(parse_csv_columns(args.rolling_columns)),
                "windows": list(parse_csv_ints(args.rolling_windows)),
                "aggregation": args.rolling_aggregation,
            },
            "expanding": {
                "enabled": bool(args.add_expanding_features),
                "columns": list(parse_csv_columns(args.expanding_columns)),
                "min_periods": int(args.expanding_min_periods),
                "aggregation": args.expanding_aggregation,
            },
        },
        "splits": {
            "range_anchor_year": int(export_config.range_anchor_year),
            "holdout_last_year": True,
            "drop_export_columns": list(export_config.drop_export_columns),
        },
        "column_rules": {
            "enabled": bool(column_rules_enabled),
            "preset": str(args.column_rules_preset),
            "rules": column_rules,
            "report_file": column_rules_report_filename,
        },
    }
