from __future__ import annotations

import argparse
import re
from typing import Any

import pandas as pd
from dataset_exports import DatasetExportConfig
from temporal_features import ColumnMethodSpec, TemporalFeatureConfig

_ALLOWED_CLIP_METHODS = {"quantile", "iqr", "absolute"}
_ALLOWED_TRANSFORM_METHODS = {"log1p", "sqrt", "yeo-johnson", "boxcox"}
_ALLOWED_SCALE_METHODS = {"standard", "minmax", "robust"}
_ALLOWED_PRESETS = {"scale", "scale-transform", "scale-transform-clip"}


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


def parse_method_specs(
    value: list[list[str]] | None,
    *,
    option_name: str,
    allowed_methods: set[str],
) -> tuple[ColumnMethodSpec, ...]:
    """Parse repeatable CLI specs shaped as `<columns> <method>`."""
    if not value:
        return ()

    specs: list[ColumnMethodSpec] = []
    for pair in value:
        if len(pair) != 2:
            raise argparse.ArgumentTypeError(
                f"{option_name} expects exactly two values: <columns> <method>."
            )

        columns = parse_csv_columns(pair[0])
        if not columns:
            raise argparse.ArgumentTypeError(f"{option_name} columns cannot be empty.")

        method = pair[1].strip().lower()
        if method not in allowed_methods:
            rendered = ", ".join(sorted(allowed_methods))
            raise argparse.ArgumentTypeError(
                f"Unsupported {option_name} method '{method}'. Allowed: {rendered}."
            )

        specs.append(ColumnMethodSpec(columns=columns, method=method))

    return tuple(specs)


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
    clip_specs = parse_method_specs(
        args.clip,
        option_name="--clip",
        allowed_methods=_ALLOWED_CLIP_METHODS,
    )
    transform_specs = parse_method_specs(
        args.transform,
        option_name="--transform",
        allowed_methods=_ALLOWED_TRANSFORM_METHODS,
    )
    scale_specs = parse_method_specs(
        args.scale,
        option_name="--scale",
        allowed_methods=_ALLOWED_SCALE_METHODS,
    )

    preset = args.preset
    if preset is not None and preset not in _ALLOWED_PRESETS:
        rendered = ", ".join(sorted(_ALLOWED_PRESETS))
        raise argparse.ArgumentTypeError(
            f"Unsupported --preset value '{preset}'. Allowed: {rendered}."
        )

    if preset and (clip_specs or transform_specs or scale_specs):
        raise argparse.ArgumentTypeError(
            "--preset cannot be combined with --clip/--transform/--scale in one run."
        )

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
        preset=preset,
        clip_specs=clip_specs,
        transform_specs=transform_specs,
        scale_specs=scale_specs,
        drop_columns=parse_csv_columns(args.drop_columns),
    )


def _sanitize_token(value: str) -> str:
    token = re.sub(r"[^a-zA-Z0-9]+", "-", value.strip().lower())
    token = re.sub(r"-+", "-", token).strip("-")
    return token or "x"


def build_feature_variant_stem(args: argparse.Namespace) -> str:
    """Build readable folder stem from enabled feature parameters."""
    tokens: list[str] = []

    if args.preset:
        tokens.append(f"preset-{_sanitize_token(args.preset)}")

    if args.clip:
        clip_specs = parse_method_specs(
            args.clip,
            option_name="--clip",
            allowed_methods=_ALLOWED_CLIP_METHODS,
        )
        for spec in clip_specs:
            tokens.append(f"clip-{_sanitize_token(spec.method)}-{len(spec.columns)}c")

    if args.transform:
        transform_specs = parse_method_specs(
            args.transform,
            option_name="--transform",
            allowed_methods=_ALLOWED_TRANSFORM_METHODS,
        )
        for spec in transform_specs:
            tokens.append(
                f"transform-{_sanitize_token(spec.method)}-{len(spec.columns)}c"
            )

    if args.scale:
        scale_specs = parse_method_specs(
            args.scale,
            option_name="--scale",
            allowed_methods=_ALLOWED_SCALE_METHODS,
        )
        for spec in scale_specs:
            tokens.append(f"scale-{_sanitize_token(spec.method)}-{len(spec.columns)}c")

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
            bool(args.preset),
            bool(args.clip),
            bool(args.transform),
            bool(args.scale),
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
) -> dict[str, Any]:
    """Build parse-friendly manifest of run parameters for downstream modules."""
    return {
        "schema": "preprocessing.run_params.v1",
        "variant_stem": export_config.base_stem,
        "features": {
            "enabled": bool(enabled_features),
            "preset": args.preset,
            "clip": [
                {
                    "columns": list(spec.columns),
                    "method": spec.method,
                }
                for spec in parse_method_specs(
                    args.clip,
                    option_name="--clip",
                    allowed_methods=_ALLOWED_CLIP_METHODS,
                )
            ],
            "transform": [
                {
                    "columns": list(spec.columns),
                    "method": spec.method,
                }
                for spec in parse_method_specs(
                    args.transform,
                    option_name="--transform",
                    allowed_methods=_ALLOWED_TRANSFORM_METHODS,
                )
            ],
            "scale": [
                {
                    "columns": list(spec.columns),
                    "method": spec.method,
                }
                for spec in parse_method_specs(
                    args.scale,
                    option_name="--scale",
                    allowed_methods=_ALLOWED_SCALE_METHODS,
                )
            ],
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
    }
