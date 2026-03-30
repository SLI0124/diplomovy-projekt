from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import PowerTransformer


@dataclass(frozen=True)
class ColumnRulesResult:
    dataframe: pd.DataFrame
    report: dict[str, Any]


def _as_method_spec(spec: Any) -> dict[str, Any]:
    if isinstance(spec, str):
        return {"method": spec}
    if isinstance(spec, dict):
        method = str(spec.get("method", "")).strip().lower()
        if not method:
            raise ValueError(
                "Rule specification dictionary must include non-empty 'method'."
            )
        normalized = {"method": method}
        for key, value in spec.items():
            if key == "method":
                continue
            normalized[str(key)] = value
        return normalized
    raise ValueError(
        "Rule specification must be either string method name or dictionary with method."
    )


def _normalize_stage(stage_spec: Any) -> dict[str, dict[str, Any]]:
    if stage_spec is None:
        return {}
    if not isinstance(stage_spec, dict):
        raise ValueError("Rule stage payload must be a dictionary.")

    normalized: dict[str, dict[str, Any]] = {}
    for raw_column, raw_spec in stage_spec.items():
        column = str(raw_column).strip()
        if not column:
            continue
        normalized[column] = _as_method_spec(raw_spec)
    return normalized


def _normalize_payload(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        "clip": _normalize_stage(payload.get("clip")),
        "transform": _normalize_stage(payload.get("transform")),
        "scale": _normalize_stage(payload.get("scale")),
    }


def _preset_payload(preset: str) -> dict[str, Any]:
    if preset == "none":
        return {}

    scale_standard = {
        "temperature_2m": {"method": "standard"},
        "pressure_msl": {"method": "standard"},
        "surface_pressure": {"method": "standard"},
        "dew_point_2m": {"method": "standard"},
        "apparent_temperature": {"method": "standard"},
    }
    scale_robust = {
        "consumption_gasnet": {"method": "robust"},
        "consumption_jmpnet": {"method": "robust"},
        "consumption_smpnet": {"method": "robust"},
        "consumption_vcpnet": {"method": "robust"},
        "consumption_total": {"method": "robust"},
        "traded_volume_mwh": {"method": "robust"},
        "weighted_avg_price_eur_mwh": {"method": "robust"},
        "min_price_eur_mwh": {"method": "robust"},
        "max_price_eur_mwh": {"method": "robust"},
        "wind_speed_10m": {"method": "robust"},
        "wind_gusts_10m": {"method": "robust"},
        "precipitation": {"method": "robust"},
        "snowfall": {"method": "robust"},
        "snow_depth": {"method": "robust"},
    }
    scale_minmax = {
        "relative_humidity_2m": {"method": "minmax"},
        "cloud_cover": {"method": "minmax"},
        "wind_direction_10m": {"method": "minmax"},
    }

    if preset == "scale_only":
        return {
            "scale": {
                **scale_standard,
                **scale_robust,
                **scale_minmax,
            }
        }

    transform_rules = {
        "consumption_gasnet": {"method": "log1p"},
        "consumption_jmpnet": {"method": "log1p"},
        "consumption_smpnet": {"method": "log1p"},
        "consumption_vcpnet": {"method": "log1p"},
        "consumption_total": {"method": "log1p"},
        "traded_volume_mwh": {"method": "log1p"},
        "weighted_avg_price_eur_mwh": {"method": "log1p"},
        "min_price_eur_mwh": {"method": "log1p"},
        "max_price_eur_mwh": {"method": "log1p"},
        "wind_speed_10m": {"method": "log1p"},
        "wind_gusts_10m": {"method": "log1p"},
        "precipitation": {"method": "log1p"},
        "snowfall": {"method": "log1p"},
        "snow_depth": {"method": "log1p"},
        "relative_humidity_2m": {"method": "yeo-johnson"},
        "cloud_cover": {"method": "yeo-johnson"},
    }

    if preset == "scale_transform":
        return {
            "transform": transform_rules,
            "scale": {
                **scale_standard,
                **scale_robust,
                "relative_humidity_2m": {"method": "robust"},
                "cloud_cover": {"method": "robust"},
                "wind_direction_10m": {"method": "minmax"},
            },
        }

    if preset == "scale_transform_clip":
        return {
            "clip": {
                "consumption_gasnet": {
                    "method": "quantile",
                    "lower_q": 0.001,
                    "upper_q": 0.999,
                },
                "consumption_jmpnet": {
                    "method": "quantile",
                    "lower_q": 0.001,
                    "upper_q": 0.999,
                },
                "consumption_smpnet": {
                    "method": "quantile",
                    "lower_q": 0.001,
                    "upper_q": 0.999,
                },
                "consumption_vcpnet": {
                    "method": "quantile",
                    "lower_q": 0.001,
                    "upper_q": 0.999,
                },
                "consumption_total": {
                    "method": "quantile",
                    "lower_q": 0.001,
                    "upper_q": 0.999,
                },
                "traded_volume_mwh": {
                    "method": "quantile",
                    "lower_q": 0.001,
                    "upper_q": 0.999,
                },
                "weighted_avg_price_eur_mwh": {
                    "method": "quantile",
                    "lower_q": 0.001,
                    "upper_q": 0.999,
                },
                "min_price_eur_mwh": {
                    "method": "quantile",
                    "lower_q": 0.001,
                    "upper_q": 0.999,
                },
                "max_price_eur_mwh": {
                    "method": "quantile",
                    "lower_q": 0.001,
                    "upper_q": 0.999,
                },
                "wind_speed_10m": {
                    "method": "quantile",
                    "lower_q": 0.001,
                    "upper_q": 0.999,
                },
                "wind_gusts_10m": {
                    "method": "quantile",
                    "lower_q": 0.001,
                    "upper_q": 0.999,
                },
                "precipitation": {
                    "method": "quantile",
                    "lower_q": 0.0,
                    "upper_q": 0.999,
                },
                "snowfall": {
                    "method": "quantile",
                    "lower_q": 0.0,
                    "upper_q": 0.999,
                },
                "snow_depth": {
                    "method": "quantile",
                    "lower_q": 0.0,
                    "upper_q": 0.999,
                },
            },
            "transform": transform_rules,
            "scale": {
                **scale_standard,
                **scale_robust,
                "relative_humidity_2m": {"method": "robust"},
                "cloud_cover": {"method": "robust"},
                "wind_direction_10m": {"method": "minmax"},
            },
        }

    raise ValueError(
        f"Unsupported column rules preset '{preset}'. Supported: none, scale_only, scale_transform, scale_transform_clip"
    )


def load_column_rules(
    rules_payload: dict[str, Any] | None,
    *,
    preset: str = "none",
) -> dict[str, Any]:
    base = _normalize_payload(_preset_payload(preset))
    override = _normalize_payload(rules_payload or {})
    return {
        "clip": {**base.get("clip", {}), **override.get("clip", {})},
        "transform": {
            **base.get("transform", {}),
            **override.get("transform", {}),
        },
        "scale": {**base.get("scale", {}), **override.get("scale", {})},
    }


def _fit_clip_bounds(
    series: np.ndarray, method_spec: dict[str, Any]
) -> tuple[float, float]:
    method = method_spec["method"]
    finite = series[np.isfinite(series)]
    if finite.size == 0:
        return float("-inf"), float("inf")

    if method == "bounds":
        lower = method_spec.get("lower")
        upper = method_spec.get("upper")
        lo = float(lower) if lower is not None else float("-inf")
        hi = float(upper) if upper is not None else float("inf")
    elif method == "quantile":
        lower_q = float(method_spec.get("lower_q", 0.01))
        upper_q = float(method_spec.get("upper_q", 0.99))
        lo = float(np.quantile(finite, lower_q))
        hi = float(np.quantile(finite, upper_q))
    elif method == "iqr":
        iqr_k = float(method_spec.get("k", 1.5))
        q1 = float(np.quantile(finite, 0.25))
        q3 = float(np.quantile(finite, 0.75))
        iqr = q3 - q1
        lo = q1 - iqr_k * iqr
        hi = q3 + iqr_k * iqr
    else:
        raise ValueError(
            f"Unsupported clip method '{method}'. Supported: bounds, quantile, iqr"
        )

    if hi <= lo:
        return float("-inf"), float("inf")
    return lo, hi


def _apply_transform(
    series: np.ndarray, method_spec: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    method = method_spec["method"]
    finite = np.isfinite(series)
    out = series.copy()

    if not finite.any():
        return out, {"method": method, "applied": False, "reason": "no_finite_values"}

    params: dict[str, Any] = {"method": method, "applied": True}

    if method == "log1p":
        min_val = float(np.min(series[finite]))
        shift = float(max(0.0, -min_val))
        out[finite] = np.log1p(series[finite] + shift)
        params["shift"] = shift
        return out, params

    if method == "sqrt":
        min_val = float(np.min(series[finite]))
        shift = float(max(0.0, -min_val))
        out[finite] = np.sqrt(series[finite] + shift)
        params["shift"] = shift
        return out, params

    if method in {"yeo-johnson", "box-cox"}:
        transformed_input = series[finite].reshape(-1, 1)
        shift = 0.0
        if method == "box-cox":
            min_val = float(np.min(transformed_input))
            if min_val <= 0.0:
                shift = float(1.0 - min_val + 1e-6)
                transformed_input = transformed_input + shift

        transformer = PowerTransformer(method=method, standardize=False)
        transformed = transformer.fit_transform(transformed_input).reshape(-1)
        out[finite] = transformed
        params["shift"] = shift
        params["lambda"] = float(transformer.lambdas_[0])
        return out, params

    raise ValueError(
        f"Unsupported transform method '{method}'. Supported: log1p, sqrt, yeo-johnson, box-cox"
    )


def _apply_scale(
    series: np.ndarray, method_spec: dict[str, Any]
) -> tuple[np.ndarray, dict[str, Any]]:
    method = method_spec["method"]
    finite = np.isfinite(series)
    out = series.copy()

    if not finite.any():
        return out, {"method": method, "applied": False, "reason": "no_finite_values"}

    stats: dict[str, Any] = {"method": method, "applied": True}
    values = series[finite]

    if method == "standard":
        mean = float(np.mean(values))
        std = float(np.std(values))
        if std <= 1e-6:
            std = 1.0
        out[finite] = (values - mean) / std
        stats["mean"] = mean
        stats["std"] = std
        return out, stats

    if method == "minmax":
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        width = max_val - min_val
        if width <= 1e-6:
            width = 1.0
        out[finite] = (values - min_val) / width
        stats["min"] = min_val
        stats["max"] = max_val
        return out, stats

    if method == "robust":
        median = float(np.median(values))
        q1 = float(np.quantile(values, 0.25))
        q3 = float(np.quantile(values, 0.75))
        iqr = q3 - q1
        if iqr <= 1e-6:
            iqr = 1.0
        out[finite] = (values - median) / iqr
        stats["median"] = median
        stats["iqr"] = iqr
        return out, stats

    raise ValueError(
        f"Unsupported scale method '{method}'. Supported: standard, minmax, robust"
    )


def apply_column_rules(
    dataframe: pd.DataFrame,
    rules: dict[str, Any],
) -> ColumnRulesResult:
    out = dataframe.copy()
    report: dict[str, Any] = {
        "enabled": False,
        "order": ["clip", "transform", "scale"],
        "clip": {},
        "transform": {},
        "scale": {},
        "skipped": [],
    }

    if not any(rules.get(stage) for stage in ("clip", "transform", "scale")):
        return ColumnRulesResult(dataframe=out, report=report)

    report["enabled"] = True

    def _numeric_values_or_skip(column: str, stage: str) -> np.ndarray | None:
        series_raw = out[column]
        series_num = pd.to_numeric(series_raw, errors="coerce")
        if series_num.notna().sum() == 0 and series_raw.notna().sum() > 0:
            report["skipped"].append(
                {
                    "stage": stage,
                    "column": column,
                    "reason": "non_numeric_column",
                }
            )
            return None
        return series_num.to_numpy(dtype=np.float64)

    # Stage 1: clipping
    for column, method_spec in rules.get("clip", {}).items():
        if column not in out.columns:
            report["skipped"].append(
                {
                    "stage": "clip",
                    "column": column,
                    "reason": "missing_column",
                }
            )
            continue

        series = _numeric_values_or_skip(column=column, stage="clip")
        if series is None:
            continue
        lo, hi = _fit_clip_bounds(series, method_spec)
        clipped = np.clip(series, lo, hi)
        out[column] = clipped
        report["clip"][column] = {
            "method": method_spec["method"],
            "lower": lo,
            "upper": hi,
        }

    # Stage 2: transforms
    for column, method_spec in rules.get("transform", {}).items():
        if column not in out.columns:
            report["skipped"].append(
                {
                    "stage": "transform",
                    "column": column,
                    "reason": "missing_column",
                }
            )
            continue

        series = _numeric_values_or_skip(column=column, stage="transform")
        if series is None:
            continue
        transformed, params = _apply_transform(series, method_spec)
        out[column] = transformed
        report["transform"][column] = params

    # Stage 3: scaling
    for column, method_spec in rules.get("scale", {}).items():
        if column not in out.columns:
            report["skipped"].append(
                {
                    "stage": "scale",
                    "column": column,
                    "reason": "missing_column",
                }
            )
            continue

        series = _numeric_values_or_skip(column=column, stage="scale")
        if series is None:
            continue
        scaled, stats = _apply_scale(series, method_spec)
        out[column] = scaled
        report["scale"][column] = stats

    return ColumnRulesResult(dataframe=out, report=report)
