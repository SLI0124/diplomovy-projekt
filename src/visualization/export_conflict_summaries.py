#!/usr/bin/env python3
"""Export pre/post-conflict summaries for deep learning runs and SARIMAX.

The script collects all available 2022 `pre_conflict` and `post_conflict`
results from `data/results/deep_learning` and computes an equivalent split for
SARIMAX from point forecasts in `data/results/sarimax_stepup`.

Outputs are written to `data/data-exports/conflict_analysis/`:

- `conflict_2022_segments_all.csv`
  One row per run / model / segment (`all`, `pre_conflict`, `post_conflict`).
- `conflict_2022_summary_all.csv`
  One row per run / model with pre/post/all metrics and deltas.
- `conflict_2022_summary_latest.csv`
  Latest available row for each unique configuration and model.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

FOUNDATION_MODELS = {"chronos2", "granite", "moirai1"}
CUSTOM_MODELS = {"model_1", "model_2", "model_3"}


@dataclass(frozen=True)
class SegmentMetrics:
    n_points: int
    n_origins: int
    mae: float
    rmse: float
    mape: float
    smape: float
    r2: float


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RESULTS_DIR = DATA_DIR / "results"
SARIMAX_DIR = RESULTS_DIR / "sarimax_stepup"
DEEP_LEARNING_DIR = RESULTS_DIR / "deep_learning"
OUTPUT_DIR = DATA_DIR / "data-exports" / "conflict_analysis"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export pre/post-conflict summaries for chapter 6 discussion."
    )
    parser.add_argument(
        "--deep-learning-root",
        default=DEEP_LEARNING_DIR,
        help="Root directory with deep learning run folders.",
    )
    parser.add_argument(
        "--sarimax-root",
        default=SARIMAX_DIR,
        help="Directory with SARIMAX outputs.",
    )
    parser.add_argument(
        "--output-dir",
        default=OUTPUT_DIR,
        help="Directory where CSV summaries will be written.",
    )
    parser.add_argument(
        "--conflict-date",
        default="2022-02-24 00:00:00",
        help="Conflict split date in ISO format.",
    )
    parser.add_argument(
        "--test-year",
        default=2022,
        type=int,
        help="Test year for pre/post-conflict analysis.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def parse_run_timestamp(run_dir: Path) -> datetime | None:
    try:
        return datetime.strptime(run_dir.name, "%Y%m%d_%H%M%S")
    except ValueError:
        return None


def classify_family(models: Sequence[str]) -> str:
    model_set = set(models)
    if model_set and model_set <= FOUNDATION_MODELS:
        return "foundation"
    if model_set and model_set <= CUSTOM_MODELS:
        return "custom"
    return "unknown"


def to_float(value: str) -> float:
    return float(value) if value not in ("", None) else math.nan


def metrics_from_rows(rows: Sequence[dict]) -> dict[str, dict]:
    grouped: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in rows:
        if int(row["test_year"]) != 2022:
            continue
        model = row["model"]
        segment = row["segment"]
        grouped[model][segment] = row
    return grouped


def collect_deep_learning_segments(root: Path) -> list[dict]:
    records: list[dict] = []
    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue
        runtime_path = run_dir / "runtime_config.json"
        results_path = run_dir / "results.csv"
        if not runtime_path.exists() or not results_path.exists():
            continue

        config = load_json(runtime_path)
        family = classify_family(config.get("models", []))
        run_timestamp = parse_run_timestamp(run_dir)

        with results_path.open("r", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))

        grouped = metrics_from_rows(rows)
        for model, segments in grouped.items():
            for segment_name in ("all", "pre_conflict", "post_conflict"):
                if segment_name not in segments:
                    continue
                row = segments[segment_name]
                records.append(
                    {
                        "source": "deep_learning",
                        "family": family,
                        "model": model,
                        "mode": config.get("mode", ""),
                        "input_mode": config.get("training_input_mode", ""),
                        "train_epochs": config.get("train_epochs", ""),
                        "action": config.get("action", ""),
                        "run_dir": str(run_dir),
                        "run_name": run_dir.name,
                        "run_timestamp": (
                            run_timestamp.isoformat() if run_timestamp else ""
                        ),
                        "test_year": int(row["test_year"]),
                        "segment": segment_name,
                        "train_years": row.get("train_years", ""),
                        "n_windows": int(row["n_windows"]),
                        "n_points": int(row["n_points"]),
                        "mae": to_float(row["mae"]),
                        "mse": to_float(row["mse"]),
                        "rmse": math.sqrt(to_float(row["mse"])),
                        "mape": to_float(row["mape"]),
                        "smape": to_float(row["smape"]),
                        "r2": to_float(row["r2"]),
                        "note": (
                            "granite_one_shot_covariate_present_in_raw_results"
                            if model == "granite"
                            and config.get("mode") == "one-shot"
                            and config.get("training_input_mode") == "covariate"
                            else ""
                        ),
                    }
                )
    return records


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return math.nan
    return num / den


def compute_segment_metrics(rows: Sequence[dict]) -> SegmentMetrics:
    y_true = [float(row["y_true"]) for row in rows]
    y_pred = [float(row["y_pred"]) for row in rows]

    n_points = len(rows)
    origins = {row["origin_timestamp"] for row in rows}
    mae = sum(abs(a - b) for a, b in zip(y_true, y_pred, strict=False)) / n_points
    mse = sum((a - b) ** 2 for a, b in zip(y_true, y_pred, strict=False)) / n_points
    rmse = math.sqrt(mse)
    mape = (
        100.0
        * sum(abs(a - b) / abs(a) for a, b in zip(y_true, y_pred, strict=False))
        / n_points
    )
    smape = (
        100.0
        * sum(
            safe_div(2.0 * abs(a - b), abs(a) + abs(b))
            for a, b in zip(y_true, y_pred, strict=False)
        )
        / n_points
    )
    y_mean = sum(y_true) / n_points
    ss_res = sum((a - b) ** 2 for a, b in zip(y_true, y_pred, strict=False))
    ss_tot = sum((a - y_mean) ** 2 for a in y_true)
    r2 = 1.0 - safe_div(ss_res, ss_tot)

    return SegmentMetrics(
        n_points=n_points,
        n_origins=len(origins),
        mae=mae,
        rmse=rmse,
        mape=mape,
        smape=smape,
        r2=r2,
    )


def collect_sarimax_segments(
    root: Path, conflict_dt: datetime, test_year: int
) -> list[dict]:
    predictions_path = (
        root / "predictions_by_year" / f"sarimax_predictions_{test_year}.csv"
    )
    if not predictions_path.exists():
        raise FileNotFoundError(f"Missing SARIMAX predictions file: {predictions_path}")

    with predictions_path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    for row in rows:
        row["timestamp_dt"] = datetime.fromisoformat(row["timestamp"])
        row["origin_dt"] = datetime.fromisoformat(row["origin_timestamp"])

    segments = {
        "all": rows,
        "pre_conflict": [row for row in rows if row["timestamp_dt"] < conflict_dt],  # type: ignore
        "post_conflict": [row for row in rows if row["timestamp_dt"] >= conflict_dt],  # type: ignore
    }

    train_years = rows[0]["train_years"] if rows else ""
    records: list[dict] = []
    for segment_name, segment_rows in segments.items():
        metrics = compute_segment_metrics(segment_rows)
        records.append(
            {
                "source": "sarimax",
                "family": "sarimax",
                "model": "sarimax",
                "mode": "baseline",
                "input_mode": "covariate",
                "train_epochs": "",
                "action": "eval",
                "run_dir": str(root),
                "run_name": root.name,
                "run_timestamp": "",
                "test_year": test_year,
                "segment": segment_name,
                "train_years": train_years,
                "n_windows": metrics.n_origins,
                "n_points": metrics.n_points,
                "mae": metrics.mae,
                "mse": metrics.rmse**2,
                "rmse": metrics.rmse,
                "mape": metrics.mape,
                "smape": metrics.smape,
                "r2": metrics.r2,
                "note": "computed_from_point_predictions",
            }
        )
    return records


def summarise_pre_post(records: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, ...], dict[str, dict]] = defaultdict(dict)
    for row in records:
        key = (
            row["source"],
            row["family"],
            row["model"],
            row["mode"],
            row["input_mode"],
            str(row["train_epochs"]),
            row["action"],
            row["run_dir"],
        )
        grouped[key][row["segment"]] = row

    summaries: list[dict] = []
    for key, segments in grouped.items():
        if "pre_conflict" not in segments or "post_conflict" not in segments:
            continue
        source, family, model, mode, input_mode, train_epochs, action, run_dir = key
        pre = segments["pre_conflict"]
        post = segments["post_conflict"]
        all_row = segments.get("all", {})
        summaries.append(
            {
                "source": source,
                "family": family,
                "model": model,
                "mode": mode,
                "input_mode": input_mode,
                "train_epochs": train_epochs,
                "action": action,
                "run_dir": run_dir,
                "run_name": Path(run_dir).name,
                "run_timestamp": pre.get("run_timestamp", ""),
                "test_year": pre["test_year"],
                "train_years": pre["train_years"],
                "all_mape": all_row.get("mape", math.nan),
                "pre_mape": pre["mape"],
                "post_mape": post["mape"],
                "delta_mape_post_minus_pre": post["mape"] - pre["mape"],
                "all_smape": all_row.get("smape", math.nan),
                "pre_smape": pre["smape"],
                "post_smape": post["smape"],
                "delta_smape_post_minus_pre": post["smape"] - pre["smape"],
                "all_mae": all_row.get("mae", math.nan),
                "pre_mae": pre["mae"],
                "post_mae": post["mae"],
                "delta_mae_post_minus_pre": post["mae"] - pre["mae"],
                "pre_n_windows": pre["n_windows"],
                "post_n_windows": post["n_windows"],
                "pre_n_points": pre["n_points"],
                "post_n_points": post["n_points"],
                "note": pre.get("note", "") or post.get("note", ""),
            }
        )
    return summaries


def latest_per_configuration(rows: Sequence[dict]) -> list[dict]:
    grouped: dict[tuple[str, ...], list[dict]] = defaultdict(list)
    for row in rows:
        key = (
            row["source"],
            row["family"],
            row["model"],
            row["mode"],
            row["input_mode"],
            str(row["train_epochs"]),
        )
        grouped[key].append(row)

    latest_rows: list[dict] = []
    for _, items in grouped.items():
        if items[0]["source"] == "sarimax":
            latest_rows.append(items[0])
            continue
        latest_rows.append(
            max(
                items,
                key=lambda row: (
                    row.get("run_timestamp", ""),
                    row.get("run_name", ""),
                ),
            )
        )
    latest_rows.sort(
        key=lambda row: (
            row["family"],
            row["mode"],
            row["input_mode"],
            row["model"],
            row.get("run_timestamp", ""),
        )
    )
    return latest_rows


def write_csv(path: Path, rows: Sequence[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def print_preview(
    title: str, rows: Sequence[dict], columns: Sequence[str], limit: int = 12
) -> None:
    print(f"\n{title}")
    print("-" * len(title))
    for row in rows[:limit]:
        values = [str(row[col]) for col in columns]
        print(" | ".join(values))
    if len(rows) > limit:
        print(f"... ({len(rows) - limit} more rows)")


def main() -> None:
    args = parse_args()

    conflict_dt = datetime.fromisoformat(args.conflict_date)
    deep_learning_root = Path(args.deep_learning_root)
    sarimax_root = Path(args.sarimax_root)
    output_dir = Path(args.output_dir)

    dl_records = collect_deep_learning_segments(deep_learning_root)
    sarimax_records = collect_sarimax_segments(
        sarimax_root, conflict_dt, args.test_year
    )
    segment_rows = sorted(
        dl_records + sarimax_records,
        key=lambda row: (
            row["source"],
            row["family"],
            row["mode"],
            row["input_mode"],
            str(row["train_epochs"]),
            row["model"],
            row["run_name"],
            row["segment"],
        ),
    )

    summary_rows = summarise_pre_post(segment_rows)
    latest_rows = latest_per_configuration(summary_rows)

    write_csv(output_dir / "conflict_2022_segments_all.csv", segment_rows)
    write_csv(output_dir / "conflict_2022_summary_all.csv", summary_rows)
    write_csv(output_dir / "conflict_2022_summary_latest.csv", latest_rows)

    print(f"Written to: {output_dir}")
    print(f"Segment rows: {len(segment_rows)}")
    print(f"Summary rows: {len(summary_rows)}")
    print(f"Latest configurations: {len(latest_rows)}")

    print_preview(
        "Latest configuration preview",
        latest_rows,
        [
            "family",
            "model",
            "mode",
            "input_mode",
            "train_epochs",
            "pre_mape",
            "post_mape",
            "delta_mape_post_minus_pre",
        ],
    )


if __name__ == "__main__":
    main()
