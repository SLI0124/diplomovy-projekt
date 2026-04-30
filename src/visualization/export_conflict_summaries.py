#!/usr/bin/env python3
"""Export conflict-related summaries for chapter 6.

The script produces two kinds of comparisons:

1. A focused split of the year 2022 into:
   - `pre_conflict`: before 24 February 2022
   - `post_conflict`: from 24 February 2022 onwards
2. A long-range regime split across the whole evaluated period:
   - `pre_conflict`: all target timestamps before 24 February 2022
   - `post_conflict`: all target timestamps from 24 February 2022 onwards

Deep learning metrics are recomputed directly from point predictions stored in
`src/results/deep_learning/<run>/predictions/*.csv`.
SARIMAX metrics are computed from
`src/results/sarimax_stepup/sarimax_predictions_all_folds.csv`.

By default, only configurations used in the thesis are exported:
- foundation models with 10 epochs,
- custom models with 10 epochs,
- SARIMAX,
- `granite` in `one-shot covariate` mode is excluded by default.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from pathlib import Path

FOUNDATION_MODELS = {"chronos2", "granite", "moirai1"}
CUSTOM_MODELS = {"model_1", "model_2", "model_3"}


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
        "--min-test-years",
        default=12,
        type=int,
        help="Minimum number of distinct test years required for a deep learning run to be included.",
    )
    parser.add_argument(
        "--year-of-interest",
        default=2022,
        type=int,
        help="Year used for the focused pre/post-conflict split.",
    )
    parser.add_argument(
        "--include-extra-epochs",
        action="store_true",
        help="Include foundation 20 epochs and custom 50 epochs.",
    )
    parser.add_argument(
        "--include-granite-oneshot-covariate",
        action="store_true",
        help="Include raw Granite one-shot covariate runs as well.",
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


def safe_div(num: float, den: float) -> float:
    if den == 0:
        return math.nan
    return num / den


def compute_metrics(rows: Sequence[dict]) -> dict:
    if not rows:
        raise ValueError("Cannot compute metrics from an empty row set.")

    y_true = [float(row["y_true"]) for row in rows]
    y_pred = [float(row["y_pred"]) for row in rows]
    n_points = len(rows)
    n_windows = len({row["origin_timestamp"] for row in rows})

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

    return {
        "n_windows": n_windows,
        "n_points": n_points,
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "smape": smape,
        "r2": r2,
    }


def should_include_config(
    family: str,
    model: str,
    mode: str,
    input_mode: str,
    train_epochs: str,
    include_extra_epochs: bool,
    include_granite_oneshot_covariate: bool,
) -> bool:
    if family == "sarimax":
        return True
    if family not in {"foundation", "custom"}:
        return False

    if not include_extra_epochs and str(train_epochs) != "10":
        return False

    return not (
        family == "foundation"
        and model == "granite"
        and mode == "one-shot"
        and input_mode == "covariate"
        and not include_granite_oneshot_covariate
    )


def load_prediction_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    for row in rows:
        timestamp = row.get("target_timestamp") or row.get("timestamp")
        row["target_timestamp"] = timestamp
        row["target_dt"] = datetime.fromisoformat(timestamp.replace("Z", ""))  # type: ignore
        origin_timestamp = row.get("origin_timestamp")
        row["origin_timestamp"] = origin_timestamp
    return rows


def collect_run_prediction_rows(run_dir: Path, model: str) -> list[dict]:
    prediction_dir = run_dir / "predictions"
    if not prediction_dir.exists():
        return []

    rows: list[dict] = []
    for path in sorted(prediction_dir.glob(f"{model}__test-*.csv")):
        rows.extend(load_prediction_rows(path))
    return rows


def distinct_test_years(rows: Sequence[dict]) -> list[int]:
    return sorted({int(row["test_year"]) for row in rows})


def make_segment_records(
    *,
    source: str,
    family: str,
    model: str,
    mode: str,
    input_mode: str,
    train_epochs: str,
    action: str,
    run_dir: str,
    run_name: str,
    run_timestamp: str,
    scope: str,
    year_of_interest: int,
    train_years: str,
    note: str,
    segment_rows: dict[str, Sequence[dict]],
) -> list[dict]:
    records: list[dict] = []
    for segment_name, rows in segment_rows.items():
        metrics = compute_metrics(rows)
        test_year_value = year_of_interest if scope == "year_2022" else "2014-2025"
        records.append(
            {
                "source": source,
                "family": family,
                "model": model,
                "mode": mode,
                "input_mode": input_mode,
                "train_epochs": train_epochs,
                "action": action,
                "run_dir": run_dir,
                "run_name": run_name,
                "run_timestamp": run_timestamp,
                "scope": scope,
                "test_year": test_year_value,
                "train_years": train_years,
                "segment": segment_name,
                "n_windows": metrics["n_windows"],
                "n_points": metrics["n_points"],
                "mae": metrics["mae"],
                "mse": metrics["mse"],
                "rmse": metrics["rmse"],
                "mape": metrics["mape"],
                "smape": metrics["smape"],
                "r2": metrics["r2"],
                "note": note,
            }
        )
    return records


def collect_deep_learning_records(
    root: Path,
    conflict_dt: datetime,
    year_of_interest: int,
    min_test_years: int,
    include_extra_epochs: bool,
    include_granite_oneshot_covariate: bool,
) -> tuple[list[dict], list[dict]]:
    split_2022_records: list[dict] = []
    regime_records: list[dict] = []

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue

        runtime_path = run_dir / "runtime_config.json"
        if not runtime_path.exists():
            continue

        config = load_json(runtime_path)
        family = classify_family(config.get("models", []))
        run_timestamp = parse_run_timestamp(run_dir)
        run_timestamp_str = run_timestamp.isoformat() if run_timestamp else ""

        for model in config.get("models", []):
            if not should_include_config(
                family=family,
                model=model,
                mode=config.get("mode", ""),
                input_mode=config.get("training_input_mode", ""),
                train_epochs=str(config.get("train_epochs", "")),
                include_extra_epochs=include_extra_epochs,
                include_granite_oneshot_covariate=include_granite_oneshot_covariate,
            ):
                continue

            rows = collect_run_prediction_rows(run_dir, model)
            if not rows:
                continue

            years_available = distinct_test_years(rows)
            if len(years_available) < min_test_years:
                continue

            rows_2022 = [
                row for row in rows if int(row["test_year"]) == year_of_interest
            ]
            if rows_2022:
                split_2022_records.extend(
                    make_segment_records(
                        source="deep_learning",
                        family=family,
                        model=model,
                        mode=config.get("mode", ""),
                        input_mode=config.get("training_input_mode", ""),
                        train_epochs=str(config.get("train_epochs", "")),
                        action=config.get("action", ""),
                        run_dir=str(run_dir),
                        run_name=run_dir.name,
                        run_timestamp=run_timestamp_str,
                        scope="year_2022",
                        year_of_interest=year_of_interest,
                        train_years="2013-2021",
                        note="",
                        segment_rows={
                            "all": rows_2022,
                            "pre_conflict": [
                                row
                                for row in rows_2022
                                if row["target_dt"] < conflict_dt
                            ],
                            "post_conflict": [
                                row
                                for row in rows_2022
                                if row["target_dt"] >= conflict_dt
                            ],
                        },
                    )
                )

            regime_records.extend(
                make_segment_records(
                    source="deep_learning",
                    family=family,
                    model=model,
                    mode=config.get("mode", ""),
                    input_mode=config.get("training_input_mode", ""),
                    train_epochs=str(config.get("train_epochs", "")),
                    action=config.get("action", ""),
                    run_dir=str(run_dir),
                    run_name=run_dir.name,
                    run_timestamp=run_timestamp_str,
                    scope="full_regime",
                    year_of_interest=year_of_interest,
                    train_years="2014-2025",
                    note="",
                    segment_rows={
                        "all": rows,
                        "pre_conflict": [
                            row for row in rows if row["target_dt"] < conflict_dt
                        ],
                        "post_conflict": [
                            row for row in rows if row["target_dt"] >= conflict_dt
                        ],
                    },
                )
            )

    return split_2022_records, regime_records


def collect_sarimax_records(
    root: Path,
    conflict_dt: datetime,
    year_of_interest: int,
) -> tuple[list[dict], list[dict]]:
    all_predictions_path = root / "sarimax_predictions_all_folds.csv"
    if not all_predictions_path.exists():
        raise FileNotFoundError(
            f"Missing SARIMAX predictions file: {all_predictions_path}"
        )

    all_rows = load_prediction_rows(all_predictions_path)
    rows_2022 = [row for row in all_rows if int(row["test_year"]) == year_of_interest]

    split_2022_records = make_segment_records(
        source="sarimax",
        family="sarimax",
        model="sarimax",
        mode="baseline",
        input_mode="covariate",
        train_epochs="",
        action="eval",
        run_dir=str(root),
        run_name=root.name,
        run_timestamp="",
        scope="year_2022",
        year_of_interest=year_of_interest,
        train_years="2013-2021",
        note="computed_from_point_predictions",
        segment_rows={
            "all": rows_2022,
            "pre_conflict": [
                row for row in rows_2022 if row["target_dt"] < conflict_dt
            ],
            "post_conflict": [
                row for row in rows_2022 if row["target_dt"] >= conflict_dt
            ],
        },
    )

    regime_records = make_segment_records(
        source="sarimax",
        family="sarimax",
        model="sarimax",
        mode="baseline",
        input_mode="covariate",
        train_epochs="",
        action="eval",
        run_dir=str(root),
        run_name=root.name,
        run_timestamp="",
        scope="full_regime",
        year_of_interest=year_of_interest,
        train_years="2014-2025",
        note="computed_from_point_predictions",
        segment_rows={
            "all": all_rows,
            "pre_conflict": [row for row in all_rows if row["target_dt"] < conflict_dt],
            "post_conflict": [
                row for row in all_rows if row["target_dt"] >= conflict_dt
            ],
        },
    )

    return split_2022_records, regime_records


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
            row["scope"],
        )
        grouped[key][row["segment"]] = row

    summaries: list[dict] = []
    for key, segments in grouped.items():
        if "pre_conflict" not in segments or "post_conflict" not in segments:
            continue

        (
            source,
            family,
            model,
            mode,
            input_mode,
            train_epochs,
            action,
            run_dir,
            scope,
        ) = key

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
                "scope": scope,
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
            row["scope"],
        )
        grouped[key].append(row)

    latest_rows: list[dict] = []
    for items in grouped.values():
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
            row["scope"],
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
        print(" | ".join(str(row[col]) for col in columns))
    if len(rows) > limit:
        print(f"... ({len(rows) - limit} dalších řádků)")


def print_group_counts(
    title: str, rows: Sequence[dict], group_fields: Sequence[str]
) -> None:
    grouped: dict[tuple[str, ...], set[str]] = defaultdict(set)
    for row in rows:
        key = tuple(str(row[field]) for field in group_fields)
        grouped[key].add(str(row.get("run_dir", "")))

    print(f"\n{title}")
    print("-" * len(title))
    for key in sorted(grouped):
        label = " | ".join(key)
        print(f"{label}: {len(grouped[key])} runs")


def main() -> None:
    args = parse_args()

    conflict_dt = datetime.fromisoformat(args.conflict_date)
    deep_learning_root = Path(args.deep_learning_root)
    sarimax_root = Path(args.sarimax_root)
    output_dir = Path(args.output_dir)

    dl_2022, dl_regime = collect_deep_learning_records(
        deep_learning_root,
        conflict_dt,
        args.year_of_interest,
        args.min_test_years,
        include_extra_epochs=args.include_extra_epochs,
        include_granite_oneshot_covariate=args.include_granite_oneshot_covariate,
    )
    sarimax_2022, sarimax_regime = collect_sarimax_records(
        sarimax_root,
        conflict_dt,
        args.year_of_interest,
    )

    split_2022_segments = sorted(
        dl_2022 + sarimax_2022,
        key=lambda row: (
            row["family"],
            row["mode"],
            row["input_mode"],
            str(row["train_epochs"]),
            row["model"],
            row["run_name"],
            row["segment"],
        ),
    )
    regime_segments = sorted(
        dl_regime + sarimax_regime,
        key=lambda row: (
            row["family"],
            row["mode"],
            row["input_mode"],
            str(row["train_epochs"]),
            row["model"],
            row["run_name"],
            row["segment"],
        ),
    )

    split_2022_summary = summarise_pre_post(split_2022_segments)
    regime_summary = summarise_pre_post(regime_segments)
    split_2022_latest = latest_per_configuration(split_2022_summary)
    regime_latest = latest_per_configuration(regime_summary)

    write_csv(output_dir / "conflict_2022_segments_all.csv", split_2022_segments)
    write_csv(output_dir / "conflict_2022_summary_all.csv", split_2022_summary)
    write_csv(output_dir / "conflict_2022_summary_latest.csv", split_2022_latest)
    write_csv(output_dir / "conflict_regime_segments_all.csv", regime_segments)
    write_csv(output_dir / "conflict_regime_summary_all.csv", regime_summary)
    write_csv(output_dir / "conflict_regime_summary_latest.csv", regime_latest)

    print(f"Zapsáno do: {output_dir}")
    print(f"Rok 2022, segmentové řádky: {len(split_2022_segments)}")
    print(f"Rok 2022, souhrnné řádky: {len(split_2022_summary)}")
    print(f"Rok 2022, nejnovější konfigurace: {len(split_2022_latest)}")
    print(f"Dlouhý režim, segmentové řádky: {len(regime_segments)}")
    print(f"Dlouhý režim, souhrnné řádky: {len(regime_summary)}")
    print(f"Dlouhý režim, nejnovější konfigurace: {len(regime_latest)}")

    print_group_counts(
        "Run counts by model/input (2022)",
        split_2022_summary,
        ["family", "model", "input_mode"],
    )
    print_group_counts(
        "Run counts by model/input (full regime)",
        regime_summary,
        ["family", "model", "input_mode"],
    )

    print_preview(
        "Přehled nejnovějších konfigurací pro rok 2022",
        split_2022_latest,
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
    print_preview(
        "Přehled nejnovějších konfigurací pro dlouhý režim",
        regime_latest,
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
