from __future__ import annotations

import argparse
import json
from pathlib import Path

from value_cleaning import preprocess_merged_csv


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report_path = args.report or args.output.with_suffix(".report.json")

    result = preprocess_merged_csv(
        args.input,
        args.output,
    )

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result.report, indent=2), encoding="utf-8")

    print(f"Input: {args.input}")
    print(f"Output: {args.output}")
    print(f"Report: {report_path}")
    print(json.dumps(result.report, indent=2))


if __name__ == "__main__":
    main()
