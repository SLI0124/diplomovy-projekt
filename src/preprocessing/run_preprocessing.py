from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Add project root to sys.path for imports
if __package__ is None or __package__ == "":
    sys.path.insert(0, str(PROJECT_ROOT))

from src.preprocessing.pipeline import PreprocessConfig, run_preprocessing  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run preprocessing pipeline for merged time-series dataset."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=PROJECT_ROOT / "data/processed/merged/merged_all_years.csv",
        help="Path to merged input CSV.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=PROJECT_ROOT / "data/preprocessed/merged_all_years_preprocessed.csv",
        help="Path where cleaned dataset CSV is saved.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = PreprocessConfig(
        input_csv=args.input_csv,
        output_csv=args.output_csv,
    )
    result = run_preprocessing(config)
    print(f"Saved cleaned dataset: {result['cleaned_path']}")
    print(f"Rows: {result['rows']}")
    print(f"Target missing after: {result['target_missing_after']}")


if __name__ == "__main__":
    main()
