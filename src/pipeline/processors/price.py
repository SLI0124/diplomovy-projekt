"""Process raw OTE-CR gas price XLS files into hourly CSVs."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import pandas as pd
import utils.config as config
import utils.helper_functions as utils
from tqdm import tqdm
from xlrd.biffh import XLRDError

DATA_SOURCE_PATH = config.RAW_PRICE_DIR
DATA_SAVE_PATH = config.PROCESSED_PRICE_DIR


def parse_price_file(file_path: Path) -> pd.DataFrame:
    """Parse one monthly price XLS file.

    Args:
        file_path: Path to a monthly XLS file.

    Returns:
        pandas.DataFrame: Hourly rows with columns
        [year, month, day, hour, traded_volume_mwh, weighted_avg_price_eur_mwh,
        min_price_eur_mwh, max_price_eur_mwh].
    """
    try:
        # there are two different Excel formats in use, try both engines
        try:
            df = pd.read_excel(file_path, skiprows=4, engine="xlrd")
        except (ImportError, ValueError, XLRDError):
            # Fallback to openpyxl if xlrd is not available or xlrd cannot be used
            df = pd.read_excel(file_path, skiprows=4, engine="openpyxl")

        df = df.iloc[1:].copy()
        date_col = pd.to_datetime(df.iloc[:, 0], errors="coerce")

        daily = pd.DataFrame(
            {
                "year": date_col.dt.year.astype("Int64"),
                "month": date_col.dt.month.astype("Int64"),
                "day": date_col.dt.day.astype("Int64"),
                "traded_volume_mwh": pd.to_numeric(
                    df.iloc[:, 1].replace("-", pd.NA), errors="coerce"
                ),
                "weighted_avg_price_eur_mwh": pd.to_numeric(
                    df.iloc[:, 2].replace("-", pd.NA), errors="coerce"
                ),
                "min_price_eur_mwh": pd.to_numeric(
                    df.iloc[:, 3].replace("-", pd.NA), errors="coerce"
                ),
                "max_price_eur_mwh": pd.to_numeric(
                    df.iloc[:, 4].replace("-", pd.NA), errors="coerce"
                ),
            }
        )

        daily = daily.dropna(subset=["year", "month", "day"]).reset_index(drop=True)
        if daily.empty:
            return pd.DataFrame()

        # Assign hour 0..23 for each day
        hourly = daily.loc[daily.index.repeat(24)].reset_index(drop=True)
        hourly["hour"] = list(range(24)) * len(daily)
        hourly["hour"] = hourly["hour"].astype("Int64")
        return hourly[
            [
                "year",
                "month",
                "day",
                "hour",
                "traded_volume_mwh",
                "weighted_avg_price_eur_mwh",
                "min_price_eur_mwh",
                "max_price_eur_mwh",
            ]
        ]

    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        FileNotFoundError,
        ValueError,
        ImportError,
    ) as exc:
        print(f"Price: parse error ({file_path.name}: {exc})")
        return pd.DataFrame()


def process_price_data_with_range(
    source_dir: Path,
    start_date: date,
    end_date: date,
) -> pd.DataFrame | None:
    """Process price data files within the specified date range.

    Args:
        source_dir: Directory with downloaded XLS files.
        start_date: Inclusive start date.
        end_date: Inclusive end date.

    Returns:
        A combined DataFrame or None if nothing was processed.
    """
    print(f"Price: {start_date} -> {end_date}")

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Price: raw directory not found: {source_dir}. "
            "Run the downloader first (src/pipeline/main.py --download price)."
        )

    # Ensure monthly raw files exist for the requested range.
    missing: list[Path] = []
    for year, month in utils.iter_year_months(start_date, end_date):
        month_str = f"{month:02d}"
        expected = source_dir / f"VDT_plyn_{month_str}_{year}_CZ.xls"
        if not expected.is_file():
            missing.append(expected)
    utils.raise_missing_inputs(
        what="price processing",
        missing_paths=missing,
        required_range=f"{start_date}..{end_date}",
    )

    # Find all price XLS files
    price_files = list(source_dir.glob("VDT_plyn_*.xls"))

    if not price_files:
        print("Price: no input files")
        return None

    print(f"Price: {len(price_files)} files")

    all_data = []
    start_year = start_date.year
    end_year = end_date.year

    for file_path in tqdm(price_files, desc="Price: processing", leave=False):
        # Extract year from filename: VDT_plyn_MM_YYYY_CZ.xls
        try:
            parts = file_path.stem.split("_")
            file_year = int(parts[3])

            # Skip files outside the date range
            if file_year < start_year or file_year > end_year:
                continue

            # Parse the file
            file_data = parse_price_file(file_path)

            if file_data is not None and len(file_data) > 0:
                all_data.append(file_data)

        except (IndexError, ValueError) as exc:
            print(f"Price: bad filename ({file_path.name}: {exc})")
            continue

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        # Further filter by exact date range
        combined_df = combined_df[
            (combined_df["year"] >= start_year) & (combined_df["year"] <= end_year)
        ].copy()

        # Sort by date
        combined_df["timestamp"] = pd.to_datetime(
            combined_df[["year", "month", "day", "hour"]], errors="coerce"
        )
        combined_df = combined_df.sort_values(["timestamp"]).reset_index(drop=True)
        combined_df = combined_df.drop(columns=["timestamp"], errors="ignore")

        print(f"Price: processed {len(combined_df):,} hourly rows")
        return combined_df

    print("Price: no data")
    return None


def save_processed_price_data_to_csv(
    df: pd.DataFrame,
    output_dir: Path,
    file_prefix: str = "price",
) -> None:
    """Save processed price data grouped by year.

    Args:
        df: Processed hourly price data.
        output_dir: Destination directory.
        file_prefix: Output filename prefix.
    """
    utils.ensure_directory(output_dir)

    years = sorted(df["year"].unique())

    for year in tqdm(years, desc="Price: saving", unit="file", leave=False):
        year_data = df[df["year"] == year]
        filename = output_dir / f"{file_prefix}_{year}.csv"
        year_data.to_csv(filename, index=False)

    if years:
        year_min = min(years)
        year_max = max(years)
        print(
            f"Price: saved {len(years)} files \
            ({year_min}-{year_max}) -> {output_dir.resolve()}\n"
        )


def process_price_data(
    end_date_param: utils.DateLike | None = None,
) -> pd.DataFrame | None:
    """Entry point used by [src/pipeline/main.py](src/pipeline/main.py).

    Args:
        end_date_param: End date as YYYY-MM-DD string, date, datetime, or None.

    Returns:
        Processed hourly price data, or None if nothing was processed.
    """
    start_date = config.PRICE_START_DATE
    end_date = utils.resolve_end_date(end_date_param)

    processed_data = process_price_data_with_range(
        DATA_SOURCE_PATH, start_date, end_date
    )
    if processed_data is None:
        return None

    save_processed_price_data_to_csv(processed_data, DATA_SAVE_PATH)
    return processed_data
