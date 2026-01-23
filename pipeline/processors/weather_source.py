"""Process downloaded weather CSV into per-year processed files."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import config
import pandas as pd
import utils
from tqdm import tqdm

DATA_SOURCE_PATH = config.RAW_WEATHER_DIR
DATA_SAVE_PATH = config.PROCESSED_WEATHER_DIR


def parse_weather_file(file_path: Path) -> Optional[pd.DataFrame]:
    """Load the raw weather CSV and add year/month/day/hour columns.

    Args:
        file_path: Path to a raw weather CSV.

    Returns:
        A processed DataFrame, or None on read/parse error.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError as exc:
        print(f"Weather: file not found ({exc})")
        return None
    except pd.errors.EmptyDataError as exc:
        print(f"Weather: empty file ({exc})")
        return None
    except (pd.errors.ParserError, UnicodeDecodeError, PermissionError) as exc:
        print(f"Weather: read error ({exc})")
        return None

    df["date"] = pd.to_datetime(df["date"])

    dt_index = pd.DatetimeIndex(df["date"])
    df["year"] = dt_index.year.astype("Int64")
    df["month"] = dt_index.month.astype("Int64")
    df["day"] = dt_index.day.astype("Int64")
    df["hour"] = dt_index.hour.astype("Int64")

    weather_columns = ["year", "month", "day", "hour", *config.WEATHER_VARIABLES]

    return df[weather_columns].dropna(how="all")


def process_weather_data_with_range(
    source_dir: Path,
    start_date: date,
    end_date: date,
) -> Optional[pd.DataFrame]:
    """Process weather data within a specific date range.

    Args:
        source_dir: Directory containing downloaded raw weather CSV(s).
        start_date: Inclusive start date.
        end_date: Inclusive end date.

    Returns:
        Processed DataFrame filtered to the requested date range, or None.
    """

    # Find the weather CSV file (should be only one)
    weather_files = list(source_dir.glob("weather_*.csv"))

    if not weather_files:
        print("Weather: no input files")
        return None

    if len(weather_files) > 1:
        print(f"Weather: multiple files, using {weather_files[0].name}")

    weather_file = weather_files[0]
    print(f"Weather: {weather_file.name}")

    weather_data = parse_weather_file(weather_file)
    if weather_data is None or weather_data.empty:
        print("Weather: no data")
        return None

    dt_col = pd.to_datetime(weather_data[["year", "month", "day", "hour"]])
    start_dt = pd.to_datetime(start_date)
    end_dt_exclusive = pd.to_datetime(end_date + timedelta(days=1))
    mask = (dt_col >= start_dt) & (dt_col < end_dt_exclusive)
    filtered = weather_data.loc[mask].copy()
    return filtered


def save_processed_weather_data_to_csv(
    df: pd.DataFrame,
    output_dir: Path,
    file_prefix: str = "weather",
) -> None:
    """Save processed weather data grouped by year.

    Args:
        df: Processed weather data.
        output_dir: Directory to save processed files to.
        file_prefix: Prefix for output filenames.
    """
    utils.ensure_directory(output_dir)

    years = sorted(df["year"].unique())

    for year in tqdm(years, desc="Weather: saving", unit="file", leave=False):
        year_data = df[df["year"] == year]
        # Keep all datetime components in the saved data
        filename = output_dir / f"{file_prefix}_{year}.csv"
        year_data.to_csv(filename, index=False)

    if years:
        year_min = min(years)
        year_max = max(years)
        print(
            f"Weather: saved {len(years)} files \
            ({year_min}-{year_max}) -> {output_dir.resolve()}\n"
        )


def process_weather_data(
    end_date_param: Optional[utils.DateLike] = None,
) -> Optional[pd.DataFrame]:
    """Entry point used by [pipeline/main.py](pipeline/main.py).

    Args:
        end_date_param: End date as YYYY-MM-DD string, date, datetime, or None.

    Returns:
        Optional[pd.DataFrame]: Processed weather data DataFrame or None on error.
    """
    start_date = config.WEATHER_START_DATE
    end_date = utils.resolve_end_date(end_date_param)

    processed_data = process_weather_data_with_range(
        DATA_SOURCE_PATH, start_date, end_date
    )
    if processed_data is None:
        return None

    save_processed_weather_data_to_csv(processed_data, DATA_SAVE_PATH)
    return processed_data
