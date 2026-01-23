"""Process downloaded weather CSV into per-year processed files."""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Optional, Union

import config
import pandas as pd
import utils
from tqdm import tqdm

DATA_SOURCE_PATH = config.RAW_WEATHER_DIR
DATA_SAVE_PATH = config.PROCESSED_WEATHER_DIR


def _normalize_weather_datetime_local_naive(
    values: Union[pd.Series, pd.Index],
) -> pd.Series:
    """Normalize timestamps to tz-naive datetimes in configured local timezone. This was
     a new problem that appeared recently, possibly due to changes in the source data.

    Args:
        values: Series or Index with datetime-like values.

    Returns:
        pd.Series: Normalized timestamps as tz-naive pandas datetimes.
    """
    dt = pd.to_datetime(values, utc=True, errors="coerce")
    dti = pd.DatetimeIndex(dt).tz_convert(config.WEATHER_TIMEZONE).tz_localize(None)
    index = (
        values.index
        if isinstance(values, pd.Series)
        else values
        if isinstance(values, pd.Index)
        else None
    )
    return pd.Series(dti, index=index, name=getattr(values, "name", None))


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

    df["date"] = _normalize_weather_datetime_local_naive(df["date"])

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

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Weather: raw directory not found: {source_dir}. "
            "Run the downloader first (pipeline/main.py --download weather)."
        )

    # Find the weather CSV file (should be only one)
    weather_files = list(source_dir.glob("weather_*.csv"))

    if not weather_files:
        raise FileNotFoundError(
            f"Weather: no input files in {source_dir}. "
            "Expected something like 'weather_YYYY-MM-DD_YYYY-MM-DD.csv'."
        )

    if len(weather_files) > 1:
        print(f"Weather: multiple files, using {weather_files[0].name}")

    weather_file = weather_files[0]
    print(f"Weather: {weather_file.name}")

    # Validate coverage before full processing (fast-fail with clear message).
    try:
        dates_only = pd.read_csv(weather_file, usecols=["date"])
        dates_only["date"] = _normalize_weather_datetime_local_naive(dates_only["date"])
        min_dt = dates_only["date"].min()
        max_dt = dates_only["date"].max()
    except ValueError:
        # usecols failed (missing 'date' column)
        raise ValueError(
            f"Weather: raw file {weather_file.name} is missing required column 'date'."
        )

    if pd.isna(min_dt) or pd.isna(max_dt):
        raise ValueError(
            f"Weather: raw file {weather_file.name} has no valid timestamps in 'date'."
        )

    required_start = pd.to_datetime(start_date)
    # Need at least through the last hour of end_date.
    required_last_hour = (
        pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    )
    if min_dt > required_start or max_dt < required_last_hour:
        raise FileNotFoundError(
            "Weather: raw data does not cover requested range. "
            f"Have {min_dt}..{max_dt}, need {required_start}..{required_last_hour}. "
            "Run the downloader for the missing period."
        )

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
