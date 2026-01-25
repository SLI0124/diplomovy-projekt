"""
Generate hourly datetime features with Czech public holidays.

Creates features (year, month, day, hour, day_of_week, holiday, before_holiday)
from 2013-01-01 to specified end date. Outputs CSV files grouped by year to
../../data/processed/datetime_features/. Designed to be called from ../main.py.
"""

from datetime import date, timedelta
from pathlib import Path

import config
import pandas as pd
import utils
from tqdm import tqdm

DATA_SAVE_PATH = config.PROCESSED_DATETIME_FEATURES_DIR
DEFAULT_START_DATE = config.COMMON_START_DATE


def calculate_easter(year: int) -> date:
    """Calculate Easter date using the Anonymous Gregorian algorithm.

    Args:
        year: Year to calculate Easter for.

    Returns:
        date: Easter Sunday date for the given year.
    """
    a = year % 19
    b = year // 100
    c = year % 100
    d = b // 4
    e = b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i = c // 4
    k = c % 4
    leap_offset = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * leap_offset) // 451
    month = (h + leap_offset - 7 * m + 114) // 31
    day = ((h + leap_offset - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def get_czech_holidays(year: int) -> set[date]:
    """Get all Czech public holidays for the given year.

    Args:
        year: Year to get holidays for.

    Returns:
        set[date]: Set of holiday dates.
    """
    easter = calculate_easter(year)
    return {
        date(year, 1, 1),  # New Year's Day
        date(year, 5, 1),  # Labour Day
        date(year, 5, 8),  # Liberation Day
        date(year, 7, 5),  # Saints Cyril and Methodius Day
        date(year, 7, 6),  # Jan Hus Day
        date(year, 9, 28),  # St. Wenceslas Day (Czech Statehood Day)
        date(year, 10, 28),  # Independent Czechoslovak State Day
        date(year, 11, 17),  # Struggle for Freedom and Democracy Day
        date(year, 12, 24),  # Christmas Eve
        date(year, 12, 25),  # Christmas Day
        date(year, 12, 26),  # St. Stephen's Day
        easter,  # Easter Sunday
        easter + timedelta(days=1),  # Easter Monday
    }


def create_date_range(start_date: date, end_date: date) -> pd.DatetimeIndex:
    """Create an hourly date range.

    Args:
        start_date: Start date for the range.
        end_date: End date for the range.

    Returns:
        pd.DatetimeIndex: Hourly datetime index.
    """
    # End date is inclusive; generate through the last hour of end_date.
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date) + pd.Timedelta(hours=23)
    return pd.date_range(start=start_dt, end=end_dt, freq="h")


def generate_datetime_features_data(end_date: date) -> pd.DataFrame:
    """Generate hourly data with datetime features and holiday flags.

    Args:
        end_date: End date for the feature generation.

    Returns:
        pd.DataFrame: Datetime features for each hour.
    """
    date_range = create_date_range(DEFAULT_START_DATE, end_date)
    print(f"Dates: {DEFAULT_START_DATE} -> {end_date}")

    years_in_range = {dt.year for dt in date_range}
    holidays_by_year = {year: get_czech_holidays(year) for year in years_in_range}

    data = []
    for dt in tqdm(date_range, desc="Dates: processing", leave=False):
        holidays = holidays_by_year[dt.year]
        processed_date = dt.date()
        is_holiday = processed_date in holidays
        is_before_holiday = processed_date + timedelta(days=1) in holidays

        data.append(
            {
                "year": dt.year,
                "month": dt.month,
                "day": dt.day,
                "hour": dt.hour,
                "day_of_week": dt.weekday(),
                "holiday": int(is_holiday),
                "before_holiday": int(is_before_holiday),
            }
        )

    print(f"Dates: generated {len(data):,} rows")
    return pd.DataFrame(data)


def save_to_csv_files(
    df: pd.DataFrame,
    output_dir: Path,
    file_prefix: str = "datetime_features",
) -> None:
    """Save DataFrame as multiple CSV files grouped by year.

    Args:
        df: DataFrame to save.
        output_dir: Directory to save files to.
        file_prefix: Prefix for output filenames.

    Returns:
        None
    """
    utils.ensure_directory(output_dir)

    columns_to_save = [
        "year",
        "month",
        "day",
        "hour",
        "day_of_week",
        "holiday",
        "before_holiday",
    ]

    years = sorted(df["year"].unique())

    for year in tqdm(years, desc="Dates: saving", unit="file", leave=False):
        year_data = df[df["year"] == year][columns_to_save]
        filename = output_dir / f"{file_prefix}_{year}.csv"
        year_data.to_csv(filename, index=False)

    if years:
        year_min = min(years)
        year_max = max(years)
        print(
            f"Dates: saved {len(years)} files \
            ({year_min}-{year_max}) -> {output_dir.resolve()}\n"
        )


def process_datetime_features(
    end_date_param: str | None = None,
) -> pd.DataFrame:
    """Main processing function - entry point for main.py.

    Args:
        end_date_param: End date in YYYY-MM-DD format,
                        or None for last day of previous month.

    Returns:
        pd.DataFrame: Generated datetime feature data.
    """
    end_date = utils.resolve_end_date(end_date_param)

    output_dir = DATA_SAVE_PATH
    dataframe = generate_datetime_features_data(end_date=end_date)
    save_to_csv_files(dataframe, output_dir)

    return dataframe
