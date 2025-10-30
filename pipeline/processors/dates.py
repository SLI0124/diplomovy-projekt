"""
Module for generating datetime features with Czech holidays.

This file will create columns year, month, day, hour, day_of_week, holiday,
and before_holiday from 1.1.2013 to given end date, preferably yesterday.

Uses Czech public holidays as source.

All holidays have static dates except Easter.

There is algorithm to calculate easter dates:
https://en.wikipedia.org/wiki/Computus#Anonymous_Gregorian_algorithm

The structure will be year month day hour day_of_week holiday before_holiday,
so for one day you need 24 rows.

It will be saved as multiple csv files in ../../data/raw/datetime_features/
folder, preferably grouped by year.

The catch is that it will be executed from ../main.py so create some entry
points accordingly.
"""

import sys
from datetime import datetime, timedelta, date
from pathlib import Path

import pandas as pd
from tqdm import tqdm

DATA_SAVE_PATH = "../../data/processed/datetime_features/"


def calculate_easter(year):
    """Calculate Easter date using Anonymous Gregorian algorithm."""
    # Anonymous Gregorian algorithm for Easter calculation
    # Source: https://en.wikipedia.org/wiki/Computus#Anonymous_Gregorian_algorithm
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


def get_czech_holidays(year):
    """Get all Czech public holidays for given year."""
    holidays = []

    # Static holidays
    holidays.extend(
        [
            date(year, 1, 1),  # New Year's Day
            date(year, 5, 1),  # Labour Day
            date(year, 5, 8),  # Victory in Europe Day
            date(year, 7, 5),  # Saints Cyril and Methodius Day
            date(year, 7, 6),  # Jan Hus Day
            date(year, 9, 28),  # Czech Statehood Day
            date(year, 10, 28),  # Independence Day
            date(year, 11, 17),  # Struggle for Freedom and Democracy Day
            date(year, 12, 24),  # Christmas Eve
            date(year, 12, 25),  # Christmas Day
            date(year, 12, 26),  # St. Stephen's Day
        ]
    )

    # Easter-based holidays
    easter = calculate_easter(year)
    holidays.extend(
        [
            easter,  # Easter Sunday
            easter + timedelta(days=1),  # Easter Monday
        ]
    )

    return set(holidays)


def create_date_range(start_date="2013-01-01", end_date_param=None):
    """Create date range from start_date to end_date (default yesterday)."""
    if end_date_param is None:
        end_date_param = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    return pd.date_range(start=start_date, end=end_date_param, freq="h")


def generate_datetime_features_data(start_date_param=None, end_date_param=None):
    """Generate hourly data with datetime features and holiday flags."""
    # Set default parameters
    if start_date_param is None:
        start_date_param = "2013-01-01"
    if end_date_param is None:
        end_date_param = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Create hourly date range
    date_range = create_date_range(start_date_param, end_date_param)
    print(
        f"Generating datetime features from {start_date_param} to {end_date_param}..."
    )

    # Pre-compute holidays for all years in the range
    years_in_range = {dt.year for dt in date_range}
    holidays_by_year = {year: get_czech_holidays(year) for year in years_in_range}

    # Generate datetime features with progress bar
    data = []
    for dt in tqdm(date_range, desc="Processing datetime features"):
        holidays = holidays_by_year[dt.year]
        current_date = dt.date()
        next_date = current_date + timedelta(days=1)

        is_holiday = current_date in holidays
        is_before_holiday = next_date in holidays

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

    total_records = len(data)
    print(f"Generated {total_records:,} datetime feature records")
    return pd.DataFrame(data)


def save_to_csv_files(df, output_dir, file_prefix="datetime_features"):
    """Save DataFrame as multiple CSV files grouped by year."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Select only the required columns
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
    print(f"Saving data to {len(years)} files:")

    for year in tqdm(years, desc="Saving files"):
        year_data = df[df["year"] == year][columns_to_save]
        filename = output_dir / f"{file_prefix}_{year}.csv"
        year_data.to_csv(filename, index=False)

    print(f"All files saved to: {output_dir}")


def process_datetime_features(end_date_param=None):
    """Main processing function - entry point for main.py."""
    # Get the directory relative to main.py
    current_dir = Path(__file__).parent
    output_dir = current_dir / DATA_SAVE_PATH

    if end_date_param is not None:
        try:
            datetime.strptime(end_date_param, "%Y-%m-%d")
        except ValueError:
            print(
                f"ERROR: Invalid date format '{end_date_param}'. "
                f"Please use YYYY-MM-DD format."
            )
            sys.exit(1)

    dataframe = generate_datetime_features_data(end_date_param=end_date_param)
    save_to_csv_files(dataframe, output_dir)

    return dataframe


if __name__ == "__main__":
    END_DATE = None
    if len(sys.argv) >= 2:
        END_DATE = sys.argv[1]

    process_datetime_features(end_date_param=END_DATE)
