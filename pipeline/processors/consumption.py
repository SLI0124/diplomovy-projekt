"""
Module for processing raw gasnet consumption data.

This script will process raw gasnet data.

Since we want to start from 2013-01-01, but gasnet provides data from 7:00 of the current day so we need to
parse data starting from 2012-12-31 7:00 to get the full day of 2013-01-01.

There are 24 hours per day, so each day will have 24 entries.

CSV format have the following columns:

Datum,ID,Hodnota,Nazev
1.1.2013 7:00,21637,461901,Zona_Gasnet
1.1.2013 8:00,21637,426805,Zona_Gasnet
1.1.2013 9:00,21637,445164,Zona_Gasnet
1.1.2013 10:00,21637,451779,Zona_Gasnet
...

From these we are interested in 'Datum' and 'Hodnota' columns.

We need to explicitly parse 'Datum' column to year, month, day, hour.
The 'Hodnota' column does not need any processing, just convert to numeric.

Try to save data as multiple CSV files in ../../data/processed/consumption/ folder, preferably grouped by year.
The catch is that it will be executed from ../main.py so create some entry points accordingly.
"""

import sys
import os
from datetime import datetime, timedelta, date
from pathlib import Path
import glob

import pandas as pd
from tqdm import tqdm

DATA_SOURCE_PATH = "../../data/raw/gasnet/"
DATA_SAVE_PATH = "../../data/processed/consumption/"


def parse_consumption_file(file_path):
    """Parse a single consumption CSV file and return processed DataFrame."""
    df = pd.read_csv(file_path, sep=",")

    # Convert Datum to datetime to extract year
    df["Datum"] = pd.to_datetime(df["Datum"], format="%d.%m.%Y %H:%M")
    df["year"] = df["Datum"].dt.year

    # Convert Hodnota to integer
    df["consumption"] = pd.to_numeric(df["Hodnota"], errors="coerce").astype("Int64")

    # Return only year and consumption columns, drop NaN values
    result = df[["year", "consumption"]].dropna()
    return result


def process_consumption_data(start_date_param=None, end_date_param=None):
    """Main processing function - entry point for main.py"""
    # Get the directory relative to main.py
    current_dir = Path(__file__).parent
    source_dir = current_dir / DATA_SOURCE_PATH
    output_dir = current_dir / DATA_SAVE_PATH

    if start_date_param is None:
        start_date_param = "2013-01-01"
    if end_date_param is None:
        end_date_param = (date.today() - timedelta(days=1)).strftime("%Y-%m-%d")

    # Validate date format
    if end_date_param is not None:
        try:
            datetime.strptime(end_date_param, "%Y-%m-%d")
        except ValueError:
            print(
                f"ERROR: Invalid date format '{end_date_param}'. "
                f"Please use YYYY-MM-DD format."
            )
            sys.exit(1)

    if start_date_param is not None:
        try:
            datetime.strptime(start_date_param, "%Y-%m-%d")
        except ValueError:
            print(
                f"ERROR: Invalid date format '{start_date_param}'. "
                f"Please use YYYY-MM-DD format."
            )
            sys.exit(1)

    # Convert string dates to date objects
    start_date = datetime.strptime(start_date_param, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_param, "%Y-%m-%d").date()

    # Process files in date range
    processed_data = process_consumption_data_with_range(
        source_dir, start_date, end_date
    )

    if processed_data is not None:
        save_processed_data_to_csv(processed_data, output_dir)
        return processed_data
    else:
        print("No data was processed.")
        return None


def process_consumption_data_with_range(source_dir, start_date, end_date):
    """Process consumption data files within the specified date range."""
    print(f"Processing consumption data from {start_date} to {end_date}...")

    all_data = []
    current_date = start_date

    # Create list of dates to process
    dates_to_process = []
    while current_date <= end_date:
        dates_to_process.append(current_date)
        current_date += timedelta(days=1)

    # Process files with progress bar
    for current_date in tqdm(dates_to_process, desc="Processing files"):
        date_str = current_date.strftime("%Y%m%d")
        file_pattern = source_dir / f"{date_str}.csv"

        if file_pattern.exists():
            consumption_data = parse_consumption_file(file_pattern)
            if consumption_data is not None and len(consumption_data) > 0:
                all_data.append(consumption_data)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        print(f"Processed {len(combined_df):,} consumption records")
        return combined_df
    else:
        print("No data found")
        return None


def save_processed_data_to_csv(df, output_dir, file_prefix="consumption"):
    """Save consumption data split by year."""
    output_dir.mkdir(parents=True, exist_ok=True)

    years = sorted(df["year"].unique())
    print(f"Saving data to {len(years)} files:")

    for year in tqdm(years, desc="Saving files"):
        year_data = df[df["year"] == year]["consumption"]
        filename = output_dir / f"{file_prefix}_{year}.csv"
        year_data.to_csv(filename, index=False, header=["consumption"])

    print(f"All files saved to: {output_dir}")


if __name__ == "__main__":
    END_DATE = None
    if len(sys.argv) >= 2:
        END_DATE = sys.argv[1]

    process_consumption_data(end_date_param=END_DATE)
