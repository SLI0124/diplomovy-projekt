"""
Module for merging processed data from dates, consumption, and weather sources.

This script will merge the processed data from:
- ../../data/processed/datetime_features/ (year, month, day, hour, day_of_week, holiday, before_holiday)
- ../../data/processed/consumption/ (consumption values)
- ../../data/processed/weather/ (20 weather features)

The merge will be done row-by-row based on the order in files (hourly data),
assuming all files have the same temporal order and length.

The merged data will be saved as multiple CSV files in ../../data/processed/merged/ folder,
preferably grouped by year.

The catch is that it will be executed from ../main.py so create some entry points accordingly.
"""

import sys
import os
from datetime import datetime, timedelta, date
from pathlib import Path
import glob

import pandas as pd
from tqdm import tqdm

DATETIME_FEATURES_PATH = "../../data/processed/datetime_features/"
CONSUMPTION_PATH = "../../data/processed/consumption/"
WEATHER_PATH = "../../data/processed/weather/"
MERGED_SAVE_PATH = "../../data/processed/merged/"


def load_year_data(year, datetime_dir, consumption_dir, weather_dir):
    """Load data for a specific year from all three sources."""
    datetime_file = datetime_dir / f"datetime_features_{year}.csv"
    consumption_file = consumption_dir / f"consumption_{year}.csv"
    weather_file = weather_dir / f"weather_{year}.csv"

    # Check if all required files exist
    missing_files = []
    if not datetime_file.exists():
        missing_files.append(str(datetime_file))
    if not consumption_file.exists():
        missing_files.append(str(consumption_file))
    if not weather_file.exists():
        missing_files.append(str(weather_file))

    if missing_files:
        print(f"Warning: Missing files for year {year}: {missing_files}")
        return None

    # Load the data
    try:
        datetime_df = pd.read_csv(datetime_file)
        consumption_df = pd.read_csv(consumption_file)
        weather_df = pd.read_csv(weather_file)

        print(f"Year {year} - Initial data sizes:")
        print(f"  Datetime features: {len(datetime_df)} rows")
        print(f"  Consumption: {len(consumption_df)} rows")
        print(f"  Weather: {len(weather_df)} rows")

        # Create datetime join keys for each dataframe
        # Datetime features already has year, month, day, hour
        datetime_df["join_key"] = (
            datetime_df["year"].astype(str)
            + "_"
            + datetime_df["month"].astype(str)
            + "_"
            + datetime_df["day"].astype(str)
            + "_"
            + datetime_df["hour"].astype(str)
        )

        # Consumption data should now have year, month, day, hour
        consumption_df["join_key"] = (
            consumption_df["year"].astype(str)
            + "_"
            + consumption_df["month"].astype(str)
            + "_"
            + consumption_df["day"].astype(str)
            + "_"
            + consumption_df["hour"].astype(str)
        )

        # Weather data should now have year, month, day, hour
        weather_df["join_key"] = (
            weather_df["year"].astype(str)
            + "_"
            + weather_df["month"].astype(str)
            + "_"
            + weather_df["day"].astype(str)
            + "_"
            + weather_df["hour"].astype(str)
        )

        # Perform proper joins based on datetime
        # Start with datetime features as base
        merged_df = datetime_df.copy()

        # Join consumption data
        consumption_join = consumption_df[["join_key", "consumption"]].copy()
        merged_df = merged_df.merge(consumption_join, on="join_key", how="left")

        # Join weather data (excluding duplicate datetime columns)
        weather_columns = [
            col
            for col in weather_df.columns
            if col not in ["year", "month", "day", "hour"]
        ]
        weather_join = weather_df[weather_columns].copy()
        merged_df = merged_df.merge(weather_join, on="join_key", how="left")

        # Drop the temporary join key
        merged_df = merged_df.drop("join_key", axis=1)

        print(f"Year {year} - After join: {len(merged_df)} rows")

        # Report missing data
        consumption_missing = merged_df["consumption"].isna().sum()
        if consumption_missing > 0:
            print(f"  Warning: {consumption_missing} rows missing consumption data")

        # Check for missing weather data (exclude join_key which was dropped)
        weather_data_columns = [col for col in weather_columns if col != "join_key"]
        if weather_data_columns:
            weather_missing = merged_df[weather_data_columns].isna().any(axis=1).sum()
            if weather_missing > 0:
                print(f"  Warning: {weather_missing} rows missing weather data")

        return merged_df

    except Exception as e:
        print(f"Error loading data for year {year}: {e}")
        return None


def get_available_years(datetime_dir, consumption_dir, weather_dir):
    """Get years that have data available in all three sources."""
    datetime_years = set()
    consumption_years = set()
    weather_years = set()

    # Extract years from datetime features files
    for file in datetime_dir.glob("datetime_features_*.csv"):
        year = file.stem.split("_")[-1]
        if year.isdigit():
            datetime_years.add(int(year))

    # Extract years from consumption files
    for file in consumption_dir.glob("consumption_*.csv"):
        year = file.stem.split("_")[-1]
        if year.isdigit():
            consumption_years.add(int(year))

    # Extract years from weather files
    for file in weather_dir.glob("weather_*.csv"):
        year = file.stem.split("_")[-1]
        if year.isdigit():
            weather_years.add(int(year))

    # Return intersection of all three sets (years available in all sources)
    common_years = datetime_years & consumption_years & weather_years
    return sorted(list(common_years))


def merge_data_for_range(
    start_date, end_date, datetime_dir, consumption_dir, weather_dir
):
    """Merge data for all years within the specified date range."""
    start_year = start_date.year
    end_year = end_date.year

    # Get available years
    available_years = get_available_years(datetime_dir, consumption_dir, weather_dir)

    # Filter years within the date range
    years_to_process = [
        year for year in available_years if start_year <= year <= end_year
    ]

    if not years_to_process:
        print(f"No data available for years {start_year}-{end_year}")
        return None

    print(f"Processing years: {years_to_process}")

    merged_data_by_year = {}
    total_records = 0

    for year in tqdm(years_to_process, desc="Merging data by year"):
        merged_df = load_year_data(year, datetime_dir, consumption_dir, weather_dir)
        if merged_df is not None:
            merged_data_by_year[year] = merged_df
            total_records += len(merged_df)
            print(f"  Year {year}: {len(merged_df):,} records merged")

    if merged_data_by_year:
        print(f"Total merged records: {total_records:,}")
        return merged_data_by_year
    else:
        print("No data was successfully merged")
        return None


def save_merged_data_to_csv(merged_data_by_year, output_dir, file_prefix="merged"):
    """Save merged data split by year and also as one combined file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Saving merged data to {len(merged_data_by_year)} yearly files:")

    # Save individual year files
    for year in tqdm(sorted(merged_data_by_year.keys()), desc="Saving yearly files"):
        year_data = merged_data_by_year[year]
        filename = output_dir / f"{file_prefix}_{year}.csv"
        year_data.to_csv(filename, index=False)

    # Combine all years into one DataFrame and save as single file
    print("Creating combined file with all years...")
    all_years_data = []
    for year in sorted(merged_data_by_year.keys()):
        all_years_data.append(merged_data_by_year[year])

    if all_years_data:
        combined_df = pd.concat(all_years_data, ignore_index=True)
        combined_filename = output_dir / f"{file_prefix}_all_years.csv"

        print(f"Saving combined file with {len(combined_df):,} total records...")
        combined_df.to_csv(combined_filename, index=False)
        print(f"Combined file saved: {combined_filename}")

    print(f"All files saved to: {output_dir}")
    print(f"  - {len(merged_data_by_year)} yearly files")
    print(f"  - 1 combined file with all years")


def merge_processed_data(start_date_param=None, end_date_param=None):
    """Main merging function - entry point for main.py"""
    # Get the directory relative to main.py
    current_dir = Path(__file__).parent
    datetime_dir = current_dir / DATETIME_FEATURES_PATH
    consumption_dir = current_dir / CONSUMPTION_PATH
    weather_dir = current_dir / WEATHER_PATH
    output_dir = current_dir / MERGED_SAVE_PATH

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

    # Merge data in date range
    merged_data = merge_data_for_range(
        start_date, end_date, datetime_dir, consumption_dir, weather_dir
    )

    if merged_data is not None:
        save_merged_data_to_csv(merged_data, output_dir)
        return merged_data
    else:
        print("No data was merged.")
        return None


if __name__ == "__main__":
    END_DATE = None
    if len(sys.argv) >= 2:
        END_DATE = sys.argv[1]

    merge_processed_data(end_date_param=END_DATE)
