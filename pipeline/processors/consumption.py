"""
Module for processing raw gasnet consumption data.

This script processes raw gasnet consumption data files from ../../data/raw/gasnet/.

The gasnet data has a unique structure where each day's 24-hour period is split across two files:
- Hours 0-6 of the current day are in the previous day's file
- Hours 7-23 of the current day are in the current day's file

CSV format:
Datum,ID,Hodnota,Nazev
1.1.2013 7:00,21637,461901,Zona_Gasnet
1.1.2013 8:00,21637,426805,Zona_Gasnet
...

We extract 'Datum' (datetime) and 'Hodnota' (consumption value) columns.
The processed data maintains complete 24-hour structures with NA values for missing hours.

Processed data is saved as multiple CSV files in ../../data/processed/consumption/, grouped by year.
The catch is that it will be executed from ../main.py so create entry points accordingly.
"""

import sys
from datetime import datetime, timedelta, date
from pathlib import Path

import pandas as pd
from tqdm import tqdm

DATA_SOURCE_PATH = "../../data/raw/gasnet/"
DATA_SAVE_PATH = "../../data/processed/consumption/"


def parse_gasnet_file(file_path):
    """Parse a single gasnet CSV file and return processed DataFrame with data quality checks."""
    df = pd.read_csv(file_path, sep=",")

    # Check for files with unexpected number of lines
    total_lines = len(df)
    if total_lines < 24 or total_lines > 26:
        print("WARNING: Suspicious file detected!")
        print(f"\tFile: {file_path.name}")
        print("\tExpected: 24-26 lines (24 hours + header + padding)")
        print(f"\tFound: {total_lines} lines")
        print("\tThis file may have missing data or malformed content!")
        print("-" * 60)

    # Convert Datum to datetime to extract datetime components
    df["Datum"] = pd.to_datetime(df["Datum"], format="%d.%m.%Y %H:%M", errors="coerce")

    # Extract datetime components and handle potential NaN values
    df["year"] = df["Datum"].dt.year
    df["month"] = df["Datum"].dt.month
    df["day"] = df["Datum"].dt.day
    df["hour"] = df["Datum"].dt.hour

    # Convert Hodnota to integer, keeping NaN for missing values
    df["consumption"] = pd.to_numeric(df["Hodnota"], errors="coerce")

    # Keep all rows, including those with NaN consumption values
    result = df[["year", "month", "day", "hour", "consumption"]].copy()

    # Convert datetime components to int where possible, keep NaN as NaN
    result["year"] = result["year"].astype("Int64")
    result["month"] = result["month"].astype("Int64")
    result["day"] = result["day"].astype("Int64")
    result["hour"] = result["hour"].astype("Int64")
    result["consumption"] = result["consumption"].astype("Int64")

    return result


def create_complete_day_structure(target_date):
    """Create a complete 24-hour DataFrame structure for a given date with NA consumption values."""
    hours = list(range(24))
    complete_day = pd.DataFrame(
        {
            "year": [target_date.year] * 24,
            "month": [target_date.month] * 24,
            "day": [target_date.day] * 24,
            "hour": hours,
            "consumption": [pd.NA] * 24,
        }
    )

    # Convert to proper dtypes
    complete_day["year"] = complete_day["year"].astype("Int64")
    complete_day["month"] = complete_day["month"].astype("Int64")
    complete_day["day"] = complete_day["day"].astype("Int64")
    complete_day["hour"] = complete_day["hour"].astype("Int64")
    complete_day["consumption"] = complete_day["consumption"].astype("Int64")

    return complete_day


def process_consumption_data(end_date_param=None):
    """Main processing function - entry point for main.py"""
    # Get the directory relative to main.py
    current_dir = Path(__file__).parent
    source_dir = current_dir / DATA_SOURCE_PATH
    output_dir = current_dir / DATA_SAVE_PATH

    # Set default date range
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

    # Convert string dates to date objects
    start_date = datetime.strptime(start_date_param, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_param, "%Y-%m-%d").date()

    # Generate consumption data
    processed_data = generate_consumption_data_with_range(
        source_dir, start_date, end_date
    )

    if processed_data is not None:
        save_consumption_data_to_csv(processed_data, output_dir)
        return processed_data

    print("No consumption data was processed.")
    return None


def get_hours_from_file(file_path, target_date, hour_range):
    """Extract specific hours from a gasnet file for the target date."""
    if not file_path.exists():
        return pd.DataFrame()

    consumption_data = parse_gasnet_file(file_path)
    if consumption_data is None or len(consumption_data) == 0:
        return pd.DataFrame()

    # Filter for the target date and hour range
    target_year = target_date.year
    target_month = target_date.month
    target_day = target_date.day

    if hour_range == "early":  # hours 0-6
        filtered_data = consumption_data[
            (consumption_data["year"] == target_year)
            & (consumption_data["month"] == target_month)
            & (consumption_data["day"] == target_day)
            & (consumption_data["hour"] <= 6)
        ]
    else:  # hours 7-23
        filtered_data = consumption_data[
            (consumption_data["year"] == target_year)
            & (consumption_data["month"] == target_month)
            & (consumption_data["day"] == target_day)
            & (consumption_data["hour"] >= 7)
        ]

    return filtered_data if len(filtered_data) > 0 else pd.DataFrame()


def process_single_date(source_dir, current_date):
    """Process consumption data for a single date."""
    # Create complete 24-hour structure for this date
    complete_day = create_complete_day_structure(current_date)

    # Get file paths for previous and current day
    prev_date = current_date - timedelta(days=1)
    prev_date_str = prev_date.strftime("%Y%m%d")
    curr_date_str = current_date.strftime("%Y%m%d")

    prev_file = source_dir / f"{prev_date_str}.csv"
    curr_file = source_dir / f"{curr_date_str}.csv"

    # Collect available data from both files
    available_data = []

    # Get hours 0-6 from previous day's file
    hours_0_6 = get_hours_from_file(prev_file, current_date, "early")
    if not hours_0_6.empty:
        available_data.append(hours_0_6)

    # Get hours 7-23 from current day's file
    hours_7_23 = get_hours_from_file(curr_file, current_date, "late")
    if not hours_7_23.empty:
        available_data.append(hours_7_23)

    # Merge available data into the complete day structure
    if available_data:
        available_combined = pd.concat(available_data, ignore_index=True)
        # Update the complete day structure with available data
        for _, row in available_combined.iterrows():
            hour_idx = complete_day["hour"] == row["hour"]
            complete_day.loc[hour_idx, "consumption"] = row["consumption"]

    return complete_day


def generate_consumption_data_with_range(source_dir, start_date, end_date):
    """
    Process gasnet consumption files within date range,
    handling cross-file temporal alignment.
    """
    print(f"Processing consumption data from {start_date} to {end_date}...")

    # Create list of dates to process
    current_date = start_date
    dates_to_process = []
    while current_date <= end_date:
        dates_to_process.append(current_date)
        current_date += timedelta(days=1)

    # Process files with progress bar
    # NOTE: Gasnet data has a unique structure where each day's 24-hour period is split:
    # - Hours 0-6 of current date are in the previous day's file
    # - Hours 7-23 of current date are in the current day's file
    all_data = []
    for process_date in tqdm(dates_to_process, desc="Processing files"):
        day_data = process_single_date(source_dir, process_date)
        all_data.append(day_data)

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)
        total_hours = len(combined_df)
        missing_hours = combined_df["consumption"].isna().sum()
        available_hours = total_hours - missing_hours
        print(
            f"Processed {total_hours:,} total hours "
            f"({available_hours:,} with data, {missing_hours:,} with NA)"
        )
        return combined_df

    print("No data found")
    return None


def save_consumption_data_to_csv(df, output_dir, file_prefix="consumption"):
    """Save consumption data to CSV files grouped by year."""
    output_dir.mkdir(parents=True, exist_ok=True)

    years = sorted(df["year"].unique())
    print(f"Saving data to {len(years)} files:")

    for year in tqdm(years, desc="Saving files"):
        year_data = df[df["year"] == year][
            ["year", "month", "day", "hour", "consumption"]
        ]
        filename = output_dir / f"{file_prefix}_{int(year)}.csv"
        year_data.to_csv(filename, index=False)

    print(f"All files saved to: {output_dir}")


if __name__ == "__main__":
    END_DATE = None
    if len(sys.argv) >= 2:
        END_DATE = sys.argv[1]

    process_consumption_data(end_date_param=END_DATE)
