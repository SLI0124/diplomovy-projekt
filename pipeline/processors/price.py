"""
Module for processing raw price data.

This script processes raw price data files from ../../data/raw/price/.

The XLS files contain gas trading data with the following structure:
- Multiple header rows, data starts from row 5 (0-indexed row 4)
- Columns are:
  - Plynárenský den (trading day)
  - Zobchodované množství (MWh) (traded volume)
  - Vážený průměr cen (EUR/MWh) (weighted average price)
  - Min. cena (EUR/MWh) (minimum price)
  - Max. cena (EUR/MWh) (maximum price)

The catch is that dates are in daily format, not hourly like in weather or consumption data.
Data is available for each day: 01/01/2013, 02/01/2013, ..., 31/01/2013, etc.

Files are organized by month: VDT_plyn_MM_YYYY_CZ.xls

Processed data is saved as multiple CSV files in ../../data/processed/price/, grouped by year.
The catch is that it will be executed from ../main.py so create entry points accordingly.
"""

import sys
from datetime import datetime, timedelta, date
from pathlib import Path

import pandas as pd
from tqdm import tqdm


def get_last_day_of_previous_month():
    """Calculate the last day of the previous month."""
    today = date.today()
    # First day of current month
    first_day_current_month = today.replace(day=1)
    # Last day of previous month
    last_day_previous_month = first_day_current_month - timedelta(days=1)
    return last_day_previous_month.strftime("%Y-%m-%d")


DATA_SOURCE_PATH = "../../data/raw/price/"
DATA_SAVE_PATH = "../../data/processed/price/"


def parse_price_file(file_path):
    """Parse a price XLS file and return processed DataFrame with exactly 7 columns."""
    try:
        # Try xlrd first for .xls files, then openpyxl for .xlsx
        try:
            df = pd.read_excel(file_path, skiprows=4, engine="xlrd")
        except Exception:
            df = pd.read_excel(file_path, skiprows=4, engine="openpyxl")

        # Skip the first row which contains the Czech headers
        df = df.iloc[1:].copy()

        # Convert date column (first column) to datetime and extract components
        date_col = pd.to_datetime(df.iloc[:, 0], errors="coerce")

        # Create daily result DataFrame with converted data
        daily_result = pd.DataFrame(
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

        # Remove rows with invalid dates
        daily_result = daily_result.dropna(subset=["year", "month", "day"])

        # Expand each day to 24 hours (0-23) with same price values
        hourly_data = []
        for _, row in daily_result.iterrows():
            for hour in range(24):
                hourly_row = {
                    "year": row["year"],
                    "month": row["month"],
                    "day": row["day"],
                    "hour": hour,
                    "traded_volume_mwh": row["traded_volume_mwh"],
                    "weighted_avg_price_eur_mwh": row["weighted_avg_price_eur_mwh"],
                    "min_price_eur_mwh": row["min_price_eur_mwh"],
                    "max_price_eur_mwh": row["max_price_eur_mwh"],
                }
                hourly_data.append(hourly_row)

        result = pd.DataFrame(hourly_data)
        return result

    except (
        pd.errors.EmptyDataError,
        pd.errors.ParserError,
        FileNotFoundError,
        ValueError,
        ImportError,
    ) as e:
        print(f"\tError parsing {file_path}: {e}")
        return pd.DataFrame()


def process_price_data_with_range(source_dir, start_date, end_date):
    """Process price data files within the specified date range."""
    print(f"Processing price data from {start_date} to {end_date}...")

    # Find all price XLS files
    price_files = list(source_dir.glob("VDT_plyn_*.xls"))

    if not price_files:
        print("\tError: No price data files found")
        return None

    print(f"\tFound {len(price_files)} price files")

    all_data = []
    start_year = start_date.year
    end_year = end_date.year

    # Process each file
    for file_path in tqdm(price_files, desc="Processing price files"):
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

        except (IndexError, ValueError) as e:
            print(f"\tWarning: Could not parse filename {file_path.name}: {e}")
            continue

    if all_data:
        combined_df = pd.concat(all_data, ignore_index=True)

        # Further filter by exact date range
        combined_df = combined_df[
            (combined_df["year"] >= start_year) & (combined_df["year"] <= end_year)
        ].copy()

        # Sort by date
        combined_df = combined_df.sort_values(["year", "month", "day"]).reset_index(
            drop=True
        )

        print(f"\tProcessed {len(combined_df):,} price records")
        return combined_df

    print("\tNo price data found")
    return None


def save_processed_price_data_to_csv(df, output_dir, file_prefix="price"):
    """Save price data split by year."""
    print("Saving processed price data...")
    output_dir.mkdir(parents=True, exist_ok=True)

    years = sorted(df["year"].unique())
    print(
        f"\tSplitting data into {len(years)} yearly files ({min(years)}-{max(years)})"
    )

    for year in tqdm(years, desc="Saving price files", unit="file"):
        year_data = df[df["year"] == year]
        filename = output_dir / f"{file_prefix}_{year}.csv"
        year_data.to_csv(filename, index=False)

    print(f"\tAll price files saved to: {output_dir}")


def process_price_data(end_date_param=None):
    """Main processing function - entry point for main.py"""
    # Get the directory relative to main.py
    current_dir = Path(__file__).parent
    source_dir = current_dir / DATA_SOURCE_PATH
    output_dir = current_dir / DATA_SAVE_PATH

    start_date_param = "2013-01-01"
    if end_date_param is None:
        end_date_param = get_last_day_of_previous_month()

    print(f"Date range: {start_date_param} to {end_date_param}")
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")

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

    # Process price data in date range
    processed_data = process_price_data_with_range(source_dir, start_date, end_date)

    if processed_data is not None:
        save_processed_price_data_to_csv(processed_data, output_dir)
        return processed_data

    return None


if __name__ == "__main__":
    END_DATE = None
    if len(sys.argv) >= 2:
        END_DATE = sys.argv[1]

    process_price_data(end_date_param=END_DATE)
