"""
Module for downloading gas price data from OTE-CR.

Downloads monthly Excel files from ote-cr.cz for gas price data.
Data is available from 2013-01 onwards.
Each month has its own Excel file in the format: VDT_plyn_MM_YYYY_CZ.xls

Sample URL: https://www.ote-cr.cz/pubweb/attachments/127/2024/month06/VDT_plyn_06_2024_CZ.xls
"""

import sys
import datetime
from pathlib import Path

import requests
from tqdm import tqdm

# Get the project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_SAVE_PATH = PROJECT_ROOT / "data" / "raw" / "price"


def ensure_directory(path):
    """Ensure that the directory exists, creating it if necessary."""
    if isinstance(path, str):
        dir_path = Path(path)
    else:
        dir_path = path
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)


def _generate_months_to_download(start_date: datetime.date, end_date: datetime.date):
    """Generate list of (year, month) tuples for the given date range."""
    months_to_download = []
    current_date = start_date.replace(day=1)  # Start from first day of start month
    end_month = end_date.replace(day=1)  # End at first day of end month

    while current_date <= end_month:
        months_to_download.append((current_date.year, current_date.month))
        # Move to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)

    return months_to_download


def _download_single_file(year: int, month: int):
    """Download a single price file for the given year and month."""
    month_str = f"{month:02d}"
    file_url = (
        f"https://www.ote-cr.cz/pubweb/attachments/127/{year}/"
        f"month{month_str}/VDT_plyn_{month_str}_{year}_CZ.xls"
    )
    file_path = DATA_SAVE_PATH / f"VDT_plyn_{month_str}_{year}_CZ.xls"

    if not file_path.is_file():
        try:
            response = requests.get(file_url, timeout=30)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(response.content)

        except (requests.RequestException, ConnectionError, TimeoutError) as e:
            print(
                f"Failed to download data for {year}-{month:02d} from {file_url}: {e}"
            )


def download_price_data(end_date_param=None):
    """Main download function - entry point for main.py"""
    start_date_param = "2013-01-01"
    if end_date_param is None:
        # Get last day of previous month
        today = datetime.date.today()
        first_day_current_month = today.replace(day=1)
        end_date_param = (
            first_day_current_month - datetime.timedelta(days=1)
        ).strftime("%Y-%m-%d")

    # Convert string dates to date objects
    start_date = datetime.datetime.strptime(start_date_param, "%Y-%m-%d").date()

    if isinstance(end_date_param, str):
        end_date = datetime.datetime.strptime(end_date_param, "%Y-%m-%d").date()
    else:
        end_date = end_date_param

    # Validate date format if needed
    if end_date_param is not None and isinstance(end_date_param, str):
        try:
            datetime.datetime.strptime(end_date_param, "%Y-%m-%d")
        except ValueError:
            print(
                f"ERROR: Invalid date format '{end_date_param}'. "
                f"Please use YYYY-MM-DD format."
            )
            sys.exit(1)

    download_price_data_with_range(start_date, end_date)


def download_price_data_with_range(start_date: datetime.date, end_date_param=None):
    """Download price data for a specific date range."""
    if end_date_param is None:
        # Get last day of previous month
        today = datetime.date.today()
        first_day_current_month = today.replace(day=1)
        end_date_param = first_day_current_month - datetime.timedelta(days=1)
        print(
            f"No end date provided. Using last day of previous month: {end_date_param}"
        )

    # normalize if user passed a datetime
    if isinstance(end_date_param, datetime.datetime):
        end_date_param = end_date_param.date()

    if start_date < datetime.date(2013, 1, 1):  # hardcoded based on data availability
        print(
            "Start date cannot be before 01.01.2013 since it is the first "
            "available data from OTE-CR price dataset."
        )
        delta_days = (datetime.date(2013, 1, 1) - start_date).days
        print(f"Adjusting start date by {delta_days} days to 01.01.2013.")
        start_date = datetime.date(2013, 1, 1)

    if start_date > end_date_param:
        print(
            f"Start date {start_date} is after end date {end_date_param}. Nothing to download."
        )
        return

    print(f"Downloading price data from {start_date} to {end_date_param}...")

    ensure_directory(DATA_SAVE_PATH)

    # Generate list of months to download
    months_to_download = _generate_months_to_download(start_date, end_date_param)
    print(f"Total months to download: {len(months_to_download)}")

    # Download each file
    for year, month in tqdm(months_to_download, desc="Downloading"):
        _download_single_file(year, month)


if __name__ == "__main__":
    END_DATE = None
    if len(sys.argv) >= 2:
        END_DATE = sys.argv[1]

    download_price_data(end_date_param=END_DATE)
