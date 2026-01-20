"""
Module for downloading gas price data from OTE-CR.

Downloads monthly Excel files from ote-cr.cz for gas price data.
Data is available from 2013-01 onwards.
Each month has its own Excel file in the format: VDT_plyn_MM_YYYY_CZ.xls

Sample URL: https://www.ote-cr.cz/pubweb/attachments/127/2024/month06/VDT_plyn_06_2024_CZ.xls
"""

import datetime
import sys

import config
import requests
import utils
from tqdm import tqdm

DATA_SAVE_PATH = config.RAW_PRICE_DIR


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
    start_date = config.PRICE_START_DATE
    try:
        end_date = utils.resolve_end_date(end_date_param)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    download_price_data_with_range(start_date, end_date)


def download_price_data_with_range(start_date: datetime.date, end_date_param=None):
    """Download price data for a specific date range."""
    if end_date_param is None:
        end_date_param = utils.get_last_day_of_previous_month()
        print(
            f"No end date provided. Using last day of previous month: {end_date_param}"
        )

    # normalize if user passed a datetime
    if isinstance(end_date_param, datetime.datetime):
        end_date_param = end_date_param.date()

    if start_date < config.PRICE_START_DATE:
        print(
            "Start date cannot be before 01.01.2013 since it is the first "
            "available data from OTE-CR price dataset."
        )
        delta_days = (config.PRICE_START_DATE - start_date).days
        print(f"Adjusting start date by {delta_days} days to 01.01.2013.")
        start_date = config.PRICE_START_DATE

    if start_date > end_date_param:
        print(
            f"Start date {start_date} is after end date {end_date_param}. \
                Nothing to download."
        )
        return

    print(f"Downloading price data from {start_date} to {end_date_param}...")

    utils.ensure_directory(DATA_SAVE_PATH)

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
