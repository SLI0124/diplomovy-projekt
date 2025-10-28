"""
Module for downloading gas consumption data from Gasnet.

Downloads daily CSV files from gasnet.cz for gas consumption data.
Data is available from 2013-01-01 onwards.
"""

import sys
import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

DATA_SAVE_PATH = "../data/raw/gasnet/"


def ensure_directory(path: str):
    """Ensure that the directory exists, creating it if necessary."""
    dir_path = Path(path)
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)


def download_consumption_data(start_date_param=None, end_date_param=None):
    """Main download function - entry point for main.py"""
    if start_date_param is None:
        start_date_param = "2013-01-01"
    if end_date_param is None:
        end_date_param = (datetime.date.today() - datetime.timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )

    # Convert string dates to date objects
    if isinstance(start_date_param, str):
        start_date = datetime.datetime.strptime(start_date_param, "%Y-%m-%d").date()
    else:
        start_date = start_date_param

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

    download_consumption_data_with_range(start_date, end_date)


def download_consumption_data_with_range(
    start_date: datetime.date, end_date_param=None
):
    """Download consumption data for a specific date range."""
    if end_date_param is None:
        end_date_param = datetime.date.today() - datetime.timedelta(days=1)
        print(
            f"No end date provided. Using last complete day, which is yesterday: {end_date_param}"
        )

    # normalize if user passed a datetime
    if isinstance(end_date_param, datetime.datetime):
        end_date_param = end_date_param.date()

    if start_date < datetime.date(
        2013, 1, 1
    ):  # hardcoded based on data availability from previous downloader
        print(
            "Start date cannot be before 1.1.2013 since it is the first "
            "available data from previous dataset."
        )
        delta_days = (datetime.date(2013, 1, 1) - start_date).days
        print(f"Adjusting start date by {delta_days} days to 1.1.2013.")
        start_date = datetime.date(2013, 1, 1)

    if start_date > end_date_param:
        print(
            f"Start date {start_date} is after end date {end_date_param}. Nothing to download."
        )
        return

    print(f"Downloading data from {start_date} to {end_date_param}...")

    ensure_directory(DATA_SAVE_PATH)

    total_days = (end_date_param - start_date).days + 1
    print(f"Total days to download: {total_days}")
    for i in tqdm(range(total_days), desc="Downloading"):
        current_date = start_date + datetime.timedelta(days=i)
        date_str = current_date.strftime('%Y%m%d')
        file_url = f"https://www.gasnet.cz/storage/online-toky/gasnet/{date_str}.csv"
        file_path = Path(DATA_SAVE_PATH) / f"{date_str}.csv"
        if not file_path.is_file():
            try:
                df = pd.read_csv(file_url, sep=";")
                df.to_csv(file_path, index=False)
            except (pd.errors.ParserError, ConnectionError, FileNotFoundError) as e:
                print(
                    f"Failed to download data for {current_date} from {file_url}: {e}"
                )


if __name__ == "__main__":
    END_DATE = None
    if len(sys.argv) >= 2:
        END_DATE = sys.argv[1]

    download_consumption_data(end_date_param=END_DATE)
