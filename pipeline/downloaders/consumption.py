"""
Module for downloading gas consumption data from Gasnet.

Downloads daily CSV files from gasnet.cz for gas consumption data.
Data is available from 2013-01-01 onwards.
Gasnet provides data from 7:00 of the current day to 6:00 of the next day.
So for to start from 2013-01-01, we need to download from 2012-12-31.
"""

import sys
import datetime
from pathlib import Path

import pandas as pd
from tqdm import tqdm

# Get the project root directory (two levels up from this file)
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_SAVE_PATH = PROJECT_ROOT / "data" / "raw" / "gasnet"


def ensure_directory(path):
    """Ensure that the directory exists, creating it if necessary."""
    if isinstance(path, str):
        dir_path = Path(path)
    else:
        dir_path = path
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)


def download_consumption_data(end_date_param=None):
    """Main download function - entry point for main.py"""
    start_date_param = "2012-12-31"
    if end_date_param is None:
        end_date_param = (datetime.date.today() - datetime.timedelta(days=1)).strftime(
            "%Y-%m-%d"
        )

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
        2012, 12, 31
    ):  # hardcoded based on data availability from previous downloader
        print(
            "Start date cannot be before 31.12.2012 since it is the first "
            "available data from previous dataset."
        )
        delta_days = (datetime.date(2012, 12, 31) - start_date).days
        print(f"Adjusting start date by {delta_days} days to 31.12.2012.")
        start_date = datetime.date(2012, 12, 31)

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
        date_str = current_date.strftime("%Y%m%d")
        file_url = f"https://www.gasnet.cz/storage/online-toky/gasnet/{date_str}.csv"
        file_path = DATA_SAVE_PATH / f"{date_str}.csv"
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
