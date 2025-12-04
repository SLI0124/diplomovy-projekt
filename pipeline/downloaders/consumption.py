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
DATA_CONSUMPTION_ROOT = PROJECT_ROOT / "data" / "raw" / "consumption"

NETWORK_URLS = {
    "gasnet": "https://www.gasnet.cz/storage/online-toky/gasnet/{date}.csv",
    "vcpnet": "https://www.gasnet.cz/storage/online-toky/vcpnet/{date}.csv",
    "jmpnet": "https://www.gasnet.cz/storage/online-toky/jmpnet/{date}.csv",
    "smpnet": "https://www.gasnet.cz/storage/online-toky/smpnet/{date}.csv",
}

ENCODING_FALLBACKS = ("utf-8", "cp1250", "iso-8859-2")


def ensure_directory(path):
    """Ensure that the directory exists, creating it if necessary."""
    if isinstance(path, str):
        dir_path = Path(path)
    else:
        dir_path = path
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)


def _read_csv_with_fallback(url):
    """Read CSV trying multiple encodings before failing."""
    last_error = None
    for encoding in ENCODING_FALLBACKS:
        try:
            return pd.read_csv(url, sep=";", encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    # Propagate the last decode error if all fallbacks failed.
    if last_error is not None:
        raise last_error
    # Should not get here, but keep default behaviour for other exceptions.
    return pd.read_csv(url, sep=";")


def _resolve_networks(networks):
    """Normalize requested networks and filter out unknown ones."""
    if networks is None:
        return list(NETWORK_URLS)

    resolved = []
    for network in networks:
        key = network.lower()
        if key not in NETWORK_URLS:
            print(f"WARNING: Unknown network '{network}'. Skipping.")
            continue
        if key not in resolved:
            resolved.append(key)

    if not resolved:
        print("No valid networks requested. Nothing to download.")

    return resolved


def download_consumption_data(end_date_param=None, networks=None):
    """Main download function - entry point for main.py.

    Args:
        end_date_param: Inclusive end date in YYYY-MM-DD format. Defaults to last
            day of previous month when omitted.
        networks: Iterable of network identifiers to download. Defaults to all
            supported networks when omitted.
    """
    start_date_param = "2012-12-31"
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

    resolved_networks = _resolve_networks(networks)
    if not resolved_networks:
        return

    download_consumption_data_with_range(
        start_date, end_date, networks=resolved_networks
    )


def download_consumption_data_with_range(
    start_date: datetime.date, end_date_param=None, networks=None
):
    """Download consumption data for a specific date range."""
    if networks is None:
        networks = _resolve_networks(None)
    else:
        networks = _resolve_networks(networks)

    if not networks:
        return
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

    total_days = (end_date_param - start_date).days + 1
    print(f"Total days to download per network: {total_days}")

    ensure_directory(DATA_CONSUMPTION_ROOT)

    for network in networks:
        print(f"\nProcessing network '{network}'...")
        save_dir = DATA_CONSUMPTION_ROOT / network
        ensure_directory(save_dir)
        url_template = NETWORK_URLS[network]

        for i in tqdm(range(total_days), desc=f"{network.upper()} downloads"):
            current_date = start_date + datetime.timedelta(days=i)
            date_str = current_date.strftime("%Y%m%d")
            file_url = url_template.format(date=date_str)
            file_path = save_dir / f"{date_str}.csv"
            if file_path.is_file():
                continue
            try:
                df = _read_csv_with_fallback(file_url)
                df.to_csv(file_path, index=False)
            except Exception as e:
                print(
                    f"Failed to download data for {current_date} from {file_url}: {e}"
                )


if __name__ == "__main__":
    END_DATE = None
    NETWORK_ARGS = None
    if len(sys.argv) >= 2:
        END_DATE = sys.argv[1]
    if len(sys.argv) >= 3:
        NETWORK_ARGS = sys.argv[2:]

    download_consumption_data(end_date_param=END_DATE, networks=NETWORK_ARGS)
