"""Download monthly gas price data from OTE-CR."""

from __future__ import annotations

import datetime

import requests
import utils.config as config
import utils.helper_functions as utils
from tqdm import tqdm

DATA_SAVE_PATH = config.RAW_PRICE_DIR
BASE_URL = "https://www.ote-cr.cz/pubweb/attachments/127"


def _generate_months_to_download(
    start_date: datetime.date,
    end_date: datetime.date,
) -> list[tuple[int, int]]:
    """Generate (year, month) tuples for the given date range.

    Args:
        start_date: First date to include.
        end_date: Last date to include.

    Returns:
        List of (year, month) tuples.
    """
    months_to_download: list[tuple[int, int]] = []
    current_date = start_date.replace(day=1)
    end_month = end_date.replace(day=1)

    while current_date <= end_month:
        months_to_download.append((current_date.year, current_date.month))
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)

    return months_to_download


def _download_single_file(year: int, month: int) -> bool:
    """Download a single price file for the given year and month.

    Args:
        year: Target year.
        month: Target month.

    Returns:
        True if a file was downloaded, False otherwise.
    """
    month_str = f"{month:02d}"
    file_url = f"{BASE_URL}/{year}/month{month_str}/VDT_plyn_{month_str}_{year}_CZ.xls"
    file_path = DATA_SAVE_PATH / f"VDT_plyn_{month_str}_{year}_CZ.xls"

    if not file_path.is_file():
        try:
            response = requests.get(file_url, timeout=30)
            response.raise_for_status()

            with open(file_path, "wb") as f:
                f.write(response.content)

            return True

        except (requests.RequestException, ConnectionError, TimeoutError) as exc:
            print(
                f"Failed to download data for {year}-{month:02d} from {file_url}: {exc}"
            )
    return False


def download_price_data_with_range(
    start_date: datetime.date,
    end_date_param: utils.DateLike | None = None,
) -> int:
    """Download price data for a specific date range.

    Args:
        start_date: First date to include.
        end_date_param: End date as YYYY-MM-DD string, date, datetime, or None.

    Returns:
        Count of files downloaded.
    """
    end_date = utils.resolve_end_date(end_date_param)

    if start_date < config.PRICE_START_DATE:
        print(
            f"Start date cannot be before {config.PRICE_START_DATE} since it is the "
            "first available data from OTE-CR price dataset."
        )
        delta_days = (config.PRICE_START_DATE - start_date).days
        print(
            f"Adjusting start date by {delta_days} days to {config.PRICE_START_DATE}."
        )
        start_date = config.PRICE_START_DATE

    if start_date > end_date:
        print(
            f"Start date {start_date} is after end date {end_date}. "
            "Nothing to download."
        )
        return 0

    print(f"Downloading price data from {start_date} to {end_date}...")

    utils.ensure_directory(DATA_SAVE_PATH)

    months_to_download = _generate_months_to_download(start_date, end_date)
    print(f"Total months to download: {len(months_to_download)}")

    downloaded = 0
    for year, month in tqdm(months_to_download, desc="Downloading"):
        if _download_single_file(year, month):
            downloaded += 1

    return downloaded


def download_price_data(end_date_param: utils.DateLike | None = None) -> int:
    """Download price data using default start date.

    Args:
        end_date_param: End date as YYYY-MM-DD string, date, datetime, or None.

    Returns:
        Count of files downloaded.
    """
    return download_price_data_with_range(config.PRICE_START_DATE, end_date_param)
