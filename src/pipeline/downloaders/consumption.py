"""Download daily gas consumption data from distribution networks.

GasNet provides data from 07:00 of the current day to 06:00 of the next day.
To cover 2013-01-01, the downloader starts at 2012-12-31.

This module performs robust remote CSV fetching:
- Sends browser-like request headers to reduce the chance of server-side blocking
  (some servers block non-browser User-Agents).
- Retries transient HTTP/URL errors (e.g., 403, 429, 5xx) with exponential backoff.
- Uses a per-request timeout and reads response bytes before parsing with pandas
  using multiple encoding fallbacks.
"""

import datetime
import io
import socket
import ssl
import time
import urllib.error
import urllib.request
from collections.abc import Iterable

import pandas as pd
import utils.config as config
import utils.helper_functions as utils
from tqdm import tqdm
from utils.helper_functions import DateLike

DATA_CONSUMPTION_ROOT = config.RAW_CONSUMPTION_DIR

NETWORK_URLS = {
    "gasnet": "https://www.gasnet.cz/storage/online-toky/gasnet/{date}.csv",
    "vcpnet": "https://www.gasnet.cz/storage/online-toky/vcpnet/{date}.csv",
    "jmpnet": "https://www.gasnet.cz/storage/online-toky/jmpnet/{date}.csv",
    "smpnet": "https://www.gasnet.cz/storage/online-toky/smpnet/{date}.csv",
}

# Fallback encodings to try when parsing CSV bytes.
ENCODING_FALLBACKS = ("utf-8", "cp1250", "iso-8859-2")

# HTTP fetch / retry configuration:
READ_MAX_RETRIES = 5  # total attempts before giving up
READ_BACKOFF_SECONDS = 2.0  # base backoff in seconds (exponential)
READ_TIMEOUT_SECONDS = 30  # per-request timeout in seconds

# Default request headers to appear like a normal browser and avoid being blocked
# by servers that filter requests based on User-Agent or Accept headers.
DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/122.0.0.0 Safari/537.36"
    ),
    "Accept": "text/csv,text/plain,application/octet-stream,*/*",
}

MIN_DATE_BY_NETWORK = {}


def _read_csv_with_fallback(url: str) -> pd.DataFrame:
    """Read a remote CSV using encoding fallbacks and robust HTTP fetching.

    Sends browser-like headers, enforces a timeout, and will retry on transient
    HTTP/URL errors (e.g., 403, 429, 5xx) with exponential backoff. The HTTP
    response is read as bytes and parsed with pandas using multiple encodings.

    Args:
        url: Source URL.

    Returns:
        pandas.DataFrame: Parsed CSV.

    Raises:
        UnicodeDecodeError: If all fallback encodings fail while decoding.
        urllib.error.HTTPError/URLError: If the request ultimately fails after retries.
    """
    last_error: Exception | None = None
    request = urllib.request.Request(url, headers=DEFAULT_HEADERS)

    for attempt in range(1, READ_MAX_RETRIES + 1):
        try:
            with urllib.request.urlopen(request, timeout=READ_TIMEOUT_SECONDS) as response:
                payload = response.read()
        except urllib.error.HTTPError as error:
            # Treat these status codes as transient (rate limiting or server
            # issues) and retry after backoff.
            last_error = error
            if error.code in {403, 429, 500, 502, 503, 504}:
                pass
            else:
                # For other HTTP errors, re-raise immediately.
                raise
        except (
            urllib.error.URLError,
            TimeoutError,
            socket.timeout,
            ConnectionError,
            ssl.SSLError,
        ) as error:
            # urllib can surface read timeouts and TLS/network issues via several
            # exception types depending on where the request fails.
            last_error = error
        else:
            for encoding in ENCODING_FALLBACKS:
                try:
                    return pd.read_csv(
                        io.BytesIO(payload),
                        sep=";",
                        encoding=encoding,
                    )
                except UnicodeDecodeError as error:
                    last_error = error

        if attempt < READ_MAX_RETRIES:
            sleep_for = READ_BACKOFF_SECONDS * (2 ** (attempt - 1))
            time.sleep(sleep_for)

    if last_error is not None:
        raise last_error
    # Final attempt: fetch once more and let any errors propagate so caller can
    # observe the underlying reason for failure.
    with urllib.request.urlopen(request, timeout=READ_TIMEOUT_SECONDS) as response:
        payload = response.read()
    return pd.read_csv(io.BytesIO(payload), sep=";")


def _resolve_networks(networks: Iterable[str] | None) -> list[str]:
    """Normalize requested networks and filter out unknown ones.

    Args:
        networks: Iterable of network identifiers or None.

    Returns:
        List of normalized network keys.
    """
    if networks is None:
        return list(config.DEFAULT_CONSUMPTION_NETWORKS)

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


def download_consumption_data_with_range(
    start_date: datetime.date,
    end_date_param: utils.DateLike | None = None,
    networks: Iterable[str] | None = None,
) -> None:
    """Download consumption data for a specific date range.

    Args:
        start_date: Inclusive start date.
        end_date_param: Inclusive end date as YYYY-MM-DD string, date, datetime,
            or None.
        networks: Iterable of network identifiers (defaults to
            config.DEFAULT_CONSUMPTION_NETWORKS).
    """
    end_date = utils.resolve_end_date(end_date_param)
    networks = _resolve_networks(networks)
    if not networks:
        return

    if start_date < config.CONSUMPTION_MIN_DATE:
        print(
            f"Start date cannot be before {config.CONSUMPTION_MIN_DATE} since it is \
                the first available data from previous dataset."
        )
        delta_days = (config.CONSUMPTION_MIN_DATE - start_date).days
        print(
            f"Adjusting start date by {delta_days} days "
            f"to {config.CONSUMPTION_MIN_DATE}."
        )
        start_date = config.CONSUMPTION_MIN_DATE

    if start_date > end_date:
        print(
            f"Start date {start_date} is after end date {end_date}. "
            "Nothing to download."
        )
        return

    print(f"Downloading data from {start_date} to {end_date}...")

    utils.ensure_directory(DATA_CONSUMPTION_ROOT)

    for network in networks:
        network_min_date = MIN_DATE_BY_NETWORK.get(network, config.CONSUMPTION_MIN_DATE)
        network_start_date = max(start_date, network_min_date)
        if network_start_date != start_date:
            delta_days = (network_start_date - start_date).days
            print(
                f"Network '{network}' has data starting {network_min_date}. "
                f"Adjusting start date by {delta_days} days to {network_start_date}."
            )
        if network_start_date > end_date:
            print(
                f"Skipping network '{network}' because start date {network_start_date} "
                f"is after end date {end_date}."
            )
            continue

        total_days = (end_date - network_start_date).days + 1
        print(f"Total days to download for '{network}': {total_days}")

        print(f"\nProcessing network '{network}'...")
        save_dir = DATA_CONSUMPTION_ROOT / network
        utils.ensure_directory(save_dir)
        url_template = NETWORK_URLS[network]

        for i in tqdm(range(total_days), desc=f"{network.upper()} downloads"):
            current_date = network_start_date + datetime.timedelta(days=i)
            date_str = current_date.strftime("%Y%m%d")
            file_url = url_template.format(date=date_str)
            file_path = save_dir / f"{date_str}.csv"
            if file_path.is_file():
                continue
            try:
                df = _read_csv_with_fallback(file_url)
                df.to_csv(file_path, index=False)
            except (
                UnicodeDecodeError,
                ValueError,
                pd.errors.EmptyDataError,
                urllib.error.HTTPError,
                urllib.error.URLError,
                TimeoutError,
                socket.timeout,
                ConnectionError,
                ssl.SSLError,
            ) as exc:
                print(
                    f"Failed to download data for {current_date} from {file_url}: {exc}"
                )


def download_consumption_data(
    end_date_param: DateLike | None = None,
    networks: Iterable[str] | None = None,
) -> None:
    """Download consumption data using configured defaults.

    Args:
        end_date_param: Inclusive end date as YYYY-MM-DD string, date, datetime,
            or None (defaults to last day of previous month).
        networks: Iterable of network identifiers to download (defaults to
            config.DEFAULT_CONSUMPTION_NETWORKS).
    """
    start_date = config.CONSUMPTION_DOWNLOAD_START_DATE
    download_consumption_data_with_range(start_date, end_date_param, networks=networks)
