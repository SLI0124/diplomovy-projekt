"""Download daily gas consumption data from distribution networks.

This module downloads per-day CSV files for multiple networks (GasNet variants
and PP Distribuce) into the raw consumption directory.
"""

import datetime
from typing import Iterable, Optional

import config
import pandas as pd
import utils
from tqdm import tqdm

DATA_CONSUMPTION_ROOT = config.RAW_CONSUMPTION_DIR

NETWORK_URLS = {
    "gasnet": "https://www.gasnet.cz/storage/online-toky/gasnet/{date}.csv",
    "vcpnet": "https://www.gasnet.cz/storage/online-toky/vcpnet/{date}.csv",
    "jmpnet": "https://www.gasnet.cz/storage/online-toky/jmpnet/{date}.csv",
    "smpnet": "https://www.gasnet.cz/storage/online-toky/smpnet/{date}.csv",
    "ppnet": "https://www.ppdistribuce.cz/online-toky/csv.php?date={date}",
}

ENCODING_FALLBACKS = ("utf-8", "cp1250", "iso-8859-2")

MIN_DATE_BY_NETWORK = {
    "ppnet": config.PPNET_MIN_DATE,
}


def _read_csv_with_fallback(url: str) -> pd.DataFrame:
    """Read a remote CSV using encoding fallbacks.

    Args:
        url: Source URL.

    Returns:
        pandas.DataFrame: Parsed CSV.

    Raises:
        UnicodeDecodeError: If all fallback encodings fail.
    """
    last_error = None
    for encoding in ENCODING_FALLBACKS:
        try:
            return pd.read_csv(url, sep=";", encoding=encoding)
        except UnicodeDecodeError as error:
            last_error = error
    # Propagate the last decode error if all fallbacks failed.
    if last_error is not None:
        raise last_error
    # Should not get here, but keep default behavior for other exceptions.
    return pd.read_csv(url, sep=";")


def _normalize_ppnet_datetime(datum_series: pd.Series) -> pd.Series:
    """Normalize PPNET timestamps by inferring AM/PM from row order.

    Args:
        datum_series: Series with timestamps in "%Y-%m-%d %H:%M:%S" format.

    Returns:
        pandas.Series: Corrected timestamps as pandas datetimes.
    """
    parsed = pd.to_datetime(datum_series, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    corrected = []
    previous = None

    for value in parsed:
        if pd.isna(value):
            corrected.append(pd.NaT)
            continue

        candidate = value
        if previous is not None:
            while candidate < previous:
                candidate += pd.Timedelta(hours=12)

        corrected.append(candidate)
        previous = candidate

    return pd.Series(corrected)


def _prepare_ppnet_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize a PPNET CSV DataFrame to match downstream processing.

    Args:
        df: Raw DataFrame loaded from PPNET.

    Returns:
        pandas.DataFrame: DataFrame with columns ["Datum", "Hodnota"].

    Raises:
        ValueError: If required columns are missing.
    """
    if "Datum" not in df.columns or "Hodnota" not in df.columns:
        raise ValueError("PPNET CSV is missing required columns 'Datum'/'Hodnota'.")

    normalized_dt = pd.to_datetime(_normalize_ppnet_datetime(df["Datum"]))
    df = df.copy()
    df["Datum"] = normalized_dt.dt.strftime("%d.%m.%Y %H:%M")
    df["Hodnota"] = pd.to_numeric(df["Hodnota"], errors="coerce")
    return df[["Datum", "Hodnota"]]


def _resolve_networks(networks: Optional[Iterable[str]]) -> list[str]:
    """Normalize requested networks and filter out unknown ones.

    Args:
        networks: Iterable of network identifiers or None.

    Returns:
        List of normalized network keys.
    """
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


def download_consumption_data(
    end_date_param: Optional[utils.DateLike] = None,
    networks: Optional[Iterable[str]] = None,
) -> None:
    """Download consumption data using configured defaults.

    Args:
        end_date_param: Inclusive end date as YYYY-MM-DD string, date, datetime,
            or None (defaults to last day of previous month).
        networks: Iterable of network identifiers to download (defaults to all).
    """
    start_date = config.CONSUMPTION_DOWNLOAD_START_DATE
    end_date = utils.resolve_end_date(end_date_param)

    resolved_networks = _resolve_networks(networks)
    if not resolved_networks:
        return

    download_consumption_data_with_range(
        start_date, end_date, networks=resolved_networks
    )


def download_consumption_data_with_range(
    start_date: datetime.date,
    end_date_param: Optional[utils.DateLike] = None,
    networks: Optional[Iterable[str]] = None,
) -> None:
    """Download consumption data for a specific date range.

    Args:
        start_date: Inclusive start date.
        end_date_param: Inclusive end date as YYYY-MM-DD string, date, datetime,
            or None.
        networks: Iterable of network identifiers (defaults to all).
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
            url_date = (
                current_date.strftime("%Y-%m-%d") if network == "ppnet" else date_str
            )
            file_url = url_template.format(date=url_date)
            file_path = save_dir / f"{date_str}.csv"
            if file_path.is_file():
                continue
            try:
                df = _read_csv_with_fallback(file_url)
                if network == "ppnet":
                    df = _prepare_ppnet_dataframe(df)
                df.to_csv(file_path, index=False)
            except (UnicodeDecodeError, pd.errors.ParserError, ValueError) as e:
                print(
                    f"Failed to download data for {current_date} from {file_url}: {e}"
                )
