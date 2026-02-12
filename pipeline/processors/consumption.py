"""Process raw network consumption CSVs into hourly per-network series."""

from collections.abc import Iterable
from datetime import date, timedelta
from pathlib import Path

import config
import pandas as pd
from tqdm import tqdm

import pipeline.utils as utils

DATA_SOURCE_ROOT = config.RAW_CONSUMPTION_DIR
DATA_SAVE_PATH = config.PROCESSED_CONSUMPTION_DIR

MIN_FILE_DATE_BY_NETWORK = {
    "ppnet": config.PPNET_MIN_DATE,
}


def _assert_raw_files_cover_range(
    network_dirs: dict[str, Path],
    start_date: date,
    end_date: date,
) -> None:
    """Ensure raw daily CSV files exist for the requested processing window.

    Consumption processing needs daily files from (start_date - 1) through end_date
    due to the 07:00..06:00 day overlap.
    """
    if not config.RAW_CONSUMPTION_DIR.exists():
        raise FileNotFoundError(
            f"Consumption: raw directory not found: {config.RAW_CONSUMPTION_DIR}. "
            "Run the downloader first (pipeline/main.py --download consumption)."
        )

    file_start = start_date - timedelta(days=1)
    file_end = end_date

    for network, directory in network_dirs.items():
        if not directory.exists():
            raise FileNotFoundError(
                f"Consumption: raw directory for network '{network}' not found: "
                f"{directory}"
            )

        min_available = MIN_FILE_DATE_BY_NETWORK.get(
            network, config.CONSUMPTION_MIN_DATE
        )
        required_start = max(file_start, min_available)
        required_end = file_end
        if required_start > required_end:
            continue

        missing: list[Path] = []
        for d in utils.iter_dates(required_start, required_end):
            expected = directory / f"{d.strftime('%Y%m%d')}.csv"
            if not expected.is_file():
                missing.append(expected)

        utils.raise_missing_inputs(
            what=f"consumption processing ({network})",
            missing_paths=missing,
            required_range=f"{required_start}..{required_end}",
        )


def _normalize_ppnet_datetime(datum_series: pd.Series) -> pd.Series:
    """Parse PPNET timestamps and make them non-decreasing by adding 12h steps. Those
    timestamps lack AM/PM designations, so we must distinguish them based on order.

    Args:
        datum_series: Series with timestamps in "%Y-%m-%d %H:%M:%S" format.

    Returns:
        pd.Series: Corrected timestamps as pandas datetimes.
    """
    # Parse timestamps and fix missing AM/PM by making series non-decreasing
    parsed = pd.to_datetime(datum_series, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    out, prev = [], None
    for parsed_time in parsed:
        if pd.isna(parsed_time):
            out.append(pd.NaT)
            continue
        # If time went backwards, add 12-hour steps until it's non-decreasing
        while prev is not None and parsed_time < prev:
            parsed_time += pd.Timedelta(hours=12)
        out.append(parsed_time)
        prev = parsed_time
    return pd.Series(out)


def discover_network_paths() -> dict[str, Path]:
    """Return mapping of available consumption networks to their data directories.

    Returns:
        Dict[str, Path]: Mapping of network keys to their data directories.
    """
    if not DATA_SOURCE_ROOT.exists():
        return {}
    return {
        item.name.lower(): item for item in DATA_SOURCE_ROOT.iterdir() if item.is_dir()
    }


def parse_consumption_file(file_path: Path, network: str) -> pd.DataFrame:
    """Parse a single consumption CSV file and return normalized data.

    Args:
        file_path: Path to the consumption CSV file.
        network: Network key (e.g., "ppnet", "vcpnet", etc.).

    Returns:
        pd.DataFrame: DataFrame with columns
        ["year", "month", "day", "hour", "consumption"].
    """

    if network == "ppnet":
        df = pd.read_csv(file_path, sep=None, engine="python")
        df.columns = df.columns.str.strip().str.strip('"')
    else:
        df = pd.read_csv(file_path, sep=",")

    if network == "ppnet":
        datum_series = _normalize_ppnet_datetime(df["Datum"])
    else:
        datum_series = pd.to_datetime(
            df["Datum"], format="%d.%m.%Y %H:%M", errors="coerce"
        )
    df["Datum"] = datum_series
    datetime_index = pd.DatetimeIndex(datum_series)
    df["year"] = datetime_index.year
    df["month"] = datetime_index.month
    df["day"] = datetime_index.day
    df["hour"] = datetime_index.hour
    df["consumption"] = pd.to_numeric(df["Hodnota"], errors="coerce")

    result = df[["year", "month", "day", "hour", "consumption"]].copy()
    for column in ("year", "month", "day", "hour", "consumption"):
        result[column] = result[column].astype("Int64")

    return result


def _build_hourly_calendar_frame(start_date: date, end_date: date) -> pd.DataFrame:
    """Create a full hourly calendar frame for [start_date, end_date] inclusive.

    Args:
        start_date: Inclusive start date.
        end_date: Inclusive end date.

    Returns:
        pd.DataFrame: DataFrame with columns ["year", "month", "day", "hour"].
    """
    all_dates = pd.date_range(start_date, end_date, freq="D")
    index = pd.MultiIndex.from_product([all_dates, range(24)], names=["date", "hour"])
    frame = index.to_frame(index=False)
    frame["year"] = frame["date"].dt.year.astype("Int64")  # type: ignore[reportAttributeAccessIssue]
    frame["month"] = frame["date"].dt.month.astype("Int64")  # type: ignore[reportAttributeAccessIssue]
    frame["day"] = frame["date"].dt.day.astype("Int64")  # type: ignore[reportAttributeAccessIssue]
    frame["hour"] = frame["hour"].astype("Int64")
    return frame[["year", "month", "day", "hour"]]


def _collect_network_range_values(
    source_dir: Path,
    network: str,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Collect hourly consumption for a network in a date range.

    Important: For calendar day D, hours 0-6 are stored in file D-1 and hours 7-23
    are stored in file D. By reading files from start_date-1 through end_date and
    filtering by the parsed calendar date, we keep the original day-boundary logic.
    """
    file_start = start_date - timedelta(days=1)
    file_end = end_date

    frames = []
    current = file_start
    while current <= file_end:
        file_path = source_dir / f"{current.strftime('%Y%m%d')}.csv"
        if file_path.exists():
            frames.append(parse_consumption_file(file_path, network))
        current += timedelta(days=1)

    if not frames:
        return pd.DataFrame(
            columns=["year", "month", "day", "hour", f"consumption_{network}"]
        )

    merged = pd.concat(frames, ignore_index=True)
    merged = merged.dropna(subset=["year", "month", "day", "hour"]).copy()

    # Filter down to requested calendar window.
    date_series = pd.to_datetime(
        {
            "year": merged["year"].astype("Int64"),
            "month": merged["month"].astype("Int64"),
            "day": merged["day"].astype("Int64"),
        },
        errors="coerce",
    )
    mask = (date_series.dt.date >= start_date) & (date_series.dt.date <= end_date)
    merged = merged.loc[mask, ["year", "month", "day", "hour", "consumption"]]
    merged = merged.drop_duplicates(
        subset=["year", "month", "day", "hour"], keep="last"
    )
    merged = merged.rename(columns={"consumption": f"consumption_{network}"})
    return merged


def create_day_structure(target_date: date, networks: Iterable[str]) -> pd.DataFrame:
    """Create a complete 24-hour structure with columns for each network.

    Args:
        target_date: Date for which to create the structure.
        networks: Iterable of network keys to create columns for.

    Returns:
        pd.DataFrame: DataFrame with 24 rows and columns for year, month, day, hour,
            and consumption per network.
    """
    networks = list(networks)
    hours = list(range(24))
    day_frame = pd.DataFrame(
        {
            "year": pd.Series([target_date.year] * 24, dtype="Int64"),
            "month": pd.Series([target_date.month] * 24, dtype="Int64"),
            "day": pd.Series([target_date.day] * 24, dtype="Int64"),
            "hour": pd.Series(hours, dtype="Int64"),
        }
    )

    for network in networks:
        day_frame[f"consumption_{network}"] = pd.Series([pd.NA] * 24, dtype="Int64")

    return day_frame


def get_hours_from_file(
    file_path: Path, target_date: date, hour_range: str, network: str
) -> pd.DataFrame:
    """
    Extract specific hours from a network file for the target date.

    This parses the entire file and then filters rows that match the target date
    (year, month, day). After selecting the target date rows, it further filters
    by hour_range:
      - "early": select hours 0-6 (inclusive)
      - "late": select hours 7-23 (inclusive)

    Args:
        file_path: Path to the consumption CSV file.
        target_date: Date for which to extract hours.
        hour_range: Range of hours to extract ("early" or "late").
        network: Network key (e.g., "ppnet", "vcpnet", etc.).

    Returns:
        pd.DataFrame: DataFrame with columns ["hour", "consumption"]
        for the specified hours. Empty DataFrame if file missing or no data.
    """
    if not file_path.exists():
        return pd.DataFrame()

    consumption_data = parse_consumption_file(file_path, network)
    if consumption_data.empty:
        return pd.DataFrame()

    # Build a mask that selects rows matching the target date (year, month, day).
    # This ensures we only consider measurements that belong to the requested day.
    mask = (
        (consumption_data["year"] == target_date.year)
        & (consumption_data["month"] == target_date.month)
        & (consumption_data["day"] == target_date.day)
    )

    # Further restrict by hour range:
    # - "early" selects hours 0 through 6 (midnight through 6:00)
    # - otherwise select hours 7 through 23 (7:00 through 23:00)
    # it is there cause days intersect at 6:00 of the next/ previous day
    if hour_range == "early":
        mask &= consumption_data["hour"] <= 6
    else:
        mask &= consumption_data["hour"] >= 7

    filtered = consumption_data.loc[mask, ["hour", "consumption"]]
    return filtered.reset_index(drop=True)


def collect_network_day_values(
    source_dir: Path, current_date: date, network: str
) -> pd.DataFrame:
    """Collect hourly consumption data for a single network and day.

    Args:
        source_dir: Directory containing consumption CSV files.
        current_date: Date for which to collect data.
        network: Network key (e.g., "ppnet", "vcpnet", etc.).

    Returns:
        pd.DataFrame: DataFrame with columns ["hour", "consumption"]
        for the specified network and date. Empty if no data found.
    """
    prev_date = current_date - timedelta(days=1)
    prev_file = source_dir / f"{prev_date.strftime('%Y%m%d')}.csv"
    curr_file = source_dir / f"{current_date.strftime('%Y%m%d')}.csv"

    segments = []
    early_hours = get_hours_from_file(prev_file, current_date, "early", network)
    if not early_hours.empty:
        segments.append(early_hours)
    late_hours = get_hours_from_file(curr_file, current_date, "late", network)
    if not late_hours.empty:
        segments.append(late_hours)

    if not segments:
        return pd.DataFrame(columns=["hour", "consumption"])

    merged = pd.concat(segments, ignore_index=True)
    return merged.drop_duplicates(subset="hour", keep="last")


def process_single_date(
    network_dirs: dict[str, Path], current_date: date
) -> pd.DataFrame:
    """Process consumption data for all networks on a single date.

    Args:
        network_dirs: Directories for each network.
        current_date: date: Date to process.

    Returns:
        pd.DataFrame: DataFrame with consumption data for all networks for given date.
    """
    day_frame = create_day_structure(current_date, network_dirs.keys())

    for network, directory in network_dirs.items():
        column = f"consumption_{network}"
        network_data = collect_network_day_values(directory, current_date, network)
        if network_data.empty:
            continue
        for _, row in network_data.iterrows():
            hour_mask = day_frame["hour"] == row["hour"]
            day_frame.loc[hour_mask, column] = row["consumption"]

    return day_frame


def generate_consumption_data_with_range(
    network_dirs: dict[str, Path], start_date: date, end_date: date
) -> pd.DataFrame | None:
    """Process consumption files for all networks within the given date range.

    Args:
        network_dirs: Directories for each network.
        start_date: Start date of the range to process.
        end_date: End date of the range to process.

    Returns:
        Optional[pd.DataFrame]: Combined consumption data for all networks within
        the date range, or None if no data was processed.
    """
    if not network_dirs:
        print(f"Consumption: no networks in {DATA_SOURCE_ROOT}")
        return None

    networks_list = ", ".join(sorted(network_dirs.keys()))
    print(f"Consumption: {start_date} -> {end_date} ({networks_list})")

    combined_df = _build_hourly_calendar_frame(start_date, end_date)

    for network, directory in tqdm(
        list(network_dirs.items()), desc="Consumption: networks", leave=False
    ):
        network_df = _collect_network_range_values(
            directory, network, start_date, end_date
        )
        if network_df.empty:
            combined_df[f"consumption_{network}"] = pd.Series(
                [pd.NA] * len(combined_df), dtype="Int64"
            )
            continue

        combined_df = combined_df.merge(
            network_df,
            on=["year", "month", "day", "hour"],
            how="left",
        )

    if combined_df.empty:
        print("Consumption: no data")
        return None

    consumption_columns = [f"consumption_{network}" for network in network_dirs]
    combined_df["consumption_total"] = (
        combined_df[consumption_columns].sum(axis=1, min_count=1).astype("Int64")
    )
    total_hours = len(combined_df) * len(consumption_columns)
    missing_hours = combined_df[consumption_columns].isna().sum().sum()
    available_hours = total_hours - missing_hours

    print(
        f"Consumption: processed {total_hours:,} network-hours "
        f"({available_hours:,} ok, {missing_hours:,} NA)"
    )

    return combined_df


def save_consumption_data_to_csv(
    df: pd.DataFrame, output_dir: Path, file_prefix="consumption"
) -> None:
    """Save consumption data to CSV files grouped by year."""
    utils.ensure_directory(output_dir)

    consumption_columns = sorted(
        column for column in df.columns if column.startswith("consumption_")
    )
    if (
        "consumption_total" in df.columns
        and "consumption_total" not in consumption_columns
    ):
        consumption_columns.append("consumption_total")
    years = sorted(df["year"].dropna().unique())

    for year in tqdm(years, desc="Consumption: saving", unit="file", leave=False):
        year_mask = df["year"] == year
        year_data = df.loc[
            year_mask, ["year", "month", "day", "hour", *consumption_columns]
        ]
        filename = output_dir / f"{file_prefix}_{int(year)}.csv"
        year_data.to_csv(filename, index=False)

    if years:
        year_min = int(min(years))
        year_max = int(max(years))
        print(
            f"Consumption: saved {len(years)} files ({year_min}-{year_max}) "
            f"-> {output_dir}"
        )


def process_consumption_data(
    end_date_param: utils.DateLike | None = None,
    networks: Iterable[str] | None = None,
) -> pd.DataFrame | None:
    """Entry point used by [pipeline/main.py](pipeline/main.py).

    Args:
        end_date_param: Inclusive end date as YYYY-MM-DD string, date, datetime,
            or None.
        networks: Optional list of network keys to process.

    Returns:
        Processed DataFrame or None if nothing was processed.
    """
    start_date = config.CONSUMPTION_PROCESS_START_DATE
    end_date = utils.resolve_end_date(end_date_param)

    available_dirs = discover_network_paths()
    if not available_dirs:
        print(f"Consumption: no data directories in {DATA_SOURCE_ROOT}")
        return None

    selected_dirs = available_dirs
    if networks:
        normalized = [network.lower() for network in networks]
        missing = [network for network in normalized if network not in available_dirs]
        if missing:
            print("Consumption: missing networks: " + ", ".join(sorted(set(missing))))
        selected_dirs = {
            name: path for name, path in available_dirs.items() if name in normalized
        }
        if not selected_dirs:
            print("Consumption: no valid networks selected")
            return None
    else:
        preferred = [
            network
            for network in config.DEFAULT_CONSUMPTION_NETWORKS
            if network in available_dirs
        ]
        if preferred:
            selected_dirs = {name: available_dirs[name] for name in preferred}
        else:
            selected_dirs = {
                name: path for name, path in available_dirs.items() if name != "ppnet"
            }
            if not selected_dirs:
                print(
                    "Consumption: only 'ppnet' raw data found. "
                    "PPNET is opt-in; explicitly select it via the CLI."
                )
                return None

    _assert_raw_files_cover_range(
        selected_dirs, start_date=start_date, end_date=end_date
    )

    processed_data = generate_consumption_data_with_range(
        selected_dirs, start_date, end_date
    )

    if processed_data is not None:
        save_consumption_data_to_csv(processed_data, DATA_SAVE_PATH)
    else:
        print("Consumption: nothing processed")

    return processed_data
