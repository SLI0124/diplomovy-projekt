"""
Module for processing raw consumption data from all available distribution networks.

This script processes raw consumption CSV files stored under data/raw/consumption/<network>/.

Each network's data has a unique structure where each day's 24-hour period is split across two files:
- Hours 0-6 of the current day are in the previous day's file
- Hours 7-23 of the current day are in the current day's file

CSV format:
Datum,ID,Hodnota,Nazev
1.1.2013 7:00,21637,461901,Zona_Gasnet
1.1.2013 8:00,21637,426805,Zona_Gasnet
...

We extract 'Datum' (datetime) and 'Hodnota' (consumption value) columns for each network.
The processed data maintains complete 24-hour structures with NA values for missing hours
and provides a dedicated consumption column per network.

Processed data is saved as multiple CSV files in ../../data/processed/consumption/, grouped by year.
The catch is that it will be executed from ../main.py so create entry points accordingly.
"""

import sys
from datetime import datetime, timedelta, date
from pathlib import Path
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Tuple

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


# Resolve project structure relative to this file
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_SOURCE_ROOT = PROJECT_ROOT / "data" / "raw" / "consumption"
DATA_SAVE_PATH = PROJECT_ROOT / "data" / "processed" / "consumption"

# Track suspicious file metadata per network
SUSPICIOUS_FILES: Dict[str, List[Tuple[str, int]]] = defaultdict(list)


def discover_network_paths() -> Dict[str, Path]:
    """Return mapping of available consumption networks to their data directories."""
    if not DATA_SOURCE_ROOT.exists():
        return {}
    return {
        item.name.lower(): item for item in DATA_SOURCE_ROOT.iterdir() if item.is_dir()
    }


def print_suspicious_files() -> None:
    """Print all misaligned files and their line counts after parsing."""
    if not SUSPICIOUS_FILES:
        return

    print("WARNING: Suspicious files detected!")
    for network, entries in SUSPICIOUS_FILES.items():
        if not entries:
            continue
        print(f"\tNetwork: {network}")
        for filename, line_count in entries:
            print(f"\t  File: {filename}")
            print("\t  Expected: 24-26 lines (24 hours + header + padding)")
            print(f"\t  Found: {line_count} lines")
        print("-" * 60)


def parse_consumption_file(file_path: Path, network: str) -> pd.DataFrame:
    """Parse a single consumption CSV file and return normalized data."""
    df = pd.read_csv(file_path, sep=",")

    total_lines = len(df)
    if total_lines < 24 or total_lines > 26:
        SUSPICIOUS_FILES[network].append((file_path.name, total_lines))

    datum_series: pd.Series = pd.to_datetime(
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


def create_day_structure(target_date: date, networks: Iterable[str]) -> pd.DataFrame:
    """Create a complete 24-hour structure with columns for each network."""
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
    """Extract specific hours from a network file for the target date."""
    if not file_path.exists():
        return pd.DataFrame()

    consumption_data = parse_consumption_file(file_path, network)
    if consumption_data.empty:
        return pd.DataFrame()

    mask = (
        (consumption_data["year"] == target_date.year)
        & (consumption_data["month"] == target_date.month)
        & (consumption_data["day"] == target_date.day)
    )

    if hour_range == "early":
        mask &= consumption_data["hour"] <= 6
    else:
        mask &= consumption_data["hour"] >= 7

    filtered = consumption_data.loc[mask, ["hour", "consumption"]]
    return filtered.reset_index(drop=True)


def collect_network_day_values(
    source_dir: Path, current_date: date, network: str
) -> pd.DataFrame:
    """Collect hourly consumption data for a single network and day."""
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
    network_dirs: Dict[str, Path], current_date: date
) -> pd.DataFrame:
    """Process consumption data for all networks on a single date."""
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
    network_dirs: Dict[str, Path], start_date: date, end_date: date
) -> Optional[pd.DataFrame]:
    """Process consumption files for all networks within the given date range."""
    if not network_dirs:
        print(
            f"No consumption networks found in {DATA_SOURCE_ROOT}. "
            "Nothing to process."
        )
        return None

    networks_list = ", ".join(sorted(network_dirs.keys()))
    print(
        f"Processing consumption data from {start_date} to {end_date} "
        f"for networks: {networks_list}"
    )

    dates_to_process = []
    current_date = start_date
    while current_date <= end_date:
        dates_to_process.append(current_date)
        current_date += timedelta(days=1)

    all_data = []
    for process_date in tqdm(dates_to_process, desc="Processing files"):
        day_data = process_single_date(network_dirs, process_date)
        all_data.append(day_data)

    if not all_data:
        print("No data found")
        return None

    combined_df = pd.concat(all_data, ignore_index=True)
    consumption_columns = [f"consumption_{network}" for network in network_dirs]
    combined_df["consumption_total"] = (
        combined_df[consumption_columns].sum(axis=1, min_count=1).astype("Int64")
    )
    total_hours = len(combined_df) * len(consumption_columns)
    missing_hours = combined_df[consumption_columns].isna().sum().sum()
    available_hours = total_hours - missing_hours

    print(
        f"Processed {total_hours:,} network-hours "
        f"({available_hours:,} with data, {missing_hours:,} with NA)"
    )

    print_suspicious_files()
    return combined_df


def process_consumption_data(
    end_date_param: Optional[str] = None, networks: Optional[Iterable[str]] = None
) -> Optional[pd.DataFrame]:
    """Main processing function - entry point for main.py."""
    SUSPICIOUS_FILES.clear()

    start_date_param = "2013-01-01"
    if end_date_param is None:
        end_date_param = get_last_day_of_previous_month()

    try:
        end_date = datetime.strptime(end_date_param, "%Y-%m-%d").date()
    except ValueError:
        print(
            f"ERROR: Invalid date format '{end_date_param}'. Please use YYYY-MM-DD format."
        )
        sys.exit(1)

    start_date = datetime.strptime(start_date_param, "%Y-%m-%d").date()

    available_dirs = discover_network_paths()
    if not available_dirs:
        print(f"No consumption data directories found in {DATA_SOURCE_ROOT}.")
        return None

    selected_dirs = available_dirs
    if networks:
        normalized = [network.lower() for network in networks]
        missing = [network for network in normalized if network not in available_dirs]
        if missing:
            print(
                "WARNING: Requested consumption networks not found: "
                + ", ".join(sorted(set(missing)))
            )
        selected_dirs = {
            name: path for name, path in available_dirs.items() if name in normalized
        }
        if not selected_dirs:
            print("No valid consumption networks selected. Nothing to process.")
            return None

    processed_data = generate_consumption_data_with_range(
        selected_dirs, start_date, end_date
    )

    if processed_data is not None:
        save_consumption_data_to_csv(processed_data, DATA_SAVE_PATH)
    else:
        print("No consumption data was processed.")

    return processed_data


def save_consumption_data_to_csv(
    df: pd.DataFrame, output_dir: Path, file_prefix="consumption"
) -> None:
    """Save consumption data to CSV files grouped by year."""
    output_dir.mkdir(parents=True, exist_ok=True)

    consumption_columns = sorted(
        column for column in df.columns if column.startswith("consumption_")
    )
    if (
        "consumption_total" in df.columns
        and "consumption_total" not in consumption_columns
    ):
        consumption_columns.append("consumption_total")
    years = sorted(df["year"].dropna().unique())
    print(f"Saving data to {len(years)} files:")

    for year in tqdm(years, desc="Saving files"):
        year_mask = df["year"] == year
        year_data = df.loc[
            year_mask, ["year", "month", "day", "hour", *consumption_columns]
        ]
        filename = output_dir / f"{file_prefix}_{int(year)}.csv"
        year_data.to_csv(filename, index=False)

    print(f"All files saved to: {output_dir}")


if __name__ == "__main__":
    END_DATE = None
    if len(sys.argv) >= 2:
        END_DATE = sys.argv[1]

    process_consumption_data(end_date_param=END_DATE)
