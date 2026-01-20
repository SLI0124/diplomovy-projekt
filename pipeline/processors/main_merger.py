"""pipeline.processors.main_merger

Merges processed data from dates, consumption, weather, and price sources.

Key requirement: do not drop any row coming from any input file. This merger
therefore performs FULL OUTER merges on a normalized timestamp key (derived
from year/month/day/hour). When a source doesn't have a matching row for a
timestamp, the merged output keeps the row and leaves empty values for that
source's columns.

Outputs are written to data/processed/merged/ as yearly files plus one combined
file with all years.
"""

import sys
from datetime import date
from functools import reduce
from pathlib import Path
from typing import Dict, Iterable, Optional

import config
import pandas as pd
import utils
from tqdm import tqdm

DATETIME_FEATURES_DIR = config.PROCESSED_DATETIME_FEATURES_DIR
CONSUMPTION_DIR = config.PROCESSED_CONSUMPTION_DIR
WEATHER_DIR = config.PROCESSED_WEATHER_DIR
PRICE_DIR = config.PROCESSED_PRICE_DIR
MERGED_SAVE_DIR = config.PROCESSED_MERGED_DIR


def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    """Read a CSV file if it exists, otherwise return None."""
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as e:
        print(f"Error loading '{path}': {e}")
        return None


def _ensure_datetime_parts(df: pd.DataFrame, *, source: str) -> None:
    """Ensure that the DataFrame has the required datetime part columns."""
    required = {"year", "month", "day", "hour"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            f"Source '{source}' is missing required columns: "
            + ", ".join(sorted(missing))
        )


def _add_timestamp_and_dup_index(df: pd.DataFrame, *, source: str) -> pd.DataFrame:
    """Create a normalized timestamp key and a duplicate index.

    Duplicate index preserves rows for repeated timestamps (e.g. DST-related
    duplicated hours) without dropping data.
    """
    # Validate required columns
    _ensure_datetime_parts(df, source=source)

    # Work on a copy to avoid mutating callers.
    out = df.copy()

    for col in ["year", "month", "day", "hour"]:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["timestamp"] = pd.to_datetime(
        {
            "year": out["year"],
            "month": out["month"],
            "day": out["day"],
            "hour": out["hour"],
        },
        errors="coerce",
    )

    nat_count = out["timestamp"].isna().sum()
    if nat_count:
        print(
            f"Warning: source '{source}' has {nat_count:,} rows with invalid "
            "timestamp parts"
        )

    out["dup_idx"] = out.groupby("timestamp", dropna=False).cumcount()
    return out


def _prepare_source_for_merge(
    df: pd.DataFrame,
    *,
    source: str,
    year: int,
) -> pd.DataFrame:
    """Prepare a source DataFrame for merging by adding timestamp and dup_idx."""
    df2 = _add_timestamp_and_dup_index(df, source=source)
    # Filter to the requested year as a safety check.
    df2 = df2[df2["year"] == year].copy()

    # Keep only payload columns (drop year/month/day/hour to prevent collisions).
    payload_columns = [
        c
        for c in df2.columns
        if c not in {"year", "month", "day", "hour"}
        and c not in {"timestamp", "dup_idx"}
    ]

    result = df2[["timestamp", "dup_idx", *payload_columns]].copy()
    result = result.sort_values(["timestamp", "dup_idx"], kind="stable")
    return result


def _outer_merge_all(sources: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Perform outer merges on all provided source DataFrames."""
    prepared = [df for df in sources.values() if df is not None and not df.empty]

    if not prepared:
        return pd.DataFrame(columns=["year", "month", "day", "hour"])

    merged = reduce(
        lambda left, right: left.merge(right, on=["timestamp", "dup_idx"], how="outer"),
        prepared,
    )

    merged = merged.sort_values(["timestamp", "dup_idx"], kind="stable")

    # Ensure dtype is datetime64[ns] for .dt accessors.
    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")

    # Recreate calendar columns from timestamp; NaT stays empty.
    merged.insert(0, "hour", merged["timestamp"].dt.hour)  # type: ignore[attr-defined]
    merged.insert(0, "day", merged["timestamp"].dt.day)  # type: ignore[attr-defined]
    merged.insert(0, "month", merged["timestamp"].dt.month)  # type: ignore[attr-defined]
    merged.insert(0, "year", merged["timestamp"].dt.year)  # type: ignore[attr-defined]

    return merged


def _year_file(dir_path: Path, prefix: str, year: int) -> Path:
    """Construct the file path for a given year and prefix in the directory."""
    return dir_path / f"{prefix}_{year}.csv"


def _summarize_missing(df: pd.DataFrame) -> dict[str, int]:
    """Summarize missing data counts for key columns in the merged DataFrame."""
    out: dict[str, int] = {}
    consumption_cols = [
        c
        for c in df.columns
        if c.startswith("consumption_") or c == "consumption_total"
    ]
    if consumption_cols:
        out["missing_consumption_rows"] = int(
            df[consumption_cols].isna().any(axis=1).sum()
        )
    if "weighted_avg_price_eur_mwh" in df.columns:
        out["missing_price_rows"] = int(df["weighted_avg_price_eur_mwh"].isna().sum())
    return out


def load_year_data(
    year,
    datetime_dir,
    consumption_dir,
    weather_dir,
    price_dir,
    consumption_networks: Optional[Iterable[str]] = None,
):
    """Load and outer-merge data for a specific year.

    Returns a tuple of (merged_df, stats_dict) or (None, stats_dict) if nothing
    can be loaded.
    """
    raw_sources: dict[str, Optional[pd.DataFrame]] = {
        "datetime_features": _read_csv_if_exists(
            _year_file(datetime_dir, "datetime_features", year)
        ),
        "consumption": _read_csv_if_exists(
            _year_file(consumption_dir, "consumption", year)
        ),
        "weather": _read_csv_if_exists(_year_file(weather_dir, "weather", year)),
        "price": _read_csv_if_exists(_year_file(price_dir, "price", year)),
    }

    stats: dict[str, object] = {
        "year": year,
        "input_rows": {k: (0 if v is None else len(v)) for k, v in raw_sources.items()},
    }

    if all(df is None for df in raw_sources.values()):
        return None, stats

    prepared: Dict[str, pd.DataFrame] = {
        name: _prepare_source_for_merge(df, source=name, year=year)
        for name, df in raw_sources.items()
        if df is not None
    }

    merged_df = _outer_merge_all(prepared)

    if consumption_networks:
        expected = {f"consumption_{n}" for n in consumption_networks}
        expected.add("consumption_total")
        missing_expected = sorted(expected.difference(merged_df.columns))
        if missing_expected:
            stats["missing_expected_consumption_columns"] = missing_expected

    stats["merged_rows"] = len(merged_df)
    stats["missing"] = _summarize_missing(merged_df)
    return merged_df, stats


def get_available_years(datetime_dir, consumption_dir, weather_dir, price_dir):
    """Get years that have data available in any source.

    We merge using OUTER joins and keep missing values, so a year can be merged
    even if some sources are missing.
    """
    years: set[int] = set()

    for pattern, dir_path in [
        ("datetime_features_*.csv", datetime_dir),
        ("consumption_*.csv", consumption_dir),
        ("weather_*.csv", weather_dir),
        ("price_*.csv", price_dir),
    ]:
        for file in dir_path.glob(pattern):
            year_str = file.stem.split("_")[-1]
            if year_str.isdigit():
                years.add(int(year_str))

    return sorted(years)


def merge_data_for_range(
    start_date: date,
    end_date: date,
    datetime_dir: Path,
    consumption_dir: Path,
    weather_dir: Path,
    price_dir: Path,
    consumption_networks: Optional[Iterable[str]] = None,
):
    """Merge data for all years within the specified date range."""
    start_year = start_date.year
    end_year = end_date.year

    # Get available years
    available_years = get_available_years(
        datetime_dir, consumption_dir, weather_dir, price_dir
    )

    # Filter years within the date range
    years_to_process = [
        year for year in available_years if start_year <= year <= end_year
    ]

    if not years_to_process:
        print(f"No data available for years {start_year}-{end_year}")
        return None

    merged_data_by_year = {}

    stats_by_year: list[dict[str, object]] = []
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(hours=23)

    for year in tqdm(years_to_process, desc="Merging years", unit="year"):
        merged_df, stats = load_year_data(
            year,
            datetime_dir,
            consumption_dir,
            weather_dir,
            price_dir,
            consumption_networks,
        )
        stats_by_year.append(stats)
        if merged_df is None:
            continue

        if "timestamp" in merged_df.columns:
            keep = merged_df["timestamp"].isna() | (
                (merged_df["timestamp"] >= start_dt)
                & (merged_df["timestamp"] <= end_dt)
            )
            merged_df = merged_df.loc[keep].copy()

        merged_data_by_year[year] = merged_df

    if not merged_data_by_year:
        print("No data was successfully merged")
        return None

    merged_years = sorted(merged_data_by_year)
    total_records = sum(len(merged_data_by_year[y]) for y in merged_years)
    print(f"Merged years: {merged_years}")
    print(f"Total merged records: {total_records:,}")

    missing_cols_years: list[tuple[int, list[str]]] = []
    for s in stats_by_year:
        year_val = s.get("year")
        cols = s.get("missing_expected_consumption_columns")
        if isinstance(year_val, int) and isinstance(cols, list) and cols:
            missing_cols_years.append((year_val, [str(c) for c in cols]))
    if missing_cols_years:
        print("Missing expected consumption columns (by year):")
        for y, cols in missing_cols_years:
            print(f"\t{y}: {', '.join(cols)}")

    return merged_data_by_year


def save_merged_data_to_csv(
    merged_data_by_year: Dict[int, pd.DataFrame],
    output_dir: Path,
    file_prefix: str = "merged",
):
    """Save merged data split by year and also as one combined file."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save individual year files
    sorted_years = sorted(merged_data_by_year.keys())
    for year in tqdm(sorted_years, desc="Saving yearly files", unit="file"):
        year_data = merged_data_by_year[year].copy()
        year_data = year_data.drop(columns=["timestamp", "dup_idx"], errors="ignore")
        filename = output_dir / f"{file_prefix}_{year}.csv"
        year_data.to_csv(filename, index=False)

    # Combine all years into one DataFrame and save as single file
    all_years_data = [merged_data_by_year[year] for year in sorted_years]

    if all_years_data:
        combined_df = pd.concat(all_years_data, ignore_index=True)
        combined_df = combined_df.drop(
            columns=["timestamp", "dup_idx"], errors="ignore"
        )
        combined_filename = output_dir / f"{file_prefix}_all_years.csv"
        combined_df.to_csv(combined_filename, index=False)

    print(f"Saved: {len(sorted_years)} yearly files + merged_all_years.csv")


def merge_processed_data(
    end_date_param=None, consumption_networks: Optional[Iterable[str]] = None
):
    """Main merging function - entry point for main.py."""
    datetime_dir = DATETIME_FEATURES_DIR
    consumption_dir = CONSUMPTION_DIR
    weather_dir = WEATHER_DIR
    price_dir = PRICE_DIR
    output_dir = MERGED_SAVE_DIR

    start_date = config.MERGE_START_DATE
    try:
        end_date = utils.resolve_end_date(end_date_param)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        return None

    # Merge data in date range
    normalized_networks = None
    if consumption_networks:
        normalized_networks = [network.lower() for network in consumption_networks]

    merged_data = merge_data_for_range(
        start_date,
        end_date,
        datetime_dir,
        consumption_dir,
        weather_dir,
        price_dir,
        normalized_networks,
    )

    if merged_data is not None:
        save_merged_data_to_csv(merged_data, output_dir)
        return merged_data

    print("No data was merged.")
    return None


if __name__ == "__main__":
    END_DATE = None
    if len(sys.argv) >= 2:
        END_DATE = sys.argv[1]

    merge_processed_data(end_date_param=END_DATE)
