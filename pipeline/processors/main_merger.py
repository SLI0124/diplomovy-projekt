"""Merge processed datasets (datetime features, consumption, weather, price).

Key requirement: do not drop any rows from inputs. We therefore do FULL OUTER
merges on a normalized timestamp and keep duplicates via an extra `dup_idx`.
"""

from __future__ import annotations

from datetime import date
from functools import reduce
from pathlib import Path
from typing import Iterable, Optional

import config
import pandas as pd
import utils
from tqdm import tqdm

DATETIME_FEATURES_DIR = config.PROCESSED_DATETIME_FEATURES_DIR
CONSUMPTION_DIR = config.PROCESSED_CONSUMPTION_DIR
WEATHER_DIR = config.PROCESSED_WEATHER_DIR
PRICE_DIR = config.PROCESSED_PRICE_DIR
MERGED_SAVE_DIR = config.PROCESSED_MERGED_DIR

KEY_COLS = ("year", "month", "day", "hour")


def _read_csv_if_exists(path: Path) -> Optional[pd.DataFrame]:
    """Read a CSV file if it exists.

    Args:
        path: CSV file path.

    Returns:
        DataFrame if the file exists and can be parsed, otherwise None.
    """
    if not path.exists():
        return None
    try:
        return pd.read_csv(path)
    except (pd.errors.EmptyDataError, pd.errors.ParserError) as exc:
        print(f"Merge: load error ({path.name}: {exc})")
        return None


def _ensure_datetime_parts(df: pd.DataFrame, source: str) -> None:
    """Raise if a source is missing required calendar columns.

    Args:
        df: Input DataFrame.
        source: Human-readable source name.

    Raises:
        ValueError: If required columns are missing.
    """
    missing = set(KEY_COLS).difference(df.columns)
    if missing:
        raise ValueError(
            f"Source '{source}' is missing required columns: "
            + ", ".join(sorted(missing))
        )


def _add_timestamp_and_dup_index(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Add a normalized timestamp key and duplicate index.

    The `dup_idx` preserves repeated timestamps without dropping rows.

    Args:
        df: Input DataFrame.
        source: Human-readable source name.

    Returns:
        Copy of the input DataFrame with added `timestamp` and `dup_idx`.
    """
    # Validate required columns.
    _ensure_datetime_parts(df, source=source)
    out = df.copy()

    for col in KEY_COLS:
        out[col] = pd.to_numeric(out[col], errors="coerce")

    out["timestamp"] = pd.to_datetime(out[list(KEY_COLS)], errors="coerce")

    nat_count = out["timestamp"].isna().sum()
    if nat_count:
        print(f"Merge: {source} has {nat_count:,} invalid timestamps")

    # If a timestamp occurs multiple times (e.g., DST duplication), keep all rows.
    out["dup_idx"] = out.groupby("timestamp", dropna=False).cumcount()
    return out


def _prepare_source_for_merge(
    df: pd.DataFrame,
    source: str,
    year: int,
) -> pd.DataFrame:
    """Prepare one source DataFrame for merging.

    Args:
        df: Source DataFrame.
        source: Source name (used for validation messages).
        year: Year to filter to.

    Returns:
        Merge-ready DataFrame with keys and payload columns.
    """
    df2 = _add_timestamp_and_dup_index(df, source=source)
    # Filter to the requested year as a safety check.
    df2 = df2[df2["year"] == year].copy()

    payload = []
    for col in df2.columns:
        if col in KEY_COLS or col in ("timestamp", "dup_idx"):
            continue
        payload.append(col)

    result = df2[["timestamp", "dup_idx", *payload]].copy()
    result = result.sort_values(["timestamp", "dup_idx"], kind="stable")
    return result


def _outer_merge_all(sources: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """FULL OUTER merge of all provided source DataFrames.

    Args:
        sources: Source DataFrames to merge.

    Returns:
        Merged DataFrame containing all sources.
    """
    frames = []
    for df in sources.values():
        if df is not None and not df.empty:
            frames.append(df)
    if not frames:
        return pd.DataFrame(columns=list(KEY_COLS))

    merged = reduce(
        lambda left, right: left.merge(right, on=["timestamp", "dup_idx"], how="outer"),
        frames,
    )
    merged = merged.sort_values(["timestamp", "dup_idx"], kind="stable")

    merged["timestamp"] = pd.to_datetime(merged["timestamp"], errors="coerce")

    # Calendar columns are derived from timestamp for consistency.
    dt_index = pd.DatetimeIndex(merged["timestamp"])
    merged.insert(0, "hour", dt_index.hour)
    merged.insert(0, "day", dt_index.day)
    merged.insert(0, "month", dt_index.month)
    merged.insert(0, "year", dt_index.year)
    return merged


def _year_file(dir_path: Path, prefix: str, year: int) -> Path:
    """Return `<dir>/<prefix>_<year>.csv`.

    Args:
        dir_path: Directory path for the file.
        prefix: File prefix.
        year: Year for the file.

    Returns:
        Path to the yearly CSV.
    """
    return dir_path / f"{prefix}_{year}.csv"


def _summarize_missing(df: pd.DataFrame) -> dict[str, int]:
    """Summarize missing data counts for a few important columns.

    Args:
        df: Merged DataFrame.

    Returns:
        Dict with missing-value metrics.
    """
    out: dict[str, int] = {}
    consumption_cols = []
    for col in df.columns:
        if col.startswith("consumption_") or col == "consumption_total":
            consumption_cols.append(col)
    if consumption_cols:
        out["missing_consumption_rows"] = int(
            df[consumption_cols].isna().any(axis=1).sum()
        )
    if "weighted_avg_price_eur_mwh" in df.columns:
        out["missing_price_rows"] = int(df["weighted_avg_price_eur_mwh"].isna().sum())
    return out


def load_year_data(
    year: int,
    datetime_dir: Path,
    consumption_dir: Path,
    weather_dir: Path,
    price_dir: Path,
    consumption_networks: Optional[Iterable[str]] = None,
) -> tuple[Optional[pd.DataFrame], dict[str, object]]:
    """Load and FULL OUTER merge one year's inputs.

    Args:
        year: Target year.
        datetime_dir: Directory with processed datetime feature CSVs.
        consumption_dir: Directory with processed consumption CSVs.
        weather_dir: Directory with processed weather CSVs.
        price_dir: Directory with processed price CSVs.
        consumption_networks: Optional list of consumption network names to validate.

    Returns:
        (merged_df, stats). If all inputs are missing, merged_df is None.
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

    input_rows: dict[str, int] = {}
    for name, df in raw_sources.items():
        input_rows[name] = 0 if df is None else len(df)
    stats: dict[str, object] = {"year": year, "input_rows": input_rows}

    if all(df is None for df in raw_sources.values()):
        return None, stats

    prepared: dict[str, pd.DataFrame] = {}
    for name, df in raw_sources.items():
        if df is None:
            continue
        prepared[name] = _prepare_source_for_merge(df, source=name, year=year)

    merged_df = _outer_merge_all(prepared)

    if consumption_networks:
        expected = set()
        for network in consumption_networks:
            expected.add(f"consumption_{network}")
        expected.add("consumption_total")
        missing_expected = sorted(expected.difference(merged_df.columns))
        if missing_expected:
            stats["missing_expected_consumption_columns"] = missing_expected

    stats["merged_rows"] = len(merged_df)
    stats["missing"] = _summarize_missing(merged_df)
    return merged_df, stats


def get_available_years(
    datetime_dir: Path, consumption_dir: Path, weather_dir: Path, price_dir: Path
) -> list[int]:
    """Get years that have data available in any source.

    We merge using OUTER joins and keep missing values, so a year can be merged
    even if some sources are missing.

    Args:
        datetime_dir: Directory with processed datetime feature CSVs.
        consumption_dir: Directory with processed consumption CSVs.
        weather_dir: Directory with processed weather CSVs.
        price_dir: Directory with processed price CSVs.

    Returns:
        Sorted list of years that appear in at least one directory.
    """
    years: set[int] = set()
    patterns = [
        ("datetime_features_*.csv", datetime_dir),
        ("consumption_*.csv", consumption_dir),
        ("weather_*.csv", weather_dir),
        ("price_*.csv", price_dir),
    ]
    for pattern, dir_path in patterns:
        if not dir_path.exists():
            continue
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
) -> Optional[dict[int, pd.DataFrame]]:
    """Merge all years intersecting the given date range.

    Years are merged independently, then trimmed to the exact date window.

    Args:
        start_date: Start date (inclusive).
        end_date: End date (inclusive).
        datetime_dir: Directory with processed datetime feature CSVs.
        consumption_dir: Directory with processed consumption CSVs.
        weather_dir: Directory with processed weather CSVs.
        price_dir: Directory with processed price CSVs.
        consumption_networks: Optional list of networks to validate.

    Returns:
        Mapping of year to merged DataFrame, or None if nothing could be merged.
    """
    start_year = start_date.year
    end_year = end_date.year

    available_years = get_available_years(
        datetime_dir, consumption_dir, weather_dir, price_dir
    )
    years_to_process = []
    for year in available_years:
        if start_year <= year <= end_year:
            years_to_process.append(year)

    if not years_to_process:
        print(f"Merge: no data for years {start_year}-{end_year}")
        return None

    merged_data_by_year: dict[int, pd.DataFrame] = {}
    stats_by_year: list[dict[str, object]] = []
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date) + pd.Timedelta(hours=23)

    for year in tqdm(years_to_process, desc="Merge: years", unit="year", leave=False):
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
            ts = merged_df["timestamp"]
            keep = ts.isna() | ((ts >= start_dt) & (ts <= end_dt))
            merged_df = merged_df.loc[keep].copy()

        merged_data_by_year[year] = merged_df

    if not merged_data_by_year:
        print("Merge: nothing merged")
        return None

    merged_years = sorted(merged_data_by_year)
    total_records = sum(len(merged_data_by_year[y]) for y in merged_years)
    print(
        f"Merge: years={min(merged_years)}-{max(merged_years)} ({len(merged_years)}), "
        f"rows={total_records:,}"
    )

    missing_cols_years: list[tuple[int, list[str]]] = []
    for stats in stats_by_year:
        year_val = stats.get("year")
        cols = stats.get("missing_expected_consumption_columns")
        if isinstance(year_val, int) and isinstance(cols, list) and cols:
            missing_cols_years.append((year_val, [str(c) for c in cols]))
    if missing_cols_years:
        print("Merge: missing expected consumption columns:")
        for y, cols in missing_cols_years:
            print(f"\t{y}: {', '.join(cols)}")

    return merged_data_by_year


def save_merged_data_to_csv(
    merged_data_by_year: dict[int, pd.DataFrame],
    output_dir: Path,
    file_prefix: str = "merged",
) -> None:
    """Save yearly merged CSVs and one combined file.

    Drops helper keys (timestamp, dup_idx) from outputs.

    Args:
        merged_data_by_year: Mapping of year to merged DataFrame.
        output_dir: Output directory.
        file_prefix: Prefix for output files.
    """
    utils.ensure_directory(output_dir)

    sorted_years = sorted(merged_data_by_year.keys())
    for year in tqdm(
        sorted_years, desc="Merge: saving yearly", unit="file", leave=False
    ):
        year_df = merged_data_by_year[year].copy()
        year_df = year_df.drop(columns=["timestamp", "dup_idx"], errors="ignore")
        year_df.to_csv(output_dir / f"{file_prefix}_{year}.csv", index=False)

    combined_list = []
    for year in sorted_years:
        combined_list.append(merged_data_by_year[year])
    combined_df = pd.concat(combined_list, ignore_index=True)
    combined_df = combined_df.drop(columns=["timestamp", "dup_idx"], errors="ignore")
    combined_df.to_csv(output_dir / f"{file_prefix}_all_years.csv", index=False)

    print(
        f"Merge: saved {len(sorted_years)} yearly + merged_all_years.csv "
        f"-> {output_dir}"
    )


def merge_processed_data(
    end_date_param: Optional[utils.DateLike] = None,
    consumption_networks: Optional[Iterable[str]] = None,
) -> Optional[dict[int, pd.DataFrame]]:
    """Merge all processed datasets and write merged CSV outputs.

    Args:
        end_date_param: End date (inclusive). Defaults to "today" semantics from utils.
        consumption_networks: Optional list of networks to validate (column presence).

    Returns:
        Mapping of year to merged DataFrame on success, otherwise None.
    """
    start_date = config.MERGE_START_DATE
    end_date = utils.resolve_end_date(end_date_param)

    normalized_networks = None
    if consumption_networks:
        normalized_networks = [network.lower() for network in consumption_networks]

    merged_data = merge_data_for_range(
        start_date,
        end_date,
        DATETIME_FEATURES_DIR,
        CONSUMPTION_DIR,
        WEATHER_DIR,
        PRICE_DIR,
        normalized_networks,
    )

    if merged_data is not None:
        save_merged_data_to_csv(merged_data, MERGED_SAVE_DIR)
        return merged_data

    print("Merge: no data")
    return None
