"""Process downloaded weather CSV into per-year processed files."""

from __future__ import annotations

from datetime import date
from pathlib import Path

import config
import pandas as pd
from tqdm import tqdm

import pipeline.utils as utils

DATA_SOURCE_PATH = config.RAW_WEATHER_DIR
DATA_SAVE_PATH = config.PROCESSED_WEATHER_DIR


def _hourly_index(start_date: date, end_date: date) -> pd.DatetimeIndex:
    """Full local-naive hourly index for [start_date, end_date] inclusive."""
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date) + pd.Timedelta(hours=23)
    return pd.date_range(start=start_dt, end=end_dt, freq="h")


def _to_local_naive(values: pd.Series | pd.Index) -> pd.Series:
    """Convert UTC timestamps to local tz, then drop tz info (local-naive)."""
    dt = pd.to_datetime(values, utc=True, errors="coerce")
    dti = pd.DatetimeIndex(dt).tz_convert(config.WEATHER_TIMEZONE).tz_localize(None)
    index = values.index if isinstance(values, pd.Series) else None
    return pd.Series(dti, index=index, name=getattr(values, "name", None))


def _process_weather_hourly(
    raw_df: pd.DataFrame,
    start_date: date,
    end_date: date,
) -> pd.DataFrame:
    """Normalize raw weather observations to a strict 24h/day local-naive calendar.

    Core idea:
    - Source timestamps are UTC.
    - We convert UTC -> local time and then drop tz info so merges use
      year/month/day/hour.
    - DST fall-back creates a duplicated local hour; we collapse it.
    - DST spring-forward creates a missing local hour; we fill only that single
      missing hour (limit=1) because the underlying UTC series is continuous.
    """
    payload = list(config.WEATHER_VARIABLES)
    if raw_df.empty:
        return pd.DataFrame(columns=["year", "month", "day", "hour", *payload])
    if "date" not in raw_df.columns:
        raise ValueError("Weather: raw data missing required column 'date'.")

    ts_local = _to_local_naive(raw_df["date"])
    df = raw_df.loc[:, payload].copy()
    df["timestamp"] = ts_local
    df = df.dropna(subset=["timestamp"]).copy()

    # Collapse duplicated local timestamps (DST fall-back).
    agg: dict[str, str] = {col: "mean" for col in payload}
    if "weather_code" in agg:
        # Keep it discrete-ish; averaging duplicates can create fractions.
        agg["weather_code"] = "first"
    hourly = df.groupby("timestamp", as_index=False).agg(agg).sort_values("timestamp")
    hourly = hourly.set_index("timestamp").reindex(_hourly_index(start_date, end_date))

    numeric_cols = [c for c in payload if c != "weather_code"]
    if numeric_cols:
        hourly[numeric_cols] = hourly[numeric_cols].interpolate(
            method="time",
            limit=1,
            limit_area="inside",
        )
    if "weather_code" in payload:
        hourly["weather_code"] = hourly["weather_code"].ffill(limit=1).bfill(limit=1)

    out = hourly.reset_index().rename(columns={"index": "timestamp"})
    dti = pd.DatetimeIndex(out["timestamp"])
    out["year"] = dti.year.astype("Int64")
    out["month"] = dti.month.astype("Int64")
    out["day"] = dti.day.astype("Int64")
    out["hour"] = dti.hour.astype("Int64")
    out["weather_code"] = out["weather_code"].astype("Int64")
    return out[["year", "month", "day", "hour", *payload]]


def parse_weather_file(file_path: Path) -> pd.DataFrame | None:
    """Load the raw weather CSV.

    Args:
        file_path: Path to a raw weather CSV.

    Returns:
        A processed DataFrame, or None on read/parse error.
    """
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError as exc:
        print(f"Weather: file not found ({exc})")
        return None
    except pd.errors.EmptyDataError as exc:
        print(f"Weather: empty file ({exc})")
        return None
    except (pd.errors.ParserError, UnicodeDecodeError, PermissionError) as exc:
        print(f"Weather: read error ({exc})")
        return None

    # Keep only the columns we need.
    keep = ["date", *config.WEATHER_VARIABLES]
    missing = [c for c in keep if c not in df.columns]
    if missing:
        raise ValueError(
            f"Weather: raw file is missing required columns: {', '.join(missing)}"
        )
    return df[keep].dropna(how="all")


def process_weather_data_with_range(
    source_dir: Path,
    start_date: date,
    end_date: date,
) -> pd.DataFrame | None:
    """Process weather data within a specific date range.

    Args:
        source_dir: Directory containing downloaded raw weather CSV(s).
        start_date: Inclusive start date.
        end_date: Inclusive end date.

    Returns:
        Processed DataFrame filtered to the requested date range, or None.
    """

    if not source_dir.exists():
        raise FileNotFoundError(
            f"Weather: raw directory not found: {source_dir}. "
            "Run the downloader first (pipeline/main.py --download weather)."
        )

    # Find the weather CSV file (should be only one)
    weather_files = list(source_dir.glob("weather_*.csv"))

    if not weather_files:
        raise FileNotFoundError(
            f"Weather: no input files in {source_dir}. "
            "Expected something like 'weather_YYYY-MM-DD_YYYY-MM-DD.csv'."
        )

    if len(weather_files) > 1:
        print(f"Weather: multiple files, using {weather_files[0].name}")

    weather_file = weather_files[0]
    print(f"Weather: {weather_file.name}")

    # Validate coverage before full processing (fast-fail with clear message).
    try:
        dates_only = pd.read_csv(weather_file, usecols=["date"])
        dates_only["date"] = _to_local_naive(dates_only["date"])
        min_dt = dates_only["date"].min()
        max_dt = dates_only["date"].max()
    except ValueError:
        # usecols failed (missing 'date' column)
        raise ValueError(
            f"Weather: raw file {weather_file.name} is missing required column 'date'."
        )

    if pd.isna(min_dt) or pd.isna(max_dt):
        raise ValueError(
            f"Weather: raw file {weather_file.name} has no valid timestamps in 'date'."
        )

    required_start = pd.to_datetime(start_date)
    # Need at least through the last hour of end_date.
    required_last_hour = (
        pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(hours=1)
    )
    if min_dt > required_start or max_dt < required_last_hour:
        raise FileNotFoundError(
            "Weather: raw data does not cover requested range. "
            f"Have {min_dt}..{max_dt}, need {required_start}..{required_last_hour}. "
            "Run the downloader for the missing period."
        )

    raw_weather = parse_weather_file(weather_file)
    if raw_weather is None or raw_weather.empty:
        print("Weather: no data")
        return None

    processed = _process_weather_hourly(
        raw_weather,
        start_date=start_date,
        end_date=end_date,
    )
    return processed


def save_processed_weather_data_to_csv(
    df: pd.DataFrame,
    output_dir: Path,
    file_prefix: str = "weather",
) -> None:
    """Save processed weather data grouped by year.

    Args:
        df: Processed weather data.
        output_dir: Directory to save processed files to.
        file_prefix: Prefix for output filenames.
    """
    utils.ensure_directory(output_dir)

    years = sorted(df["year"].unique())

    for year in tqdm(years, desc="Weather: saving", unit="file", leave=False):
        year_data = df[df["year"] == year]
        # Keep all datetime components in the saved data
        filename = output_dir / f"{file_prefix}_{year}.csv"
        year_data.to_csv(filename, index=False)

    if years:
        year_min = min(years)
        year_max = max(years)
        print(f"Weather: saved {len(years)} files \
            ({year_min}-{year_max}) -> {output_dir.resolve()}\n")


def process_weather_data(
    end_date_param: utils.DateLike | None = None,
) -> pd.DataFrame | None:
    """Entry point used by [pipeline/main.py](pipeline/main.py).

    Args:
        end_date_param: End date as YYYY-MM-DD string, date, datetime, or None.

    Returns:
        Optional[pd.DataFrame]: Processed weather data DataFrame or None on error.
    """
    start_date = config.WEATHER_START_DATE
    end_date = utils.resolve_end_date(end_date_param)

    processed_data = process_weather_data_with_range(
        DATA_SOURCE_PATH, start_date, end_date
    )
    if processed_data is None:
        return None

    save_processed_weather_data_to_csv(processed_data, DATA_SAVE_PATH)
    return processed_data
