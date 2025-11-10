"""
Module for processing raw weather data.

This script will load ../../data/raw/weather/ weather data

The CSV file contains the following columns:
- date: datetime of the measurement, pretty self-explanatory
- temperature_2m: air temperature at 2 meters above ground level in °C
- wind_gusts_10m: maximum wind gust at 10 meters above ground level in km/h
- wind_direction_100m: wind direction at 100 meters above ground level
- wind_direction_10m: wind direction at 10 meters above ground level
- wind_speed_100m: wind speed at 100 meters above ground level in km/h
- wind_speed_10m: wind speed at 10 meters above ground level in km/h
- weather_code: WMO weather interpretation code, see
  https://open-meteo.com/en/docs for details
- pressure_msl: mean sea level pressure in hPa
- surface_pressure: surface pressure in hPa
- cloud_cover: total cloud cover in %
- cloud_cover_low: low level cloud cover in %
- cloud_cover_mid: mid level cloud cover in %
- cloud_cover_high: high level cloud cover in %
- relative_humidity_2m: relative humidity at 2 meters above ground level in %
- dew_point_2m: dew point temperature at 2 meters above ground level in °C
- apparent_temperature: perceived temperature at 2 meters above ground level in °C
- precipitation: total precipitation in mm
- rain: total rain in mm
- snowfall: total snowfall in cm
- snow_depth: snow depth in cm

Since everything is already in one csv, this file will provide every function
needed to load the data and save it as a DataFrame.

date can be thrown away, since we will use datetime features from processors/dates.py

since we are making hourly predictions, for one day we will have 24 entries

The processed data will be saved as multiple CSV files in
../../data/processed/weather/ folder, preferably grouped by year.
The catch is that it will be executed from ../main.py so create some entry points
accordingly.
"""

import sys
from datetime import datetime, timedelta, date
from pathlib import Path

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


DATA_SOURCE_PATH = "../../data/raw/weather/"
DATA_SAVE_PATH = "../../data/processed/weather/"


def parse_weather_file(file_path):
    """Parse the weather CSV file and return processed DataFrame."""
    print(f"\tReading weather data from {file_path}")

    try:
        df = pd.read_csv(file_path)
        print(f"\tLoaded {len(df):,} raw weather records")
    except FileNotFoundError as e:
        print(f"\tError: Weather file not found: {e}")
        return None
    except pd.errors.EmptyDataError as e:
        print(f"\tError: Weather file is empty: {e}")
        return None
    except (pd.errors.ParserError, UnicodeDecodeError, PermissionError) as e:
        print(f"\tError reading weather file: {e}")
        return None

    # Convert date column to datetime
    print("\tProcessing datetime components...")
    df["date"] = pd.to_datetime(df["date"])

    # Extract datetime components
    df["year"] = df["date"].dt.year.astype(int)
    df["month"] = df["date"].dt.month.astype(int)
    df["day"] = df["date"].dt.day.astype(int)
    df["hour"] = df["date"].dt.hour.astype(int)

    # Select weather features (including datetime components, excluding original date)
    weather_columns = [
        "year",
        "month",
        "day",
        "hour",
        "temperature_2m",
        "wind_gusts_10m",
        "wind_direction_100m",
        "wind_direction_10m",
        "wind_speed_100m",
        "wind_speed_10m",
        "weather_code",
        "pressure_msl",
        "surface_pressure",
        "cloud_cover",
        "cloud_cover_low",
        "cloud_cover_mid",
        "cloud_cover_high",
        "relative_humidity_2m",
        "dew_point_2m",
        "apparent_temperature",
        "precipitation",
        "rain",
        "snowfall",
        "snow_depth",
    ]

    # Return only the weather columns and drop any rows with all NaN values
    print("\tFiltering weather columns and removing empty rows...")
    result = df[weather_columns].dropna(how="all")
    print(f"\tProcessed {len(result):,} weather records")
    return result


def process_weather_data_with_range(source_dir, start_date, end_date):
    """Process weather data files within the specified date range."""
    print(f"Processing weather data from {start_date} to {end_date}...")

    # Find the weather CSV file (should be only one)
    weather_files = list(source_dir.glob("weather_*.csv"))

    if not weather_files:
        print("\tError: No weather data files found")
        return None

    if len(weather_files) > 1:
        print(f"\tWarning: Multiple weather files found, using {weather_files[0]}")

    weather_file = weather_files[0]
    print(f"\tFound weather file: {weather_file.name}")

    # Parse the weather file
    weather_data = parse_weather_file(weather_file)

    if weather_data is None or len(weather_data) == 0:
        print("\tError: No weather data found")
        return None

    # Filter data by year range and exclude first day of next month
    start_year = start_date.year
    end_year = end_date.year

    print(f"\tFiltering data for years {start_year}-{end_year}...")
    filtered_data = weather_data[
        (weather_data["year"] >= start_year) & (weather_data["year"] <= end_year)
    ]

    # Remove first day of the next month if it exists in the data
    # (this removes extra data downloaded due to the 1-day buffer in downloader)
    next_month_start = end_date.replace(day=1) + timedelta(days=32)
    next_month_start = next_month_start.replace(day=1)
    
    if len(filtered_data) > 0:
        # Create datetime column for precise filtering
        filtered_data_copy = filtered_data.copy()
        filtered_data_copy['temp_datetime'] = pd.to_datetime(
            filtered_data_copy[['year', 'month', 'day', 'hour']]
        )
        
        # Filter out first day of next month
        next_month_datetime = pd.to_datetime(next_month_start)
        next_day_datetime = next_month_datetime + timedelta(days=1)
        
        # Keep only data before the first day of next month
        filtered_data = filtered_data_copy[
            filtered_data_copy['temp_datetime'] < next_month_datetime
        ].drop('temp_datetime', axis=1)
        
        print(f"\tRemoved first day of next month ({next_month_start})")

    print(f"\tFiltered to {len(filtered_data):,} weather records")
    return filtered_data


def save_processed_weather_data_to_csv(df, output_dir, file_prefix="weather"):
    """Save weather data split by year."""
    print("Saving processed weather data...")
    output_dir.mkdir(parents=True, exist_ok=True)

    years = sorted(df["year"].unique())
    print(
        f"\tSplitting data into {len(years)} yearly files ({min(years)}-{max(years)})"
    )

    for year in tqdm(years, desc="Saving weather files", unit="file"):
        year_data = df[df["year"] == year]
        # Keep all datetime components in the saved data
        filename = output_dir / f"{file_prefix}_{year}.csv"
        year_data.to_csv(filename, index=False)

    print(f"\tAll weather files saved to: {output_dir}")


def process_weather_data(end_date_param=None):
    """Main processing function - entry point for main.py"""
    # Get the directory relative to main.py
    current_dir = Path(__file__).parent
    source_dir = current_dir / DATA_SOURCE_PATH
    output_dir = current_dir / DATA_SAVE_PATH

    start_date_param = "2013-01-01"
    if end_date_param is None:
        end_date_param = get_last_day_of_previous_month()

    print(f"Date range: {start_date_param} to {end_date_param}")
    print(f"Source directory: {source_dir}")
    print(f"Output directory: {output_dir}")

    # Validate date format
    if end_date_param is not None:
        try:
            datetime.strptime(end_date_param, "%Y-%m-%d")
        except ValueError:
            print(
                f"ERROR: Invalid date format '{end_date_param}'. "
                f"Please use YYYY-MM-DD format."
            )
            sys.exit(1)

    # Convert string dates to date objects
    start_date = datetime.strptime(start_date_param, "%Y-%m-%d").date()
    end_date = datetime.strptime(end_date_param, "%Y-%m-%d").date()

    # Process weather data in date range
    processed_data = process_weather_data_with_range(source_dir, start_date, end_date)

    if processed_data is not None:
        save_processed_weather_data_to_csv(processed_data, output_dir)

        return processed_data

    return None


if __name__ == "__main__":
    END_DATE = None
    if len(sys.argv) >= 2:
        END_DATE = sys.argv[1]

    process_weather_data(end_date_param=END_DATE)
