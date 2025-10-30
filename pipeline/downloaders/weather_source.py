"""
Module for downloading weather data from Open-Meteo API.

Downloads hourly weather data for the Czech Republic region using the
Open-Meteo Archive API. Data includes temperature, humidity, pressure, wind,
precipitation, and other meteorological variables.
Data is available from 2013-01-01 onwards.
"""

import sys
import datetime
from pathlib import Path

import openmeteo_requests
import pandas as pd
from retry_requests import retry

PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_SAVE_PATH = PROJECT_ROOT / "data" / "raw" / "weather"


def ensure_directory(path):
    """Ensure that the directory exists, creating it if necessary."""
    if isinstance(path, str):
        dir_path = Path(path)
    else:
        dir_path = path
    if not dir_path.exists():
        dir_path.mkdir(parents=True, exist_ok=True)


def _setup_api_client():
    """Setup the Open-Meteo API client with retry configuration."""
    retry_session = retry(retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)


def _build_api_params(start_date_val, end_date_val):
    """Build API parameters for weather data request."""
    return {
        "latitude": 49.5667,  # Czech Republic coordinates
        "longitude": 13.6333,
        "start_date": start_date_val.strftime("%Y-%m-%d"),
        "end_date": end_date_val.strftime("%Y-%m-%d"),
        "hourly": [
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
        ],
        "timezone": "Europe/Berlin",
    }


def _process_api_response(response):
    """Process API response and extract weather data."""
    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")

    # Process hourly data. The order of variables needs to be the same as requested.
    hourly = response.Hourly()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }

    # Map all weather variables
    variable_names = [
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

    for i, var_name in enumerate(variable_names):
        hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()

    return pd.DataFrame(data=hourly_data)


def download_weather_data(end_date_param=None):
    """Main download function - entry point for main.py"""
    start_date_param = "2013-01-01"
    if end_date_param is None:
        end_date_param = datetime.date.today().strftime("%Y-%m-%d")

    # Convert string dates to date objects if needed
    start_date_obj = datetime.datetime.strptime(start_date_param, "%Y-%m-%d").date()

    if isinstance(end_date_param, str):
        end_date_obj = datetime.datetime.strptime(end_date_param, "%Y-%m-%d").date()
    else:
        end_date_obj = end_date_param

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

    download_weather_data_with_range(start_date_obj, end_date_obj)


def download_weather_data_with_range(
    start_date_val: datetime.date, end_date_val: datetime.date
):
    """Download weather data for a specific date range."""

    # normalize if user passed a datetime
    if isinstance(end_date_val, datetime.datetime):
        end_date_val = end_date_val.date()

    if start_date_val < datetime.date(
        2013, 1, 1
    ):  # hardcoded based on data availability
        print(
            "Start date cannot be before 01.01.2013 since it is the first "
            "available data from weather API."
        )
        delta_days = (datetime.date(2013, 1, 1) - start_date_val).days
        print(f"Adjusting start date by {delta_days} days to 01.01.2013.")
        start_date_val = datetime.date(2013, 1, 1)

    if start_date_val > end_date_val:
        print(
            f"Start date {start_date_val} is after end date {end_date_val}. Nothing to download."
        )
        return

    print(f"Downloading weather data from {start_date_val} to {end_date_val}...")

    ensure_directory(DATA_SAVE_PATH)

    try:
        openmeteo = _setup_api_client()
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = _build_api_params(start_date_val, end_date_val)

        responses = openmeteo.weather_api(url, params=params)

        # Process first location
        hourly_dataframe = _process_api_response(responses[0])

        # Save to CSV file
        output_file = DATA_SAVE_PATH / f"weather_{start_date_val}_{end_date_val}.csv"
        hourly_dataframe.to_csv(output_file, index=False)

        print(f"Weather data saved to: {output_file}")
        print(f"Downloaded {len(hourly_dataframe)} hourly records")

    except (ConnectionError, TimeoutError, ValueError, KeyError) as e:
        print(f"Failed to download weather data: {e}")
        sys.exit(1)


if __name__ == "__main__":
    end_date = None
    if len(sys.argv) >= 2:
        end_date = sys.argv[1]

    download_weather_data(end_date_param=end_date)
