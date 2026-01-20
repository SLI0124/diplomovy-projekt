"""
Module for downloading weather data from Open-Meteo API.

Downloads hourly weather data for the Czech Republic region using the
Open-Meteo Archive API. Data includes temperature, humidity, pressure, wind,
precipitation, and other meteorological variables.
Data is available from 2013-01-01 onwards.
"""

import datetime
import sys

import config
import openmeteo_requests
import pandas as pd
import utils
from retry_requests import retry

DATA_SAVE_PATH = config.RAW_WEATHER_DIR


def _setup_api_client():
    """Setup the Open-Meteo API client with retry configuration."""
    retry_session = retry(retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)  # type: ignore


def _build_api_params(start_date_val, end_date_val):
    """Build API parameters for weather data request."""
    return {
        "latitude": config.WEATHER_LATITUDE,
        "longitude": config.WEATHER_LONGITUDE,  # Kbely Airport
        "start_date": start_date_val.strftime("%Y-%m-%d"),
        "end_date": end_date_val.strftime("%Y-%m-%d"),
        "hourly": config.WEATHER_VARIABLES,
        "timezone": config.WEATHER_TIMEZONE,
    }


def _process_api_response(response):
    """Process API response and extract weather data.

    print(f"Coordinates: {response.Latitude()}°N {response.Longitude()}°E")
    print(f"Elevation: {response.Elevation()} m asl")
    print(f"Timezone: {response.Timezone()}{response.TimezoneAbbreviation()}")
    print(f"Timezone difference to GMT+0: {response.UtcOffsetSeconds()}s")
    """

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
    for i, var_name in enumerate(config.WEATHER_VARIABLES):
        hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()

    return pd.DataFrame(data=hourly_data)


def download_weather_data(end_date_param=None):
    """Main download function - entry point for main.py"""
    start_date_obj = config.WEATHER_START_DATE
    try:
        end_date_obj = utils.resolve_end_date(end_date_param)
    except ValueError as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    # since the weather source provides data up to 22:00 of the day
    # we need to adjust end_date by one day to include the last day's data
    end_date_obj += datetime.timedelta(days=1)

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
            f"Start date {start_date_val} is after end date {end_date_val}. \
                Nothing to download."
        )
        return

    print(f"Downloading weather data from {start_date_val} to {end_date_val}...")

    utils.ensure_directory(DATA_SAVE_PATH)

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
