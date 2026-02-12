"""Download historical hourly weather data from Open-Meteo.

The location and variables are defined in `pipeline/config.py`.
"""

import datetime

import config
import openmeteo_requests
import pandas as pd
from retry_requests import retry

import pipeline.utils as utils

DATA_SAVE_PATH = config.RAW_WEATHER_DIR


def _setup_api_client() -> openmeteo_requests.Client:
    """Create an Open-Meteo API client with retries.

    Returns:
        openmeteo_requests.Client: Configured API client.
    """
    retry_session = retry(retries=5, backoff_factor=0.2)
    return openmeteo_requests.Client(session=retry_session)  # type: ignore


def _build_api_params(
    start_date_val: datetime.date,
    end_date_val: datetime.date,
) -> dict[str, object]:
    """Build parameters for the Open-Meteo archive request.

    Args:
        start_date_val: Inclusive start date.
        end_date_val: Inclusive end date.

    Returns:
        Dict of request parameters for `Client.weather_api`.
    """
    return {
        "latitude": config.WEATHER_LATITUDE,
        "longitude": config.WEATHER_LONGITUDE,
        "start_date": start_date_val.strftime("%Y-%m-%d"),
        "end_date": end_date_val.strftime("%Y-%m-%d"),
        "hourly": config.WEATHER_VARIABLES,
        "timezone": config.WEATHER_TIMEZONE,
    }


def _process_api_response(response) -> pd.DataFrame:
    """Convert a single Open-Meteo response into a DataFrame.

    Args:
        response: One response returned by `Client.weather_api`.

    Returns:
        pandas.DataFrame: Hourly time series with a `date` column and one column
            per variable from `config.WEATHER_VARIABLES`.
    """
    hourly = response.Hourly()

    hourly_data = {
        "date": pd.date_range(
            start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
            end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
            freq=pd.Timedelta(seconds=hourly.Interval()),
            inclusive="left",
        )
    }

    for i, var_name in enumerate(config.WEATHER_VARIABLES):
        hourly_data[var_name] = hourly.Variables(i).ValuesAsNumpy()

    return pd.DataFrame(data=hourly_data)


def download_weather_data(end_date_param: utils.DateLike | None = None) -> None:
    """Download weather data using configured defaults.

    Args:
        end_date_param: End date as YYYY-MM-DD string, date, datetime, or None.
            When None, defaults to the last day of the previous month.
    """
    download_weather_data_with_range(config.WEATHER_START_DATE, end_date_param)


def download_weather_data_with_range(
    start_date_val: datetime.date,
    end_date_param: utils.DateLike | None = None,
) -> None:
    """Download weather data for a specific date range.

    Args:
        start_date_val: Inclusive start date.
        end_date_param: Inclusive end date as YYYY-MM-DD string, date, datetime,
            or None. The implementation adds 1 day internally to cover the last
            day's late-night hours.
    """
    end_date_val = utils.resolve_end_date(end_date_param) + datetime.timedelta(days=1)

    if start_date_val < config.WEATHER_START_DATE:
        print(
            f"Start date cannot be before {config.WEATHER_START_DATE} since it is the "
            "first available data from weather API."
        )
        delta_days = (config.WEATHER_START_DATE - start_date_val).days
        print(
            f"Adjusting start date by {delta_days} days to {config.WEATHER_START_DATE}."
        )
        start_date_val = config.WEATHER_START_DATE

    if start_date_val > end_date_val:
        print(
            f"Start date {start_date_val} is after end date {end_date_val}. "
            "Nothing to download."
        )
        return

    print(f"Downloading weather data from {start_date_val} to {end_date_val}...")

    utils.ensure_directory(DATA_SAVE_PATH)

    output_file = DATA_SAVE_PATH / f"weather_{start_date_val}_{end_date_val}.csv"
    if output_file.exists():
        print(f"Weather data already exists, skipping download: {output_file}")
        return

    try:
        openmeteo = _setup_api_client()
        url = "https://archive-api.open-meteo.com/v1/archive"
        params = _build_api_params(start_date_val, end_date_val)

        responses = openmeteo.weather_api(url, params=params)

        # Process first location
        hourly_dataframe = _process_api_response(responses[0])

        # Save to CSV file
        hourly_dataframe.to_csv(output_file, index=False)

        print(f"Weather data saved to: {output_file}")
        print(f"Downloaded {len(hourly_dataframe)} hourly records")

    except (ConnectionError, TimeoutError, ValueError, KeyError) as e:
        print(f"Failed to download weather data: {e}")
