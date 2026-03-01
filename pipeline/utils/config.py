"""Central configuration for pipeline defaults and paths."""

from datetime import date, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

# Raw data directories
RAW_CONSUMPTION_DIR = RAW_DIR / "consumption"
RAW_PRICE_DIR = RAW_DIR / "price"
RAW_WEATHER_DIR = RAW_DIR / "weather"

# Processed data directories
PROCESSED_CONSUMPTION_DIR = PROCESSED_DIR / "consumption"
PROCESSED_PRICE_DIR = PROCESSED_DIR / "price"
PROCESSED_WEATHER_DIR = PROCESSED_DIR / "weather"
PROCESSED_DATETIME_FEATURES_DIR = PROCESSED_DIR / "datetime_features"
PROCESSED_MERGED_DIR = PROCESSED_DIR / "merged"

# Date defaults - date(2013, 1, 1) is the first date with data in all datasets
COMMON_START_DATE = date(2013, 1, 1)
CONSUMPTION_DOWNLOAD_START_DATE = COMMON_START_DATE - timedelta(days=1)

# CONSUMPTION_DOWNLOAD_START_DATE = date(2012, 12, 31)
CONSUMPTION_PROCESS_START_DATE = COMMON_START_DATE
PRICE_START_DATE = COMMON_START_DATE
WEATHER_START_DATE = COMMON_START_DATE
MERGE_START_DATE = COMMON_START_DATE

CONSUMPTION_MIN_DATE = CONSUMPTION_DOWNLOAD_START_DATE

# Default consumption networks.
DEFAULT_CONSUMPTION_NETWORKS = (
    "gasnet",
    "vcpnet",
    "jmpnet",
    "smpnet",
)

# Weather API settings
WEATHER_LATITUDE = 50.1333
WEATHER_LONGITUDE = 14.55  # Kbely airport
WEATHER_TIMEZONE = "Europe/Berlin"  # nearest to the Prague
WEATHER_VARIABLES = [
    "temperature_2m",
    "relative_humidity_2m",
    "pressure_msl",
    "surface_pressure",
    "dew_point_2m",
    "cloud_cover",
    "wind_speed_10m",
    "wind_direction_10m",
    "apparent_temperature",
    "wind_gusts_10m",
    "precipitation",
    "snowfall",
    "snow_depth",
]