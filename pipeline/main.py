"""
Main pipeline entry point for data processing and downloading.

This module provides command-line interface for running various data processing
and downloading tasks including consumption data, weather data, datetime features,
and data merging operations.
"""

import argparse
from datetime import datetime
from typing import Optional, Sequence

import processors.dates
import processors.consumption
import processors.weather_source
import processors.price
import processors.main_merger
import downloaders.consumption
import downloaders.weather_source
import downloaders.price


def download_data(
    download_type: Optional[str] = None,
    end_date: Optional[str] = None,
    consumption_networks: Optional[Sequence[str]] = None,
) -> None:
    """Download data based on specified type, end date, and requested networks."""
    if end_date:  # validate date format to be YYYY-MM-DD
        try:
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            print("Invalid date format. Please use 'YYYY-MM-DD'.")
            return

    match download_type:
        case "consumption":
            print("Downloading gas consumption data...")
            downloaders.consumption.download_consumption_data(
                end_date_param=end_date, networks=consumption_networks
            )
        case "weather":
            print("Downloading weather data...")
            downloaders.weather_source.download_weather_data(end_date_param=end_date)
        case "price":
            print("Downloading gas price data...")
            downloaders.price.download_price_data(end_date_param=end_date)
        case "all":
            print("Downloading all data types...")
            print("Downloading gas consumption data...")
            downloaders.consumption.download_consumption_data(
                end_date_param=end_date, networks=consumption_networks
            )
            print("Downloading weather data...")
            downloaders.weather_source.download_weather_data(end_date_param=end_date)
            print("Downloading gas price data...")
            downloaders.price.download_price_data(end_date_param=end_date)
        case _:
            raise NotImplementedError(
                f"Download type '{download_type}' is not implemented."
            )


def process_data(
    process_type: Optional[str] = None, end_date: Optional[str] = None
) -> None:
    """Process data based on specified type and end date."""
    if end_date:  # validate date format to be YYYY-MM-DD
        try:
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            print("Invalid date format. Please use 'YYYY-MM-DD'.")
            return

    match process_type:
        case "dates":
            print("Processing datetime features data...")
            processors.dates.process_datetime_features(end_date_param=end_date)
        case "consumption":
            print("Processing gas consumption data...")
            processors.consumption.process_consumption_data(end_date_param=end_date)
        case "weather":
            print("Processing weather data...")
            processors.weather_source.process_weather_data(end_date_param=end_date)
        case "price":
            print("Processing gas price data...")
            processors.price.process_price_data(end_date_param=end_date)
        case "merge":
            print("Merging all processed data...")
            processors.main_merger.merge_processed_data(end_date_param=end_date)
        case "all":
            print("Processing all data types...")
            print("Processing datetime features data...")
            processors.dates.process_datetime_features(end_date_param=end_date)
            print("Processing gas consumption data...")
            processors.consumption.process_consumption_data(end_date_param=end_date)
            print("Processing weather data...")
            processors.weather_source.process_weather_data(end_date_param=end_date)
            print("Processing gas price data...")
            processors.price.process_price_data(end_date_param=end_date)
            print("Merging all processed data...")
            processors.main_merger.merge_processed_data(end_date_param=end_date)
        case _:
            raise NotImplementedError(
                f"Process type '{process_type}' is not implemented."
            )


def main():
    """Main entry point for the data pipeline manager."""
    # this will run all the modules: downloaders, processors, ...
    parser = argparse.ArgumentParser(description="Data Pipeline Manager")
    parser.add_argument(
        "--download",
        choices=["consumption", "weather", "price", "all"],
        help=(
            "Download specific data type: 'consumption' for gas consumption data, "
            "'weather' for weather data, 'price' for gas price data, "
            "'all' for all data types"
        ),
    )
    parser.add_argument(
        "--consumption-networks",
        nargs="+",
        choices=list(downloaders.consumption.NETWORK_URLS),
        metavar="NETWORK",
        help=(
            "Optional list of consumption networks to download "
            "(default downloads all supported networks)."
        ),
    )
    parser.add_argument(
        "--process",
        choices=["dates", "consumption", "weather", "price", "merge", "all"],
        help=(
            "Process specific data type: 'dates' for datetime features, "
            "'consumption' for gas consumption data, 'weather' for weather data, "
            "'price' for gas price data, 'merge' for merging all processed data"
        ),
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Download and process all data types (consumption, weather, price, dates, and merge)",
    )
    parser.add_argument(
        "--end-date",
        help=(
            "End date in YYYY-MM-DD format (e.g., 2025-10-28). "
            "If not specified, defaults to last day of previous month."
        ),
    )

    args = parser.parse_args()

    if args.all:
        # When --all is specified, download and process all data types
        download_data("all", args.end_date, args.consumption_networks)
        process_data("all", args.end_date)
    else:
        # Handle individual download and process flags
        if args.download:
            download_data(args.download, args.end_date, args.consumption_networks)

        if args.process:
            process_data(args.process, args.end_date)

        if not (args.download or args.process):
            print("No action specified. Use --download, --process or --all.")


if __name__ == "__main__":
    main()
