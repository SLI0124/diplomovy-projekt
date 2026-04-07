"""Command-line entry point for the pipeline.

Runs downloaders and processors with a small CLI wrapper.
"""

import argparse
import sys
from collections.abc import Sequence

import downloaders.consumption
import downloaders.price
import downloaders.weather_source
import processors.consumption
import processors.dates
import processors.main_merger
import processors.price
import processors.weather_source
import utils.config as config
import utils.helper_functions as utils

DOWNLOAD_ORDER = ("consumption", "weather", "price")
PROCESS_ORDER = ("dates", "consumption", "weather", "price", "merge")


def download_data(
    download_type: str | None = None,
    end_date: str | None = None,
    consumption_networks: Sequence[str] | None = None,
) -> None:
    """Download data.

    Args:
        download_type: One of: consumption, weather, price, all.
        end_date: End date used by downloaders.
        consumption_networks: Optional list of consumption networks.
    """
    match download_type:
        case "all":
            print("Downloading all data types...")
            for item in DOWNLOAD_ORDER:
                download_data(item, end_date, consumption_networks)
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
        case _:
            print(f"Download type '{download_type}' is not implemented.")


def process_data(
    process_type: str | None = None,
    end_date: str | None = None,
    consumption_networks: Sequence[str] | None = None,
) -> None:
    """Process data.

    Args:
        process_type: One of: dates, consumption, weather, price, merge, all.
        end_date: End date used by processors.
        consumption_networks: Optional list of consumption networks.
    """
    try:
        match process_type:
            case "all":
                print("Processing all data types...")
                for item in PROCESS_ORDER:
                    process_data(item, end_date, consumption_networks)
            case "dates":
                print("Process: dates")
                processors.dates.process_datetime_features(end_date_param=end_date)
            case "consumption":
                print("Process: consumption")
                processors.consumption.process_consumption_data(
                    end_date_param=end_date, networks=consumption_networks
                )
            case "weather":
                print("Process: weather")
                processors.weather_source.process_weather_data(end_date_param=end_date)
            case "price":
                print("Process: price")
                processors.price.process_price_data(end_date_param=end_date)
            case "merge":
                print("Process: merge")
                processors.main_merger.merge_processed_data(
                    end_date_param=end_date, consumption_networks=consumption_networks
                )
            case _:
                print(f"Process type '{process_type}' is not implemented.")
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")


def main(argv: Sequence[str] | None = None) -> None:
    """Main entry point for the pipeline CLI.

    Args:
        argv: Optional argv override (used for testing or edge cases).
    """
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
            "(default downloads stable networks: "
            f"{', '.join(config.DEFAULT_CONSUMPTION_NETWORKS)}). "
            f"Current options are: {', '.join(downloaders.consumption.NETWORK_URLS)}"
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
        help=(
            "Download and process all data types (consumption, weather, price, dates, "
            "and merge)."
        ),
    )
    parser.add_argument(
        "--end-date",
        default="2025-12-31",
        help=(
            "End date in YYYY-MM-DD format (e.g., 2025-10-28). "
            "If not specified, defaults to last day of previous month."
        ),
    )

    cli_args = list(argv) if argv is not None else sys.argv[1:]
    if not cli_args:
        print("No arguments provided.")
        print("Default action is to run with '--all' (download + process everything).")

        try:
            answer = input("Proceed with '--all' now? [y/N]: ").strip().lower()
        except EOFError:
            print("No input available. Aborting.")
            return

        if answer not in {"y", "yes"}:
            print("Aborted.")
            return

        cli_args = ["--all"]

    args = parser.parse_args(cli_args)

    validated_end_date = None
    if args.end_date is not None:
        try:
            validated_end_date = utils.validate_date_str(args.end_date)
        except ValueError as exc:
            print(exc)
            return

    if args.all:
        # When --all is specified, download and process all data types
        download_data("all", validated_end_date, args.consumption_networks)
        process_data("all", validated_end_date, args.consumption_networks)
    else:
        # Handle individual download and process flags
        if args.download:
            download_data(args.download, validated_end_date, args.consumption_networks)

        if args.process:
            process_data(args.process, validated_end_date, args.consumption_networks)

        if not (args.download or args.process):
            print("No action specified. Use --download, --process or --all.")


if __name__ == "__main__":
    main()
