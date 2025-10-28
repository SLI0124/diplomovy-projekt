import argparse
from datetime import datetime

import processors.dates
import processors.consumption
import downloaders.consumption


def download_data(download_type: str = None, end_date: str = None) -> None:
    if end_date:  # validate date format to be YYYY-MM-DD
        try:
            datetime.strptime(end_date, "%Y-%m-%d")
        except ValueError:
            print("Invalid date format. Please use 'YYYY-MM-DD'.")
            return

    match download_type:
        case "consumption":
            print("Downloading gas consumption data...")
            downloaders.consumption.download_consumption_data(end_date_param=end_date)
        case _:
            raise NotImplementedError("Downloading all data is not implemented yet.")


def process_data(process_type: str = None, end_date: str = None) -> None:
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
        case _:
            raise NotImplementedError("Processing all data is not implemented yet.")


def main():
    # this will run all the modules: downloaders, processors, ...
    parser = argparse.ArgumentParser(description="Data Pipeline Manager")
    parser.add_argument(
        "--download",
        choices=["consumption"],
        help="Download specific data type: 'consumption' for gas consumption data",
    )
    parser.add_argument(
        "--process",
        choices=["dates", "consumption"],
        help="Process specific data type: 'dates' for datetime features, 'consumption' for gas consumption data",
    )
    parser.add_argument("--all", action="store_true", help="Download and process data")
    parser.add_argument(
        "--end-date",
        help="End date in YYYY-MM-DD format (e.g., 2025-10-28). If not specified, defaults to yesterday.",
    )

    args = parser.parse_args()

    if args.all:
        args.download = "consumption"
        if not args.process:
            args.process = "consumption"

    if args.download:
        download_data(args.download, args.end_date)

    if args.process:
        process_data(args.process, args.end_date)

    if not (args.download or args.process):
        print("No action specified. Use --download, --process or --all.")


if __name__ == "__main__":
    main()
