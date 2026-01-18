"""
PPNet legacy consumption extractor (one-shot).

Converts the legacy input file into daily PPNet-export-shaped CSVs.

Data is not available from PPNet directly anymore so I am not sure about the legality
of distributing it. Hence this script only converts an existing personal file.
If you wanna have the original data file, ask me directly. I think it won't be a problem
because the final merged dataset will be the main deliverable of this project anyway.

Paths
-----
This script is meant to be run from `tools/` or from the repository root.
All relative paths are resolved from the repository root.

Output file shape (matches PPNet exports)
----------------------------------------
- One file per day: `YYYYMMDD.csv`
- 24 rows for a 24-hour window: 06:00..23:00 (same day) + 00:00..05:00 (next day)
- Timestamps are written in PPNet's 12-hour style (no AM/PM), e.g. `01:00` and `12:00`.

Run
---
`python ppnet_data_extractor.py`

"""

import argparse
import csv
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def root_path(path: Path) -> Path:
    """
    Resolve path relative to project root if not absolute.

    This is used to run this script from either `tools/` or project root.

    """
    return path if path.is_absolute() else (PROJECT_ROOT / path)


@dataclass
class DayState:
    """
    State while parsing a single day's entries since hours repeat in 12-hour format.
    """

    period: int = 0  # 0=night, 1=morning, 2=afternoon, 3=evening


def map_hour(hour_12: int, state: DayState) -> int:
    """
    Map 12-hour format hour to 24-hour format using the current day state.
    """
    if hour_12 == 12:
        if state.period >= 1:
            state.period = 2
            return 12
        return 0
    if 1 <= hour_12 <= 5:
        return hour_12 + 12 if state.period >= 2 else hour_12
    if 6 <= hour_12 <= 11:
        if state.period >= 2:
            state.period = 3
            return hour_12 + 12
        state.period = 1
        return hour_12
    raise ValueError(f"Invalid 12-hour hour: {hour_12}")


def parse_legacy_file(path: Path) -> dict[datetime, int]:
    """
    Parse the legacy PPNet consumption file into a datetime-indexed dictionary.
    """
    states: dict[str, DayState] = {}
    data: dict[datetime, int] = {}
    bad_lines = 0

    with path.open("r", encoding="utf-8", errors="replace") as f:
        for raw in f:
            line = raw.strip()
            if not line:
                continue
            parts = [p.strip() for p in line.split(";")]
            if len(parts) < 2:
                bad_lines += 1
                continue

            dt_str, val_str = parts[0], parts[1]
            try:
                date_str, time_str = dt_str.split()
                day_dt = datetime.strptime(date_str, "%d.%m.%Y")
                hour_12 = int(time_str.split(":", 1)[0])
                state = states.setdefault(date_str, DayState())
                hour_24 = map_hour(hour_12, state)
                if val_str:
                    dt = day_dt.replace(hour=hour_24, minute=0, second=0, microsecond=0)
                    data[dt] = int(val_str)
            except (ValueError, IndexError):
                bad_lines += 1

    if bad_lines:
        print(f"WARNING: Skipped {bad_lines} unparsable lines")
    return data


def _to_date(value: str) -> datetime:
    """
    Convert a 'YYYY-MM-DD' string to a datetime at midnight.
    """
    return datetime.strptime(value, "%Y-%m-%d").replace(hour=0, minute=0, second=0)


def write_daily_files(
    data: dict[datetime, int],
    out_dir: Path,
    start: str,
    end: str,
    nan_string: str,
) -> None:
    """
    Write daily CSV files in PPNet export format.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    start_day = _to_date(start)
    end_day = _to_date(end)

    day = start_day
    files = 0
    while day <= end_day:
        file_path = out_dir / f"{day.strftime('%Y%m%d')}.csv"
        with file_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.writer(
                f, delimiter=";", quoting=csv.QUOTE_ALL, lineterminator="\n"
            )
            writer.writerow(["Datum", "Hodnota"])

            window = [day + timedelta(hours=h) for h in range(6, 24)] + [
                day + timedelta(days=1, hours=h) for h in range(0, 6)
            ]
            for actual in window:
                val = data.get(actual)
                h = actual.hour
                disp_hour = 12 if h in (0, 12) else (h if 1 <= h <= 11 else h - 12)
                disp = actual.replace(hour=disp_hour)
                writer.writerow(
                    [
                        disp.strftime("%Y-%m-%d %H:%M:%S"),
                        str(val) if val is not None else nan_string,
                    ]
                )

        files += 1
        day += timedelta(days=1)

    print(f"Wrote {files:,} daily files to: {out_dir}")


def main() -> None:
    """
    Main entry point for PPNet legacy extractor.
    """
    parser = argparse.ArgumentParser(description="PPNet legacy extractor (one-shot).")
    parser.add_argument(
        "--in",
        dest="input_path",
        type=Path,
        default=Path("data/input_ppnet_consumption.csv"),
        help="Legacy input file (relative to repo root by default).",
    )
    parser.add_argument(
        "--out-dir",
        dest="out_dir",
        type=Path,
        default=Path("data/raw/consumption/ppnet"),
        help="Output directory (relative to repo root by default).",
    )
    parser.add_argument("--start", default="2013-01-01")
    parser.add_argument("--end", default="2015-12-31")
    parser.add_argument("--nan", dest="nan_string", default="")

    args = parser.parse_args()
    input_path = root_path(args.input_path)
    out_dir = root_path(args.out_dir)
    data = parse_legacy_file(input_path)
    write_daily_files(data, out_dir, args.start, args.end, args.nan_string)


if __name__ == "__main__":
    main()
