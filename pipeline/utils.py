"""Shared utility helpers for the pipeline."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Union

DateLike = Union[str, date, datetime]


def ensure_directory(path: str | Path) -> Path:
    """Ensure that the directory exists, creating it if necessary."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_last_day_of_previous_month(today: date | None = None) -> date:
    """Calculate the last day of the previous month."""
    if today is None:
        today = date.today()
    first_day_current_month = today.replace(day=1)
    return first_day_current_month - timedelta(days=1)


def resolve_end_date(end_date_param: DateLike | None) -> date:
    """Resolve an end date with validation and default fallback.

    Args:
        end_date_param: None, YYYY-MM-DD string, date, or datetime.

    Returns:
        date: Resolved date value.

    Raises:
        ValueError: If a string value is not in YYYY-MM-DD format.
    """
    if end_date_param is None:
        return get_last_day_of_previous_month()

    if isinstance(end_date_param, datetime):
        return end_date_param.date()

    if isinstance(end_date_param, date):
        return end_date_param

    if isinstance(end_date_param, str):
        try:
            return datetime.strptime(end_date_param, "%Y-%m-%d").date()
        except ValueError as exc:
            raise ValueError(
                f"Invalid date format '{end_date_param}'. Please use YYYY-MM-DD."
            ) from exc

    raise ValueError("Invalid end_date_param type. Expected str, date, or datetime.")


def validate_date_str(value: str) -> str:
    """Validate YYYY-MM-DD string format and return the value."""
    try:
        datetime.strptime(value, "%Y-%m-%d")
    except ValueError as exc:
        raise ValueError(
            f"Invalid date format '{value}'. Please use YYYY-MM-DD."
        ) from exc
    return value


def iter_dates(start_date: date, end_date: date) -> Iterable[date]:
    """Yield each calendar date in [start_date, end_date] inclusive."""
    if start_date > end_date:
        return
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def iter_year_months(start_date: date, end_date: date) -> Iterable[tuple[int, int]]:
    """Yield (year, month) pairs intersecting [start_date, end_date] inclusive."""
    if start_date > end_date:
        return
    current = start_date.replace(day=1)
    end_month = end_date.replace(day=1)
    while current <= end_month:
        yield current.year, current.month
        if current.month == 12:
            current = current.replace(year=current.year + 1, month=1)
        else:
            current = current.replace(month=current.month + 1)


def _summarize_paths(paths: list[Path], limit: int = 12) -> str:
    shown = paths[:limit]
    more = len(paths) - len(shown)
    lines = "\n".join(f"  - {p}" for p in shown)
    if more > 0:
        lines += f"\n  ... and {more} more"
    return lines


def raise_missing_inputs(
    *,
    what: str,
    missing_paths: list[Path],
    required_range: str | None = None,
) -> None:
    """Raise a FileNotFoundError with a helpful, compact message."""
    if not missing_paths:
        return
    header = f"Missing required input files for {what}."
    if required_range:
        header += f" Required range: {required_range}."
    details = _summarize_paths(missing_paths)
    raise FileNotFoundError(f"{header}\nMissing ({len(missing_paths)}):\n{details}")
