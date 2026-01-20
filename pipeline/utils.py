"""Shared utility helpers for the pipeline."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional, Union

DateLike = Union[str, date, datetime]


def ensure_directory(path: Union[str, Path]) -> Path:
    """Ensure that the directory exists, creating it if necessary."""
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


def get_last_day_of_previous_month(today: Optional[date] = None) -> date:
    """Calculate the last day of the previous month."""
    if today is None:
        today = date.today()
    first_day_current_month = today.replace(day=1)
    return first_day_current_month - timedelta(days=1)


def resolve_end_date(end_date_param: Optional[DateLike]) -> date:
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
