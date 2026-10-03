"""Small label and duration helpers for report joins.

Example: ``normalize_name("Credit-G") == "credit_g"``.
"""

import math


def normalize_name(name: str) -> str:
    """Normalize dataset and experiment labels for table joins."""
    return name.lower().replace("-", "_").replace(" ", "_")


def finite_float(value: object) -> float:
    """Return a finite float or NaN for invalid values."""
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return math.nan
    return numeric if math.isfinite(numeric) else math.nan


def format_duration_tag(value: object) -> str:
    """Format seconds as a compact runtime tag."""
    seconds = finite_float(value)
    if math.isnan(seconds):
        return "unknown"
    if seconds < 60:
        return f"{seconds:.0f}s"
    total_seconds = round(seconds)
    minutes, second_remainder = divmod(total_seconds, 60)
    if minutes < 60:
        return f"{minutes}m {second_remainder:02d}s"
    hours, minute_remainder = divmod(minutes, 60)
    if hours < 24:
        return f"{hours}h {minute_remainder:02d}m"
    days, hour_remainder = divmod(hours, 24)
    return f"{days}d {hour_remainder:02d}h"
