"""Spreadsheet formula-injection protection for any future tabular exports."""
from __future__ import annotations

from io import StringIO
import pandas as pd


FORMULA_PREFIXES = ("=", "+", "-", "@")


def protect_spreadsheet_values(df: pd.DataFrame) -> pd.DataFrame:
    """Return a copy with risky string cells rendered as literal spreadsheet text."""
    safe = df.copy()
    for column in safe.columns:
        safe[column] = safe[column].map(_protect_value)
    return safe


def _protect_value(value):
    if isinstance(value, str) and value.startswith(FORMULA_PREFIXES):
        return "'" + value
    return value


def dataframe_to_safe_csv(df: pd.DataFrame, **kwargs) -> str:
    """Serialize a formula-safe CSV; HTML reports are already escaped separately."""
    output = StringIO()
    protect_spreadsheet_values(df).to_csv(output, index=False, **kwargs)
    return output.getvalue()
