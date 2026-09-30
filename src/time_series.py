"""Safe, explicit time-series preparation and period calculations."""
import pandas as pd


class TimeSeriesError(ValueError):
    pass


def datetime_series(df: pd.DataFrame, column: str, validity_threshold: float = 0.8) -> tuple[pd.Series, list[str]]:
    if column not in df.columns:
        raise TimeSeriesError(f"Date column '{column}' is not present.")
    source = df[column]
    converted = source if pd.api.types.is_datetime64_any_dtype(source) else pd.to_datetime(source, errors="coerce", format="mixed")
    non_missing = source.notna().sum()
    valid_ratio = 1.0 if non_missing == 0 else converted.notna().sum() / non_missing
    if valid_ratio < validity_threshold:
        raise TimeSeriesError(f"'{column}' is not a sufficiently valid date column ({valid_ratio:.0%} valid).")
    warnings = []
    if source.isna().any() or converted.isna().any(): warnings.append("Rows with missing or invalid dates were excluded.")
    if converted.duplicated().any(): warnings.append("Duplicate dates were aggregated.")
    return converted, warnings


def periodize(dates: pd.Series, frequency: str) -> pd.Series:
    aliases = {"month": "M", "quarter": "Q", "year": "Y", "M": "M", "Q": "Q", "Y": "Y"}
    if frequency not in aliases: raise TimeSeriesError("frequency must be month, quarter, or year.")
    return dates.dt.to_period(aliases[frequency]).astype(str)


def growth(current: float, previous: float) -> tuple[float | None, str]:
    if pd.isna(previous) or previous == 0: return None, "unavailable_zero_or_missing_previous"
    return ((current - previous) / abs(previous)) * 100, "ok"
