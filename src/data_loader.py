"""Safe, reusable dataframe file loading with centralized input limits."""
from __future__ import annotations

from pathlib import Path
from typing import BinaryIO
import os
import pandas as pd

from src.config import DEFAULT_DATA_LIMITS, DataLimits


SUPPORTED_EXTENSIONS = {".csv", ".xlsx", ".json", ".parquet"}


class DataLoadError(ValueError):
    """A controlled file loading failure suitable for display to an application user."""


class UnsupportedFileFormatError(DataLoadError):
    pass


class DataLimitError(DataLoadError):
    pass


def load_dataframe(source: str | Path | BinaryIO, *, filename: str | None = None,
                   limits: DataLimits = DEFAULT_DATA_LIMITS) -> pd.DataFrame:
    """Load CSV, XLSX, JSON, or Parquet without evaluating file-provided code."""
    name = filename or getattr(source, "name", None) or (str(source) if isinstance(source, (str, Path)) else "")
    suffix = Path(name).suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        supported = ", ".join(sorted(SUPPORTED_EXTENSIONS))
        raise UnsupportedFileFormatError(f"Unsupported file format '{suffix or 'unknown'}'. Supported formats: {supported}.")
    _validate_file_size(source, limits)
    try:
        if suffix == ".csv":
            df = pd.read_csv(source)
        elif suffix == ".xlsx":
            df = pd.read_excel(source, engine="openpyxl")
        elif suffix == ".json":
            df = pd.read_json(source)
        else:
            df = pd.read_parquet(source)
    except (OSError, ValueError, TypeError, ImportError) as exc:
        raise DataLoadError(f"Could not read '{Path(name).name or 'dataset'}': {exc}") from exc
    _validate_dataframe_limits(df, limits)
    return df


def _validate_file_size(source: str | Path | BinaryIO, limits: DataLimits) -> None:
    try:
        if isinstance(source, (str, Path)):
            size = os.path.getsize(source)
        else:
            current = source.tell()
            source.seek(0, 2)
            size = source.tell()
            source.seek(current)
    except (AttributeError, OSError):
        # A non-seekable source is still read by pandas; callers can provide a
        # seekable upload to receive a pre-read size check.
        return
    if size > limits.max_file_size_bytes:
        raise DataLimitError(f"File size {size} bytes exceeds the {limits.max_file_size_bytes} byte limit.")


def _validate_dataframe_limits(df: pd.DataFrame, limits: DataLimits) -> None:
    rows, columns = df.shape
    if rows > limits.max_rows:
        raise DataLimitError(f"Dataset has {rows} rows and exceeds the {limits.max_rows} row limit.")
    if columns > limits.max_columns:
        raise DataLimitError(f"Dataset has {columns} columns and exceeds the {limits.max_columns} column limit.")
