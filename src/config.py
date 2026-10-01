"""Central, conservative limits for dataframe inputs."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class DataLimits:
    """Limits are deliberately enforced before a dataframe reaches analysis."""
    max_file_size_bytes: int = 100 * 1024 * 1024
    max_rows: int = 1_000_000
    max_columns: int = 1_000


DEFAULT_DATA_LIMITS = DataLimits()
