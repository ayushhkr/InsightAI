"""Content and schema fingerprints used to safely key dataset-specific caches."""
from __future__ import annotations

import hashlib
import json
import pandas as pd


def dataframe_fingerprint(df: pd.DataFrame) -> str:
    """Return a deterministic hash that changes for index, values, columns, or dtypes.

    ``hash_pandas_object`` avoids building an enormous CSV/JSON representation while
    preserving the dataframe's row order and values.
    """
    schema = {
        "columns": [str(column) for column in df.columns],
        "dtypes": [str(dtype) for dtype in df.dtypes],
        "index_dtype": str(df.index.dtype),
    }
    digest = hashlib.sha256(json.dumps(schema, sort_keys=True).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return digest.hexdigest()
