"""Small trusted-operation backend abstraction for pandas and optional DuckDB."""
from __future__ import annotations

import pandas as pd

from src.analyzer import execute_analysis_plan
from src.filters import apply_filter
from src.plan_validation import validate_analysis_plan


class BackendUnavailableError(RuntimeError):
    pass


def execute_with_backend(df: pd.DataFrame, plan: dict, backend: str = "pandas") -> tuple[pd.DataFrame, dict]:
    """Execute a validated, narrow operation set without accepting arbitrary SQL."""
    if backend == "pandas":
        return execute_analysis_plan(df, plan)
    if backend != "duckdb":
        raise ValueError(f"Unsupported analytics backend: {backend}")
    try:
        import duckdb
    except ImportError as exc:
        raise BackendUnavailableError("DuckDB is optional; install the 'duckdb' package to use this backend.") from exc

    validated = validate_analysis_plan(plan, df)
    if validated["operation"] not in {"groupby", "aggregate", "top_n", "bottom_n", "ranking"}:
        # Existing pandas implementation remains the canonical executor for the
        # broader advanced operation set.
        return execute_analysis_plan(df, validated)
    work = apply_filter(df, validated.get("filter"))
    operation, metric, group = validated["operation"], validated.get("metric"), validated.get("group_column")
    aggregation = validated.get("aggregation")
    if metric == "revenue" and metric not in work.columns and {"price", "quantity"}.issubset(work.columns):
        work = work.assign(revenue=work["price"] * work["quantity"])
    connection = duckdb.connect(":memory:")
    try:
        connection.register("dataset", work)
        metric_sql = _quote_identifier(metric)
        aggregate_sql = f"{aggregation.upper()}({metric_sql})"
        if operation == "aggregate":
            result = connection.execute(f"SELECT {aggregate_sql} AS {metric_sql} FROM dataset").fetchdf()
        else:
            group_sql = _quote_identifier(group)
            direction = "ASC" if operation == "bottom_n" else "DESC"
            sql = f"SELECT {group_sql}, {aggregate_sql} AS {metric_sql} FROM dataset GROUP BY {group_sql} ORDER BY {metric_sql} {direction}"
            result = connection.execute(sql).fetchdf()
            if operation == "ranking":
                result["rank"] = range(1, len(result) + 1)
            if validated.get("limit"):
                result = result.head(validated["limit"])
    finally:
        connection.close()
    return result, {"status": "success", "operation": operation, "backend": "duckdb"}


def _quote_identifier(identifier: str) -> str:
    # Called only after plan validation confirms this exact dataframe column.
    return '"' + identifier.replace('"', '""') + '"'
