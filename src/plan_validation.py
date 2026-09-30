"""Central validation and normalization of LLM analysis plans."""
from __future__ import annotations

from typing import Any
import pandas as pd
from src.filters import validate_filter

OPERATIONS = {"groupby", "aggregate", "filter", "correlation", "trend", "describe", "top_n", "bottom_n", "ranking", "percentage_contribution", "growth_rate", "period_comparison", "rolling_average", "time_trend", "distribution"}
AGGREGATIONS = {"sum", "mean", "count", "min", "max"}


class PlanValidationError(ValueError):
    pass


def validate_analysis_plan(plan: Any, df: pd.DataFrame) -> dict:
    if not isinstance(plan, dict):
        raise PlanValidationError("Analysis plan must be a JSON object.")
    normalized = dict(plan)
    # Legacy test/app clients used action; retain it as a non-public compatibility alias.
    normalized["operation"] = normalized.get("operation", normalized.get("action"))
    operation = normalized["operation"]
    if operation not in OPERATIONS:
        raise PlanValidationError(f"Unsupported operation: {operation}")
    dimensions = normalized.get("dimensions")
    if dimensions is not None:
        if not isinstance(dimensions, list) or not all(isinstance(c, str) for c in dimensions):
            raise PlanValidationError("dimensions must be a list of column names.")
        _validate_columns(dimensions, df, "dimension")
        if len(dimensions) > 1:
            raise PlanValidationError("Only one dimension is supported.")
        normalized.setdefault("group_column", dimensions[0] if dimensions else None)
    group, metric, aggregation = (normalized.get("group_column"), normalized.get("metric"), normalized.get("aggregation"))
    _validate_optional_column(group, df, "group_column")
    _validate_optional_column(metric, df, "metric", allow_derived_revenue=True)
    if aggregation is not None and aggregation not in AGGREGATIONS:
        raise PlanValidationError(f"Unsupported aggregation: {aggregation}")
    grouped_operations = {"groupby", "trend", "top_n", "bottom_n", "ranking", "percentage_contribution"}
    if operation in grouped_operations and (not group or not metric or not aggregation):
        raise PlanValidationError(f"{operation} requires group_column, metric, and aggregation.")
    if operation == "aggregate" and (not metric or not aggregation):
        raise PlanValidationError("aggregate requires metric and aggregation.")
    if aggregation and metric and aggregation != "count" and metric in df.columns and not pd.api.types.is_numeric_dtype(df[metric]):
        raise PlanValidationError(f"Metric '{metric}' must be numeric for {aggregation}.")
    normalized["filter"] = validate_filter(normalized.get("filter"), df)
    sort = normalized.get("sort")
    if sort is not None and sort not in {"ascending", "descending"}:
        raise PlanValidationError("sort must be ascending, descending, or null.")
    limit = normalized.get("limit", normalized.get("top_n"))
    if limit is not None and (not isinstance(limit, int) or isinstance(limit, bool) or limit <= 0):
        raise PlanValidationError("limit/top_n must be a positive integer or null.")
    normalized["limit"] = limit
    normalized["top_n"] = limit
    time_operations = {"trend", "growth_rate", "period_comparison", "rolling_average", "time_trend", "distribution"}
    date_column = normalized.get("date_column", group if operation == "trend" else None)
    if date_column is not None:
        _validate_optional_column(date_column, df, "date_column")
    if operation in time_operations and operation != "distribution" and (not date_column or not metric or not aggregation):
        raise PlanValidationError(f"{operation} requires date_column, metric, and aggregation.")
    normalized["date_column"] = date_column
    if normalized.get("frequency", "month") not in {"month", "quarter", "year", "M", "Q", "Y"}:
        raise PlanValidationError("frequency must be month, quarter, or year.")
    if operation == "period_comparison":
        periods = normalized.get("comparison_periods")
        if not isinstance(periods, list) or len(periods) != 2 or not all(isinstance(item, str) for item in periods):
            raise PlanValidationError("period_comparison requires two comparison_periods.")
    if operation == "rolling_average" and (not isinstance(normalized.get("window", 3), int) or normalized.get("window", 3) <= 0):
        raise PlanValidationError("rolling_average window must be a positive integer.")
    if operation == "distribution":
        if not metric or metric not in df.columns or not pd.api.types.is_numeric_dtype(df[metric]):
            raise PlanValidationError("distribution requires a numeric metric.")
        bins = normalized.get("bins", 10)
        if not isinstance(bins, int) or bins <= 0: raise PlanValidationError("distribution bins must be a positive integer.")
    date_range = normalized.get("date_range")
    if date_range is not None:
        if not date_column or not isinstance(date_range, dict) or set(date_range) - {"start", "end"}:
            raise PlanValidationError("date_range requires date_column and start/end fields.")
        try: pd.to_datetime([date_range.get("start"), date_range.get("end")], errors="raise")
        except (TypeError, ValueError) as exc: raise PlanValidationError("date_range values must be valid dates.") from exc
    correlation_columns = normalized.get("correlation_columns")
    if correlation_columns is not None:
        if not isinstance(correlation_columns, list) or len(correlation_columns) < 2:
            raise PlanValidationError("correlation_columns must contain at least two columns.")
        _validate_columns(correlation_columns, df, "correlation")
        if not all(pd.api.types.is_numeric_dtype(df[c]) for c in correlation_columns):
            raise PlanValidationError("correlation columns must be numeric.")
    return normalized


def _validate_columns(columns: list[str], df: pd.DataFrame, label: str) -> None:
    for column in columns: _validate_optional_column(column, df, label)


def _validate_optional_column(column: Any, df: pd.DataFrame, label: str, allow_derived_revenue: bool = False) -> None:
    if column is None: return
    if not isinstance(column, str): raise PlanValidationError(f"{label} must be a column name or null.")
    if allow_derived_revenue and column == "revenue" and {"price", "quantity"}.issubset(df.columns): return
    if column not in df.columns: raise PlanValidationError(f"Column '{column}' referenced as {label} is not present in the dataset.")
