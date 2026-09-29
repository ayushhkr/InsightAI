"""Safe parsing and execution of structured dataframe filters."""
from __future__ import annotations

from typing import Any
import pandas as pd


SUPPORTED_OPERATORS = {
    "==", "!=", ">", "<", ">=", "<=", "contains", "startswith", "endswith",
    "in", "not_in", "is_null", "is_not_null", "between",
}


class FilterValidationError(ValueError):
    """A structured filter cannot safely be applied to the dataset."""


def validate_filter(filter_spec: Any, df: pd.DataFrame) -> dict | None:
    if filter_spec is None:
        return None
    if not isinstance(filter_spec, dict):
        raise FilterValidationError("filter must be an object with conditions and logic.")
    conditions = filter_spec.get("conditions")
    logic = filter_spec.get("logic", "AND")
    if not isinstance(conditions, list) or not conditions:
        raise FilterValidationError("filter.conditions must be a non-empty list.")
    if logic not in {"AND", "OR"}:
        raise FilterValidationError("filter.logic must be AND or OR.")

    normalized = []
    for condition in conditions:
        if not isinstance(condition, dict):
            raise FilterValidationError("Each filter condition must be an object.")
        column, operator = condition.get("column"), condition.get("operator")
        if not isinstance(column, str) or column not in df.columns:
            raise FilterValidationError(f"Filter column '{column}' is not present in the dataset.")
        if operator not in SUPPORTED_OPERATORS:
            raise FilterValidationError(f"Unsupported filter operator: {operator}")
        value = condition.get("value")
        if operator == "between":
            if not isinstance(value, list) or len(value) != 2:
                raise FilterValidationError("between requires a list with exactly two values.")
        elif operator in {"in", "not_in"}:
            if not isinstance(value, list):
                raise FilterValidationError(f"{operator} requires value to be a list.")
        elif operator not in {"is_null", "is_not_null"} and "value" not in condition:
            raise FilterValidationError(f"{operator} requires a value.")
        _validate_value_compatibility(df[column], operator, value)
        normalized.append({"column": column, "operator": operator, "value": value})
    return {"conditions": normalized, "logic": logic}


def _validate_value_compatibility(series: pd.Series, operator: str, value: Any) -> None:
    if operator in {"is_null", "is_not_null"}:
        return
    values = value if operator in {"between", "in", "not_in"} else [value]
    if operator in {">", "<", ">=", "<=", "between"}:
        if pd.api.types.is_numeric_dtype(series):
            try:
                pd.to_numeric(pd.Series(values), errors="raise")
            except (TypeError, ValueError) as exc:
                raise FilterValidationError("Filter values must be numeric for this column.") from exc
        elif pd.api.types.is_datetime64_any_dtype(series):
            try:
                pd.to_datetime(values, errors="raise")
            except (TypeError, ValueError) as exc:
                raise FilterValidationError("Filter values must be dates for this column.") from exc
        else:
            raise FilterValidationError("Comparison operators require numeric or datetime columns.")
    if operator in {"contains", "startswith", "endswith"} and not pd.api.types.is_string_dtype(series):
        raise FilterValidationError(f"{operator} requires a string column.")


def apply_filter(df: pd.DataFrame, filter_spec: dict | None) -> pd.DataFrame:
    """Apply a validated filter using pandas operations only (never expression evaluation)."""
    spec = validate_filter(filter_spec, df)
    if spec is None:
        return df.copy()
    masks = [_condition_mask(df[part["column"]], part["operator"], part["value"])
             for part in spec["conditions"]]
    mask = masks[0]
    for next_mask in masks[1:]:
        mask = mask & next_mask if spec["logic"] == "AND" else mask | next_mask
    return df.loc[mask].copy()


def _condition_mask(series: pd.Series, operator: str, value: Any) -> pd.Series:
    if operator == "is_null": return series.isna()
    if operator == "is_not_null": return series.notna()
    if operator == "contains": return series.astype("string").str.contains(str(value), regex=False, na=False)
    if operator == "startswith": return series.astype("string").str.startswith(str(value), na=False)
    if operator == "endswith": return series.astype("string").str.endswith(str(value), na=False)
    if operator == "in": return series.isin(value)
    if operator == "not_in": return ~series.isin(value)
    if operator == "between": return series.between(value[0], value[1], inclusive="both")
    return {"==": series.eq, "!=": series.ne, ">": series.gt, "<": series.lt,
            ">=": series.ge, "<=": series.le}[operator](value)
