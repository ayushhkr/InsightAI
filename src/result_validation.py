"""Post-execution safeguards for analysis results."""
from __future__ import annotations
import pandas as pd

class ResultValidationError(ValueError):
    pass

def validate_result(result: object, plan: dict, filtered_row_count: int) -> list[str]:
    if not isinstance(result, pd.DataFrame): raise ResultValidationError("Expected a pandas DataFrame result.")
    if result.empty and filtered_row_count > 0: raise ResultValidationError("Analysis unexpectedly produced no rows.")
    op, metric, group = plan["operation"], plan.get("metric"), plan.get("group_column")
    expected = [c for c in ([group, metric] if op in {"groupby", "trend"} else [metric] if op == "aggregate" else []) if c]
    missing = [c for c in expected if c not in result.columns]
    if missing: raise ResultValidationError(f"Result is missing requested columns: {missing}")
    if plan.get("aggregation") and op in {"groupby", "aggregate", "trend"} and metric not in result.columns:
        raise ResultValidationError("Requested aggregation did not occur.")
    if metric in result.columns and plan.get("aggregation") != "count" and op in {"groupby", "aggregate", "trend"} and not pd.api.types.is_numeric_dtype(result[metric]):
        raise ResultValidationError(f"Aggregated metric '{metric}' is not numeric.")
    if plan.get("limit") and len(result) > plan["limit"]: raise ResultValidationError("Result limit was not respected.")
    if plan.get("sort") and metric in result.columns and len(result) > 1:
        values = result[metric].dropna()
        if not (values.is_monotonic_increasing if plan["sort"] == "ascending" else values.is_monotonic_decreasing):
            raise ResultValidationError("Result sorting was not respected.")
    return ["Filter legitimately produced zero rows."] if result.empty and filtered_row_count == 0 and plan.get("filter") else []
