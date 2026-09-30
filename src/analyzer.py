"""
analyzer.py
Contains generic data-analysis functions for processing the Pandas dataframes based on user requests.
"""
import pandas as pd
from src.filters import apply_filter
from src.plan_validation import validate_analysis_plan
from src.result_validation import validate_result
from src.analytics import execute_advanced

def execute_analysis_plan(df: pd.DataFrame, plan: dict) -> tuple[pd.DataFrame, dict]:
    """Validate then execute a plan without evaluating arbitrary expressions."""
    validated = validate_analysis_plan(plan, df)
    source = df.copy()
    if "revenue" in {validated.get("metric"), validated.get("group_column")} and "revenue" not in source.columns:
        source["revenue"] = source["price"] * source["quantity"]
    validated = validate_analysis_plan(validated, source)
    filtered = apply_filter(source, validated.get("filter"))
    date_column, date_range = validated.get("date_column"), validated.get("date_range")
    if date_range:
        dates = pd.to_datetime(filtered[date_column], errors="coerce")
        filtered = filtered.loc[dates.between(pd.Timestamp(date_range["start"]), pd.Timestamp(date_range["end"]), inclusive="both")].copy()
    op, group, metric, aggregation = validated["operation"], validated.get("group_column"), validated.get("metric"), validated.get("aggregation")
    try:
        if op in {"top_n", "bottom_n", "ranking", "percentage_contribution", "growth_rate", "period_comparison", "rolling_average", "time_trend", "distribution"}:
            result, advanced_warnings = execute_advanced(filtered, validated)
        elif op == "groupby": result = filtered.groupby(group, dropna=False)[metric].agg(aggregation).reset_index()
        elif op == "aggregate": result = pd.DataFrame({metric: [getattr(filtered[metric], aggregation)()], "aggregation": [aggregation]})
        elif op == "filter": result = filtered
        elif op == "correlation":
            columns = validated.get("correlation_columns") or filtered.select_dtypes(include="number").columns.tolist()
            if len(columns) < 2: raise ValueError("Need at least two numerical columns for correlation.")
            result = filtered[columns].corr().reset_index()
        elif op == "trend":
            dates = pd.to_datetime(filtered[date_column], errors="raise")
            result = filtered.assign(__trend_date__=dates.dt.normalize()).groupby("__trend_date__")[metric].agg(aggregation).reset_index().rename(columns={"__trend_date__": group})
        else:
            selected = filtered[[metric]] if metric else filtered.select_dtypes(include="number")
            result = selected.describe().reset_index() if not selected.empty else filtered.head(0)
        if validated.get("sort") and metric in result.columns: result = result.sort_values(metric, ascending=validated["sort"] == "ascending", kind="stable")
        if validated.get("limit"): result = result.head(validated["limit"])
        warnings = (advanced_warnings if 'advanced_warnings' in locals() else []) + validate_result(result, validated, len(filtered))
    except (ValueError, TypeError, KeyError) as exc:
        raise ValueError(f"Execution Error: {exc}") from exc
    metadata = {"status": "success", "operation": op}
    if warnings: metadata["warnings"] = warnings; metadata["warning"] = warnings[0]
    return result, metadata
