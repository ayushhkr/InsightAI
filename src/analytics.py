"""Deterministic advanced analysis operations built on pandas."""
import pandas as pd
from src.time_series import datetime_series, growth, periodize


def grouped(df, dimension, metric, aggregation):
    return df.groupby(dimension, dropna=False)[metric].agg(aggregation).reset_index()


def contribution_analysis(df, metric, dimension, date_column, previous_period, current_period, aggregation="sum"):
    dates, warnings = datetime_series(df, date_column)
    work = df.assign(__period__=dates.dt.to_period("Q").astype(str)).dropna(subset=["__period__"])
    values = work.pivot_table(index=dimension, columns="__period__", values=metric, aggfunc=aggregation, fill_value=0)
    previous = values[previous_period] if previous_period in values else pd.Series(0, index=values.index)
    current = values[current_period] if current_period in values else pd.Series(0, index=values.index)
    result = pd.DataFrame({dimension: values.index, "previous_value": previous.values, "current_value": current.values})
    result["absolute_change"] = result.current_value - result.previous_value
    result["percentage_change"] = [growth(c, p)[0] for c, p in zip(result.current_value, result.previous_value)]
    total_change = result.absolute_change.sum()
    result["contribution_to_total_change"] = None if total_change == 0 else result.absolute_change / total_change * 100
    return result.sort_values("absolute_change", key=lambda s: s.abs(), ascending=False), warnings


def execute_advanced(df, plan):
    op, metric, dimension = plan["operation"], plan.get("metric"), plan.get("group_column")
    limit, aggregation = plan.get("limit"), plan.get("aggregation", "sum")
    if op in {"top_n", "bottom_n", "ranking", "percentage_contribution"}:
        result = grouped(df, dimension, metric, aggregation)
        if op == "top_n": result = result.sort_values(metric, ascending=False)
        elif op == "bottom_n": result = result.sort_values(metric, ascending=True)
        elif op == "ranking":
            result = result.sort_values(metric, ascending=False); result["rank"] = range(1, len(result) + 1)
        else:
            total = result[metric].sum(); result["percentage_contribution"] = None if total == 0 else result[metric] / total * 100
            result = result.sort_values("percentage_contribution", ascending=False)
        return result.head(limit) if limit else result, []
    if op == "distribution":
        counts = pd.cut(df[metric], bins=plan.get("bins", 10), include_lowest=True, duplicates="drop").value_counts(sort=False)
        return counts.rename_axis("bin").reset_index(name="count"), []
    dates, warnings = datetime_series(df, plan["date_column"])
    work = df.assign(__date__=dates).dropna(subset=["__date__"])
    frequency = plan.get("frequency", "month")
    if op in {"growth_rate", "time_trend", "rolling_average"}:
        period = periodize(work.__date__, frequency)
        result = work.assign(period=period).groupby("period")[metric].agg(aggregation).reset_index().sort_values("period")
        if op == "growth_rate":
            result["previous_value"] = result[metric].shift(1)
            computed = [growth(c, p) for c, p in zip(result[metric], result.previous_value)]
            result["growth_pct"] = [item[0] for item in computed]; result["growth_status"] = [item[1] for item in computed]
        elif op == "rolling_average": result["rolling_average"] = result[metric].rolling(plan.get("window", 3), min_periods=1).mean()
        return result, warnings
    if op == "period_comparison":
        periods = plan["comparison_periods"]
        values = work.assign(period=periodize(work.__date__, frequency)).groupby("period")[metric].agg(aggregation)
        previous, current = values.get(periods[0], 0), values.get(periods[1], 0)
        pct, status = growth(current, previous)
        return pd.DataFrame({"previous_period": [periods[0]], "current_period": [periods[1]], "previous_value": [previous], "current_value": [current], "absolute_change": [current - previous], "percentage_change": [pct], "growth_status": [status]}), warnings
    raise ValueError(f"Unsupported advanced operation: {op}")
