"""Offline deterministic benchmark runner for structured plan execution."""
from __future__ import annotations

from dataclasses import asdict
from typing import Callable
import numpy as np
import pandas as pd

from eval.benchmark import EvaluationCase, benchmark_cases, controlled_dataset
from src.analyzer import execute_analysis_plan
from src.plan_validation import validate_analysis_plan


def run_evaluation(cases: list[EvaluationCase] | None = None, *, tolerance: float = 1e-9,
                   planner: Callable[[EvaluationCase], dict] | None = None) -> dict:
    """Run offline reference plans; pass a planner only for a separate integration run."""
    dataset = controlled_dataset()
    cases = cases or benchmark_cases()
    planner = planner or (lambda case: dict(case.expected_plan))
    checks = {"plan_validity": [0, 0], "operation": [0, 0], "columns": [0, 0], "filters": [0, 0], "numerical": [0, 0], "ranking": [0, 0]}
    failures = []
    for case in cases:
        try:
            actual_plan = planner(case)
            actual = validate_analysis_plan(actual_plan, dataset)
            expected = validate_analysis_plan(case.expected_plan, dataset)
            case_checks = {"plan_validity": True, "operation": actual["operation"] == expected["operation"],
                           "columns": _same_columns(actual, expected), "filters": actual.get("filter") == expected.get("filter")}
            _record(checks["plan_validity"], case_checks["plan_validity"])
            _record(checks["operation"], case_checks["operation"])
            _record(checks["columns"], case_checks["columns"])
            _record(checks["filters"], case_checks["filters"])
            actual_result, _ = execute_analysis_plan(dataset, actual)
            expected_result = _reference_execute(dataset, expected)
            case_checks["numerical"] = dataframe_matches(actual_result, expected_result, tolerance)
            _record(checks["numerical"], case_checks["numerical"])
            needs_ranking = expected["operation"] in {"top_n", "bottom_n", "ranking"}
            case_checks["ranking"] = _ranking_matches(actual_result, expected_result) if needs_ranking else True
            _record(checks["ranking"], case_checks["ranking"])
            if not all(case_checks.values()):
                failures.append({"question": case.question, "failed_checks": [key for key, value in case_checks.items() if not value]})
        except Exception as exc:
            for value in checks.values(): _record(value, False)
            failures.append({"question": case.question, "error": str(exc)})
    passed = len(cases) - len(failures)
    return {"total_cases": len(cases), "passed": passed, "failed": len(failures), "checks": {name: tuple(value) for name, value in checks.items()}, "failures": failures}


def dataframe_matches(actual: pd.DataFrame, expected: pd.DataFrame, tolerance: float = 1e-9) -> bool:
    """Compare relevant columns and values; only ordered comparisons use row order."""
    if set(actual.columns) != set(expected.columns) or len(actual) != len(expected): return False
    actual, expected = actual[expected.columns].reset_index(drop=True), expected.reset_index(drop=True)
    for column in expected.columns:
        if pd.api.types.is_numeric_dtype(expected[column]):
            if not np.allclose(actual[column].astype(float), expected[column].astype(float), rtol=0, atol=tolerance, equal_nan=True): return False
        elif not actual[column].equals(expected[column]): return False
    return True


def format_results(result: dict) -> str:
    checks = result["checks"]
    return "\n".join([f"Total cases: {result['total_cases']}", f"Passed: {result['passed']}", f"Failed: {result['failed']}", "",
        f"Plan validity: {checks['plan_validity'][0]}/{checks['plan_validity'][1]}", f"Operation accuracy: {checks['operation'][0]}/{checks['operation'][1]}",
        f"Column accuracy: {checks['columns'][0]}/{checks['columns'][1]}", f"Filter accuracy: {checks['filters'][0]}/{checks['filters'][1]}",
        f"Numerical accuracy: {checks['numerical'][0]}/{checks['numerical'][1]}", f"Ranking accuracy: {checks['ranking'][0]}/{checks['ranking'][1]}"])


def _record(counter, passed): counter[1] += 1; counter[0] += int(passed)
def _same_columns(actual, expected): return all(actual.get(key) == expected.get(key) for key in ("group_column", "metric", "date_column", "correlation_columns"))
def _ranking_matches(actual, expected): return actual.reset_index(drop=True).equals(expected.reset_index(drop=True))


def _reference_execute(df, plan):
    # Independent, deliberately small oracle for benchmark operations.
    filtered = df
    spec = plan.get("filter")
    if spec:
        for cond in spec["conditions"]:
            if cond["operator"] == "==": filtered = filtered[filtered[cond["column"]] == cond["value"]]
    op, metric, group, agg = plan["operation"], plan.get("metric"), plan.get("group_column"), plan.get("aggregation")
    if op == "aggregate": return pd.DataFrame({metric: [getattr(filtered[metric], agg)()], "aggregation": [agg]})
    if op in {"groupby", "top_n", "bottom_n", "ranking", "percentage_contribution"}:
        result = filtered.groupby(group, dropna=False)[metric].agg(agg).reset_index()
        if op == "top_n": return result.sort_values(metric, ascending=False).head(plan.get("limit")).reset_index(drop=True)
        if op == "bottom_n": return result.sort_values(metric, ascending=True).head(plan.get("limit")).reset_index(drop=True)
        if op == "ranking": result = result.sort_values(metric, ascending=False).reset_index(drop=True); result["rank"] = range(1, len(result) + 1)
        if op == "percentage_contribution": result["percentage_contribution"] = result[metric] / result[metric].sum() * 100; result = result.sort_values("percentage_contribution", ascending=False).reset_index(drop=True)
        return result
    if op == "correlation": return filtered[plan["correlation_columns"]].corr().reset_index()
    if op in {"growth_rate", "time_trend"}:
        work = filtered.assign(period=filtered[plan["date_column"]].dt.to_period("M").astype(str))
        result = work.groupby("period")[metric].agg(agg).reset_index().sort_values("period").reset_index(drop=True)
        if op == "growth_rate":
            result["previous_value"] = result[metric].shift(1); result["growth_pct"] = result[metric].pct_change() * 100; result["growth_status"] = ["unavailable_zero_or_missing_previous"] + ["ok"] * (len(result) - 1)
        return result
    if op == "period_comparison":
        values = filtered.assign(period=filtered[plan["date_column"]].dt.to_period("M").astype(str)).groupby("period")[metric].agg(agg)
        old, new = (values[p] for p in plan["comparison_periods"]); pct = (new - old) / old * 100
        return pd.DataFrame({"previous_period": [plan["comparison_periods"][0]], "current_period": [plan["comparison_periods"][1]], "previous_value": [old], "current_value": [new], "absolute_change": [new-old], "percentage_change": [pct], "growth_status": ["ok"]})
    if op == "describe": return filtered[[metric]].describe().reset_index() if metric else filtered.select_dtypes(include="number").describe().reset_index()
    raise AssertionError(op)


if __name__ == "__main__":
    print(format_results(run_evaluation()))
