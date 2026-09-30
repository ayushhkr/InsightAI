"""Deterministic backend for period-change investigations."""
import json
import re
import pandas as pd
from src.analytics import contribution_analysis
from src.time_series import datetime_series, growth


def serialize_investigation_evidence(result: dict, max_contributors: int = 10) -> str:
    compact = {key: result.get(key) for key in ("metric", "date_column", "previous_period", "current_period", "previous_value", "current_value", "absolute_change", "percentage_change", "limitations")}
    compact["contributors"] = result.get("contributing_dimensions", [])[:max_contributors]
    return json.dumps(compact, default=str, separators=(",", ":"))


def investigate_change(question: str, df: pd.DataFrame, profile: dict | None = None, metric: str | None = None, date_column: str | None = None, frequency: str = "quarter") -> dict:
    result = {"status": "clarification_required", "metric": None, "date_column": None, "previous_period": None, "current_period": None, "previous_value": None, "current_value": None, "absolute_change": None, "percentage_change": None, "contributing_dimensions": [], "supporting_statistics": {}, "limitations": []}
    numeric = df.select_dtypes(include="number").columns.tolist()
    question_lower = question.casefold()
    metric = metric or next((column for column in numeric if column.casefold() in question_lower), None)
    if metric is None:
        result["limitations"].append("No unambiguous numeric metric was identified.")
        return result
    date_candidates = [column for column in df.columns if pd.api.types.is_datetime64_any_dtype(df[column])]
    if date_column is None:
        date_column = date_candidates[0] if len(date_candidates) == 1 else None
    if not date_column:
        result["limitations"].append("No unambiguous datetime column was identified.")
        return result
    try: dates, warnings = datetime_series(df, date_column)
    except ValueError as exc:
        result["limitations"].append(str(exc)); return result
    work = df.assign(__date__=dates).dropna(subset=["__date__"])
    periods = work.__date__.dt.to_period({"month": "M", "quarter": "Q", "year": "Y"}[frequency]).astype(str)
    totals = work.assign(__period__=periods).groupby("__period__")[metric].sum().sort_index()
    if len(totals) < 2:
        result["limitations"].append("At least two periods are required for an investigation."); return result
    previous_period, current_period = totals.index[-2], totals.index[-1]
    previous, current = float(totals.iloc[-2]), float(totals.iloc[-1])
    pct, _ = growth(current, previous)
    result.update({"status": "success", "metric": metric, "date_column": date_column, "previous_period": previous_period, "current_period": current_period, "previous_value": previous, "current_value": current, "absolute_change": current - previous, "percentage_change": pct, "limitations": warnings})
    dimensions = [column for column in df.select_dtypes(include=["object", "category", "string", "bool"]).columns if column != date_column]
    for dimension in dimensions:
        contributors, _ = contribution_analysis(work, metric, dimension, date_column, previous_period, current_period)
        for row in contributors.head(5).to_dict("records"):
            row["dimension"] = dimension
            result["contributing_dimensions"].append(row)
    result["contributing_dimensions"].sort(key=lambda item: abs(item["absolute_change"]), reverse=True)
    result["supporting_statistics"] = {"period_count": len(totals), "candidate_dimensions": dimensions}
    return result


def build_investigation_prompt(evidence: dict) -> str:
    return "Use only this calculated evidence. Do not calculate numbers, invent causes, or claim causation. Distinguish evidence from possible explanations and state limitations.\nEvidence: " + serialize_investigation_evidence(evidence)
