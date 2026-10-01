"""Small realistic benchmark with structured expected behavior, not LLM wording."""
from __future__ import annotations

from dataclasses import dataclass
import pandas as pd


def controlled_dataset() -> pd.DataFrame:
    return pd.DataFrame({
        "Date": pd.to_datetime(["2024-01-05", "2024-01-20", "2024-02-10", "2024-02-20", "2024-03-15", "2024-03-25"]),
        "Region": ["North", "South", "North", "West", "South", "West"],
        "Product": ["A", "A", "B", "B", "A", "C"],
        "Total Revenue": [100.0, 80.0, 140.0, 60.0, 120.0, 40.0],
        "Quantity": [10, 8, 14, 6, 12, 4],
        "Cost": [70.0, 60.0, 90.0, 50.0, 85.0, 30.0],
    })


@dataclass(frozen=True)
class EvaluationCase:
    question: str
    category: str
    expected_plan: dict


def benchmark_cases() -> list[EvaluationCase]:
    return [
        EvaluationCase("What is total revenue?", "aggregation", {"operation": "aggregate", "metric": "Total Revenue", "aggregation": "sum"}),
        EvaluationCase("What revenue did North generate?", "filtering", {"operation": "aggregate", "metric": "Total Revenue", "aggregation": "sum", "filter": {"conditions": [{"column": "Region", "operator": "==", "value": "North"}], "logic": "AND"}}),
        EvaluationCase("Revenue by region", "grouping", {"operation": "groupby", "group_column": "Region", "metric": "Total Revenue", "aggregation": "sum"}),
        EvaluationCase("Top two regions by revenue", "top_n", {"operation": "top_n", "group_column": "Region", "metric": "Total Revenue", "aggregation": "sum", "limit": 2}),
        EvaluationCase("Bottom two products by revenue", "bottom_n", {"operation": "bottom_n", "group_column": "Product", "metric": "Total Revenue", "aggregation": "sum", "limit": 2}),
        EvaluationCase("Rank regions by revenue", "ranking", {"operation": "ranking", "group_column": "Region", "metric": "Total Revenue", "aggregation": "sum"}),
        EvaluationCase("Monthly revenue growth", "growth", {"operation": "growth_rate", "date_column": "Date", "metric": "Total Revenue", "aggregation": "sum", "frequency": "month"}),
        EvaluationCase("Compare January and February revenue", "period_comparison", {"operation": "period_comparison", "date_column": "Date", "metric": "Total Revenue", "aggregation": "sum", "frequency": "month", "comparison_periods": ["2024-01", "2024-02"]}),
        EvaluationCase("Show the revenue time trend", "time_trend", {"operation": "time_trend", "date_column": "Date", "metric": "Total Revenue", "aggregation": "sum", "frequency": "month"}),
        EvaluationCase("How much does each region contribute?", "contribution_analysis", {"operation": "percentage_contribution", "group_column": "Region", "metric": "Total Revenue", "aggregation": "sum"}),
        EvaluationCase("Is revenue correlated with quantity and cost?", "correlation", {"operation": "correlation", "correlation_columns": ["Total Revenue", "Quantity", "Cost"]}),
        # These are intentionally represented as safe descriptive plans: anomaly
        # detection and investigation are deterministic modules, not LLM SQL plans.
        EvaluationCase("Are there anomalous numerical records?", "anomaly_analysis", {"operation": "describe", "metric": "Total Revenue"}),
        EvaluationCase("What should I investigate about unusual results?", "root_cause_investigation", {"operation": "describe", "metric": "Total Revenue"}),
        EvaluationCase("Analyze this", "ambiguous", {"operation": "describe", "metric": None}),
    ]
