import pandas as pd
import pytest
from src.analyzer import execute_analysis_plan
from src.time_series import TimeSeriesError, datetime_series, growth


@pytest.fixture
def data():
    return pd.DataFrame({"region": ["A", "A", "B", "B", "C", "C"], "value": [10, 20, 30, 10, -5, 5], "date": ["2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01", "2024-01-01", "2024-02-01"]})

@pytest.mark.parametrize("operation", ["top_n", "bottom_n", "ranking", "percentage_contribution"])
def test_grouped_advanced_operations(data, operation):
    result, _ = execute_analysis_plan(data, {"operation": operation, "group_column": "region", "metric": "value", "aggregation": "sum", "limit": 2})
    assert len(result) == 2
    if operation == "ranking": assert "rank" in result
    if operation == "percentage_contribution": assert "percentage_contribution" in result

@pytest.mark.parametrize("operation", ["growth_rate", "time_trend", "rolling_average"])
def test_time_advanced_operations(data, operation):
    result, _ = execute_analysis_plan(data, {"operation": operation, "date_column": "date", "metric": "value", "aggregation": "sum", "frequency": "month", "window": 2})
    assert not result.empty

def test_period_comparison_and_distribution(data):
    comparison, _ = execute_analysis_plan(data, {"operation": "period_comparison", "date_column": "date", "metric": "value", "aggregation": "sum", "frequency": "month", "comparison_periods": ["2024-01", "2024-02"]})
    distribution, _ = execute_analysis_plan(data, {"operation": "distribution", "metric": "value", "bins": 3})
    assert comparison.iloc[0].absolute_change == 0
    assert distribution["count"].sum() == len(data)

def test_time_safety_invalid_missing_duplicates_and_zero_previous():
    with pytest.raises(TimeSeriesError): datetime_series(pd.DataFrame({"date": ["not-date", "also-bad"]}), "date")
    dates, warnings = datetime_series(pd.DataFrame({"date": ["2024-01-01", None, "2024-01-01"]}), "date")
    assert dates.notna().sum() == 2 and warnings
    assert growth(10, 0) == (None, "unavailable_zero_or_missing_previous")
    assert growth(-5, -10)[0] == 50
