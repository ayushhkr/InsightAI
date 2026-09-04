import pandas as pd
from src.analyzer import execute_analysis_plan

def test_analyzer_empty():
    df = pd.DataFrame()
    plan = {"action": "groupby", "group_column": "x", "metric": "y", "aggregation": "sum"}
    try:
        res, meta = execute_analysis_plan(df, plan)
        assert res.empty
    except pd.errors.EmptyDataError:
        pass  # Also valid handling
    except ValueError:
        pass  # Known handling in app.py

def test_analyzer_groupby():
    df = pd.DataFrame({
        "A": ["foo", "bar", "foo"],
        "B": [10, 20, 30]
    })
    plan = {"action": "groupby", "group_column": "A", "metric": "B", "aggregation": "sum"}
    res, meta = execute_analysis_plan(df, plan)
    assert len(res) == 2
