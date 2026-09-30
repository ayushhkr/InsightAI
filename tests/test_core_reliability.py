import pandas as pd
import pytest

from src.analyzer import execute_analysis_plan
from src.filters import FilterValidationError, apply_filter
from src.llm import LLMServiceError, LLMTransientError, call_groq_with_retry
from src.llm_cache import LLMResponseCache
from src.plan_validation import PlanValidationError, validate_analysis_plan
from src.result_validation import ResultValidationError, validate_result


@pytest.fixture
def frame():
    return pd.DataFrame({"country": ["India", "USA", "India", None], "name": ["Alpha", "Beta", "Alpine", None], "value": [1, 2, 3, 4]})

@pytest.mark.parametrize("operator,value,expected", [
    ("==", "India", 2), ("!=", "India", 2), (">", 2, 2), ("<", 3, 2),
    (">=", 3, 2), ("<=", 2, 2), ("contains", "lp", 2), ("startswith", "Al", 2),
    ("endswith", "a", 2), ("in", ["India"], 2), ("not_in", ["India"], 2),
    ("is_null", None, 1), ("is_not_null", None, 3), ("between", [2, 3], 2),
])
def test_each_safe_filter_operator(frame, operator, value, expected):
    column = "value" if operator in {">", "<", ">=", "<=", "between"} else "name" if operator in {"contains", "startswith", "endswith"} else "country"
    spec = {"conditions": [{"column": column, "operator": operator, "value": value}], "logic": "AND"}
    assert len(apply_filter(frame, spec)) == expected

def test_and_or_filters(frame):
    conditions = [{"column": "country", "operator": "==", "value": "India"}, {"column": "value", "operator": ">", "value": 1}]
    assert len(apply_filter(frame, {"conditions": conditions, "logic": "AND"})) == 1
    assert len(apply_filter(frame, {"conditions": conditions, "logic": "OR"})) == 4

def test_invalid_filters_are_controlled(frame):
    for spec in [{"conditions": [{"column": "bad", "operator": "==", "value": 1}]}, {"conditions": [{"column": "value", "operator": "bad", "value": 1}]}, {"conditions": [{"column": "value", "operator": "between", "value": [1]}]}]:
        with pytest.raises(FilterValidationError): apply_filter(frame, spec)

def test_malformed_plan_and_legitimate_empty_filter(frame):
    with pytest.raises(PlanValidationError): validate_analysis_plan({"operation": "groupby", "metric": "value"}, frame)
    result, metadata = execute_analysis_plan(frame, {"operation": "filter", "filter": {"conditions": [{"column": "country", "operator": "==", "value": "Japan"}], "logic": "AND"}})
    assert result.empty and "warning" in metadata

def test_result_validation_checks_contract(frame):
    plan = validate_analysis_plan({"operation": "groupby", "group_column": "country", "metric": "value", "aggregation": "sum", "sort": "descending", "limit": 1}, frame)
    result, _ = execute_analysis_plan(frame, plan)
    assert result.iloc[0]["value"] == 4
    with pytest.raises(ResultValidationError): validate_result(pd.DataFrame({"country": ["India"]}), plan, 3)

def test_cache_hit_miss_and_secret_exclusion():
    cache = LLMResponseCache()
    metadata = {"columns": ["value"]}
    key = cache.key(metadata, " Total   Value ", [], "model")
    cache.set(key, {"operation": "aggregate"})
    assert cache.get(cache.key(metadata, "total value", [], "model")) == {"operation": "aggregate"}
    assert cache.get(cache.key(metadata, "other", [], "model")) is None
    assert cache.key(metadata, "api_key=private", [], "model") is None

def test_llm_errors_are_generic_and_transient():
    class Failure(Exception):
        status_code = 500
    class Client:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs): raise Failure()
    with pytest.raises(LLMServiceError): call_groq_with_retry(Client(), "model", "x", 0, max_retries=0)
    class Busy(Failure): status_code = 503
    class BusyClient:
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs): raise Busy()
    with pytest.raises(LLMTransientError): call_groq_with_retry(BusyClient(), "model", "x", 0, max_retries=0)
