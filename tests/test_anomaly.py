import pandas as pd
import pytest
from src.anomaly import detect_anomalies, build_anomaly_evidence

def test_anomaly_empty_df():
    df = pd.DataFrame()
    res = detect_anomalies(df)
    assert res.get("total_anomalies") == 0

def test_anomaly_only_text():
    df = pd.DataFrame({
        "category": ["A", "B", "C", "A", "B"]
    })
    res = detect_anomalies(df)
    assert res.get("total_anomalies", 0) == 0

def test_anomaly_with_nan():
    df = pd.DataFrame({
        "values": [10, 12, 10, None, 100, 11]
    })
    res = detect_anomalies(df)
    assert res["total_anomalies"] == 1
    assert "values" in res["columns"]
    
    ev = build_anomaly_evidence(df, res)
    assert isinstance(ev, dict)
    assert ev["summary"]["total_anomalies_found"] == 1

def test_anomaly_constant_value():
    df = pd.DataFrame({
        "values": [10, 10, 10, 10, 10]
    })
    res = detect_anomalies(df)
    assert res.get("total_anomalies") == 0

def test_isolation_forest_returns_consistent_structure():
    df = pd.DataFrame({"value": [1, 1, 1, 1, 100]})
    result = detect_anomalies(df, contamination=.2)
    assert result["method"] == "isolation_forest"
    assert {"total_anomalies", "anomalous_indices", "anomaly_scores", "columns", "metadata"}.issubset(result)
    assert result["total_anomalies"] >= 1

def test_anomaly_configuration_and_severity():
    df = pd.DataFrame({"value": [1, 1, 1, 1, 100]})
    with pytest.raises(ValueError): detect_anomalies(df, contamination=0)
    result = detect_anomalies(df, contamination=.2)
    assert result["severity"]
    assert {"mean", "q1", "q3", "iqr"}.issubset(result["columns"]["value"])
