import pandas as pd
from src.analytics import contribution_analysis
from src.investigation import investigate_change, serialize_investigation_evidence, build_investigation_prompt


def sample():
    return pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-02", "2024-04-01", "2024-04-02"]), "region": ["A", "B", "A", "B"], "value": [100, 50, 40, 70]})

def test_contribution_and_investigation():
    df = sample()
    contribution, _ = contribution_analysis(df, "value", "region", "date", "2024Q1", "2024Q2")
    assert contribution.iloc[0]["region"] == "A"
    result = investigate_change("Why did value decrease?", df)
    assert result["status"] == "success" and result["absolute_change"] == -40
    payload = serialize_investigation_evidence(result)
    assert "contributors" in payload and "Do not calculate" in build_investigation_prompt(result)

def test_investigation_clarifies_ambiguous_or_missing_dates():
    assert investigate_change("why did it change", sample())["status"] == "clarification_required"
    assert investigate_change("why did value change", sample().assign(date="x"))["status"] == "clarification_required"
