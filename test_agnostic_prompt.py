"""Offline contract test for domain-agnostic anomaly explanations."""
from types import SimpleNamespace
import re

import pandas as pd
import pytest

from src.anomaly import build_anomaly_evidence, detect_anomalies
from src.llm import explain_anomalies

UNSUPPORTED_DOMAIN_WORDS = (
    "order", "transaction", "customer", "employee", "financial",
    "operational", "sales", "performance", "team",
)


def _assert_domain_agnostic(explanation: str) -> None:
    for word in UNSUPPORTED_DOMAIN_WORDS:
        assert not re.search(rf"\b{re.escape(word)}s?\b", explanation, re.IGNORECASE), (
            f"Explanation introduced unsupported domain term: {word}"
        )


def test_prompt_safety(monkeypatch):
    """Retain the guardrails and validate output without a Groq request."""
    captured = {}
    safe_response = (
        "### Summary\nThe model flagged unusual records.\n\n"
        "### Key Findings\n- Values differ from the typical pattern.\n\n"
        "### What This Could Mean\nThe available data does not establish the cause.\n\n"
        "### What to Investigate\n- Review the affected records."
    )

    def fake_completion(**kwargs):
        captured.update(kwargs)
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=safe_response))])

    monkeypatch.setattr("src.llm.get_client", lambda: object())
    monkeypatch.setattr("src.llm.call_groq_with_retry", fake_completion)
    df = pd.read_csv("data/sample_sales.csv")
    explanation = explain_anomalies(build_anomaly_evidence(df, detect_anomalies(df)))

    prompt = captured["contents"]
    assert "NEVER call them \"orders\", \"transactions\", \"customers\", \"employees\"" in prompt
    assert "NEVER categorize columns as financial, operational, sales, performance" in prompt
    assert "The available data does not establish the cause." in prompt
    _assert_domain_agnostic(explanation)


@pytest.mark.parametrize("term", UNSUPPORTED_DOMAIN_WORDS)
def test_domain_assumption_detector_rejects_unsupported_language(term):
    """Ensure the domain-language assertion cannot be silently weakened."""
    with pytest.raises(AssertionError):
        _assert_domain_agnostic(f"This is an unsupported {term} assumption.")
