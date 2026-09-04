import pandas as pd
import json
from src.anomaly import detect_anomalies, build_anomaly_evidence
from src.llm import explain_anomalies

def test_prompt_safety():
    df = pd.read_csv("data/sample_sales.csv")
    anomaly_result = detect_anomalies(df)
    evidence = build_anomaly_evidence(df, anomaly_result)
    
    explanation = explain_anomalies(evidence)
    print("=== EXPLANATION ===")
    print(explanation)
    print("===================")
    
    bad_words = ["order", "transaction", "customer", "employee", "financial", "operational", "sales", "performance", "team"]
    found_bad = False
    for word in bad_words:
        if word.lower() in explanation.lower():
            print(f"FAILED: Found unsupported term '{word}' in explanation!")
            found_bad = True
            
    if not found_bad:
        print("SUCCESS: Prompt constraints held. No domain assumptions detected.")

if __name__ == "__main__":
    test_prompt_safety()
