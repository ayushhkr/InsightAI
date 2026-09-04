import json
from src.llm import explain_anomalies

def main():
    evidence = {
      "total_anomalies": 2,
      "columns": {
        "value": {
          "anomaly_count": 2,
          "lower_bound": 7,
          "upper_bound": 15
        }
      },
      "anomalous_rows": [
        {"index": 1, "value": 100},
        {"index": 2, "value": 120}
      ]
    }
    try:
        res = explain_anomalies(evidence)
        print("=== RESULT ===")
        print(res)
    except Exception as e:
        print("=== ERROR ===")
        print(type(e).__name__)
        print(str(e))
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
