import pandas as pd
import numpy as np
import json
from src.anomaly import detect_anomalies

def main():
    print("=== TEST 1: Basic Outliers and NaNs ===")
    df1 = pd.DataFrame({
        "outlier_col": [10, 12, 11, 10, 13, 1000, 11, 9, 12, -500],
        "normal_col": [50, 52, 51, 49, 50, 53, 51, 52, 50, 51],
        "cat_col": ["A", "B", "A", "B", "A", "B", "A", "B", "A", "B"],
        "nan_col": [1, 2, np.nan, 4, 5, np.nan, 7, 8, 9, 10]
    })
    res1 = detect_anomalies(df1)
    print(json.dumps(res1, indent=2))

    print("\n=== TEST 2: Empty DataFrame ===")
    df2 = pd.DataFrame()
    res2 = detect_anomalies(df2)
    print(json.dumps(res2, indent=2))

    print("\n=== TEST 3: Only Categorical ===")
    df3 = pd.DataFrame({"cat": ["A", "B", "C"]})
    res3 = detect_anomalies(df3)
    print(json.dumps(res3, indent=2))

    print("\n=== TEST 4: Constant Numerical Column ===")
    df4 = pd.DataFrame({"const": [5, 5, 5, 5, 5]})
    res4 = detect_anomalies(df4)
    print(json.dumps(res4, indent=2))
    
    print("\n=== TEST 5: Constant with one anomaly ===")
    df5 = pd.DataFrame({"const": [5, 5, 5, 5, 100]})
    res5 = detect_anomalies(df5)
    print(json.dumps(res5, indent=2))

if __name__ == "__main__":
    main()
