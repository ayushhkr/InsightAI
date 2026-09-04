"""
anomaly.py
Contains Isolation Forest anomaly detection logic for numerical dataset features.
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

def detect_anomalies(df: pd.DataFrame) -> dict:
    """
    Identifies anomalous rows with Isolation Forest using all usable numerical columns.
    Missing and non-finite values are median-imputed before fitting the model.
    """
    result = {
        "total_anomalies": 0,
        "anomalous_indices": [],
        "columns": {},
        "anomaly_scores": [],
        "method": "Isolation Forest",
        "contamination": 0.05,
    }

    if df is None or df.empty:
        return result

    numeric_data = df.select_dtypes(include=["number"])
    numeric_data = numeric_data.loc[:, [
        column for column in numeric_data.columns if str(column).casefold() != "order id"
    ]]
    numeric_data = numeric_data.replace([np.inf, -np.inf], np.nan)
    medians = numeric_data.median()
    usable_columns = medians.dropna().index.tolist()
    if not usable_columns:
        return result

    features = numeric_data[usable_columns].fillna(medians[usable_columns])
    model = IsolationForest(contamination=0.05, random_state=42)
    predictions = model.fit_predict(features)
    scores = model.decision_function(features)
    anomaly_mask = predictions == -1

    anomalous_indices = df.index[anomaly_mask].tolist()
    result["anomalous_indices"] = anomalous_indices
    result["total_anomalies"] = len(anomalous_indices)
    result["columns"] = {
        column: {
            "anomaly_count": int(anomaly_mask.sum()),
            "median_imputation_value": float(medians[column]),
            "missing_values_imputed": int(numeric_data[column].isna().sum()),
        }
        for column in usable_columns
    }
    result["anomaly_scores"] = [
        {
            "index": index.item() if isinstance(index, np.generic) else index,
            "score": float(score),
            "prediction": int(prediction),
        }
        for index, score, prediction in zip(df.index, scores, predictions)
    ]

    return result

def build_anomaly_evidence(df: pd.DataFrame, anomaly_result: dict) -> dict:
    """
    Constructs a JSON-serializable evidence package for the detected anomalous rows.
    """
    evidence = {
        "summary": {
            "method": anomaly_result.get("method", "Isolation Forest"),
            "contamination": anomaly_result.get("contamination", 0.05),
            "total_anomalies_found": anomaly_result.get("total_anomalies", 0),
            "affected_columns": [],
        },
        "feature_details": {},
        "anomalous_data_samples": [],
    }

    total = anomaly_result.get("total_anomalies", 0)
    if total == 0 or df is None or df.empty:
        return evidence

    columns = anomaly_result.get("columns", {})
    evidence["summary"]["affected_columns"] = list(columns)
    evidence["feature_details"] = columns

    score_by_index = {
        score["index"]: score["score"]
        for score in anomaly_result.get("anomaly_scores", [])
        if score["prediction"] == -1
    }
    subset = df.loc[anomaly_result.get("anomalous_indices", [])].copy()
    subset = subset.where(pd.notnull(subset), None)
    subset["__anomaly_score__"] = [score_by_index.get(index) for index in subset.index]
    subset["__original_index__"] = subset.index
    evidence["anomalous_data_samples"] = subset.to_dict(orient="records")

    return evidence
