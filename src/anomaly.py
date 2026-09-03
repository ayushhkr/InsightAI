"""
anomaly.py
Contains anomaly detection logic for the dataset.
Uses the Interquartile Range (IQR) method for identifying numerical outliers.
"""
import pandas as pd
import numpy as np

def detect_anomalies(df: pd.DataFrame) -> dict:
    """
    Dynamically identifies numerical anomalies (outliers) in a dataframe using the IQR method.
    Returns a dictionary of statistics and anomalous row indices.
    """
    result = {
        "total_anomalies": 0,
        "anomalous_indices": [],
        "columns": {}
    }
    
    if df is None or df.empty:
        return result
        
    num_cols = df.select_dtypes(include=['number']).columns.tolist()
    
    if not num_cols:
        return result
        
    all_anomalous_indices = set()
    
    for col in num_cols:
        col_data = df[col].dropna()
        if col_data.empty:
            continue
            
        q1 = col_data.quantile(0.25)
        q3 = col_data.quantile(0.75)
        iqr = q3 - q1
        
        # Calculate bounds
        lower_bound = q1 - 1.5 * iqr
        upper_bound = q3 + 1.5 * iqr
        
        # Identify outliers logically outside [Q1 - 1.5*IQR, Q3 + 1.5*IQR]
        outliers = col_data[(col_data < lower_bound) | (col_data > upper_bound)]
        outlier_indices = outliers.index.tolist()
        
        # Convert NumPy types strictly to base Python types for JSON compatibility
        col_result = {
            "q1": float(q1) if not pd.isna(q1) else None,
            "q3": float(q3) if not pd.isna(q3) else None,
            "iqr": float(iqr) if not pd.isna(iqr) else None,
            "lower_bound": float(lower_bound) if not pd.isna(lower_bound) else None,
            "upper_bound": float(upper_bound) if not pd.isna(upper_bound) else None,
            "anomaly_count": len(outlier_indices),
            "anomalous_indices": outlier_indices
        }
        
        result["columns"][col] = col_result
        all_anomalous_indices.update(outlier_indices)
        
    result["anomalous_indices"] = sorted(list(all_anomalous_indices))
    result["total_anomalies"] = len(result["anomalous_indices"])
    
    return result
