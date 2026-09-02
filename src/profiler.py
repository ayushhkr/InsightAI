"""
profiler.py
Responsible for analyzing and profiling uploaded datasets (e.g., getting missing values, data types, summary statistics).
"""
import pandas as pd

def profile_dataset(df: pd.DataFrame) -> dict:
    """
    Analyzes a given Pandas DataFrame and returns a dictionary of metrics.
    Attempts to convert evident date columns to datetime.
    """
    # Attempt to convert categorical/object columns strings that look like dates to datetime
    for col in df.columns:
        if df[col].dtype == 'object':
            try:
                # We attempt a light conversion; if it fails, we keep as object
                df[col] = pd.to_datetime(df[col], infer_datetime_format=True)
            except (ValueError, TypeError):
                pass
                
    num_rows, num_cols = df.shape
    columns = list(df.columns)
    data_types = df.dtypes.astype(str).to_dict()
    missing_counts = df.isna().sum().to_dict()
    duplicate_rows = df.duplicated().sum()
    
    numerical_cols = df.select_dtypes(include=['number']).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
    datetime_cols = df.select_dtypes(include=['datetime']).columns.tolist()
    
    # Descriptive statistics
    if len(numerical_cols) > 0:
        desc_stats = df[numerical_cols].describe().to_dict()
    else:
        desc_stats = {}
        
    return {
        "num_rows": num_rows,
        "num_cols": num_cols,
        "columns": columns,
        "data_types": data_types,
        "missing_counts": missing_counts,
        "duplicate_rows": int(duplicate_rows),
        "numerical_cols": numerical_cols,
        "categorical_cols": categorical_cols,
        "datetime_cols": datetime_cols,
        "descriptive_statistics": desc_stats
    }
