"""
analyzer.py
Contains generic data-analysis functions for processing the Pandas dataframes based on user requests.
"""
import pandas as pd

def execute_analysis_plan(df: pd.DataFrame, plan: dict) -> tuple[pd.DataFrame, dict]:
    """
    Executes the JSON analysis plan safely without using eval or exec.
    Returns (result_dataframe, execution_metadata).
    """
    operation = plan.get("operation")
    group_column = plan.get("group_column")
    metric = plan.get("metric")
    aggregation = plan.get("aggregation")
    sort_dir = plan.get("sort")
    top_n = plan.get("top_n")
    
    if not isinstance(group_column, str): group_column = None
    if not isinstance(metric, str): metric = None
    
    df_exec = df.copy()
    
    # Auto-calculate common derived columns if requested but missing
    if metric == "revenue" and "revenue" not in df_exec.columns:
        if "price" in df_exec.columns and "quantity" in df_exec.columns:
            df_exec["revenue"] = df_exec["price"] * df_exec["quantity"]
            
    if group_column == "revenue" and "revenue" not in df_exec.columns:
        if "price" in df_exec.columns and "quantity" in df_exec.columns:
            df_exec["revenue"] = df_exec["price"] * df_exec["quantity"]
    
    # Validate columns
    columns_to_check = [col for col in [group_column, metric] if col]
    for col in columns_to_check:
        if col not in df_exec.columns:
            raise ValueError(f"Column '{col}' referenced in plan is not present in the dataset.")
            
    result_df = pd.DataFrame()
    metadata = {"status": "success", "operation": operation}
    
    try:
        if operation == "groupby":
            if not group_column or not metric or not aggregation:
                raise ValueError("Groupby requires group_column, metric, and aggregation.")
            if aggregation == "sum":
                result_df = df_exec.groupby(group_column)[metric].sum().reset_index()
            elif aggregation == "mean":
                result_df = df_exec.groupby(group_column)[metric].mean().reset_index()
            elif aggregation == "count":
                result_df = df_exec.groupby(group_column)[metric].count().reset_index()
            elif aggregation == "min":
                result_df = df_exec.groupby(group_column)[metric].min().reset_index()
            elif aggregation == "max":
                result_df = df_exec.groupby(group_column)[metric].max().reset_index()
            else:
                raise ValueError(f"Unsupported aggregation: {aggregation}")
                
        elif operation == "aggregate":
            if not metric or not aggregation:
                raise ValueError("Aggregate requires metric and aggregation.")
            val = None
            if aggregation == "sum": val = df_exec[metric].sum()
            elif aggregation == "mean": val = df_exec[metric].mean()
            elif aggregation == "count": val = df_exec[metric].count()
            elif aggregation == "min": val = df_exec[metric].min()
            elif aggregation == "max": val = df_exec[metric].max()
            else:
                raise ValueError(f"Unsupported aggregation: {aggregation}")
            result_df = pd.DataFrame({metric: [val], "aggregation": [aggregation]})
            
        elif operation == "filter":
            if metric:
                result_df = df_exec.sort_values(by=metric, ascending=False).head(50)
            else:
                result_df = df_exec.head(50)
            metadata["warning"] = "Safe filtering is limited without eval; showing fallback sample records."
            
        elif operation == "correlation":
            num_cols = df_exec.select_dtypes(include=['number']).columns
            if len(num_cols) < 2:
                raise ValueError("Need at least two numerical columns for correlation.")
            result_df = df_exec[num_cols].corr().reset_index()
            
        elif operation == "trend":
            if not group_column or not metric or not aggregation:
                raise ValueError("Trend requires group_column, metric, and aggregation.")
            if not pd.api.types.is_datetime64_any_dtype(df_exec[group_column]):
                try:
                    df_exec[group_column] = pd.to_datetime(df_exec[group_column])
                except Exception:
                    raise ValueError(f"'{group_column}' cannot be converted to datetime for trend analysis.")
            
            temp_df = df_exec.copy()
            temp_df["__trend_date__"] = temp_df[group_column].dt.date
            
            if aggregation == "sum":
                result_df = temp_df.groupby("__trend_date__")[metric].sum().reset_index()
            elif aggregation == "mean":
                result_df = temp_df.groupby("__trend_date__")[metric].mean().reset_index()
            elif aggregation == "count":
                result_df = temp_df.groupby("__trend_date__")[metric].count().reset_index()
            elif aggregation == "min":
                result_df = temp_df.groupby("__trend_date__")[metric].min().reset_index()
            elif aggregation == "max":
                result_df = temp_df.groupby("__trend_date__")[metric].max().reset_index()
            else:
                result_df = temp_df.groupby("__trend_date__")[metric].sum().reset_index()
                
            result_df.rename(columns={"__trend_date__": group_column}, inplace=True)
            result_df[group_column] = pd.to_datetime(result_df[group_column])
            result_df = result_df.sort_values(group_column)
            
        elif operation == "describe":
            if metric:
                result_df = df_exec[[metric]].describe().reset_index()
            else:
                num_cols = df_exec.select_dtypes(include=['number']).columns
                if len(num_cols) > 0:
                    result_df = df_exec[num_cols].describe().reset_index()
                else:
                    result_df = df_exec.head(10)
        else:
            raise ValueError(f"Unsupported operation: {operation}")
            
        # Apply sorting and top_n safely
        if sort_dir in ["ascending", "descending"] and metric and metric in result_df.columns:
            is_ascending = (sort_dir == "ascending")
            result_df = result_df.sort_values(by=metric, ascending=is_ascending)
            
        if isinstance(top_n, int) and top_n > 0:
            result_df = result_df.head(top_n)
            
    except Exception as e:
        raise ValueError(f"Execution Error: {str(e)}")
        
    return result_df, metadata
