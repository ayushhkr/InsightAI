"""
visualizer.py
Contains Plotly visualization functions for rendering dynamic charts.
"""
import plotly.express as px
import pandas as pd

def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str):
    return px.bar(df, x=x_col, y=y_col, title=f"Bar Chart: {y_col} by {x_col}")

def create_line_chart(df: pd.DataFrame, x_col: str, y_col: str):
    return px.line(df, x=x_col, y=y_col, title=f"Line Chart: {y_col} over {x_col}")

def create_scatter_chart(df: pd.DataFrame, x_col: str, y_col: str):
    return px.scatter(df, x=x_col, y=y_col, title=f"Scatter Chart: {y_col} vs {x_col}")

def create_histogram(df: pd.DataFrame, x_col: str):
    return px.histogram(df, x=x_col, title=f"Histogram: Distribution of {x_col}")

def create_chart(result_df: pd.DataFrame, plan: dict):
    """
    Creates a Plotly chart dynamically based on the analysis plan provided by Gemini.
    """
    chart_type = plan.get("chart", "none").lower()
    title = plan.get("title", "Analysis Result")
    
    if chart_type == "none" or result_df.empty:
        return None
        
    cols = list(result_df.columns)
    
    if len(cols) == 0:
        return None
        
    x_col = plan.get("group_column")
    y_col = plan.get("metric")
    
    if not x_col and len(cols) > 0:
        x_col = cols[0]
    if not y_col and len(cols) > 1:
        y_col = cols[1]
        
    if x_col not in result_df.columns:
        x_col = cols[0]
    if y_col not in result_df.columns and len(cols) > 1:
        y_col = cols[1]

    try:
        if chart_type == "bar":
            return px.bar(result_df, x=x_col, y=y_col, title=title)
        elif chart_type == "line":
            return px.line(result_df, x=x_col, y=y_col, title=title)
        elif chart_type == "scatter":
            return px.scatter(result_df, x=x_col, y=y_col, title=title)
        elif chart_type == "histogram":
            return px.histogram(result_df, x=x_col, title=title)
        else:
            return None
    except Exception as e:
        print(f"Error creating chart: {e}")
        return None
