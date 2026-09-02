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
