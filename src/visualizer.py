"""
visualizer.py
Contains Plotly visualization functions for rendering dynamic charts.
"""
import plotly.express as px
import pandas as pd

def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str = None):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
        
    plot_df = df.copy()
    if pd.api.types.is_numeric_dtype(plot_df[y_col]):
        plot_df = plot_df.sort_values(by=y_col, ascending=False)
        
    text_auto = True if len(plot_df) <= 15 else False
    if not title:
        title = f"Bar Chart: {y_col} by {x_col}"
        
    fig = px.bar(plot_df, x=x_col, y=y_col, title=title, text_auto=text_auto)
    
    if plot_df[x_col].astype(str).str.len().max() > 15:
        fig.update_layout(xaxis_tickangle=-45)
        
    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, title_font_size=16)
    return fig

def create_line_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str = None):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
        
    plot_df = df.copy()
    if pd.api.types.is_datetime64_any_dtype(plot_df[x_col]):
        plot_df = plot_df.sort_values(by=x_col)
    else:
        try:
            temp_dt = pd.to_datetime(plot_df[x_col])
            plot_df = plot_df.assign(__temp_date_sort=temp_dt)
            plot_df = plot_df.sort_values(by="__temp_date_sort").drop(columns=["__temp_date_sort"])
        except Exception:
            pass
            
    if not title:
        title = f"Line Chart: {y_col} over {x_col}"
        
    fig = px.line(plot_df, x=x_col, y=y_col, title=title, markers=True)
    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, title_font_size=16)
    return fig

def create_scatter_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str = None):
    if df.empty or x_col not in df.columns or y_col not in df.columns:
        return None
        
    plot_df = df.dropna(subset=[x_col, y_col])
    if plot_df.empty:
        return None
        
    hover_data = [c for c in plot_df.columns if c not in [x_col, y_col]][:3]
    
    if not title:
        title = f"Scatter Chart: {y_col} vs {x_col}"
        
    fig = px.scatter(plot_df, x=x_col, y=y_col, title=title, hover_data=hover_data)
    fig.update_layout(xaxis_title=x_col, yaxis_title=y_col, title_font_size=16)
    return fig

def create_histogram(df: pd.DataFrame, x_col: str, title: str = None):
    if df.empty or x_col not in df.columns:
        return None
        
    plot_df = df.dropna(subset=[x_col])
    if plot_df.empty:
        return None
        
    num_unique = plot_df[x_col].nunique()
    nbins = max(10, min(50, num_unique // 2))
    
    if not title:
        title = f"Histogram: Distribution of {x_col}"
        
    fig = px.histogram(plot_df, x=x_col, title=title, nbins=nbins)
    fig.update_layout(xaxis_title=x_col, yaxis_title="Count", title_font_size=16)
    return fig

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
            return create_bar_chart(result_df, x_col, y_col, title)
        elif chart_type == "line":
            return create_line_chart(result_df, x_col, y_col, title)
        elif chart_type == "scatter":
            return create_scatter_chart(result_df, x_col, y_col, title)
        elif chart_type == "histogram":
            return create_histogram(result_df, x_col, title)
        else:
            return None
    except Exception as e:
        print(f"Error creating chart: {e}")
        return None
