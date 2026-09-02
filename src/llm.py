"""
llm.py
Will contain the Gemini API functionality (using google-genai) to answer natural language questions.
"""
import os
import json
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def get_client():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY is not set in environment variables.")
    return genai.Client(api_key=api_key)

def generate_analysis_plan(question: str, dataframe_metadata: dict) -> dict:
    client = get_client()
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    prompt = f"""
    You are an AI Data Analyst. Your job is to convert a user question into a structured JSON analysis plan.
    
    Dataset Metadata:
    Columns: {dataframe_metadata['columns']}
    Data Types: {dataframe_metadata['data_types']}
    Numerical Columns: {dataframe_metadata['numerical_cols']}
    Categorical Columns: {dataframe_metadata['categorical_cols']}
    Datetime Columns: {dataframe_metadata['datetime_cols']}
    
    User Question: "{question}"
    
    Return ONLY a JSON object with the following schema:
    {{
      "operation": "groupby" | "aggregate" | "filter" | "correlation" | "trend" | "describe",
      "group_column": "<column name or null>",
      "metric": "<column name or null>",
      "aggregation": "sum" | "mean" | "count" | "min" | "max" | null,
      "filter": "<simple condition or null>",
      "chart": "bar" | "line" | "scatter" | "histogram" | "none",
      "title": "<A short title for the chart/analysis>"
    }}
    
    If the question cannot be answered using the available columns, return {{"operation": "describe", "title": "Cannot Answer", "chart": "none", "group_column": null, "metric": null, "aggregation": null, "filter": null}}.
    """
    
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            temperature=0.1
        )
    )
    
    try:
        plan = json.loads(response.text)
        return plan
    except Exception as e:
        raise ValueError(f"Failed to parse Gemini response as JSON: {response.text}") from e

def generate_insight(question: str, analysis_result: str) -> str:
    client = get_client()
    model_name = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    
    prompt = f"""
    You are an AI Data Analyst. You executed an analysis plan based on a user's question.
    
    User Question: "{question}"
    Analysis Result (CSV format representation of top rows):
    {analysis_result}
    
    Provide a concise explanation containing:
    1. A direct answer to the user's question.
    2. Important numbers.
    3. A notable trend or pattern.
    4. One useful observation.
    
    Be extremely concise. Do NOT hallucinate information not present in the result.
    """
    
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2
        )
    )
    
    return response.text
