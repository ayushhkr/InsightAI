"""
llm.py
Handles interactions with the Gemini API (via google-genai); converts natural language queries into analysis plans.
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

def generate_analysis_plan(question: str, dataframe_metadata: dict, history: list = None) -> dict:
    client = get_client()
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    
    if history is None:
        history = []
        
    history_str = ""
    for idx, turn in enumerate(history):
        history_str += f"\nTurn {idx+1}:\nUser: {turn['question']}\nPlan: {json.dumps(turn.get('plan', {}))}\n"

    prompt = f"""
    You are an AI Data Analyst. Your job is to convert a user question into a structured JSON analysis plan.
    
    Dataset Metadata:
    Columns: {dataframe_metadata['columns']}
    Data Types: {dataframe_metadata['data_types']}
    Numerical Columns: {dataframe_metadata['numerical_cols']}
    Categorical Columns: {dataframe_metadata['categorical_cols']}
    Datetime Columns: {dataframe_metadata['datetime_cols']}
    
    Conversation History:
    {history_str}
    
    Current User Question: "{question}"
    
    Return ONLY a valid JSON object with the following schema:
    {{
      "operation": "groupby" | "aggregate" | "filter" | "correlation" | "trend" | "describe",
      "group_column": "<column name or null>",
      "metric": "<column name or null>",
      "aggregation": "sum" | "mean" | "count" | "min" | "max" | null,
      "filter": "<simple condition or null>",
      "sort": "ascending" | "descending" | null,
      "top_n": <integer or null>,
      "chart": "bar" | "line" | "scatter" | "histogram" | "none",
      "title": "<A short title for the chart/analysis>"
    }}
    
    Rules:
    - Use the conversation history to understand context if the query is a follow-up (e.g. "What about quantity?").
    - If the user implies looking at the highest or lowest values, use `sort` and `top_n` respectively.
    - Select a sensible `chart` type (e.g. bar for categorical comparison, line for trend/time-based, scatter for 2 numerical, histogram for distribution).
    - If a column like 'revenue' is asked for but does not exist in columns, you can specify `"metric": "revenue"` and the analyzer will attempt to calculate it automatically if possible.
    
    If the question cannot be answered, return {{"operation": "describe", "title": "Cannot Answer", "chart": "none", "group_column": null, "metric": null, "aggregation": null, "filter": null, "sort": null, "top_n": null}}.
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

def generate_insight(question: str, analysis_result: str, history: list = None) -> str:
    client = get_client()
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    
    if history is None:
        history = []
        
    history_str = ""
    for idx, turn in enumerate(history):
        history_str += f"\nTurn {idx+1}:\nUser: {turn['question']}\nAI Insight: {turn.get('insight', '')}\n"

    prompt = f"""
    You are an AI Data Analyst. You executed an analysis plan based on a user's question.
    
    Conversation History:
    {history_str}
    
    Current User Question: "{question}"
    Analysis Result (CSV format representation of top rows):
    {analysis_result}
    
    Provide a concise explanation containing:
    1. A direct answer to the user's question.
    2. Important numbers.
    3. A notable trend, pattern, or key finding.
    
    Be extremely concise. Do NOT hallucinate information not present in the result. Keep it conversational but professional.
    """
    
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2
        )
    )
    
    return response.text
