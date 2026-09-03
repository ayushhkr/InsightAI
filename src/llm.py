"""
llm.py
Handles interactions with the Gemini API (via google-genai); converts natural language queries into analysis plans.
"""
import os
import json
import re
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

def clean_markdown_output(text: str) -> str:
    """
    Sanitizes and cleans AI markdown output:
    - Ensures section headers (**Key Findings:**, **Recommendations:**) are on their own lines.
    - Fixes attached section headers like '...94.**Recommendations:**1.' -> '...94.\n\n**Recommendations:**\n1.'
    - Cleans up weird multi-space artifacts.
    """
    if not text:
        return ""
    
    cleaned = text.strip()
    
    # Ensure **Key Findings:** starts on a new line
    cleaned = re.sub(r'([^\n])\s*\*\*Key\s*Findings\s*:\*\*', r'\1\n\n**Key Findings:**', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\*\*Key\s*Findings\s*:\*\*\s*', r'**Key Findings:**\n', cleaned, flags=re.IGNORECASE)
    
    # Ensure **Recommendations:** starts on a new line
    cleaned = re.sub(r'([^\n])\s*\*\*Recommendations\s*:\*\*', r'\1\n\n**Recommendations:**', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\*\*Recommendations\s*:\*\*\s*', r'**Recommendations:**\n', cleaned, flags=re.IGNORECASE)
    
    # Fix numbered lists touching **Recommendations:** (e.g. **Recommendations:**1. -> **Recommendations:**\n1. )
    cleaned = re.sub(r'(\*\*Recommendations:\*\*)\s*([0-9]+\.)', r'\1\n\2', cleaned)
    
    # Normalize clean line breaks (max 2 consecutive newlines)
    cleaned = re.sub(r'\n{3,}', '\n\n', cleaned)
    
    return cleaned.strip()

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
    - For strategy or business questions (e.g. "How can I increase sales in [X]?"), use `"operation": "groupby"`, set `"group_column"` to a categorical column like region, category, or product, `"metric"` to "revenue" (or quantity), `"aggregation"` to "sum", and `"sort"` to "descending".
    - Select a sensible `chart` type (e.g. bar for categorical comparison, line for trend/time-based, scatter for 2 numerical, histogram for distribution).
    - If a column like 'revenue' is asked for but does not exist in columns, specify `"metric": "revenue"` and the analyzer will attempt to calculate it automatically if price and quantity exist.
    
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
    Analysis Result (CSV format representation of dataset result):
    {analysis_result}
    
    CRITICAL FORMATTING INSTRUCTIONS:
    1. Write standard, plain English sentences. ALWAYS put normal spaces between words. NEVER concatenate words together (e.g. NEVER write "youshouldanalyze" or "in crease").
    2. Format numbers normally with standard currency or comma notation (e.g. "$5,643,356.55" or "5,643,356 units"). NEVER format numbers as "5, 643, 356.55" with spaces after commas.
    3. Structure your response EXACTLY as follows:

    [Short 1-sentence overall conclusion]

    **Key Findings:**
    - [Key finding 1 with exact number/metric]
    - [Key finding 2 with exact number/metric]
    - [Key finding 3 with exact number/metric]

    **Recommendations:**
    1. [Actionable recommendation 1]
    2. [Actionable recommendation 2]
    3. [Actionable recommendation 3]

    Rules:
    - Base all numbers ONLY on the Analysis Result provided above. Do NOT invent numbers or facts.
    - For simple questions (e.g. "Which region generated the least revenue?"), you can omit the Recommendations block and keep the response concise.
    - Make sure every heading (**Key Findings:** and **Recommendations:**) is on its own separate line.
    """
    
    response = client.models.generate_content(
        model=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2
        )
    )
    
    raw_text = response.text
    cleaned_text = clean_markdown_output(raw_text)
    return cleaned_text
