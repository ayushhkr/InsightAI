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
    Sanitizes and cleans AI markdown output to fix common formatting glitches:
    - Removes duplicated/malformed asterisks like '* *' or '* *in'
    - Ensures proper space separation around bold markers '**'
    - Fixes attached words like '**intotalrevenue**' -> '**in total revenue**'
    """
    if not text:
        return ""
    
    cleaned = text
    
    # Replace multiple spaces/asterisks artifacts like '* *' with '**'
    cleaned = re.sub(r'\*\s+\*', '**', cleaned)
    
    # Fix concatenated bold markers where space is missing before text or after text
    # e.g., "* *intotalrevenue" -> "**in total revenue"
    cleaned = re.sub(r'\*\*([a-zA-Z0-9_]+)', r'** \1', cleaned)
    cleaned = re.sub(r'([a-zA-Z0-9_]+)\*\*', r'\1 **', cleaned)
    
    # Separate camelCase or glued words inside bold tags if needed
    # Insert spaces between lowercase followed by uppercase or glued words like 'intotalrevenue'
    # Specifically fix common pattern: 'in' 'total' 'revenue'
    cleaned = re.sub(r'\b(in|total|revenue|by|with|for|and|of|the|is|are|was|were|most|least|highest|lowest)([a-zA-Z]+)\b', r'\1 \2', cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r'\b(in|total|revenue|by|with|for|and|of|the|is|are|was|were|most|least|highest|lowest)([a-zA-Z]+)\b', r'\1 \2', cleaned, flags=re.IGNORECASE)
    
    # Clean up double spaces caused by regex replacements
    cleaned = re.sub(r' +', ' ', cleaned)
    cleaned = re.sub(r'\s*\*\*\s*', '**', cleaned)  # normalize bold tag attachments cleanly
    
    # Ensure markdown headers/lists have clean spacing
    cleaned = re.sub(r'(\n-[^\n]+)\n([^\n-])', r'\1\n\n\2', cleaned)
    
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
    - For questions asking "How can I increase sales in [X]?" or strategy/business questions about a specific region, category, or segment:
      * Choose `"operation": "groupby"`
      * `"group_column"`: choose product, category, or appropriate column to break down performance (e.g., product or category)
      * `"metric"`: "revenue" (or "quantity" / "price")
      * `"aggregation"`: "sum"
      * `"sort"`: "descending"
    - If the user implies looking at the highest or lowest values, use `sort` and `top_n` respectively.
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
    
    CRITICAL FORMATTING & CONTENT REQUIREMENTS:
    1. Base all observations and recommendations ONLY on the actual numbers present in the Analysis Result. Do NOT invent metrics, sales figures, or facts not present in the data.
    2. Format the response strictly using clean markdown as follows:
    
    [Short 1-sentence conclusion directly answering the question]

    **Key Findings:**
    - [Finding 1 + exact metric/number]
    - [Finding 2 + exact metric/number]
    - [Finding 3 + exact metric/number]

    **Recommendations:**
    1. [Actionable recommendation clearly based on the observed data]
    2. [Actionable recommendation clearly based on the observed data]
    3. [Actionable recommendation clearly based on the observed data]

    Rules:
    - Keep formatting crisp. Do not double asterisks or combine words into things like "* *intotalrevenue" or "**97million**". Write normal numbers with proper currency/units (e.g. "$1,200.00" or "10 units").
    - For non-recommendation questions (e.g., simple facts like "Which region generated the least revenue?"), you may omit the Recommendations block and keep the Key Findings concise.
    - Clearly distinguish observed facts from recommendations.
    - Stay concise and professional.
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
