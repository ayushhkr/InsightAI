"""
llm.py
Handles interactions with the Gemini API (via google-genai); converts natural language queries into analysis plans.
"""
import os
import json
import re
import time
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()

class GeminiBusyError(Exception):
    """Raised when the Gemini API is temporarily unavailable or experiencing high demand after retries."""
    pass

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

def call_gemini_with_retry(client, model_name: str, contents: str, config: types.GenerateContentConfig, max_retries: int = 3):
    """
    Calls Gemini API with automatic exponential backoff retries for 503/UNAVAILABLE temporary errors.
    Backoff delays: 2s -> 4s -> 8s.
    Does NOT retry 400, 401, 403, 404 client/authentication errors.
    """
    delays = [2, 4, 8]
    
    for attempt in range(max_retries + 1):
        try:
            return client.models.generate_content(
                model=model_name,
                contents=contents,
                config=config
            )
        except Exception as e:
            err_str = str(e).upper()
            status_code = getattr(e, 'code', None) or getattr(e, 'status_code', None)
            
            # Check if non-retryable error (400, 401, 403, 404)
            is_non_retryable = False
            if status_code in (400, 401, 403, 404):
                is_non_retryable = True
            elif any(code_str in err_str for code_str in ["400", "401", "403", "404", "INVALID_ARGUMENT", "PERMISSION_DENIED", "NOT_FOUND"]):
                is_non_retryable = True
                
            if is_non_retryable:
                raise e
                
            # Check if retryable 503 or transient server busy error
            is_transient_error = (
                status_code == 503 or
                "503" in err_str or
                "UNAVAILABLE" in err_str or
                "HIGH DEMAND" in err_str or
                "TEMPORARILY" in err_str or
                "RESOURCE_EXHAUSTED" in err_str or
                "OVERLOADED" in err_str
            )
            
            if is_transient_error and attempt < max_retries:
                delay = delays[attempt] if attempt < len(delays) else 8
                time.sleep(delay)
                continue
            elif is_transient_error:
                raise GeminiBusyError("Gemini is temporarily busy. Please try again in a moment.") from e
            else:
                raise e

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
    
    response = call_gemini_with_retry(
        client=client,
        model_name=model_name,
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
    
    response = call_gemini_with_retry(
        client=client,
        model_name=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2
        )
    )
    
    raw_text = response.text
    cleaned_text = clean_markdown_output(raw_text)
    return cleaned_text

def explain_anomalies(evidence: dict) -> str:
    """
    Generates a natural-language explanation of anomaly results deterministically mapped via IQR without recalculating statistics.
    """
    client = get_client()
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    
    prompt = f"""
    You are an AI Data Analyst communicating to a non-technical user.
    The following evidence was ALREADY calculated deterministically using the IQR (Interquartile Range) algorithm.
    
    Evidence Pack:
    {json.dumps(evidence, indent=2)}
    
    CRITICAL INSTRUCTIONS:
    1. Do NOT recalculate or invent anomaly results.
    2. Only describe facts contained in the evidence above.
    3. Refer to observations ONLY as "rows", "records", "values", or the exact column name. NEVER call them "orders", "transactions", "customers", "employees", etc.
    4. NEVER categorize columns as financial, operational, sales, performance, etc.
    5. Use exact column names from the evidence. Do not infer what a numerical column represents beyond its name and observed values.
    6. Do NOT claim a cause for an anomaly. You must explicitly state: "The available data does not establish the cause."
    7. Do not recommend checking formulas or contacting specific teams/departments/regions unless the evidence explicitly establishes they are relevant.
    8. Recommendations must be generic investigation steps grounded in the available evidence (e.g. "review the affected records", "verify the unusual values against the original source", "investigate whether multiple anomalous columns are related").
    9. Every numerical claim must come directly from the supplied evidence. Do not invent business context.
    10. Do not generate SQL, Python, formulas, or implementation code.
    11. The explanation should work unchanged in structure for completely unrelated datasets.
    
    Format your response EXACTLY with these sections:
    
    ### Summary
    [Briefly explain what was detected]
    
    ### Key Findings
    - [Concise evidence-based observation 1]
    - [Concise evidence-based observation 2]
    
    ### What This Could Mean
    [Explain the statistical significance in plain language without inventing causes]
    
    ### What to Investigate
    - [Practical investigation step 1 based only on available evidence]
    - [Practical investigation step 2 based only on available evidence]
    """
    
    response = call_gemini_with_retry(
        client=client,
        model_name=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2
        )
    )
    
    return clean_markdown_output(response.text)

def answer_anomaly_question(question: str, evidence: dict) -> str:
    """
    Answers a follow-up user question about the detected anomalies based purely on the generated evidence package.
    """
    client = get_client()
    model_name = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
    
    prompt = f"""
    You are an AI Data Analyst answering a follow-up question about detected anomalies.
    
    Evidence Pack (Calculated deterministically via IQR):
    {json.dumps(evidence, indent=2)}
    
    User Question: "{question}"
    
    CRITICAL INSTRUCTIONS:
    1. Answer ONLY using the supplied evidence. Never invent numbers.
    2. Use exact column names from the evidence.
    3. Explain statistical anomaly detection using the supplied IQR bounds (lower_bound, upper_bound) when relevant to the question.
    4. Never invent business/domain causes for why an anomaly occurred.
    5. Never refer to rows as orders, customers, transactions, employees, etc., unless explicitly stated in the evidence.
    6. If the evidence is insufficient to answer the question, explicitly say so.
    7. If asked WHY an anomaly happened in the real world, explicitly state that the supplied evidence does not establish the real-world cause.
    8. Do not generate SQL, Python, or implementation code.
    9. Keep the answer concise.
    """
    
    response = call_gemini_with_retry(
        client=client,
        model_name=model_name,
        contents=prompt,
        config=types.GenerateContentConfig(
            temperature=0.2
        )
    )
    
    return clean_markdown_output(response.text)
