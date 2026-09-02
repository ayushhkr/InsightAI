import os
from dotenv import load_dotenv
import pandas as pd
from src.profiler import profile_dataset
from src.llm import generate_analysis_plan, generate_insight
from src.analyzer import execute_analysis_plan

def run_tests():
    load_dotenv()
    df = pd.read_csv("data/sample_sales.csv")
    
    # Calculate revenue if needed (quantity * price), but wait, there is no revenue column!
    # A user might ask "revenue", Gemini might try to use price * quantity, or just sum price.
    # We will see what happens!
    
    profile = profile_dataset(df)
    metadata = {
        "columns": profile["columns"],
        "data_types": profile["data_types"],
        "numerical_cols": profile["numerical_cols"],
        "categorical_cols": profile["categorical_cols"],
        "datetime_cols": profile["datetime_cols"]
    }
    
    queries = [
        "Which region generated the most revenue?",
        "What are the top 5 products by sales?",
        "Show sales trends over time.",
        "What is the average price by category?",
        "Which region has the highest quantity sold?"
    ]
    
    for q in queries:
        print(f"\n--- Question: {q}")
        try:
            plan = generate_analysis_plan(q, metadata)
            print("Plan:", plan)
            res_df, exe_meta = execute_analysis_plan(df, plan)
            print("Result Head:\n", res_df.head(2))
            
            insight = generate_insight(q, res_df.head(5).to_csv(index=False))
            print("Insight:", insight.replace('\n', ' ')[:150], "...")
        except Exception as e:
            print(f"FAILED: {e}")

if __name__ == "__main__":
    run_tests()
