import pandas as pd
from src.profiler import profile_dataset
from src.llm import generate_analysis_plan, generate_insight, clean_markdown_output
from src.analyzer import execute_analysis_plan

def test_formatting():
    df = pd.read_csv("data/sample_sales.csv")
    profile = profile_dataset(df)
    metadata = {
        "columns": profile["columns"],
        "data_types": profile["data_types"],
        "numerical_cols": profile["numerical_cols"],
        "categorical_cols": profile["categorical_cols"],
        "datetime_cols": profile["datetime_cols"]
    }

    test_queries = [
        "How can I increase my sales in North America?",
        "Which region generated the least revenue?",
        "Top 5 products by revenue?"
    ]

    for q in test_queries:
        print(f"\n==========================================")
        print(f"QUERY: {q}")
        print(f"==========================================")
        try:
            plan = generate_analysis_plan(q, metadata)
            print(f"PLAN: {plan}")
            res_df, exe_meta = execute_analysis_plan(df, plan)
            print(f"RESULT DF:\n{res_df}\n")
            
            small_res = res_df.head(20).to_csv(index=False)
            insight = generate_insight(q, small_res)
            print(f"RAW INSIGHT:\n{insight}\n")
            
        except Exception as e:
            print(f"ERROR: {e}")

if __name__ == "__main__":
    test_formatting()
