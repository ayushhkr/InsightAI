import pandas as pd
from src.profiler import profile_dataset
from src.llm import generate_analysis_plan, generate_insight
from src.analyzer import execute_analysis_plan

def test():
    df = pd.read_csv("data/sample_sales.csv")
    profile = profile_dataset(df)
    metadata = {
        "columns": profile["columns"],
        "data_types": profile["data_types"],
        "numerical_cols": profile["numerical_cols"],
        "categorical_cols": profile["categorical_cols"],
        "datetime_cols": profile["datetime_cols"]
    }

    history = []
    queries = [
        "Which region generated the most revenue?",
        "What about quantity?",
        "What are the top 3 products by sales?",
    ]

    for q in queries:
        print(f"\n--- Question: {q}")
        try:
            plan = generate_analysis_plan(q, metadata, history)
            print("Plan:", plan)
            res_df, exe_meta = execute_analysis_plan(df, plan)
            print("Result Head:\n", res_df.head(3))
            
            insight = generate_insight(q, res_df.head(5).to_csv(index=False), history)
            print("Insight:", insight.replace('\n', ' ')[:100], "...")
            
            history.append({
                "question": q,
                "plan": plan,
                "res_df": res_df,
                "insight": insight
            })
        except Exception as e:
            print(f"FAILED: {e}")

if __name__ == "__main__":
    test()
