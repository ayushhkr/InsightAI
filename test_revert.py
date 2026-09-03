import pandas as pd
import json
from src.profiler import profile_dataset
from src.analyzer import execute_analysis_plan
import src.router as router

def test_queries():
    df = pd.read_csv("data/sample_sales.csv")
    profile = profile_dataset(df)
    
    # Remove physical revenue column just in case to verify the working derived metric layer
    if "revenue" in df.columns:
        df = df.drop(columns=["revenue"])

    questions = [
        "Which region generated the least revenue?",
        "Which region generated the most revenue?",
        "Which region has the least sales and how do I increase sales in that region?"
    ]

    for user_question in questions:
        print(f"\n==========================================")
        print(f"QUESTION: {user_question}")
        print(f"==========================================")
        
        try:
            # 1. Route Intent
            route_info = router.route_intent(user_question, profile, [])
            plan = route_info["plan"]
            is_diagnostic = (route_info["route"] == "diagnostic")

            # 2. Execute
            res_df, exe_meta = execute_analysis_plan(df, plan)
            
            # 3. Output
            if is_diagnostic or plan.get("operation") == "diagnostic":
                ai_payload = json.dumps(exe_meta.get("evidence_pack", {}), indent=2)
                cleaned = f"DIAGNOSTIC EVIDENCE PACK DELIVERED:\n{ai_payload[:300]}..."
            else:
                cleaned = router.format_deterministic_answer(res_df, plan)

            print(f"Result:\n{cleaned}")
            
        except Exception as e:
            print(f"TEST FAILED: {e}")

if __name__ == "__main__":
    test_queries()
