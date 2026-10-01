"""Optional live LLM-planning evaluation; excluded from normal pytest and CI."""
from __future__ import annotations

from eval.benchmark import controlled_dataset
from eval.runner import format_results, run_evaluation
from src.llm import generate_analysis_plan
from src.profiler import profile_dataset


def run_llm_planning_evaluation() -> dict:
    """Evaluate live plan generation only when a caller has configured Groq.

    The normal benchmark deliberately does not import this function or make any
    network request.
    """
    profile = profile_dataset(controlled_dataset())
    metadata = {key: profile[key] for key in ("dataset_fingerprint", "columns", "data_types", "numerical_cols", "categorical_cols", "datetime_cols")}
    return run_evaluation(planner=lambda case: generate_analysis_plan(case.question, metadata))


if __name__ == "__main__":
    print(format_results(run_llm_planning_evaluation()))
