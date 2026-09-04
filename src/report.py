"""HTML report rendering for completed InsightAI analyses."""
from html import escape

import pandas as pd
from src.visualizer import create_chart


def _paragraphs(text) -> str:
    return "<br>".join(escape(str(text)).splitlines())


def build_report(
    profile: dict,
    question: str | None = None,
    analysis_result: pd.DataFrame | None = None,
    insight: str | None = None,
    anomaly_result: dict | None = None,
    anomaly_explanation: str | None = None,
    dataset: pd.DataFrame | None = None,
    analysis_history: list | None = None,
) -> str:
    """Build a self-contained, safely escaped HTML report from existing data."""
    sections = [
        "<h1>InsightAI Analysis Report</h1>",
        "<h2>Dataset Summary</h2>",
        "<ul>"
        f"<li>Rows: {escape(str(profile.get('num_rows', '')))}</li>"
        f"<li>Columns: {escape(str(profile.get('num_cols', '')))}</li>"
        f"<li>Fields: {escape(', '.join(map(str, profile.get('columns', []))))}</li>"
        "</ul>",
    ]

    if profile.get("missing_counts") or profile.get("duplicate_rows") is not None:
        missing_values = sum(profile.get("missing_counts", {}).values())
        sections.append(
            "<h2>Data Quality</h2>"
            "<ul>"
            f"<li>Missing values: {escape(str(missing_values))}</li>"
            f"<li>Duplicate rows: {escape(str(profile.get('duplicate_rows', 0)))}</li>"
            "</ul>"
        )

    if profile.get("data_types"):
        column_details = pd.DataFrame({
            "Column": list(profile["data_types"]),
            "Data Type": list(profile["data_types"].values()),
            "Missing Values": [profile.get("missing_counts", {}).get(column, 0) for column in profile["data_types"]],
        })
        sections.extend(["<h2>Column Details</h2>", column_details.to_html(index=False, escape=True, border=0)])

    if profile.get("descriptive_statistics"):
        statistics = pd.DataFrame(profile["descriptive_statistics"]).T
        sections.extend(["<h2>Descriptive Statistics</h2>", statistics.to_html(escape=True, border=0)])

    if dataset is not None and not dataset.empty:
        sections.extend(["<h2>Dataset Preview</h2>", dataset.head(10).to_html(index=False, escape=True, border=0)])

    if question:
        sections.append(f"<h2>Question</h2><p>{_paragraphs(question)}</p>")

    if analysis_result is not None and not analysis_result.empty:
        sections.extend([
            "<h2>Analysis Result</h2>",
            analysis_result.to_html(index=False, escape=True, border=0),
        ])

    if insight:
        sections.append(f"<h2>AI Insight</h2><p>{_paragraphs(insight)}</p>")

    if analysis_history:
        sections.append("<h2>Analysis History</h2>")
        include_plotlyjs = True
        for number, turn in enumerate(analysis_history, start=1):
            sections.append(f"<h3>Analysis {number}</h3>")
            if turn.get("question"):
                sections.append(f"<h4>Question</h4><p>{_paragraphs(turn['question'])}</p>")
            if turn.get("insight"):
                sections.append(f"<h4>AI Insight</h4><p>{_paragraphs(turn['insight'])}</p>")
            result = turn.get("res_df")
            if result is not None and not result.empty:
                sections.append(result.to_html(index=False, escape=True, border=0))
                figure = create_chart(result, turn.get("plan", {}))
                if figure:
                    sections.append("<h4>Chart</h4>")
                    sections.append(figure.to_html(full_html=False, include_plotlyjs=include_plotlyjs))
                    include_plotlyjs = False

    if anomaly_result and anomaly_result.get("total_anomalies", 0):
        columns = anomaly_result.get("columns", {})
        sections.append(
            "<h2>Anomaly Summary</h2>"
            "<ul>"
            f"<li>Method: {escape(str(anomaly_result.get('method', '')))}</li>"
            f"<li>Anomalous rows: {escape(str(anomaly_result.get('total_anomalies', 0)))}</li>"
            f"<li>Model features: {escape(', '.join(map(str, columns)))}</li>"
            "</ul>"
        )

    if anomaly_explanation:
        sections.append(f"<h2>Anomaly Explanation</h2><p>{_paragraphs(anomaly_explanation)}</p>")

    body = "\n".join(sections)
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>InsightAI Analysis Report</title>
<style>
body {{ font-family: Arial, sans-serif; line-height: 1.5; margin: 2rem; color: #222; }}
table {{ border-collapse: collapse; width: 100%; }}
th, td {{ border: 1px solid #ccc; padding: 0.5rem; text-align: left; }}
th {{ background: #f2f2f2; }}
</style>
</head>
<body>
{body}
</body>
</html>"""
