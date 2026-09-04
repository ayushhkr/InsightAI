"""
app.py
Main Streamlit UI application for InsightAI.
"""
import streamlit as st
import pandas as pd
from src.profiler import profile_dataset
from src.visualizer import create_bar_chart, create_line_chart, create_scatter_chart, create_histogram, create_chart
from src.llm import generate_analysis_plan, generate_insight, clean_markdown_output, GeminiBusyError
from src.analyzer import execute_analysis_plan
from src.anomaly import detect_anomalies, build_anomaly_evidence
from src.report import build_report

st.set_page_config(page_title="InsightAI - Data Analyst", page_icon="📊", layout="wide")

def apply_dashboard_styles():
    st.markdown("""
    <style>
    .block-container { max-width: 1280px; padding-top: 2.25rem; padding-bottom: 3rem; }
    .stApp h1 { font-size: clamp(2rem, 3vw, 2.5rem); letter-spacing: -0.04em; margin-bottom: 0.1rem; }
    .insightai-eyebrow { color: #0f766e; font-size: 0.74rem; font-weight: 700; letter-spacing: 0.12em; margin: 0 0 0.35rem; }
    .insightai-subtitle { color: var(--secondary-text-color); font-size: 1.05rem; margin: 0 0 2rem; max-width: 46rem; }
    .section-heading { font-size: 1.4rem; font-weight: 700; margin: 0; }
    .section-copy { color: var(--secondary-text-color); margin: 0.25rem 0 1.15rem; }
    [data-testid="stMetric"] { border: 1px solid rgba(128, 128, 128, 0.22); border-radius: 0.8rem; padding: 0.85rem 1rem; background: rgba(15, 118, 110, 0.05); }
    [data-testid="stMetricLabel"] { font-size: 0.78rem; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; }
    [data-testid="stDataFrame"], [data-testid="stPlotlyChart"] { border: 1px solid rgba(128, 128, 128, 0.2); border-radius: 0.75rem; overflow: hidden; }
    [data-testid="stChatMessage"] { border: 1px solid rgba(128, 128, 128, 0.18); border-radius: 0.85rem; padding: 0.6rem 0.8rem; }
    [data-testid="stExpander"] { border-radius: 0.75rem; }
    .stButton > button, .stDownloadButton > button { border-radius: 0.55rem; font-weight: 650; min-height: 2.55rem; }
    .stDownloadButton > button { border-color: #0f766e; color: #0f766e; }
    </style>
    """, unsafe_allow_html=True)

def section_heading(title: str, description: str):
    st.markdown(
        f'<p class="section-heading">{title}</p><p class="section-copy">{description}</p>',
        unsafe_allow_html=True,
    )

def initialize_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def main():
    apply_dashboard_styles()
    st.markdown('<p class="insightai-eyebrow">AI ANALYTICS WORKSPACE</p>', unsafe_allow_html=True)
    st.title("📊 InsightAI")
    st.markdown('<p class="insightai-subtitle">AI-powered data analysis and revenue intelligence</p>', unsafe_allow_html=True)
    
    initialize_chat_history()
    
    # Upload section
    uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # Rehydrate context on new upload or load
            if "df" not in st.session_state or st.session_state.get("uploaded_filename") != uploaded_file.name:
                df = pd.read_csv(uploaded_file)
                st.session_state["df"] = df
                st.session_state["uploaded_filename"] = uploaded_file.name
                st.session_state["profile"] = profile_dataset(df)
                st.session_state.chat_history = []
                
            df = st.session_state["df"]
            profile = st.session_state["profile"]
            
            if df.empty:
                st.error("The uploaded CSV is empty. Please upload a valid dataset.")
                return
                
            # Dashboard KPI Cards
            st.divider()
            section_heading("Dataset Overview", "A compact read on shape, completeness, and data quality.")
            kpi_col1, kpi_col2, kpi_col3, kpi_col4 = st.columns(4)
            
            kpi_col1.metric("Rows", f"{profile['num_rows']:,}")
            kpi_col2.metric("Columns", f"{profile['num_cols']:,}")
            
            missing_val_count = sum(profile["missing_counts"].values())
            dup_count = profile["duplicate_rows"]
            
            kpi_col3.metric("Missing Values", f"{missing_val_count:,}", 
                            delta="Perfect" if missing_val_count == 0 else "Action Needed",
                            delta_color="normal" if missing_val_count == 0 else "inverse")
                            
            kpi_col4.metric("Duplicate Rows", f"{dup_count:,}",
                            delta="Clean" if dup_count == 0 else "Review Needed",
                            delta_color="normal" if dup_count == 0 else "inverse")

            ask_section = st.container()
            results_section = st.container()
            explore_section = st.container()
            anomaly_section = st.container()
            details_section = st.container()

            with details_section.expander("Dataset Details", expanded=False):
                # --- Section 1: Dataset Preview ---
                st.subheader("Dataset Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # --- Section 2: Column Information ---
                st.subheader("Column Information")
                col_info_df = pd.DataFrame({
                    "Data Type": profile["data_types"],
                    "Missing Values": profile["missing_counts"]
                })
                st.dataframe(col_info_df, use_container_width=True)
                
                # --- Section 3: Summary Statistics ---
                st.subheader("Statistics")
                if profile["descriptive_statistics"]:
                    stats_df = pd.DataFrame(profile["descriptive_statistics"])
                    st.dataframe(stats_df, use_container_width=True)
                else:
                    st.info("No numerical columns available for summary statistics.")
                    
                # --- Section 4: Data Quality ---
                st.subheader("Data Quality")
                dq_col1, dq_col2 = st.columns(2)
                
                with dq_col1:
                    if profile['duplicate_rows'] == 0:
                        st.success("✅ **No duplicate rows found.**")
                    else:
                        st.warning(f"⚠️ **{profile['duplicate_rows']:,} duplicate rows** detected.")
                        
                with dq_col2:
                    if sum(profile['missing_counts'].values()) == 0:
                        st.success("✅ **No missing values found in any column.**")
                    else:
                        st.warning(f"⚠️ **{sum(profile['missing_counts'].values()):,} missing values** detected across the dataset.")
                
                # --- Explore Data ---
                explore_section.divider()
                explore_section.subheader("Explore Data")
                explore_section.caption("Build a quick visual check with the existing dataset fields.")
                if profile["num_cols"] > 0:
                    explore_col1, explore_col2, explore_col3 = explore_section.columns(3)
                    with explore_col1:
                        chart_type = st.selectbox("Select Chart Type", ["Bar", "Line", "Scatter", "Histogram"])
                    with explore_col2:
                        x_col = st.selectbox("Select X-axis", profile["columns"])
                    with explore_col3:
                        if chart_type != "Histogram":
                            y_col = st.selectbox("Select Y-axis", profile["columns"], index=min(1, len(profile["columns"])-1))
                        else:
                            y_col = None
                            
                    # Rendering chart
                    try:
                        if chart_type == "Bar":
                            fig = create_bar_chart(df, x_col, y_col)
                        elif chart_type == "Line":
                            fig = create_line_chart(df, x_col, y_col)
                        elif chart_type == "Scatter":
                            fig = create_scatter_chart(df, x_col, y_col)
                        elif chart_type == "Histogram":
                            fig = create_histogram(df, x_col)
                            
                        explore_section.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        explore_section.error(f"Could not generate {chart_type} chart. Error: {e}")

            # --- Anomaly Detection ---
            with anomaly_section:
                section_heading("Anomaly Detection", "Scan numerical features for unusual patterns with Isolation Forest.")
            
            if anomaly_section.button("Detect Anomalies", type="primary"):
                with anomaly_section.spinner("Scanning dataset for anomalies..."):
                    st.session_state["anomaly_results"] = detect_anomalies(df)
                    st.session_state["anomaly_explanation"] = None
                    
            if "anomaly_results" in st.session_state:
                anomaly_results = st.session_state["anomaly_results"]
                total = anomaly_results.get("total_anomalies", 0)
                
                if total == 0:
                    anomaly_section.success("✅ No numerical outliers were detected in this dataset.")
                else:
                    anomaly_section.error(f"🚨 **Found {total:,} anomalous rows**")
                    
                    anomaly_section.markdown("#### Model Features")
                    cols = anomaly_results.get("columns", {})
                    for col_name, stats in cols.items():
                        if stats["anomaly_count"] > 0:
                            anomaly_section.write(f"- **`{col_name}`**: included in the model "
                                                  f"*({stats['missing_values_imputed']} missing values median-imputed)*")
                                     
                    anomaly_section.markdown("#### Anomalous Records Preview")
                    anomaly_idx = anomaly_results.get("anomalous_indices", [])
                    anomaly_section.dataframe(df.loc[anomaly_idx], use_container_width=True)
                    
                    evidence = build_anomaly_evidence(df, anomaly_results)
                    
                    if anomaly_section.button("Explain Anomalies with AI"):
                        with anomaly_section.spinner("Analyzing anomalies with Gemini..."):
                            try:
                                from src.llm import explain_anomalies
                                st.session_state["anomaly_explanation"] = explain_anomalies(evidence)
                            except GeminiBusyError:
                                anomaly_section.error("Gemini is temporarily busy. Please try again in a moment.")
                            except Exception as e:
                                anomaly_section.error(f"Failed to generate explanation. {e}")
                                
                    if st.session_state.get("anomaly_explanation"):
                        anomaly_section.markdown(st.session_state["anomaly_explanation"])
                        
                    anomaly_section.write("### Ask about these anomalies")
                    anomaly_q = anomaly_section.text_input("Ask a follow-up question", placeholder="Why was row 13 flagged?", label_visibility="collapsed")
                    
                    if anomaly_section.button("Ask AI about Anomalies"):
                        if not anomaly_q.strip():
                            anomaly_section.warning("Please enter a question.")
                        else:
                            with anomaly_section.spinner("Asking Gemini..."):
                                try:
                                    from src.llm import answer_anomaly_question
                                    ans = answer_anomaly_question(anomaly_q, evidence)
                                    st.session_state["anomaly_qa_answer"] = ans
                                except GeminiBusyError:
                                    anomaly_section.error("Gemini is temporarily busy. Please try again in a moment.")
                                except Exception as e:
                                    anomaly_section.error(f"Failed to generate answer. {e}")
                                    
                    if st.session_state.get("anomaly_qa_answer"):
                        anomaly_section.markdown(st.session_state["anomaly_qa_answer"])
                        
                    with anomaly_section.expander("Technical Details"):
                        anomaly_section.json(evidence)

            # --- AI Analysis (Conversational) ---
            
            colA, colB = ask_section.columns([0.8, 0.2])
            with colA:
                section_heading("Ask Your Data", "Ask a question in plain language to generate an analysis, chart, and AI insight.")
            with colB:
                if len(st.session_state.chat_history) > 0:
                    if st.button("🗑️ Clear Chat History"):
                        st.session_state.chat_history = []
                        st.rerun()

            with ask_section:
                user_question = st.chat_input("Ask a question about your dataset (e.g. Which region generated the most revenue?)")

            # Render Past History
            if st.session_state.chat_history:
                with results_section:
                    section_heading("Analysis Results", "Review prior questions, generated insights, charts, and supporting data.")
            for turn in st.session_state.chat_history:
                with results_section.chat_message("user"):
                    st.write(turn["question"])
                with results_section.chat_message("assistant"):
                    st.markdown(clean_markdown_output(turn["insight"]))
                    
                    if turn.get('res_df') is not None and not turn['res_df'].empty:
                        with st.expander("📊 View Data & Chart Details"):
                            st.markdown(f"**{turn.get('plan', {}).get('title', 'Analysis Results')}**")
                            fig = create_chart(turn['res_df'], turn['plan'])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                            st.dataframe(turn['res_df'], use_container_width=True)
                            
                            st.markdown("##### 🔍 How I Analyzed This:")
                            st.json(turn['plan'])

            with results_section.container(border=True):
                st.caption("Export the complete workspace, including the dataset profile, analyses, and anomaly findings.")
                report_slot = st.empty()

            if user_question:
                # Show instantly
                with results_section.chat_message("user"):
                    st.write(user_question)
                    
                with results_section.chat_message("assistant"):
                    with st.spinner("Analyzing with AI..."):
                        try:
                            # 1. Provide Context
                            metadata = {
                                "columns": profile["columns"],
                                "data_types": profile["data_types"],
                                "numerical_cols": profile["numerical_cols"],
                                "categorical_cols": profile["categorical_cols"],
                                "datetime_cols": profile["datetime_cols"]
                            }
                            recent_history = st.session_state.chat_history[-10:]
                            
                            # 2. Get Plan
                            plan = generate_analysis_plan(user_question, metadata, history=recent_history)
                            
                            # 3. Execute
                            res_df, exe_meta = execute_analysis_plan(df, plan)
                            
                            # 4. Display
                            if "warning" in exe_meta:
                                st.warning(exe_meta["warning"])
                                
                            small_res = res_df.head(20).to_csv(index=False)
                            insight = generate_insight(user_question, small_res, history=recent_history)
                            cleaned_insight = clean_markdown_output(insight)
                            
                            st.markdown(cleaned_insight)
                            
                            with st.expander("📊 View Data & Chart Details", expanded=True):
                                st.markdown(f"**{plan.get('title', 'Analysis Results')}**")
                                fig = create_chart(res_df, plan)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True, theme="streamlit")
                                st.dataframe(res_df, use_container_width=True)
                                
                                st.markdown("##### 🔍 How I Analyzed This:")
                                st.json(plan)

                            # Append history
                            st.session_state.chat_history.append({
                                "question": user_question,
                                "plan": plan,
                                "res_df": res_df,
                                "insight": cleaned_insight
                            })
                            
                            # Trim to 10
                            if len(st.session_state.chat_history) > 10:
                                st.session_state.chat_history = st.session_state.chat_history[-10:]
                                
                        except GeminiBusyError:
                            st.error("Gemini is temporarily busy. Please try again in a moment.")
                            st.stop()
                        except ValueError as e:
                            st.error(f"Analysis could not be completed: {e}")
                            st.stop()
                        except Exception as e:
                            st.error("An unexpected error occurred during AI analysis. Please try again.")
                            st.stop()

            report_html = build_report(
                profile=profile,
                dataset=df,
                analysis_history=st.session_state.chat_history,
                anomaly_result=st.session_state.get("anomaly_results"),
                anomaly_explanation=st.session_state.get("anomaly_explanation"),
            )
            report_slot.download_button(
                "Download Full Report",
                data=report_html,
                file_name="insightai_full_report.html",
                mime="text/html",
            )

        except pd.errors.EmptyDataError:
            st.error("Uploaded file is empty or invalid.")
        except Exception as e:
            st.error(f"An error occurred reading the file: {e}")

if __name__ == "__main__":
    main()
