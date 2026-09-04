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

st.set_page_config(page_title="InsightAI - Data Analyst", page_icon="📊", layout="wide")

def initialize_chat_history():
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

def main():
    st.title("InsightAI")
    st.subheader("AI-Powered Data Analyst")
    
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
                
            # Metric cards
            col1, col2, col3, col4 = st.columns(4)
            col1.metric("Rows", profile["num_rows"])
            col2.metric("Columns", profile["num_cols"])
            col3.metric("Missing Values", sum(profile["missing_counts"].values()))
            col4.metric("Duplicate Rows", profile["duplicate_rows"])
            
            with st.expander("Explore Dashboard", expanded=False):
                # --- Section 1: Dataset Preview ---
                st.header("1. Dataset Preview")
                st.dataframe(df.head(10), use_container_width=True)
                
                # --- Section 2: Column Information ---
                st.header("2. Column Information")
                col_info_df = pd.DataFrame({
                    "Data Type": profile["data_types"],
                    "Missing Values": profile["missing_counts"]
                })
                st.dataframe(col_info_df, use_container_width=True)
                
                # --- Section 3: Summary Statistics ---
                st.header("3. Summary Statistics")
                if profile["descriptive_statistics"]:
                    stats_df = pd.DataFrame(profile["descriptive_statistics"])
                    st.dataframe(stats_df, use_container_width=True)
                else:
                    st.info("No numerical columns available for summary statistics.")
                    
                # --- Section 4: Data Quality ---
                st.header("4. Data Quality")
                st.write(f"**Total duplicate rows:** {profile['duplicate_rows']}")
                st.write(f"**Total missing values across all columns:** {sum(profile['missing_counts'].values())}")
                
                st.divider()
                
                # --- Explore Data ---
                st.header("Explore Data")
                if profile["num_cols"] > 0:
                    explore_col1, explore_col2, explore_col3 = st.columns(3)
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
                            
                        st.plotly_chart(fig, use_container_width=True)
                    except Exception as e:
                        st.error(f"Could not generate {chart_type} chart. Error: {e}")

            st.divider()
            
            # --- Anomaly Detection ---
            st.header("Anomaly Detection")
            if st.button("Detect Anomalies"):
                with st.spinner("Scanning dataset for anomalies..."):
                    anomaly_results = detect_anomalies(df)
                    
                    total = anomaly_results.get("total_anomalies", 0)
                    if total == 0:
                        st.success("No numerical outliers were detected in this dataset.")
                    else:
                        st.warning(f"Found {total} anomalous rows.")
                        
                        cols = anomaly_results.get("columns", {})
                        for col_name, stats in cols.items():
                            if stats["anomaly_count"] > 0:
                                st.write(f"- **{col_name}**: {stats['anomaly_count']} anomalies "
                                         f"(IQR Bounds: [{stats['lower_bound']}, {stats['upper_bound']}])")
                                         
                        st.write("### Anomalous Rows")
                        anomaly_idx = anomaly_results.get("anomalous_indices", [])
                        st.dataframe(df.loc[anomaly_idx], use_container_width=True)
                        
                        evidence = build_anomaly_evidence(df, anomaly_results)
                        with st.expander("Show Evidence"):
                            st.json(evidence)

            st.divider()
            
            # --- AI Analysis (Conversational) ---
            colA, colB = st.columns([0.8, 0.2])
            with colA:
                st.header("Ask your data")
            with colB:
                if len(st.session_state.chat_history) > 0:
                    if st.button("Clear conversation"):
                        st.session_state.chat_history = []
                        st.rerun()

            # Render Past History
            for turn in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.write(turn["question"])
                with st.chat_message("assistant"):
                    st.markdown(clean_markdown_output(turn["insight"]))
                    
                    if turn.get('res_df') is not None and not turn['res_df'].empty:
                        with st.expander("Analysis Results & Details"):
                            st.write(f"### {turn.get('plan', {}).get('title', 'Analysis Results')}")
                            fig = create_chart(turn['res_df'], turn['plan'])
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                            st.dataframe(turn['res_df'], use_container_width=True)
                            st.json(turn['plan'])
            
            # New message input
            user_question = st.chat_input("Ask a question about your dataset (e.g. Which region generated the most revenue?)")
            
            if user_question:
                # Show instantly
                with st.chat_message("user"):
                    st.write(user_question)
                    
                with st.chat_message("assistant"):
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
                            
                            with st.expander("Analysis Results & Details", expanded=True):
                                st.write(f"### {plan.get('title', 'Analysis Results')}")
                                fig = create_chart(res_df, plan)
                                if fig:
                                    st.plotly_chart(fig, use_container_width=True)
                                st.dataframe(res_df, use_container_width=True)
                                st.write("**How I analyzed this:**")
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
                            
        except pd.errors.EmptyDataError:
            st.error("Uploaded file is empty or invalid.")
        except Exception as e:
            st.error(f"An error occurred reading the file: {e}")

if __name__ == "__main__":
    main()
