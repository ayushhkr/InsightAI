"""
app.py
Main Streamlit UI application for InsightAI.
"""
import streamlit as st
import pandas as pd
from src.profiler import profile_dataset
from src.visualizer import create_bar_chart, create_line_chart, create_scatter_chart, create_histogram, create_chart
from src.llm import generate_analysis_plan, generate_insight
from src.analyzer import execute_analysis_plan

st.set_page_config(page_title="InsightAI - Data Analyst", page_icon="📊", layout="wide")

def main():
    st.title("InsightAI")
    st.subheader("AI-Powered Data Analyst")
    
    # Upload section
    uploaded_file = st.file_uploader("Upload a CSV dataset", type=["csv"])
    
    if uploaded_file is not None:
        try:
            # We only load it once to save performance/session state
            if "df" not in st.session_state or st.session_state.get("uploaded_filename") != uploaded_file.name:
                df = pd.read_csv(uploaded_file)
                st.session_state["df"] = df
                st.session_state["uploaded_filename"] = uploaded_file.name
                st.session_state["profile"] = profile_dataset(df)
                
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
            
            st.divider()
            
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
            
            # --- AI Analysis ---
            st.header("Ask your data")
            user_question = st.text_input("Ask a question about your dataset", placeholder="e.g. Which region generated the most revenue?")
            
            if st.button("Analyze"):
                if user_question:
                    with st.spinner("Analyzing with AI..."):
                        try:
                            # 1. Get Plan
                            metadata = {
                                "columns": profile["columns"],
                                "data_types": profile["data_types"],
                                "numerical_cols": profile["numerical_cols"],
                                "categorical_cols": profile["categorical_cols"],
                                "datetime_cols": profile["datetime_cols"]
                            }
                            plan = generate_analysis_plan(user_question, metadata)
                            
                            # Execute Plan
                            res_df, exe_meta = execute_analysis_plan(df, plan)
                            
                            # Display Result DF
                            st.write(f"### {plan.get('title', 'Analysis Results')}")
                            st.dataframe(res_df, use_container_width=True)
                            
                            if "warning" in exe_meta:
                                st.warning(exe_meta["warning"])
                                
                            # Visualizer
                            fig = create_chart(res_df, plan)
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                                
                            # AI Explanation
                            small_res = res_df.head(20).to_csv(index=False)
                            insight = generate_insight(user_question, small_res)
                            
                            st.success(f"**AI Insight:**\n\n{insight}")
                            
                            with st.expander("How I analyzed this"):
                                st.json(plan)
                                
                        except ValueError as e:
                            st.error(f"Analysis could not be completed: {e}")
                        except Exception as e:
                            st.error(f"An unexpected error occurred during AI analysis: {e}")
                    
        except pd.errors.EmptyDataError:
            st.error("Uploaded file is empty or invalid.")
        except Exception as e:
            st.error(f"An error occurred reading the file: {e}")

if __name__ == "__main__":
    main()
