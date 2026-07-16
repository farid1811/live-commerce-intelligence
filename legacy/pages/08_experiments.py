import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from src.utils.ui_utils import apply_global_styles, render_page_header, render_alert
from src.services.training_service import TrainingService

apply_global_styles()

render_page_header("Experiments Center", "Compare Linear, Polynomial, and Logarithmic architectures trained across different dataset pre-processings.")

# Active dataset
df = st.session_state.get("dataset")
if df is None:
    st.warning("No dataset loaded. Please upload a dataset in the Dataset Manager.")
    st.stop()

# Setup services
model_repo = st.session_state.get("model_repo")
training_service = TrainingService(model_repo)

# -----------------------------------------------------
# ROW 1: RUN BULK BENCHMARK
# -----------------------------------------------------
st.markdown("### Bulk Benchmark Experiment Runner")
st.markdown("Select a dataset segment to train and evaluate all three architectures (Linear, Polynomial, Logarithmic) simultaneously.")

c_sel, c_run = st.columns([3, 1])

with c_sel:
    bench_segment = st.selectbox(
        "Select Target Dataset Segment for Benchmark",
        ["All Data", "Cleaned Data", "Weekly Data", "Monthly Data"],
        index=0
    )

with c_run:
    st.markdown("<div style='height:28px;'></div>", unsafe_allow_html=True)
    run_bench_btn = st.button("⚡ Run Comparative Benchmark", use_container_width=True)

if run_bench_btn:
    with st.spinner(f"Training architectures on '{bench_segment}'..."):
        # Train Linear
        training_service.train_model(df, "Linear", bench_segment)
        # Train Polynomial
        training_service.train_model(df, "Polynomial", bench_segment)
        # Train Logarithmic
        training_service.train_model(df, "Logarithmic", bench_segment)
        
    render_alert(f"Successfully trained and registered all 3 models on segment: **{bench_segment}**!", type="success")
    st.rerun()

# -----------------------------------------------------
# ROW 2: COMPARATIVE BAR CHARTS
# -----------------------------------------------------
st.markdown("---")
st.markdown("### Architectural Metrics Comparison")

models_meta = model_repo.list_models()
if len(models_meta) < 2:
    st.info("Train multiple models using the Experiment Runner above or the Training Center to compare metrics side-by-side.")
else:
    # Build comparative dataframe
    comp_records = []
    for m in models_meta:
        comp_records.append({
            "Name": m["name"],
            "Type": m["model_type"],
            "Segment": m["data_type"],
            "R² Score": m["metrics"].get("r2", 0.0),
            "MAE": m["metrics"].get("mae", 0.0),
            "RMSE": m["metrics"].get("rmse", 0.0),
            "MAPE (%)": m["metrics"].get("mape", 0.0)
        })
    
    comp_df = pd.DataFrame(comp_records)
    
    # Filter selection to compare
    st.markdown("#### Filter Comparison View")
    selected_names = st.multiselect(
        "Select Models to Compare",
        comp_df["Name"].tolist(),
        default=comp_df["Name"].tolist()[:5] # default compare top 5
    )
    
    filtered_comp_df = comp_df[comp_df["Name"].isin(selected_names)]
    
    if len(filtered_comp_df) > 0:
        # Side by side charts for R2 and RMSE
        col_c1, col_c2 = st.columns(2)
        
        with col_c1:
            fig_r2 = go.Figure(data=go.Bar(
                x=filtered_comp_df["Name"],
                y=filtered_comp_df["R² Score"],
                marker_color='#2563EB',
                name='R² Score'
            ))
            fig_r2.update_layout(
                title="R² Comparison (Higher is Better)",
                xaxis_title="Model Spec",
                yaxis_title="R² Score",
                template="simple_white",
                height=320,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig_r2, use_container_width=True)
            
        with col_c2:
            fig_rmse = go.Figure(data=go.Bar(
                x=filtered_comp_df["Name"],
                y=filtered_comp_df["RMSE"],
                marker_color='#EF4444',
                name='RMSE'
            ))
            fig_rmse.update_layout(
                title="RMSE Comparison (Lower is Better)",
                xaxis_title="Model Spec",
                yaxis_title="RMSE (Products Sold)",
                template="simple_white",
                height=320,
                margin=dict(l=40, r=40, t=40, b=40)
            )
            st.plotly_chart(fig_rmse, use_container_width=True)
            
        # Metrics Table
        st.markdown("#### Benchmark Metrics Table")
        st.table(filtered_comp_df.style.highlight_max(subset=["R² Score"], color='#D1FAE5').highlight_min(subset=["RMSE", "MAE", "MAPE (%)"], color='#D1FAE5'))
